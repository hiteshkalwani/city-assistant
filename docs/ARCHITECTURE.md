# Architecture

## Request flow

```
browser (React UI) ──► nginx: static UI + /api proxy (rate limit, least_conn) ──► backend replica ×3 (FastAPI)
                                                │
                                                ▼
                                   CityInformationAgent (LangChain, OpenAI tools)
                                                │  picks 0..n tools per turn
              ┌──────────────────┬──────────────┼────────────────┬──────────────────┐
              ▼                  ▼              ▼                ▼
        get_weather          get_time     get_city_facts   plan_city_visit
      (OpenWeatherMap)    (WorldTimeAPI)     (GeoDB)       (runs the other three)
              └──────────── Redis cache (shared across replicas) ─┘
```

| Layer | Path | Responsibility |
|---|---|---|
| HTTP | `backend/app/api/v1/` | Validation (Pydantic), request → agent input, response shaping, SSE |
| Agent | `backend/app/agents/` | Prompt, tool list, LangChain `AgentExecutor`; one place that talks to the LLM |
| Tools | `backend/app/tools/` | One external API each, cached, falls back to static data on failure |
| Core | `backend/app/core/` | Settings, logging, Redis client, cache decorator, LangSmith tracing |

## Design decisions

**1. One agent class, used by every endpoint.**
The chat endpoints previously built their own copy of the prompt and executor (twice), so the agent the tests exercised was not the agent production ran. Now both `POST /chat` and `GET /chat/stream` go through `CityInformationAgent`. The prompt changes in one place, and tests can patch at one boundary.

**2. Tools degrade instead of failing.**
Every tool returns static fallback data when its API key is missing or the upstream call fails. The assistant still answers (and says the data may be stale) instead of returning a 500 because a free-tier weather API timed out.

**3. Cache per tool, with TTLs based on how fast the data changes.**
Weather 30 min, time 5 min, facts 2 h. The cache lives in Redis so all replicas share it. If Redis is down the cache is skipped, not fatal.

**4. Tests never touch the network.**
`tests/conftest.py` patches `AgentExecutor.ainvoke` for every API test and sets dummy credentials. The full suite runs in ~4s with no API keys, so it runs in CI on every PR. Tests that need real tool behaviour patch deeper (the tool's `_arun`).

**5. Errors are logged in full but returned generic.**
Clients get `"The assistant could not process this request."`. The stack trace goes to logs. Exception text (which can include upstream URLs or key fragments) is only returned when `DEBUG=true`. Request headers are not logged.

**6. Streaming is unbuffered end to end.**
The UI reads `POST /api/v1/chat` (`stream: true`) with `fetch` and a stream reader instead of `EventSource`, because `EventSource` can only GET and can't send the conversation history. The backend sets `X-Accel-Buffering: no` so nginx forwards each event as it is produced. `GET /chat/stream` has its own nginx block, marked `^~` so the generic `/api/v1/` regex doesn't override it.

## Known limitations / next steps

- `/chat/stream` replays the finished answer word by word. True token streaming needs `agent.astream` events. It's marked `shortcut:` in `chat.py`.
- The streaming endpoint reports zero token usage. The non-streaming endpoint reports real usage via `OpenAICallbackHandler`.
- `/api/v1/health` calls OpenWeatherMap and GeoDB on every hit. Under a load-balancer health check this spends API quota. It should cache the result or use the plain `/health` liveness probe.
- Coverage is ~68%. The uncovered parts are mostly the Redis client and the tools' upstream branches.
- The UI shows the agent's markdown as plain text. A markdown renderer is the next step if answers get richer formatting.
- Agent is constructed per request (cheap, no I/O). Pool it if profiling shows it matters.

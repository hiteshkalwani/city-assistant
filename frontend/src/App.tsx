import { FormEvent, useEffect, useRef, useState } from 'react'
import { Message, Role, ToolCall, streamChat } from './chat'

interface Turn {
  role: Role
  content: string
  tools: ToolCall[]
  error?: string
}

// Each tool is a "line" on the route, coloured like a metro line.
const LINES: Record<string, { name: string; color: string }> = {
  get_weather: { name: 'Weather', color: 'var(--line-weather)' },
  get_time: { name: 'Local time', color: 'var(--line-time)' },
  get_city_facts: { name: 'City facts', color: 'var(--line-facts)' },
  plan_city_visit: { name: 'Visit plan', color: 'var(--line-plan)' },
}

const SUGGESTIONS = ['Plan my visit to Tokyo', "What's the weather in London?", 'What time is it in New York?']

function cityOf(call: ToolCall): string | undefined {
  return typeof call.input === 'object' ? (call.input.city as string | undefined) : undefined
}

function Route({ tools, busy }: { tools: ToolCall[]; busy: boolean }) {
  if (!tools.length && !busy) return null
  return (
    <ol className="route" aria-label="Tools the assistant used">
      {tools.map((call, i) => {
        const line = LINES[call.tool] ?? { name: call.tool, color: 'var(--muted)' }
        const city = cityOf(call)
        return (
          <li key={i} className="stop" style={{ '--line': line.color } as React.CSSProperties}>
            <details>
              <summary>
                <span className="stop-name">{line.name}</span>
                {city && <span className="stop-city">{city}</span>}
              </summary>
              <pre>{call.output}</pre>
            </details>
          </li>
        )
      })}
      {busy && <li className="stop stop-pending">Working…</li>}
    </ol>
  )
}

export default function App() {
  const [turns, setTurns] = useState<Turn[]>([])
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'end' })
  }, [turns])

  async function send(text: string) {
    text = text.trim()
    if (!text || busy) return

    const history: Message[] = [
      ...turns.filter((t) => t.content && !t.error).map(({ role, content }) => ({ role, content })),
      { role: 'user', content: text },
    ]
    const update = (fn: (t: Turn) => Turn) => setTurns((ts) => [...ts.slice(0, -1), fn(ts[ts.length - 1])])

    setDraft('')
    setBusy(true)
    setTurns((ts) => [...ts, { role: 'user', content: text, tools: [] }, { role: 'assistant', content: '', tools: [] }])
    try {
      for await (const e of streamChat(history)) {
        if (e.event === 'tool_call') update((t) => ({ ...t, tools: [...t.tools, e.data.tool_call] }))
        else if (e.event === 'token') update((t) => ({ ...t, content: t.content + e.data.token }))
        else if (e.event === 'complete') update((t) => ({ ...t, content: e.data.message.content }))
        else if (e.event === 'error') update((t) => ({ ...t, error: e.data.error }))
      }
    } catch (err) {
      update((t) => ({ ...t, error: `Couldn't reach the assistant: ${err instanceof Error ? err.message : err}` }))
    } finally {
      setBusy(false)
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault()
    void send(draft)
  }

  const last = turns.length - 1
  return (
    <div className="page">
      <header className="masthead">
        <h1>City Assistant</h1>
        <p>Ask about any city. You'll see which live data the assistant checked before it answers.</p>
      </header>

      <main className="log" aria-live="polite">
        {turns.length === 0 && (
          <div className="empty">
            <p>Try one of these:</p>
            <ul className="suggestions">
              {SUGGESTIONS.map((s) => (
                <li key={s}>
                  <button type="button" onClick={() => void send(s)}>
                    {s}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}

        {turns.map((t, i) =>
          t.role === 'user' ? (
            <p key={i} className="question">
              {t.content}
            </p>
          ) : (
            <section key={i} className="answer">
              <Route tools={t.tools} busy={busy && i === last && !t.content} />
              {/* shortcut: renders the agent's markdown as plain text, add a markdown renderer if formatting matters */}
              {t.content && <div className="answer-text">{t.content}</div>}
              {t.error && (
                <p className="error" role="alert">
                  {t.error} Send your question again to retry.
                </p>
              )}
            </section>
          ),
        )}
        <div ref={endRef} />
      </main>

      <form className="composer" onSubmit={onSubmit}>
        <label htmlFor="question" className="visually-hidden">
          Ask about a city
        </label>
        <input
          id="question"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Ask about a city"
          autoComplete="off"
        />
        <button type="submit" disabled={busy || !draft.trim()}>
          {busy ? 'Asking…' : 'Ask'}
        </button>
      </form>
    </div>
  )
}

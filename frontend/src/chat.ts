export type Role = 'user' | 'assistant'

export interface Message {
  role: Role
  content: string
}

export interface ToolCall {
  tool: string
  input: Record<string, unknown> | string
  output: string
}

/** Events sent by POST /api/v1/chat with `stream: true` (see backend/app/api/v1/chat.py). */
export type ChatEvent =
  | { event: 'start'; data: Record<string, unknown> }
  | { event: 'tool_call'; data: { tool_call: ToolCall } }
  | { event: 'token'; data: { token: string } }
  | { event: 'complete'; data: { message: Message } }
  | { event: 'error'; data: { error: string } }

/** Parses complete SSE events out of `buffer`; `rest` is the unfinished tail to prepend to the next chunk. */
export function parseSSE(buffer: string): { events: ChatEvent[]; rest: string } {
  const blocks = buffer.split('\n\n')
  const rest = blocks.pop() ?? ''
  const events = blocks.flatMap((block) =>
    block
      .split('\n')
      .filter((line) => line.startsWith('data: '))
      .map((line) => JSON.parse(line.slice('data: '.length)) as ChatEvent),
  )
  return { events, rest }
}

export async function* streamChat(messages: Message[], signal?: AbortSignal): AsyncGenerator<ChatEvent> {
  const response = await fetch('/api/v1/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages, stream: true }),
    signal,
  })
  if (!response.ok || !response.body) {
    throw new Error(response.status === 429 ? 'Too many requests, slow down a little.' : `Request failed (${response.status})`)
  }

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) return
    const { events, rest } = parseSSE(buffer + value)
    buffer = rest
    yield* events
  }
}

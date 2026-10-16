import { describe, expect, it } from 'vitest'
import { parseSSE } from './chat'

const frame = (obj: unknown) => `data: ${JSON.stringify(obj)}\n\n`

describe('parseSSE', () => {
  it('parses every complete event in order', () => {
    const buffer = frame({ event: 'start', data: {} }) + frame({ event: 'token', data: { token: 'Hi ' } })
    const { events, rest } = parseSSE(buffer)

    expect(events.map((e) => e.event)).toEqual(['start', 'token'])
    expect(rest).toBe('')
  })

  it('keeps a half-received event for the next chunk', () => {
    const full = frame({ event: 'token', data: { token: 'Tokyo' } })
    const first = parseSSE(full.slice(0, 15))
    expect(first.events).toEqual([])

    const second = parseSSE(first.rest + full.slice(15))
    expect(second.events).toEqual([{ event: 'token', data: { token: 'Tokyo' } }])
  })

  it('handles unicode split across chunks once reassembled', () => {
    const { events } = parseSSE(frame({ event: 'token', data: { token: '東京 São Paulo' } }))
    expect(events[0]).toEqual({ event: 'token', data: { token: '東京 São Paulo' } })
  })
})

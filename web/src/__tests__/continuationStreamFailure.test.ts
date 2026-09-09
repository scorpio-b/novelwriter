import type { SetStateAction } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { applyStreamEvent } from '@/components/studio/stages/continuation-results/streamRuntime'
import type { VariantState } from '@/components/studio/stages/continuation-results/helpers'
import type { StreamEvent } from '@/types/api'

function runtime() {
  let variants: VariantState[] = []
  const refs = {
    continuationMapRef: { current: new Map<number, number>() },
    totalVariantsRef: { current: 0 },
    receivedStreamOutputRef: { current: false },
  }
  const setters = {
    setIsDone: vi.fn(), setStreamDebug: vi.fn(), setStreamError: vi.fn(),
    setVariants: (value: SetStateAction<VariantState[]>) => {
      variants = typeof value === 'function' ? value(variants) : value
    },
  }
  const persistResultsToUrl = vi.fn()
  return {
    refs, setters, persistResultsToUrl, variants: () => variants,
    apply: (event: StreamEvent) => applyStreamEvent(event, { refs, setters, persistResultsToUrl }),
  }
}

describe('continuation failure events', () => {
  it('ends an empty failed stream without erasing the error or inventing a saved version', () => {
    const state = runtime()
    state.apply({ type: 'start', variant: 0, total_variants: 1 })
    state.apply({ type: 'error', variant: 0, code: 'llm_stream_failed', message: 'Generation failed' })
    state.apply({ type: 'done', continuation_ids: [] })
    expect(state.setters.setIsDone).toHaveBeenCalledWith(true)
    expect(state.variants()[0]).toMatchObject({
      isStreaming: false, continuationId: null, error: 'Generation failed',
    })
    expect(state.refs.continuationMapRef.current.size).toBe(0)
  })

  it('preserves the successful variant when another variant fails', () => {
    const state = runtime()
    state.apply({ type: 'start', variant: 0, total_variants: 2 })
    state.apply({ type: 'error', variant: 0, code: 'llm_stream_failed', message: 'Generation failed' })
    state.apply({ type: 'variant_done', variant: 1, continuation_id: 42, content: 'Saved prose' })
    state.apply({ type: 'done', continuation_ids: [42] })
    expect(state.variants()[0].error).toBe('Generation failed')
    expect(state.variants()[1]).toMatchObject({ content: 'Saved prose', continuationId: 42, error: null })
    expect([...state.refs.continuationMapRef.current.entries()]).toEqual([[1, 42]])
  })
})

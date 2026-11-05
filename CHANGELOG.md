# Changelog

## 0.3.0 (unreleased)

### Breaking changes to realtime sessions

`RealtimeEvent::Done` was documented as session closure, but the OpenAI and
xAI adapters also emitted it when a response finished. Replace matches on
`Done` according to what your application needs to do:

- Use `TurnComplete` for a completed turn. The session remains open. An
  interrupted turn can also emit `TurnComplete`.
- Use `Interrupted` to discard queued playback after a confirmed interruption.
  `SpeechStarted` reports voice activity, not confirmed interruption.
- Use `ToolCallsCancelled { ids }` to cancel outstanding tool work for the
  specified calls.
- Use `SessionClosed` for a closed connection. The next `recv` returns `None`.
- Use `GenerationComplete` when generation has finished but the turn may still
  be active.

Implementors of `RealtimeSession` must add `send_tool_result` and `resume`.
`send_tool_result` accepts the call ID, tool name, and JSON result. `resume`
continues after all results in a batch have been submitted. Providers that
continue automatically can implement `resume` as a no-op. Consumers should wait
for `ToolCallBatchComplete` before executing the received calls and submit all
results before calling `resume`. Update exhaustive event matches for the new
variants, including transcription events.

Gemini Live now queues mixed content, lifecycle events, and all tool calls
instead of retaining only the first event or call.

### Verification and limitations

Event preservation and interruption are verified against local WebSocket tests
only. Live provider sessions remain unverified: Gemini rejected the available
key, and OpenAI and xAI credentials were unavailable. No live interruption was
tested. OpenAI and xAI tool batches are tracked per session rather than per
response ID; concurrent interleaved responses are not supported.

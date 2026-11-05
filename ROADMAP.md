# Multimodal and live voice roadmap

This roadmap records a source audit, not a claim that provider APIs have been
verified end to end. Check current provider documentation and test with real
services before promising compatibility.

## Reliability first

- Report malformed audio and tool arguments rather than dropping them or
  substituting empty data. Check the remaining realtime paths against current
  provider schemas and test them with real sessions.
- Known limitation: OpenAI and xAI tool batches are tracked per session, not per
  response ID. Concurrent interleaved responses are not supported.

## A portable voice agent

The core `RealtimeSession` supports returning tool results and observing turn
completion, interruption, cancellation, and connection closure. It cannot yet
finish a manually controlled audio turn or configure tools. Provider-specific
sessions expose some of these operations. A portable contract also needs the
negotiated audio format and sample rate, so playback does not depend on
undocumented provider defaults.

Event preservation and interruption are verified against local WebSocket tests
only. Live provider sessions remain unverified. Gemini rejected the available
API key, and OpenAI and xAI credentials were unavailable. No live interruption
was tested. Test with valid credentials and add a microphone-to-speaker example
that clears queued playback on interruption. The existing examples send text
and count audio bytes; they do not demonstrate a duplex voice conversation.

## Creative multimodal experiences

- **See and speak:** combine Gemini Live video frames with audio for a spoken
  description of a changing scene. Bound frame rate and buffering, and define
  how interruption affects queued speech. The provider-specific session already
  exposes `send_video_frame`; the common session does not.
- **Hands-free tools:** let a user interrupt a long-running tool, cancel work when
  the provider cancels its call, and hear a partial result without blocking the
  rest of the conversation. Keep tool execution and user approval in the host
  application.
- **Long conversations:** investigate Gemini session resumption and context
  compression rather than hiding connection expiry. Preserve explicit session
  handles and make reconnection and privacy costs visible to applications.
- **Media pipelines:** make generated media usable as subsequent image or audio
  input without silently confusing a URL with bytes. Represent remote media and
  inline data explicitly, with bounded downloads and validation.

## Provider references

- [Gemini Live capabilities](https://ai.google.dev/gemini-api/docs/live-api/capabilities)
  documents video, transcription, interruption, and tool cancellation.
- [Gemini Live session management](https://ai.google.dev/gemini-api/docs/live-api/session-management)
  describes connection limits, resumption, and context compression.
- [OpenAI Realtime conversations](https://developers.openai.com/api/docs/guides/realtime-conversations)
  describes function results, interruption, and audio playback coordination.

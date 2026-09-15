use std::collections::HashMap;

use futures::stream::{self, Stream, StreamExt};
use rs_ai_core::{AiError, FinishReason, StreamEvent, Usage};

use super::api_types::ChatCompletionChunk;
use super::convert::parse_finish_reason;

/// Parse an SSE event stream from the OpenAI-compatible API into a stream of
/// [`StreamEvent`] values.
///
/// Tool-call arguments are streamed incrementally by the API, so we accumulate
/// them here and emit a [`StreamEvent::ToolCallEnd`] once the chunk indicates
/// the call is complete (via `finish_reason == "tool_calls"`) or the next
/// tool-call index appears.
pub(crate) fn parse_sse_stream(
    event_stream: impl Stream<Item = Result<reqwest_eventsource::Event, reqwest_eventsource::Error>>
        + Send
        + 'static,
) -> impl Stream<Item = Result<StreamEvent, AiError>> + Send {
    // We keep mutable state across events via `stream::unfold`.
    struct State<S> {
        inner: S,
        message_id: Option<String>,
        /// In-progress tool calls keyed by index. Stores (call_id, name, args_buffer).
        pending_tools: HashMap<u32, (String, String, String)>,
        done: bool,
    }

    let state = State {
        inner: Box::pin(event_stream),
        message_id: None,
        pending_tools: HashMap::new(),
        done: false,
    };

    stream::unfold(state, |mut state| async move {
        if state.done {
            return None;
        }

        loop {
            let event = match state.inner.next().await {
                Some(Ok(ev)) => ev,
                Some(Err(e)) => {
                    state.done = true;
                    return Some((
                        vec![Err(AiError::StreamError {
                            message: e.to_string(),
                        })],
                        state,
                    ));
                }
                None => {
                    return None;
                }
            };

            match event {
                reqwest_eventsource::Event::Open => continue,
                reqwest_eventsource::Event::Message(msg) => {
                    let data = msg.data.trim();

                    if data == "[DONE]" {
                        // Flush any remaining pending tool calls.
                        let mut events = flush_pending_tools(&mut state.pending_tools);
                        events.push(Ok(StreamEvent::MessageEnd {
                            finish_reason: FinishReason::Stop,
                            usage: None,
                        }));
                        state.done = true;
                        return Some((events, state));
                    }

                    let chunk: ChatCompletionChunk = match serde_json::from_str(data) {
                        Ok(c) => c,
                        Err(e) => {
                            tracing::error!(data, error = %e, "failed to parse SSE chunk; terminating stream");
                            state.done = true;
                            return Some((
                                vec![Err(AiError::StreamError {
                                    message: format!("Unparseable SSE chunk: {e}"),
                                })],
                                state,
                            ));
                        }
                    };

                    let mut events = Vec::new();

                    // Emit MessageStart on first chunk.
                    if state.message_id.is_none() {
                        state.message_id = Some(chunk.id.clone());
                        events.push(Ok(StreamEvent::MessageStart {
                            message_id: chunk.id.clone(),
                        }));
                    }

                    for choice in &chunk.choices {
                        if let Some(ref delta) = choice.delta {
                            // Text content delta.
                            if let Some(ref content) = delta.content {
                                if let Some(text) = content.as_str() {
                                    if !text.is_empty() {
                                        events.push(Ok(StreamEvent::TextDelta {
                                            delta: text.to_string(),
                                        }));
                                    }
                                }
                            }

                            // Tool call deltas.
                            if let Some(ref tool_calls) = delta.tool_calls {
                                for tc in tool_calls {
                                    // OpenAI streams parallel tool calls with
                                    // `delta.tool_calls[].index`. Choice index
                                    // is the completion choice, not the tool.
                                    let idx = tc.index.unwrap_or(choice.index);

                                    if !tc.id.is_empty() {
                                        // New tool call starting — flush any
                                        // previous one at the same index.
                                        if let Some((old_id, _old_name, old_args)) =
                                            state.pending_tools.remove(&idx)
                                        {
                                            match serde_json::from_str(&old_args) {
                                                Ok(arguments) => events.push(Ok(StreamEvent::ToolCallEnd {
                                                    call_id: old_id,
                                                    arguments,
                                                })),
                                                Err(e) => events.push(Err(AiError::StreamError {
                                                    message: format!(
                                                        "Malformed tool call arguments (JSON parse error: {e})"
                                                    ),
                                                })),
                                            }
                                        }

                                        state.pending_tools.insert(
                                            idx,
                                            (
                                                tc.id.clone(),
                                                tc.function.name.clone(),
                                                tc.function.arguments.clone(),
                                            ),
                                        );

                                        events.push(Ok(StreamEvent::ToolCallStart {
                                            call_id: tc.id.clone(),
                                            tool_name: tc.function.name.clone(),
                                        }));
                                    } else if let Some((ref id, _name, ref mut args)) =
                                        state.pending_tools.get_mut(&idx)
                                    {
                                        // Continuation of an existing tool call.
                                        let arg_delta = &tc.function.arguments;
                                        if !arg_delta.is_empty() {
                                            args.push_str(arg_delta);
                                            let call_id = id.clone();
                                            events.push(Ok(StreamEvent::ToolCallDelta {
                                                call_id,
                                                delta: arg_delta.clone(),
                                            }));
                                        }
                                    }
                                }
                            }
                        }

                        // Check for finish reason.
                        if let Some(ref reason) = choice.finish_reason {
                            let fr = parse_finish_reason(Some(reason.as_str()));

                            // If finishing with tool_calls, flush pending.
                            if fr == FinishReason::ToolCall {
                                events.append(&mut flush_pending_tools(&mut state.pending_tools));
                            }

                            // Emit usage if present on this chunk.
                            let usage = chunk.usage.as_ref().map(|u| Usage {
                                prompt_tokens: Some(u.prompt_tokens),
                                completion_tokens: Some(u.completion_tokens),
                                total_tokens: Some(u.total_tokens),
                            });

                            events.push(Ok(StreamEvent::MessageEnd {
                                finish_reason: fr,
                                usage,
                            }));

                            state.done = true;
                            return Some((events, state));
                        }
                    }

                    // Usage-only chunk (when stream_options.include_usage is set
                    // and choices is empty).
                    if chunk.choices.is_empty() {
                        if let Some(ref u) = chunk.usage {
                            events.push(Ok(StreamEvent::UsageDelta {
                                usage: Usage {
                                    prompt_tokens: Some(u.prompt_tokens),
                                    completion_tokens: Some(u.completion_tokens),
                                    total_tokens: Some(u.total_tokens),
                                },
                            }));
                        }
                    }

                    if !events.is_empty() {
                        return Some((events, state));
                    }
                    // If no events were produced, continue reading.
                }
            }
        }
    })
    .flat_map(stream::iter)
}

/// Flush all pending tool calls into `ToolCallEnd` (or `StreamError`) events.
fn flush_pending_tools(
    pending: &mut HashMap<u32, (String, String, String)>,
) -> Vec<Result<StreamEvent, AiError>> {
    let mut events = Vec::new();
    let mut indices: Vec<u32> = pending.keys().cloned().collect();
    indices.sort();
    for idx in indices {
        if let Some((id, _name, args)) = pending.remove(&idx) {
            match serde_json::from_str(&args) {
                Ok(arguments) => events.push(Ok(StreamEvent::ToolCallEnd {
                    call_id: id,
                    arguments,
                })),
                Err(e) => events.push(Err(AiError::StreamError {
                    message: format!(
                        "Malformed tool call arguments for call `{id}` (JSON parse error: {e})"
                    ),
                })),
            }
        }
    }
    events
}

#[cfg(test)]
mod tests {
    use super::parse_sse_stream;
    use futures::StreamExt;
    use rs_ai_core::StreamEvent;

    fn message(data: &str) -> reqwest_eventsource::Event {
        reqwest_eventsource::Event::Message(eventsource_stream::Event {
            event: String::new(),
            data: data.into(),
            id: String::new(),
            retry: None,
        })
    }

    #[tokio::test]
    async fn parallel_tool_calls_use_tool_index_not_choice_index() {
        let events = vec![
            Ok(message(
                r#"{"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"alpha","arguments":"{"}}]}}]}"#,
            )),
            Ok(message(
                r#"{"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"beta","arguments":"{"}}]}}]}"#,
            )),
            Ok(message(
                r#"{"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"a\":1}"}}]}}]}"#,
            )),
            Ok(message(
                r#"{"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\"b\":2}"}}]}}]}"#,
            )),
            Ok(message(
                r#"{"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
            )),
        ];
        let stream = parse_sse_stream(futures::stream::iter(events));
        let collected: Vec<_> = stream.collect().await;
        let ends: Vec<_> = collected
            .into_iter()
            .filter_map(|event| match event.unwrap() {
                StreamEvent::ToolCallEnd { call_id, arguments } => Some((call_id, arguments)),
                _ => None,
            })
            .collect();
        assert_eq!(ends.len(), 2);
        assert_eq!(ends[0].0, "call_a");
        assert_eq!(ends[0].1, serde_json::json!({"a":1}));
        assert_eq!(ends[1].0, "call_b");
        assert_eq!(ends[1].1, serde_json::json!({"b":2}));
    }
}

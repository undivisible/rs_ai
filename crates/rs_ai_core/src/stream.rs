//! Streaming response types and utilities.
use std::pin::Pin;

use futures::stream::{self, Stream, StreamExt};

use crate::error::{AiError, AiResult};
use crate::structured::GenerateResult;
use crate::tool::ToolCallRequest;
use crate::types::{FinishReason, ResponseMetadata};
use crate::usage::Usage;

/// Events emitted by a streaming response.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Start of a new message.
    MessageStart {
        /// Unique message identifier.
        message_id: String,
    },
    /// Incremental text content.
    TextDelta {
        /// The text chunk.
        delta: String,
    },
    /// Start of a tool call.
    ToolCallStart {
        /// Tool call identifier.
        call_id: String,
        /// Name of the tool being called.
        tool_name: String,
    },
    /// Incremental tool call arguments.
    ToolCallDelta {
        /// Tool call identifier.
        call_id: String,
        /// The argument chunk.
        delta: String,
    },
    /// End of a tool call.
    ToolCallEnd {
        /// Tool call identifier.
        call_id: String,
        /// Final parsed arguments.
        arguments: serde_json::Value,
    },
    /// Result of a tool execution.
    ToolResult {
        /// Tool call identifier.
        call_id: String,
        /// Result content.
        content: String,
        /// Whether the tool returned an error.
        is_error: bool,
    },
    /// Incremental structured object content.
    ObjectDelta {
        /// The JSON fragment.
        delta: serde_json::Value,
    },
    /// Emitted when an extended-thinking / reasoning model produces
    /// intermediate "thinking" tokens (Anthropic, Gemini 2.5+, Ollama think).
    ThinkingDelta {
        /// The thinking token chunk.
        delta: String,
    },
    /// Incremental usage update.
    UsageDelta {
        /// Token usage so far.
        usage: Usage,
    },
    /// A non-fatal warning.
    Warning {
        /// Warning message.
        message: String,
    },
    /// End of the message stream.
    MessageEnd {
        /// Why generation stopped.
        finish_reason: FinishReason,
        /// Final usage, if provided.
        usage: Option<Usage>,
    },
    /// A fatal stream error.
    Error {
        /// Error description.
        error: String,
    },
    /// Emitted once when a local runtime falls back to non-native streaming.
    SyntheticStreamingNotice,
}

/// A boxed, pinned, sendable stream of `StreamEvent` results.
pub type AiStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, AiError>> + Send>>;

/// Collects a full `AiStream` into a single `GenerateResult`.
pub struct StreamCollector;

impl StreamCollector {
    /// Consume a stream and assemble a complete `GenerateResult`.
    pub async fn collect(mut stream: AiStream) -> AiResult<GenerateResult> {
        let mut text = String::new();
        let mut reasoning: Option<String> = None;
        let mut tool_calls: Vec<ToolCallRequest> = Vec::new();
        let mut finish_reason = FinishReason::Unknown;
        let mut usage = Usage::default();

        // Track in-progress tool calls by call_id
        let mut pending_tool_calls: std::collections::HashMap<String, (String, String)> =
            std::collections::HashMap::new();

        while let Some(event) = stream.next().await {
            let event = event?;
            match event {
                StreamEvent::TextDelta { delta } => {
                    text.push_str(&delta);
                }
                StreamEvent::ThinkingDelta { delta } => {
                    reasoning.get_or_insert_with(String::new).push_str(&delta);
                }
                StreamEvent::ToolCallStart { call_id, tool_name } => {
                    pending_tool_calls.insert(call_id, (tool_name, String::new()));
                }
                StreamEvent::ToolCallDelta { call_id, delta } => {
                    if let Some((_name, args)) = pending_tool_calls.get_mut(&call_id) {
                        args.push_str(&delta);
                    }
                }
                StreamEvent::ToolCallEnd { call_id, arguments } => {
                    if let Some((name, _partial_args)) = pending_tool_calls.remove(&call_id) {
                        tool_calls.push(ToolCallRequest {
                            id: call_id,
                            name,
                            arguments,
                        });
                    }
                }
                StreamEvent::UsageDelta { usage: u } => {
                    usage.merge(&u);
                }
                StreamEvent::MessageEnd {
                    finish_reason: fr,
                    usage: u,
                } => {
                    finish_reason = fr;
                    if let Some(u) = u {
                        usage.merge(&u);
                    }
                }
                StreamEvent::Error { error } => {
                    return Err(AiError::StreamError { message: error });
                }
                _ => {}
            }
        }

        Ok(GenerateResult {
            text: if text.is_empty() { None } else { Some(text) },
            tool_calls,
            finish_reason,
            usage,
            metadata: ResponseMetadata::default(),
            steps: Vec::new(),
            reasoning,
        })
    }
}

/// Produces a synthetic stream from a complete text response.
///
/// Useful for providers that don't support real streaming.
pub struct SyntheticStreamer;

impl SyntheticStreamer {
    /// Create a synthetic stream builder from text.
    pub fn from_text(text: &str) -> SyntheticStreamBuilder {
        SyntheticStreamBuilder {
            text: text.to_owned(),
        }
    }

    /// Create a synthetic stream by chunking the given text.
    pub fn stream(text: String, chunk_size: usize) -> AiStream {
        let chunk_size = chunk_size.max(1);
        let chunks: Vec<Result<StreamEvent, AiError>> = {
            let mut v = Vec::new();
            v.push(Ok(StreamEvent::MessageStart {
                message_id: uuid::Uuid::new_v4().to_string(),
            }));
            // Notify callers that this is simulated streaming.
            v.push(Ok(StreamEvent::SyntheticStreamingNotice));

            let mut pos = 0;
            while pos < text.len() {
                let end = (pos + chunk_size).min(text.len());
                // Ensure we don't split in the middle of a multi-byte char.
                let end = if end < text.len() {
                    let mut e = end;
                    while !text.is_char_boundary(e) && e > pos {
                        e -= 1;
                    }
                    e
                } else {
                    end
                };
                let end = if end == pos {
                    pos + text[pos..]
                        .chars()
                        .next()
                        .expect("position is within text")
                        .len_utf8()
                } else {
                    end
                };
                v.push(Ok(StreamEvent::TextDelta {
                    delta: text[pos..end].to_owned(),
                }));
                pos = end;
            }

            v.push(Ok(StreamEvent::MessageEnd {
                finish_reason: FinishReason::Stop,
                usage: None,
            }));
            v
        };

        Box::pin(stream::iter(chunks))
    }
}

/// Builder returned by [`SyntheticStreamer::from_text`].
pub struct SyntheticStreamBuilder {
    text: String,
}

impl SyntheticStreamBuilder {
    /// Build an `AiStream` that emits the text in chunks.
    pub fn stream(self) -> AiStream {
        SyntheticStreamer::stream(self.text, 50)
    }
}

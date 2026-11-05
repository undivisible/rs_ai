use async_trait::async_trait;
use futures::stream::{self, StreamExt};

use rs_ai_core::{
    AiError, AiResult, AiStream, Capability, CapabilitySet, EmbeddingModel, EmbeddingResult,
    FinishReason, GenerateOptions, GenerateResult, LanguageModel, ResponseMetadata, StreamEvent,
    Usage,
};

use super::api_types::{
    OllamaChatRequest, OllamaChatResponse, OllamaEmbedRequest, OllamaEmbedResponse,
};
use super::convert;

/// An Ollama model that can perform chat completions and embeddings against a
/// local (or remote) Ollama server.
pub struct OllamaModel {
    base_url: String,
    model_id: String,
    capabilities: CapabilitySet,
    client: reqwest::Client,
}

impl OllamaModel {
    /// Create a new `OllamaModel` pointing at `http://localhost:11434`.
    pub fn new(model_id: &str) -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            model_id: model_id.to_string(),
            capabilities: CapabilitySet::new()
                .with(Capability::TextInput)
                .with(Capability::TextOutput)
                .with(Capability::Streaming)
                .with(Capability::ToolCalling)
                .with(Capability::ImageInput)
                .with(Capability::Embeddings)
                .with(Capability::LocalExecution),
            client: reqwest::Client::new(),
        }
    }

    /// Override the base URL (e.g. `http://myserver:11434`).
    pub fn with_base_url(mut self, url: &str) -> Self {
        self.base_url = url.trim_end_matches('/').to_string();
        self
    }

    /// Build the chat request body from a prompt and options.
    fn build_chat_request(
        &self,
        prompt: rs_ai_core::Prompt,
        options: &GenerateOptions,
        stream: bool,
    ) -> AiResult<OllamaChatRequest> {
        let conversation = prompt.into_messages();
        let messages = convert::convert_messages(&conversation)?;

        Ok(OllamaChatRequest {
            model: self.model_id.clone(),
            messages,
            stream,
            options: convert::convert_options(options),
            format: options.output_schema.as_ref().map(|s| s.as_value().clone()),
            tools: convert::convert_tools(options.tools.as_deref()),
            think: options.thinking.as_ref().map(|_| true),
        })
    }

    /// Parse token usage from an Ollama chat response.
    fn parse_usage(resp: &OllamaChatResponse) -> Usage {
        Usage {
            prompt_tokens: resp.prompt_eval_count,
            completion_tokens: resp.eval_count,
            total_tokens: match (resp.prompt_eval_count, resp.eval_count) {
                (Some(p), Some(c)) => Some(p + c),
                _ => None,
            },
        }
    }

    /// Determine the finish reason from an Ollama response.
    fn parse_finish_reason(resp: &OllamaChatResponse, has_tool_calls: bool) -> FinishReason {
        if has_tool_calls {
            FinishReason::ToolCall
        } else if resp.done_reason.as_deref() == Some("length") {
            FinishReason::Length
        } else {
            FinishReason::Stop
        }
    }

    /// Send a non-streaming chat request and parse the response.
    async fn do_generate(&self, request: OllamaChatRequest) -> AiResult<GenerateResult> {
        let url = format!("{}/api/chat", self.base_url);

        let resp = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| AiError::Transport {
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        let status = resp.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let body = match resp.text().await {
                Ok(body) => body,
                Err(e) => {
                    tracing::warn!(status = status_code, error = %e, "Failed to read Ollama error response body");
                    format!("<failed to read response body: {e}>")
                }
            };
            return Err(AiError::ProviderError {
                provider: "ollama".to_string(),
                status: Some(status_code),
                message: body,
            });
        }

        let chat_resp: OllamaChatResponse = resp
            .json()
            .await
            .map_err(|e| AiError::Serialization(e.to_string()))?;

        let tool_calls = chat_resp
            .message
            .tool_calls
            .as_deref()
            .map(convert::convert_tool_calls)
            .unwrap_or_default();

        let finish_reason = Self::parse_finish_reason(&chat_resp, !tool_calls.is_empty());

        let text = if chat_resp.message.content.is_empty() {
            None
        } else {
            Some(chat_resp.message.content.clone())
        };

        let usage = Self::parse_usage(&chat_resp);

        Ok(GenerateResult {
            text,
            tool_calls,
            finish_reason,
            usage,
            metadata: ResponseMetadata {
                provider: "ollama".to_string(),
                model: self.model_id.clone(),
                ..ResponseMetadata::default()
            },
            steps: Vec::new(),
            reasoning: None,
        })
    }
}

#[async_trait]
impl LanguageModel for OllamaModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn provider_id(&self) -> &str {
        "ollama"
    }

    fn capabilities(&self) -> &CapabilitySet {
        &self.capabilities
    }

    async fn generate(
        &self,
        prompt: rs_ai_core::Prompt,
        options: GenerateOptions,
    ) -> AiResult<GenerateResult> {
        let request = self.build_chat_request(prompt, &options, false)?;
        self.do_generate(request).await
    }

    async fn stream(
        &self,
        prompt: rs_ai_core::Prompt,
        options: GenerateOptions,
    ) -> AiResult<AiStream> {
        let request = self.build_chat_request(prompt, &options, true)?;
        let url = format!("{}/api/chat", self.base_url);

        let resp = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| AiError::Transport {
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        let status = resp.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let body = match resp.text().await {
                Ok(body) => body,
                Err(e) => {
                    tracing::warn!(status = status_code, error = %e, "Failed to read Ollama error response body");
                    format!("<failed to read response body: {e}>")
                }
            };
            return Err(AiError::ProviderError {
                provider: "ollama".to_string(),
                status: Some(status_code),
                message: body,
            });
        }

        // Ollama streams NDJSON: one JSON object per line.
        let byte_stream = resp.bytes_stream();
        Ok(build_ndjson_stream(byte_stream))
    }
}

/// Build an `AiStream` from Ollama's NDJSON byte stream.
///
/// Ollama emits one JSON object per line. We accumulate bytes until we see a
/// newline, then parse each complete line as an `OllamaChatResponse`.
fn build_ndjson_stream(
    byte_stream: impl futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
) -> AiStream {
    let message_id = uuid::Uuid::new_v4().to_string();

    // State: (buffer of partial bytes, whether we already sent MessageStart)
    let event_stream = byte_stream
        .scan(
            (Vec::<u8>::new(), false, message_id),
            |state, chunk_result: Result<bytes::Bytes, reqwest::Error>| {
                let (ref mut buf, ref mut sent_start, ref message_id) = *state;

                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        return std::future::ready(Some(vec![Err(AiError::StreamError {
                            message: e.to_string(),
                        })]));
                    }
                };

                buf.extend_from_slice(&chunk);

                let mut events: Vec<Result<StreamEvent, AiError>> = Vec::new();

                // Process all complete lines in the buffer.
                while let Some(newline_pos) = buf.iter().position(|&b| b == b'\n') {
                    let line_bytes: Vec<u8> = buf.drain(..=newline_pos).collect();
                    let line = match std::str::from_utf8(&line_bytes) {
                        Ok(s) => s.trim().to_string(),
                        Err(_) => continue,
                    };

                    if line.is_empty() {
                        continue;
                    }

                    let resp: OllamaChatResponse = match serde_json::from_str(&line) {
                        Ok(r) => r,
                        Err(e) => {
                            tracing::error!(
                                line = %line,
                                error = %e,
                                "Failed to parse Ollama stream chunk; terminating stream"
                            );
                            events.push(Err(AiError::StreamError {
                                message: format!("Unparseable NDJSON chunk from Ollama: {e}"),
                            }));
                            return std::future::ready(Some(events));
                        }
                    };

                    if !*sent_start {
                        *sent_start = true;
                        events.push(Ok(StreamEvent::MessageStart {
                            message_id: message_id.clone(),
                        }));
                    }

                    // Emit thinking tokens if present (reasoning models)
                    if let Some(ref thinking) = resp.thinking {
                        if !thinking.is_empty() {
                            events.push(Ok(StreamEvent::ThinkingDelta {
                                delta: thinking.clone(),
                            }));
                        }
                    }

                    if !resp.message.content.is_empty() {
                        events.push(Ok(StreamEvent::TextDelta {
                            delta: resp.message.content.clone(),
                        }));
                    }

                    if let Some(ref calls) = resp.message.tool_calls {
                        for (i, tc) in calls.iter().enumerate() {
                            let call_id = format!("call_{}", i);
                            events.push(Ok(StreamEvent::ToolCallStart {
                                call_id: call_id.clone(),
                                tool_name: tc.function.name.clone(),
                            }));
                            events.push(Ok(StreamEvent::ToolCallEnd {
                                call_id,
                                arguments: tc.function.arguments.clone(),
                            }));
                        }
                    }

                    if resp.done {
                        let usage = Usage {
                            prompt_tokens: resp.prompt_eval_count,
                            completion_tokens: resp.eval_count,
                            total_tokens: match (resp.prompt_eval_count, resp.eval_count) {
                                (Some(p), Some(c)) => Some(p + c),
                                _ => None,
                            },
                        };

                        let has_tools = resp
                            .message
                            .tool_calls
                            .as_ref()
                            .is_some_and(|v| !v.is_empty());

                        let finish_reason = if has_tools {
                            FinishReason::ToolCall
                        } else if resp.done_reason.as_deref() == Some("length") {
                            FinishReason::Length
                        } else {
                            FinishReason::Stop
                        };

                        events.push(Ok(StreamEvent::MessageEnd {
                            finish_reason,
                            usage: Some(usage),
                        }));
                    }
                }

                std::future::ready(Some(events))
            },
        )
        .flat_map(stream::iter);

    Box::pin(event_stream)
}

#[async_trait]
impl EmbeddingModel for OllamaModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn provider_id(&self) -> &str {
        "ollama"
    }

    fn dimensions(&self) -> Option<usize> {
        // Ollama does not report embedding dimensions upfront.
        None
    }

    async fn embed(&self, texts: Vec<String>) -> AiResult<EmbeddingResult> {
        let url = format!("{}/api/embed", self.base_url);

        let request = OllamaEmbedRequest {
            model: self.model_id.clone(),
            input: texts,
        };

        let resp = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| AiError::Transport {
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        let status = resp.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let body = match resp.text().await {
                Ok(body) => body,
                Err(e) => {
                    tracing::warn!(status = status_code, error = %e, "Failed to read Ollama error response body");
                    format!("<failed to read response body: {e}>")
                }
            };
            return Err(AiError::ProviderError {
                provider: "ollama".to_string(),
                status: Some(status_code),
                message: body,
            });
        }

        let embed_resp: OllamaEmbedResponse = resp
            .json()
            .await
            .map_err(|e| AiError::Serialization(e.to_string()))?;

        // Convert f32 -> f64 to match the trait signature.
        let embeddings = embed_resp
            .embeddings
            .into_iter()
            .map(|v| v.into_iter().map(|x| x as f64).collect())
            .collect();

        Ok(EmbeddingResult {
            embeddings,
            usage: Usage::default(),
        })
    }
}

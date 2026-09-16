//! OpenAI Realtime API — WebSocket-based voice and audio streaming.
//!
//! The Realtime API maintains a persistent WebSocket connection to exchange
//! low-latency audio, text, and function-call events bidirectionally.
//!
//! Reference: <https://platform.openai.com/docs/api-reference/realtime>

#![deny(missing_docs)]
#![allow(clippy::result_large_err)]

use std::sync::Arc;

use base64::Engine;
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::Mutex;
use tokio_tungstenite::{
    connect_async_tls_with_config,
    tungstenite::{client::IntoClientRequest, http::HeaderValue, Message},
};

const REALTIME_URL: &str = "wss://api.openai.com/v1/realtime";
const BETA_HEADER: &str = "realtime=v1";

// ── Error ─────────────────────────────────────────────────────────────────────

/// Errors that can occur during a Realtime API session.
#[derive(Debug, Error)]
pub enum RealtimeError {
    /// A WebSocket transport error.
    #[error("WebSocket error: {0}")]
    WebSocket(#[from] tokio_tungstenite::tungstenite::Error),
    /// A JSON serialization or deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// A URL parsing error.
    #[error("URL parse error: {0}")]
    Url(#[from] url::ParseError),
    /// The communication channel was closed.
    #[error("Channel closed")]
    ChannelClosed,
    /// An API error returned by the server.
    #[error("API error: {code} — {message}")]
    ApiError {
        /// Error code returned by the API.
        code: String,
        /// Human-readable error message.
        message: String,
    },
}

/// Shorthand result type for Realtime API operations.
pub type RealtimeResult<T> = Result<T, RealtimeError>;

// ── Events sent TO the API ────────────────────────────────────────────────────

/// Modalities supported by the Realtime API.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    /// Text modality.
    Text,
    /// Audio modality.
    Audio,
}

/// Voice options for audio output.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Voice {
    /// Alloy voice.
    #[default]
    Alloy,
    /// Ash voice.
    Ash,
    /// Ballad voice.
    Ballad,
    /// Coral voice.
    Coral,
    /// Echo voice.
    Echo,
    /// Sage voice.
    Sage,
    /// Shimmer voice.
    Shimmer,
    /// Verse voice.
    Verse,
}

/// Audio format for input or output.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AudioFormat {
    /// 16-bit PCM.
    #[default]
    Pcm16,
    /// G.711 μ-law.
    G711Ulaw,
    /// G.711 A-law.
    G711Alaw,
}

/// Turn detection configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnDetection {
    /// Turn detection type (e.g. "server_vad").
    #[serde(rename = "type")]
    pub kind: String,
    /// Activation threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f32>,
    /// Padding before speech in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_padding_ms: Option<u32>,
    /// Silence duration to trigger end of turn in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub silence_duration_ms: Option<u32>,
    /// Whether to create a response after detecting the end of a turn.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub create_response: Option<bool>,
}

impl TurnDetection {
    /// Create a default server-side VAD configuration.
    pub fn server_vad() -> Self {
        Self {
            kind: "server_vad".into(),
            threshold: Some(0.5),
            prefix_padding_ms: Some(300),
            silence_duration_ms: Some(500),
            create_response: Some(true),
        }
    }
}

/// A function/tool definition for the Realtime session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeTool {
    /// Tool type (e.g. "function").
    #[serde(rename = "type")]
    pub kind: String,
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// JSON Schema parameters.
    pub parameters: serde_json::Value,
}

/// Session configuration update.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionConfig {
    /// Desired modalities.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<Modality>>,
    /// System instructions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// Voice for audio output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<Voice>,
    /// Input audio format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_audio_format: Option<AudioFormat>,
    /// Output audio format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_audio_format: Option<AudioFormat>,
    /// Turn detection settings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_detection: Option<TurnDetection>,
    /// Tools available to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<RealtimeTool>>,
    /// Tool choice strategy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Maximum response output tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_response_output_tokens: Option<serde_json::Value>,
}

/// Client → server events.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientEvent {
    /// Update the session configuration.
    #[serde(rename = "session.update")]
    SessionUpdate {
        /// New session configuration.
        session: SessionConfig,
    },
    /// Append audio data to the input buffer.
    #[serde(rename = "input_audio_buffer.append")]
    InputAudioBufferAppend {
        /// Base64-encoded audio data.
        audio: String,
    },
    /// Commit the current audio buffer.
    #[serde(rename = "input_audio_buffer.commit")]
    InputAudioBufferCommit,
    /// Clear the audio buffer.
    #[serde(rename = "input_audio_buffer.clear")]
    InputAudioBufferClear,
    /// Create a conversation item.
    #[serde(rename = "conversation.item.create")]
    ConversationItemCreate {
        /// The item to create.
        item: ConversationItem,
    },
    /// Truncate a conversation item.
    #[serde(rename = "conversation.item.truncate")]
    ConversationItemTruncate {
        /// ID of the item to truncate.
        item_id: String,
        /// Content index.
        content_index: u32,
        /// Audio end position in milliseconds.
        audio_end_ms: u32,
    },
    /// Delete a conversation item.
    #[serde(rename = "conversation.item.delete")]
    ConversationItemDelete {
        /// ID of the item to delete.
        item_id: String,
    },
    /// Create a response.
    #[serde(rename = "response.create")]
    ResponseCreate {
        /// Optional response configuration.
        response: Option<ResponseConfig>,
    },
    /// Cancel the current response.
    #[serde(rename = "response.cancel")]
    ResponseCancel,
}

/// A conversation item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationItem {
    /// Item type (e.g. "message").
    #[serde(rename = "type")]
    pub kind: String,
    /// Optional item ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Role (e.g. "user" or "assistant").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Item content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<ItemContent>>,
    /// Call ID for function call output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    /// Function call output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
}

/// Content within a conversation item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemContent {
    /// Content type.
    #[serde(rename = "type")]
    pub kind: String,
    /// Text content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Base64-encoded audio content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<String>,
    /// Transcript content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transcript: Option<String>,
}

/// Configuration for a response.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResponseConfig {
    /// Desired modalities.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<Modality>>,
    /// Response instructions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

// ── Events received FROM the API ──────────────────────────────────────────────

/// Server → client events.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerEvent {
    /// Session was created.
    #[serde(rename = "session.created")]
    SessionCreated {
        /// Session data.
        session: serde_json::Value,
    },
    /// Session was updated.
    #[serde(rename = "session.updated")]
    SessionUpdated {
        /// Session data.
        session: serde_json::Value,
    },
    /// Speech started in the input audio buffer.
    #[serde(rename = "input_audio_buffer.speech_started")]
    SpeechStarted {
        /// Start time in milliseconds.
        audio_start_ms: u64,
        /// Associated item ID.
        item_id: String,
    },
    /// Speech stopped in the input audio buffer.
    #[serde(rename = "input_audio_buffer.speech_stopped")]
    SpeechStopped {
        /// End time in milliseconds.
        audio_end_ms: u64,
        /// Associated item ID.
        item_id: String,
    },
    /// Audio buffer was committed.
    #[serde(rename = "input_audio_buffer.committed")]
    AudioBufferCommitted {
        /// Associated item ID.
        item_id: String,
    },
    /// A conversation item was created.
    #[serde(rename = "conversation.item.created")]
    ConversationItemCreated {
        /// Item data.
        item: serde_json::Value,
    },
    /// Input audio transcription completed.
    #[serde(rename = "conversation.item.input_audio_transcription.completed")]
    TranscriptionCompleted {
        /// Item ID.
        item_id: String,
        /// Content index.
        content_index: u32,
        /// Transcript text.
        transcript: String,
    },
    /// A response was created.
    #[serde(rename = "response.created")]
    ResponseCreated {
        /// Response data.
        response: serde_json::Value,
    },
    /// An output item was added to the response.
    #[serde(rename = "response.output_item.added")]
    ResponseOutputItemAdded {
        /// Item data.
        item: serde_json::Value,
    },
    /// Audio transcript delta.
    #[serde(rename = "response.audio_transcript.delta")]
    AudioTranscriptDelta {
        /// Item ID.
        item_id: String,
        /// Delta text.
        delta: String,
    },
    /// Audio delta (base64-encoded PCM16).
    #[serde(rename = "response.audio.delta")]
    AudioDelta {
        /// Item ID.
        item_id: String,
        /// Base64-encoded audio delta.
        delta: String,
    },
    /// Audio generation completed.
    #[serde(rename = "response.audio.done")]
    AudioDone {
        /// Item ID.
        item_id: String,
    },
    /// Text delta.
    #[serde(rename = "response.text.delta")]
    TextDelta {
        /// Item ID.
        item_id: String,
        /// Delta text.
        delta: String,
    },
    /// Text generation completed.
    #[serde(rename = "response.text.done")]
    TextDone {
        /// Item ID.
        item_id: String,
        /// Final text.
        text: String,
    },
    /// Function call arguments delta.
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        /// Item ID.
        item_id: String,
        /// Call ID.
        call_id: String,
        /// Delta arguments.
        delta: String,
    },
    /// Function call arguments completed.
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        /// Item ID.
        item_id: String,
        /// Call ID.
        call_id: String,
        /// Function name.
        name: String,
        /// Final arguments.
        arguments: String,
    },
    /// Response generation completed.
    #[serde(rename = "response.done")]
    ResponseDone {
        /// Response data.
        response: serde_json::Value,
    },
    /// An error occurred.
    #[serde(rename = "error")]
    Error {
        /// Error details.
        error: RealtimeApiError,
    },
    /// An unrecognised event type.
    #[serde(other)]
    Unknown,
}

/// Error details returned by the Realtime API.
#[derive(Debug, Clone, Deserialize)]
pub struct RealtimeApiError {
    /// Error type.
    pub r#type: String,
    /// Error code.
    pub code: String,
    /// Error message.
    pub message: String,
}

// ── RealtimeSession ───────────────────────────────────────────────────────────

type WsSink = futures::stream::SplitSink<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
    Message,
>;
type WsStream = futures::stream::SplitStream<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
>;

/// An active Realtime API session.
///
/// # Example
///
/// ```no_run
/// use rs_ai_providers::chatgpt::{ChatGptProvider, GPT_4O_REALTIME};
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let provider = ChatGptProvider::new(std::env::var("OPENAI_API_KEY")?);
/// let mut session = provider.realtime_session(GPT_4O_REALTIME).await?;
///
/// // Configure the session
/// session.configure(Default::default()).await?;
///
/// // Send a text message and get a response
/// session.send_text("Hello, how are you?").await?;
/// while let Some(event) = session.recv().await {
///     match event? {
///         rs_ai_providers::chatgpt::realtime_api::ServerEvent::TextDelta { delta, .. } => print!("{delta}"),
///         rs_ai_providers::chatgpt::realtime_api::ServerEvent::ResponseDone { .. } => break,
///         _ => {}
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub struct RealtimeSession {
    sink: Arc<Mutex<WsSink>>,
    stream: Arc<Mutex<WsStream>>,
    _model: String,
}

impl RealtimeSession {
    pub(crate) async fn connect(api_key: &str, model: &str) -> RealtimeResult<Self> {
        let url_str = format!("{REALTIME_URL}?model={model}");
        let mut request = url_str.as_str().into_client_request()?;
        request.headers_mut().insert(
            "Authorization",
            HeaderValue::from_str(&format!("Bearer {api_key}")).map_err(|e| {
                RealtimeError::ApiError {
                    code: "header".into(),
                    message: e.to_string(),
                }
            })?,
        );
        request
            .headers_mut()
            .insert("OpenAI-Beta", HeaderValue::from_static(BETA_HEADER));

        let (ws, _response) = connect_async_tls_with_config(request, None, false, None).await?;
        let (sink, stream) = ws.split();
        Ok(Self {
            sink: Arc::new(Mutex::new(sink)),
            stream: Arc::new(Mutex::new(stream)),
            _model: model.to_string(),
        })
    }

    /// Update the session configuration.
    pub async fn configure(&mut self, config: SessionConfig) -> RealtimeResult<()> {
        self.send_event(ClientEvent::SessionUpdate { session: config })
            .await
    }

    /// Send raw PCM16 audio bytes (will be base64-encoded).
    pub async fn send_audio(&mut self, pcm16: &[u8]) -> RealtimeResult<()> {
        let encoded = base64::engine::general_purpose::STANDARD.encode(pcm16);
        self.send_event(ClientEvent::InputAudioBufferAppend { audio: encoded })
            .await
    }

    /// Commit the audio buffer and request a response.
    pub async fn commit_audio(&mut self) -> RealtimeResult<()> {
        self.send_event(ClientEvent::InputAudioBufferCommit).await?;
        self.send_event(ClientEvent::ResponseCreate { response: None })
            .await
    }

    /// Send a text message as a user turn.
    pub async fn send_text(&mut self, text: &str) -> RealtimeResult<()> {
        self.send_event(ClientEvent::ConversationItemCreate {
            item: ConversationItem {
                kind: "message".into(),
                id: None,
                role: Some("user".into()),
                content: Some(vec![ItemContent {
                    kind: "input_text".into(),
                    text: Some(text.to_string()),
                    audio: None,
                    transcript: None,
                }]),
                call_id: None,
                output: None,
            },
        })
        .await?;
        self.send_event(ClientEvent::ResponseCreate { response: None })
            .await
    }

    /// Return a tool call result back to the model.
    pub async fn send_tool_result(&mut self, call_id: &str, output: &str) -> RealtimeResult<()> {
        self.send_event(ClientEvent::ConversationItemCreate {
            item: ConversationItem {
                kind: "function_call_output".into(),
                id: None,
                role: None,
                content: None,
                call_id: Some(call_id.to_string()),
                output: Some(output.to_string()),
            },
        })
        .await?;
        self.send_event(ClientEvent::ResponseCreate { response: None })
            .await
    }

    /// Receive the next server event.
    pub async fn recv(&mut self) -> Option<RealtimeResult<ServerEvent>> {
        let mut stream = self.stream.lock().await;
        loop {
            let msg = stream.next().await?;
            match msg {
                Ok(Message::Text(text)) => {
                    return Some(serde_json::from_str(&text).map_err(RealtimeError::Json));
                }
                Ok(Message::Close(_)) => return None,
                Ok(_) => continue, // ping/pong/binary — ignore
                Err(e) => return Some(Err(RealtimeError::WebSocket(e))),
            }
        }
    }

    /// Decode a base64 audio delta from the server into raw PCM16 bytes.
    #[allow(clippy::result_large_err)]
    pub fn decode_audio(base64_delta: &str) -> RealtimeResult<Vec<u8>> {
        base64::engine::general_purpose::STANDARD
            .decode(base64_delta)
            .map_err(|e| RealtimeError::ApiError {
                code: "decode_error".into(),
                message: e.to_string(),
            })
    }

    async fn send_event(&mut self, event: ClientEvent) -> RealtimeResult<()> {
        let json = serde_json::to_string(&event)?;
        let mut sink = self.sink.lock().await;
        sink.send(Message::Text(json)).await?;
        Ok(())
    }
}

// ─── Unified RealtimeSession trait impl ──────────────────────────────────

use async_trait::async_trait as async_trait_reexport;
use rs_ai_core::{AiResult, RealtimeEvent};

/// Adapter that wraps ChatGPT's RealtimeSession into the unified
/// [`rs_ai_core::RealtimeSession`] trait.
pub struct ChatGptRealtimeSession {
    inner: RealtimeSession,
    model: String,
}

impl ChatGptRealtimeSession {
    /// Wrap a [`RealtimeSession`] into the unified trait.
    pub fn new(session: RealtimeSession, model: impl Into<String>) -> Self {
        Self {
            inner: session,
            model: model.into(),
        }
    }
}

#[async_trait_reexport]
impl rs_ai_core::RealtimeSession for ChatGptRealtimeSession {
    fn model_id(&self) -> &str {
        &self.model
    }

    fn provider_id(&self) -> &str {
        "chatgpt"
    }

    async fn send_text(&mut self, text: &str) -> AiResult<()> {
        self.inner
            .send_text(text)
            .await
            .map_err(|e| rs_ai_core::AiError::BridgeError {
                bridge: "chatgpt_realtime".into(),
                message: e.to_string(),
            })
    }

    async fn send_audio(&mut self, audio: Vec<u8>, _mime_type: &str) -> AiResult<()> {
        self.inner
            .send_audio(&audio)
            .await
            .map_err(|e| rs_ai_core::AiError::BridgeError {
                bridge: "chatgpt_realtime".into(),
                message: e.to_string(),
            })
    }

    async fn recv(&mut self) -> Option<RealtimeEvent> {
        loop {
            match self.inner.recv().await {
                Some(Ok(ServerEvent::TextDelta { delta, .. })) => {
                    return Some(RealtimeEvent::TextDelta { delta });
                }
                Some(Ok(ServerEvent::TextDone { text, .. })) => {
                    return Some(RealtimeEvent::TextDone { text });
                }
                Some(Ok(ServerEvent::AudioDelta { delta, .. })) => {
                    if let Ok(bytes) = RealtimeSession::decode_audio(&delta) {
                        return Some(RealtimeEvent::AudioDelta { delta: bytes });
                    }
                }
                Some(Ok(ServerEvent::FunctionCallArgumentsDone {
                    call_id,
                    name,
                    arguments,
                    ..
                })) => {
                    let args: serde_json::Value =
                        serde_json::from_str(&arguments).unwrap_or(serde_json::Value::Null);
                    return Some(RealtimeEvent::ToolCall {
                        id: call_id,
                        name,
                        arguments: args,
                    });
                }
                Some(Ok(ServerEvent::ResponseDone { .. })) => return Some(RealtimeEvent::Done),
                Some(Ok(ServerEvent::Error { error })) => {
                    return Some(RealtimeEvent::Error {
                        message: error.message,
                    });
                }
                Some(Ok(_)) => continue, // transient events
                Some(Err(e)) => {
                    return Some(RealtimeEvent::Error {
                        message: e.to_string(),
                    });
                }
                None => return None,
            }
        }
    }

    async fn close(self: Box<Self>) -> AiResult<()> {
        // WebSocket Drop impl sends close frame
        Ok(())
    }
}

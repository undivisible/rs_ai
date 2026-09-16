//! Gemini Live API — bidirectional WebSocket streaming for voice, video, and text.
//!
//! The Live API enables low-latency, real-time interactions by streaming audio,
//! video frames, and text in both directions over a persistent WebSocket.
//!
//! Reference: <https://ai.google.dev/gemini-api/docs/live-api>

#![deny(missing_docs)]
#![allow(clippy::result_large_err)]

use base64::Engine;
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::Mutex;
use tokio_tungstenite::{
    connect_async_tls_with_config,
    tungstenite::{client::IntoClientRequest, Message},
};

const LIVE_URL: &str =
    "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent";

// ── Error ─────────────────────────────────────────────────────────────────────

/// Errors that can occur during a Gemini Live API session.
#[derive(Debug, Error)]
pub enum LiveError {
    /// A WebSocket transport error.
    #[error("WebSocket error: {0}")]
    WebSocket(#[from] tokio_tungstenite::tungstenite::Error),
    /// A JSON serialization or deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// A URL parsing error.
    #[error("URL parse error: {0}")]
    Url(#[from] url::ParseError),
    /// An API error returned by the server.
    #[error("API error: {message}")]
    ApiError {
        /// Human-readable error message.
        message: String,
    },
    /// The session was closed.
    #[error("Session closed")]
    SessionClosed,
}

/// Shorthand result type for Live API operations.
pub type LiveResult<T> = Result<T, LiveError>;

// ── Voice / audio options ─────────────────────────────────────────────────────

/// HD voice IDs available in Gemini Live (30 voices across 24 languages).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum LiveVoice {
    /// Aoede voice.
    #[default]
    Aoede,
    /// Charon voice.
    Charon,
    /// Fenrir voice.
    Fenrir,
    /// Kore voice.
    Kore,
    /// Puck voice.
    Puck,
    /// Custom voice ID.
    Custom(String),
}

impl LiveVoice {
    /// Return the voice identifier as a string slice.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Aoede => "Aoede",
            Self::Charon => "Charon",
            Self::Fenrir => "Fenrir",
            Self::Kore => "Kore",
            Self::Puck => "Puck",
            Self::Custom(v) => v.as_str(),
        }
    }
}

/// Audio encoding format.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AudioEncoding {
    /// 16-bit linear PCM.
    Linear16,
    /// μ-law encoding.
    Mulaw,
    /// A-law encoding.
    Alaw,
}

/// Speech generation config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechConfig {
    /// Voice configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_config: Option<VoiceConfig>,
    /// BCP-47 language code.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_code: Option<String>,
}

/// Voice configuration wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceConfig {
    /// Prebuilt voice configuration.
    pub prebuilt_voice_config: PrebuiltVoiceConfig,
}

/// Prebuilt voice configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrebuiltVoiceConfig {
    /// Name of the prebuilt voice.
    pub voice_name: String,
}

// ── Setup message ─────────────────────────────────────────────────────────────

/// Tool / function definition for the live session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveTool {
    /// Function declarations available to the model.
    pub function_declarations: Vec<FunctionDeclaration>,
}

/// A single function declaration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDeclaration {
    /// Function name.
    pub name: String,
    /// Function description.
    pub description: String,
    /// JSON Schema parameters.
    pub parameters: serde_json::Value,
}

/// Session configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LiveGenerationConfig {
    /// Desired response modalities (e.g. "AUDIO", "TEXT").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_modalities: Option<Vec<String>>,
    /// Speech generation configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speech_config: Option<SpeechConfig>,
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Maximum output tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
}

// ── Client → Server messages ──────────────────────────────────────────────────

/// Setup message sent when opening a Live session.
#[derive(Debug, Clone, Serialize)]
pub struct SetupMessage {
    /// Bidi setup configuration.
    pub setup: BidiSetup,
}

/// Bidirectional setup configuration.
#[derive(Debug, Clone, Serialize)]
pub struct BidiSetup {
    /// Model identifier (prefixed with "models/").
    pub model: String,
    /// Optional system instruction.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<SystemInstruction>,
    /// Optional generation configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<LiveGenerationConfig>,
    /// Optional tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<LiveTool>>,
}

/// System instruction for the Live session.
#[derive(Debug, Clone, Serialize)]
pub struct SystemInstruction {
    /// Text parts of the instruction.
    pub parts: Vec<TextPart>,
}

/// A plain text part.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextPart {
    /// Text content.
    pub text: String,
}

/// Real-time audio/video input chunk.
#[derive(Debug, Clone, Serialize)]
pub struct RealtimeInputMessage {
    /// Real-time input payload.
    #[serde(rename = "realtimeInput")]
    pub realtime_input: RealtimeInput,
}

/// Real-time input payload.
#[derive(Debug, Clone, Serialize)]
pub struct RealtimeInput {
    /// Media chunks (audio/video frames).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_chunks: Option<Vec<MediaChunk>>,
    /// Optional text input.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Signal end of audio stream.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_stream_end: Option<bool>,
}

/// A single media chunk.
#[derive(Debug, Clone, Serialize)]
pub struct MediaChunk {
    /// MIME type of the chunk (e.g. "audio/pcm" or "image/jpeg").
    pub mime_type: String,
    /// Base64-encoded data.
    pub data: String,
}

/// Turn-based text/multi-turn content.
#[derive(Debug, Clone, Serialize)]
pub struct ClientContentMessage {
    /// Client content payload.
    #[serde(rename = "clientContent")]
    pub client_content: ClientContent,
}

/// Client content payload.
#[derive(Debug, Clone, Serialize)]
pub struct ClientContent {
    /// Conversation turns.
    pub turns: Vec<ContentTurn>,
    /// Whether the turn is complete.
    pub turn_complete: bool,
}

/// A single conversation turn.
#[derive(Debug, Clone, Serialize)]
pub struct ContentTurn {
    /// Role (e.g. "user").
    pub role: String,
    /// Turn parts.
    pub parts: Vec<serde_json::Value>,
}

/// Tool call result back to the model.
#[derive(Debug, Clone, Serialize)]
pub struct ToolResponseMessage {
    /// Tool response payload.
    pub tool_response: ToolResponse,
}

/// Tool response payload.
#[derive(Debug, Clone, Serialize)]
pub struct ToolResponse {
    /// Individual function responses.
    pub function_responses: Vec<FunctionResponse>,
}

/// A single function response.
#[derive(Debug, Clone, Serialize)]
pub struct FunctionResponse {
    /// Response ID.
    pub id: String,
    /// Function name.
    pub name: String,
    /// Response data.
    pub response: serde_json::Value,
}

// ── Server → Client messages ──────────────────────────────────────────────────

/// A message received from the server.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerMessage {
    /// Setup complete marker.
    pub setup_complete: Option<serde_json::Value>,
    /// Server content payload.
    pub server_content: Option<ServerContent>,
    /// Tool call event.
    pub tool_call: Option<ToolCallEvent>,
    /// Tool call cancellation.
    pub tool_call_cancellation: Option<ToolCallCancellation>,
}

/// Server content payload.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerContent {
    /// Model turn data.
    pub model_turn: Option<ModelTurn>,
    /// Whether the turn is complete.
    pub turn_complete: Option<bool>,
    /// Whether the generation was interrupted.
    pub interrupted: Option<bool>,
    /// Whether generation is complete.
    pub generation_complete: Option<bool>,
}

/// A turn produced by the model.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelTurn {
    /// Parts of the model turn.
    pub parts: Vec<ModelPart>,
}

/// A single part within a model turn.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelPart {
    /// Text content.
    pub text: Option<String>,
    /// Inline binary data (audio/image).
    pub inline_data: Option<InlineData>,
    /// Function call.
    pub function_call: Option<FunctionCallPart>,
}

/// Inline binary data.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InlineData {
    /// MIME type.
    pub mime_type: String,
    /// Base64-encoded data.
    pub data: String,
}

/// A function call from the model.
#[derive(Debug, Clone, Deserialize)]
pub struct FunctionCallPart {
    /// Call ID.
    pub id: String,
    /// Function name.
    pub name: String,
    /// Function arguments.
    pub args: serde_json::Value,
}

/// Tool call event wrapper.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallEvent {
    /// Individual function calls.
    pub function_calls: Vec<FunctionCallPart>,
}

/// Tool call cancellation event.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallCancellation {
    /// IDs of cancelled calls.
    pub ids: Vec<String>,
}

// ── Decoded event ─────────────────────────────────────────────────────────────

/// High-level decoded event from the Live API.
#[derive(Debug, Clone)]
pub enum LiveEvent {
    /// Session setup is complete.
    SetupComplete,
    /// Text delta from the model.
    TextDelta(String),
    /// Audio delta (raw PCM bytes).
    AudioDelta(Vec<u8>),
    /// The model's turn is complete.
    TurnComplete,
    /// The generation was interrupted.
    Interrupted,
    /// The model requested tool calls.
    ToolCall(Vec<FunctionCallPart>),
    /// Tool calls were cancelled.
    ToolCallCancelled(Vec<String>),
}

// ── LiveSession ───────────────────────────────────────────────────────────────

type WsSink = futures::stream::SplitSink<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
    Message,
>;
type WsStream = futures::stream::SplitStream<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
>;

/// An active Gemini Live API session.
///
/// # Example
///
/// ```no_run
/// use rs_ai_providers::gemini::{GeminiProvider, live_api::{LiveEvent, LiveGenerationConfig}};
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let provider = GeminiProvider::new(std::env::var("GOOGLE_API_KEY")?);
/// let mut session = provider.live_session("gemini-2.5-flash-live-preview").await?;
///
/// // Stream audio in and receive text/audio back
/// session.send_audio_chunk(b"...pcm16...", "audio/pcm").await?;
/// session.end_audio_turn().await?;
///
/// while let Some(event) = session.recv().await {
///     match event? {
///         LiveEvent::TextDelta(t) => print!("{t}"),
///         LiveEvent::AudioDelta(pcm) => { /* play audio */ }
///         LiveEvent::TurnComplete => break,
///         _ => {}
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub struct LiveSession {
    sink: Arc<Mutex<WsSink>>,
    stream: Arc<Mutex<WsStream>>,
}

impl LiveSession {
    /// Open a WebSocket connection to the Live API and send the setup message.
    pub async fn connect(api_key: &str, _model: &str, config: BidiSetup) -> LiveResult<Self> {
        let url_str = format!("{LIVE_URL}?key={api_key}");
        let request = url_str
            .as_str()
            .into_client_request()
            .map_err(LiveError::WebSocket)?;

        let (ws, _) = connect_async_tls_with_config(request, None, false, None).await?;
        let (mut sink, stream) = ws.split();

        // Send the setup message immediately after connection.
        let setup = SetupMessage { setup: config };
        sink.send(Message::Text(serde_json::to_string(&setup)?))
            .await?;

        Ok(Self {
            sink: Arc::new(Mutex::new(sink)),
            stream: Arc::new(Mutex::new(stream)),
        })
    }

    /// Send a chunk of raw audio bytes.
    ///
    /// `mime_type` is typically `"audio/pcm"` (16-bit, 16 kHz, mono) or
    /// `"audio/opus"`.
    pub async fn send_audio_chunk(&mut self, pcm: &[u8], mime_type: &str) -> LiveResult<()> {
        let data = base64::engine::general_purpose::STANDARD.encode(pcm);
        let msg = RealtimeInputMessage {
            realtime_input: RealtimeInput {
                media_chunks: Some(vec![MediaChunk {
                    mime_type: mime_type.to_string(),
                    data,
                }]),
                text: None,
                audio_stream_end: None,
            },
        };
        self.send_raw(&msg).await
    }

    /// Send a video frame as JPEG bytes.
    pub async fn send_video_frame(&mut self, jpeg: &[u8]) -> LiveResult<()> {
        let data = base64::engine::general_purpose::STANDARD.encode(jpeg);
        let msg = RealtimeInputMessage {
            realtime_input: RealtimeInput {
                media_chunks: Some(vec![MediaChunk {
                    mime_type: "image/jpeg".to_string(),
                    data,
                }]),
                text: None,
                audio_stream_end: None,
            },
        };
        self.send_raw(&msg).await
    }

    /// Signal that the audio stream has ended (triggers model response).
    pub async fn end_audio_turn(&mut self) -> LiveResult<()> {
        let msg = RealtimeInputMessage {
            realtime_input: RealtimeInput {
                media_chunks: None,
                text: None,
                audio_stream_end: Some(true),
            },
        };
        self.send_raw(&msg).await
    }

    /// Send a text message as a user turn.
    pub async fn send_text(&mut self, text: &str) -> LiveResult<()> {
        let msg = ClientContentMessage {
            client_content: ClientContent {
                turns: vec![ContentTurn {
                    role: "user".into(),
                    parts: vec![serde_json::json!({"text": text})],
                }],
                turn_complete: true,
            },
        };
        self.send_raw(&msg).await
    }

    /// Return a tool/function result to the model.
    pub async fn send_tool_result(
        &mut self,
        call_id: &str,
        name: &str,
        result: serde_json::Value,
    ) -> LiveResult<()> {
        let msg = ToolResponseMessage {
            tool_response: ToolResponse {
                function_responses: vec![FunctionResponse {
                    id: call_id.to_string(),
                    name: name.to_string(),
                    response: result,
                }],
            },
        };
        self.send_raw(&msg).await
    }

    /// Receive the next high-level event.
    pub async fn recv(&mut self) -> Option<LiveResult<LiveEvent>> {
        let mut stream = self.stream.lock().await;
        loop {
            let msg = stream.next().await?;
            match msg {
                Ok(Message::Text(text)) => {
                    let server_msg: ServerMessage = match serde_json::from_str(&text) {
                        Ok(m) => m,
                        Err(e) => return Some(Err(LiveError::Json(e))),
                    };
                    if let Some(event) = Self::decode(server_msg) {
                        return Some(Ok(event));
                    }
                    // Unknown/empty message — keep looping.
                }
                Ok(Message::Close(_)) => return None,
                Ok(_) => continue,
                Err(e) => return Some(Err(LiveError::WebSocket(e))),
            }
        }
    }

    fn decode(msg: ServerMessage) -> Option<LiveEvent> {
        if msg.setup_complete.is_some() {
            return Some(LiveEvent::SetupComplete);
        }
        if let Some(tc) = msg.tool_call {
            return Some(LiveEvent::ToolCall(tc.function_calls));
        }
        if let Some(tcc) = msg.tool_call_cancellation {
            return Some(LiveEvent::ToolCallCancelled(tcc.ids));
        }
        if let Some(sc) = msg.server_content {
            if sc.interrupted == Some(true) {
                return Some(LiveEvent::Interrupted);
            }
            if let Some(turn) = sc.model_turn {
                // Collect text first
                let text: String = turn
                    .parts
                    .iter()
                    .filter_map(|p| p.text.as_deref())
                    .collect();
                if !text.is_empty() {
                    return Some(LiveEvent::TextDelta(text));
                }
                // Then audio
                for part in &turn.parts {
                    if let Some(inline) = &part.inline_data {
                        if inline.mime_type.starts_with("audio/") {
                            if let Ok(pcm) =
                                base64::engine::general_purpose::STANDARD.decode(&inline.data)
                            {
                                return Some(LiveEvent::AudioDelta(pcm));
                            }
                        }
                    }
                }
            }
            if sc.turn_complete == Some(true) {
                return Some(LiveEvent::TurnComplete);
            }
        }
        None
    }

    async fn send_raw<T: Serialize>(&mut self, msg: &T) -> LiveResult<()> {
        let json = serde_json::to_string(msg)?;
        let mut sink = self.sink.lock().await;
        sink.send(Message::Text(json)).await?;
        Ok(())
    }
}

// ─── Unified RealtimeSession trait impl ──────────────────────────────────

use async_trait::async_trait as async_trait_reexport;
use rs_ai_core::{AiResult, RealtimeEvent};

/// Adapter that wraps Gemini's LiveSession into the unified
/// [`rs_ai_core::RealtimeSession`] trait.
pub struct GeminiLiveSessionAdapter {
    inner: LiveSession,
    model: String,
}

impl GeminiLiveSessionAdapter {
    /// Wrap a [`LiveSession`] into the unified trait.
    pub fn new(session: LiveSession, model: impl Into<String>) -> Self {
        Self {
            inner: session,
            model: model.into(),
        }
    }
}

#[async_trait_reexport]
impl rs_ai_core::RealtimeSession for GeminiLiveSessionAdapter {
    fn model_id(&self) -> &str {
        &self.model
    }

    fn provider_id(&self) -> &str {
        "gemini"
    }

    async fn send_text(&mut self, text: &str) -> AiResult<()> {
        self.inner
            .send_text(text)
            .await
            .map_err(|e| rs_ai_core::AiError::BridgeError {
                bridge: "gemini_live".into(),
                message: e.to_string(),
            })
    }

    async fn send_audio(&mut self, audio: Vec<u8>, mime_type: &str) -> AiResult<()> {
        self.inner
            .send_audio_chunk(&audio, mime_type)
            .await
            .map_err(|e| rs_ai_core::AiError::BridgeError {
                bridge: "gemini_live".into(),
                message: e.to_string(),
            })
    }

    async fn recv(&mut self) -> Option<RealtimeEvent> {
        loop {
            match self.inner.recv().await {
                Some(Ok(LiveEvent::TextDelta(text))) => {
                    return Some(RealtimeEvent::TextDelta { delta: text });
                }
                Some(Ok(LiveEvent::AudioDelta(pcm))) => {
                    return Some(RealtimeEvent::AudioDelta { delta: pcm });
                }
                Some(Ok(LiveEvent::ToolCall(calls))) => {
                    if let Some(first) = calls.into_iter().next() {
                        return Some(RealtimeEvent::ToolCall {
                            id: first.id,
                            name: first.name,
                            arguments: first.args,
                        });
                    }
                }
                Some(Ok(LiveEvent::SetupComplete)) => continue,
                Some(Ok(LiveEvent::TurnComplete)) => continue,
                Some(Ok(LiveEvent::Interrupted)) => continue,
                Some(Ok(LiveEvent::ToolCallCancelled(_))) => continue,
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

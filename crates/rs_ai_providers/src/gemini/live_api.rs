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
use std::collections::VecDeque;
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
    /// Invalid base64 audio data received from the server.
    #[error("Invalid audio data: {0}")]
    AudioDecode(#[from] base64::DecodeError),
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

fn normalize_tool_result(result: serde_json::Value) -> serde_json::Value {
    if result.is_object() {
        result
    } else {
        serde_json::json!({ "result": result })
    }
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
    /// Transcription of the user's input audio.
    pub input_transcription: Option<Transcription>,
    /// Transcription of the model's output audio.
    pub output_transcription: Option<Transcription>,
}

/// Text transcription of audio.
#[derive(Debug, Clone, Deserialize)]
pub struct Transcription {
    /// Transcribed text.
    pub text: String,
    /// Whether the transcription is finished, when provided.
    pub finished: Option<bool>,
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
    /// Model generation is complete, potentially before the turn is complete.
    GenerationComplete,
    /// The generation was interrupted.
    Interrupted,
    /// The model requested tool calls.
    ToolCall(Vec<FunctionCallPart>),
    /// All calls in a tool-call batch have been delivered.
    ToolCallBatchComplete,
    /// Transcription of the user's input audio.
    InputTranscription {
        /// Transcribed text.
        text: String,
        /// Whether the transcription is finished, when provided.
        finished: Option<bool>,
    },
    /// Transcription of the model's output audio.
    OutputTranscription {
        /// Transcribed text.
        text: String,
        /// Whether the transcription is finished, when provided.
        finished: Option<bool>,
    },
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
    pending: VecDeque<LiveResult<LiveEvent>>,
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
            pending: VecDeque::new(),
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
    /// Non-object results are wrapped in an object under the `result` key.
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
                    response: normalize_tool_result(result),
                }],
            },
        };
        self.send_raw(&msg).await
    }

    /// Receive the next high-level event.
    pub async fn recv(&mut self) -> Option<LiveResult<LiveEvent>> {
        if let Some(event) = self.pending.pop_front() {
            return Some(event);
        }
        let mut stream = self.stream.lock().await;
        loop {
            let msg = stream.next().await?;
            match msg {
                Ok(Message::Text(text)) => {
                    let server_msg: ServerMessage = match serde_json::from_str(&text) {
                        Ok(m) => m,
                        Err(e) => return Some(Err(LiveError::Json(e))),
                    };
                    self.pending.extend(Self::decode(server_msg));
                    if let Some(event) = self.pending.pop_front() {
                        return Some(event);
                    }
                    // Unknown/empty message — keep looping.
                }
                Ok(Message::Close(_)) => return None,
                Ok(_) => continue,
                Err(e) => return Some(Err(LiveError::WebSocket(e))),
            }
        }
    }

    fn decode(msg: ServerMessage) -> VecDeque<LiveResult<LiveEvent>> {
        let mut events = VecDeque::new();
        if msg.setup_complete.is_some() {
            events.push_back(Ok(LiveEvent::SetupComplete));
        }
        if let Some(sc) = msg.server_content {
            let interrupted = sc.interrupted == Some(true);
            if let Some(turn) = sc.model_turn {
                let mut has_calls = false;
                for part in turn.parts {
                    if let Some(text) = part.text {
                        events.push_back(Ok(LiveEvent::TextDelta(text)));
                    }
                    if let Some(inline) = part.inline_data {
                        if inline.mime_type.starts_with("audio/") {
                            events.push_back(
                                base64::engine::general_purpose::STANDARD
                                    .decode(&inline.data)
                                    .map(LiveEvent::AudioDelta)
                                    .map_err(LiveError::AudioDecode),
                            );
                        }
                    }
                    if let Some(call) = part.function_call {
                        has_calls = true;
                        events.push_back(Ok(LiveEvent::ToolCall(vec![call])));
                    }
                }
                if has_calls && !interrupted {
                    events.push_back(Ok(LiveEvent::ToolCallBatchComplete));
                }
            }
            if let Some(transcription) = sc.input_transcription {
                events.push_back(Ok(LiveEvent::InputTranscription {
                    text: transcription.text,
                    finished: transcription.finished,
                }));
            }
            if let Some(transcription) = sc.output_transcription {
                events.push_back(Ok(LiveEvent::OutputTranscription {
                    text: transcription.text,
                    finished: transcription.finished,
                }));
            }
            if interrupted {
                events.push_back(Ok(LiveEvent::Interrupted));
            }
            if sc.generation_complete == Some(true) {
                events.push_back(Ok(LiveEvent::GenerationComplete));
            }
            if sc.turn_complete == Some(true) {
                events.push_back(Ok(LiveEvent::TurnComplete));
            }
        }
        if let Some(tc) = msg.tool_call {
            if !tc.function_calls.is_empty() {
                events.push_back(Ok(LiveEvent::ToolCall(tc.function_calls)));
                events.push_back(Ok(LiveEvent::ToolCallBatchComplete));
            }
        }
        if let Some(tcc) = msg.tool_call_cancellation {
            events.push_back(Ok(LiveEvent::ToolCallCancelled(tcc.ids)));
        }
        events
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
    pending: VecDeque<RealtimeEvent>,
    tool_responses: Vec<FunctionResponse>,
    closed: bool,
}

impl GeminiLiveSessionAdapter {
    /// Wrap a [`LiveSession`] into the unified trait.
    pub fn new(session: LiveSession, model: impl Into<String>) -> Self {
        Self {
            inner: session,
            model: model.into(),
            pending: VecDeque::new(),
            tool_responses: Vec::new(),
            closed: false,
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

    async fn send_tool_result(
        &mut self,
        call_id: &str,
        name: &str,
        result: serde_json::Value,
    ) -> AiResult<()> {
        self.tool_responses.push(FunctionResponse {
            id: call_id.to_string(),
            name: name.to_string(),
            response: normalize_tool_result(result),
        });
        Ok(())
    }

    async fn resume(&mut self) -> AiResult<()> {
        if self.tool_responses.is_empty() {
            return Ok(());
        }
        let msg = ToolResponseMessage {
            tool_response: ToolResponse {
                function_responses: self.tool_responses.clone(),
            },
        };
        self.inner
            .send_raw(&msg)
            .await
            .map_err(|e| rs_ai_core::AiError::BridgeError {
                bridge: "gemini_live".into(),
                message: e.to_string(),
            })?;
        self.tool_responses.clear();
        Ok(())
    }

    async fn recv(&mut self) -> Option<RealtimeEvent> {
        if let Some(event) = self.pending.pop_front() {
            return Some(event);
        }
        if self.closed {
            return None;
        }
        loop {
            match self.inner.recv().await {
                Some(Ok(LiveEvent::TextDelta(text))) => {
                    return Some(RealtimeEvent::TextDelta { delta: text });
                }
                Some(Ok(LiveEvent::AudioDelta(pcm))) => {
                    return Some(RealtimeEvent::AudioDelta { delta: pcm });
                }
                Some(Ok(LiveEvent::ToolCall(calls))) => {
                    self.pending
                        .extend(calls.into_iter().map(|call| RealtimeEvent::ToolCall {
                            id: call.id,
                            name: call.name,
                            arguments: call.args,
                        }));
                    if let Some(event) = self.pending.pop_front() {
                        return Some(event);
                    }
                }
                Some(Ok(LiveEvent::ToolCallBatchComplete)) => {
                    return Some(RealtimeEvent::ToolCallBatchComplete);
                }
                Some(Ok(LiveEvent::InputTranscription { text, finished })) => {
                    return Some(RealtimeEvent::InputTranscription { text, finished });
                }
                Some(Ok(LiveEvent::OutputTranscription { text, finished })) => {
                    return Some(RealtimeEvent::OutputTranscription { text, finished });
                }
                Some(Ok(LiveEvent::SetupComplete)) => continue,
                Some(Ok(LiveEvent::TurnComplete)) => return Some(RealtimeEvent::TurnComplete),
                Some(Ok(LiveEvent::GenerationComplete)) => {
                    return Some(RealtimeEvent::GenerationComplete)
                }
                Some(Ok(LiveEvent::Interrupted)) => return Some(RealtimeEvent::Interrupted),
                Some(Ok(LiveEvent::ToolCallCancelled(ids))) => {
                    self.tool_responses
                        .retain(|response| !ids.contains(&response.id));
                    return Some(RealtimeEvent::ToolCallsCancelled { ids });
                }
                Some(Err(e)) => {
                    return Some(RealtimeEvent::Error {
                        message: e.to_string(),
                    });
                }
                None => {
                    self.closed = true;
                    return Some(RealtimeEvent::SessionClosed);
                }
            }
        }
    }

    async fn close(self: Box<Self>) -> AiResult<()> {
        // WebSocket Drop impl sends close frame
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rs_ai_core::RealtimeSession;

    #[test]
    fn tool_result_normalizes_non_objects() {
        for value in [serde_json::json!(null), serde_json::json!([1, 2])] {
            assert_eq!(
                normalize_tool_result(value.clone()),
                serde_json::json!({"result": value})
            );
        }
        let object = serde_json::json!({"result": 1, "other": true});
        assert_eq!(normalize_tool_result(object.clone()), object);
    }

    #[tokio::test]
    async fn mixed_message_and_tool_results_survive_adapter() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut ws = tokio_tungstenite::accept_async(socket).await.unwrap();
            ws.send(Message::Text(
                serde_json::json!({
                    "serverContent": {
                        "modelTurn": {"parts": [
                            {"text": "first"},
                            {"inlineData": {"mimeType": "audio/pcm", "data": "AQI="}},
                            {"inlineData": {"mimeType": "audio/pcm", "data": "!"}},
                            {"inlineData": {"mimeType": "audio/pcm", "data": "AwQF"}},
                            {"functionCall": {"id": "part", "name": "f", "args": {}}},
                            {"text": "last"}
                        ]},
                        "inputTranscription": {"text": "heard", "finished": false},
                        "outputTranscription": {"text": "spoken"},
                        "interrupted": true, "generationComplete": true, "turnComplete": true
                    },
                    "toolCall": {"functionCalls": [
                        {"id": "one", "name": "f", "args": {"n": 1}},
                        {"id": "two", "name": "g", "args": {"n": 2}}
                    ]},
                    "toolCallCancellation": {"ids": ["two"]}
                })
                .to_string(),
            ))
            .await
            .unwrap();
            let response = ws.next().await.unwrap().unwrap().into_text().unwrap();
            let response: serde_json::Value = serde_json::from_str(&response).unwrap();
            assert_eq!(
                response["tool_response"]["function_responses"],
                serde_json::json!([
                    {"id":"one", "name":"f", "response":{"ok":true}},
                    {"id":"part", "name":"g", "response":{"result":42}}
                ])
            );
            ws.close(None).await.unwrap();
        });
        let (ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}"))
            .await
            .unwrap();
        let (sink, stream) = ws.split();
        let session = LiveSession {
            sink: Arc::new(Mutex::new(sink)),
            stream: Arc::new(Mutex::new(stream)),
            pending: VecDeque::new(),
        };
        let mut adapter = GeminiLiveSessionAdapter::new(session, "test");
        assert!(
            matches!(adapter.recv().await, Some(RealtimeEvent::TextDelta { delta }) if delta == "first")
        );
        assert!(
            matches!(adapter.recv().await, Some(RealtimeEvent::AudioDelta { delta }) if delta == [1, 2])
        );
        assert!(
            matches!(adapter.recv().await, Some(RealtimeEvent::Error { message }) if message.contains("Invalid audio data"))
        );
        assert!(
            matches!(adapter.recv().await, Some(RealtimeEvent::AudioDelta { delta }) if delta == [3, 4, 5])
        );
        assert!(
            matches!(adapter.recv().await, Some(RealtimeEvent::ToolCall { id, .. }) if id == "part")
        );
        assert!(
            matches!(adapter.recv().await, Some(RealtimeEvent::TextDelta { delta }) if delta == "last")
        );
        assert!(
            matches!(adapter.recv().await, Some(RealtimeEvent::InputTranscription { text, finished: Some(false) }) if text == "heard")
        );
        assert!(
            matches!(adapter.recv().await, Some(RealtimeEvent::OutputTranscription { text, finished: None }) if text == "spoken")
        );
        assert!(matches!(
            adapter.recv().await,
            Some(RealtimeEvent::Interrupted)
        ));
        assert!(matches!(
            adapter.recv().await,
            Some(RealtimeEvent::GenerationComplete)
        ));
        assert!(matches!(
            adapter.recv().await,
            Some(RealtimeEvent::TurnComplete)
        ));
        assert!(
            matches!(adapter.recv().await, Some(RealtimeEvent::ToolCall { id, .. }) if id == "one")
        );
        assert!(
            matches!(adapter.recv().await, Some(RealtimeEvent::ToolCall { id, .. }) if id == "two")
        );
        assert!(matches!(
            adapter.recv().await,
            Some(RealtimeEvent::ToolCallBatchComplete)
        ));
        adapter
            .send_tool_result("one", "f", serde_json::json!({"ok": true}))
            .await
            .unwrap();
        adapter
            .send_tool_result("part", "g", serde_json::json!(42))
            .await
            .unwrap();
        adapter
            .send_tool_result("two", "g", serde_json::json!({"cancelled": true}))
            .await
            .unwrap();
        assert!(
            matches!(adapter.recv().await, Some(RealtimeEvent::ToolCallsCancelled { ids }) if ids == ["two"])
        );
        adapter.resume().await.unwrap();
        assert!(matches!(
            adapter.recv().await,
            Some(RealtimeEvent::SessionClosed)
        ));
        assert!(adapter.recv().await.is_none());
        server.await.unwrap();
    }
}

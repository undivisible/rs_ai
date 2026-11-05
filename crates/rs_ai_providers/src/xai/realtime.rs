//! xAI Grok Voice Agent API — WebSocket voice streaming.
//!
//! Compatible with OpenAI Realtime API message format.
//! Reference: <https://docs.x.ai/developers/model-capabilities/audio/voice-agent>
//!
//! # Models
//! - `grok-voice-latest` (recommended, alias for the latest flagship)
//! - `grok-voice-think-fast-1.0` (deep reasoning, fast)
//!
//! # Features
//! - Direct speech-to-speech (no ASR/TTS pipeline)
//! - Built-in XSearch + WebSearch tools
//! - Server VAD with interrupt support
//! - In-stream reasoning

use async_trait::async_trait;
use base64::Engine;
use futures::SinkExt;
use rs_ai_core::{AiError, AiResult, RealtimeEvent, RealtimeSession};
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_tungstenite::{connect_async, tungstenite::Message as WsMessage};

#[cfg(test)]
#[path = "realtime_tool_tests.rs"]
mod realtime_tool_tests;

const DEFAULT_BASE_URL: &str = "https://api.x.ai";
const WS_BASE_URL: &str = "wss://api.x.ai";

/// xAI Grok Voice Agent session configuration.
pub struct GrokVoiceConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub voice: Option<String>,
}

impl GrokVoiceConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            model: "grok-voice-latest".into(),
            voice: None,
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = Some(voice.into());
        self
    }
}

// ── Auth: client_secret flow ─────────────────────────────────────────────────

#[derive(Deserialize)]
struct ClientSecretResponse {
    client_secret: ClientSecretValue,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct ClientSecretValue {
    value: String,
    expires_at: u64,
}

async fn create_client_secret(api_key: &str, base_url: &str) -> AiResult<String> {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/realtime/client_secrets", base_url))
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({}))
        .send()
        .await
        .map_err(|e| AiError::Transport {
            message: format!("xAI client_secret request failed: {e}"),
            source: Some(Box::new(e)),
        })?;

    if !resp.status().is_success() {
        return Err(AiError::AuthError {
            message: format!("xAI client_secret returned {}", resp.status()),
        });
    }

    let data: ClientSecretResponse = resp.json().await.map_err(|e| AiError::Transport {
        message: format!("xAI client_secret parse failed: {e}"),
        source: Some(Box::new(e)),
    })?;

    Ok(data.client_secret.value)
}

// ── WebSocket session ────────────────────────────────────────────────────────

/// An active xAI Grok Voice Agent session.
pub struct GrokVoiceSession {
    write: Arc<
        Mutex<
            tokio_tungstenite::WebSocketStream<
                tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
            >,
        >,
    >,
    model: String,
}

impl GrokVoiceSession {
    /// Connect to the xAI Grok Voice Agent API.
    ///
    /// Creates a client secret via REST, then opens a WebSocket connection.
    pub async fn connect(config: GrokVoiceConfig) -> AiResult<Self> {
        let secret = create_client_secret(&config.api_key, &config.base_url).await?;

        let ws_url = format!("{}/realtime?client_secret={}", WS_BASE_URL, secret);

        let (ws_stream, _) = connect_async(&ws_url)
            .await
            .map_err(|e| AiError::Transport {
                message: format!("xAI WebSocket connect failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        let write = Arc::new(Mutex::new(ws_stream));

        // Send session.update to configure
        let mut session_config = serde_json::json!({
            "model": config.model,
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "silence_duration_ms": 200,
                },
            }
        });
        if let Some(ref voice) = config.voice {
            session_config["session"]["voice"] = serde_json::json!(voice);
        }

        {
            let mut w = write.lock().await;
            let msg = serde_json::to_string(&session_config).map_err(|e| {
                AiError::Serialization(format!("xAI session config serialize: {e}"))
            })?;
            w.send(WsMessage::Text(msg))
                .await
                .map_err(|e| AiError::Transport {
                    message: format!("xAI WebSocket send failed: {e}"),
                    source: Some(Box::new(e)),
                })?;
        }

        Ok(Self {
            write,
            model: config.model,
        })
    }
}

#[async_trait]
impl RealtimeSession for GrokVoiceSession {
    fn model_id(&self) -> &str {
        &self.model
    }

    fn provider_id(&self) -> &str {
        "xai"
    }

    async fn send_text(&mut self, text: &str) -> AiResult<()> {
        let event = serde_json::json!({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            }
        });
        let msg =
            serde_json::to_string(&event).map_err(|e| AiError::Serialization(e.to_string()))?;
        let mut w = self.write.lock().await;
        w.send(WsMessage::Text(msg))
            .await
            .map_err(|e| AiError::Transport {
                message: format!("xAI WebSocket send_text failed: {e}"),
                source: Some(Box::new(e)),
            })?;
        Ok(())
    }

    async fn send_audio(&mut self, audio: Vec<u8>, _mime_type: &str) -> AiResult<()> {
        let base64 = base64::engine::general_purpose::STANDARD.encode(&audio);
        let event = serde_json::json!({
            "type": "input_audio_buffer.append",
            "audio": base64,
        });
        let msg =
            serde_json::to_string(&event).map_err(|e| AiError::Serialization(e.to_string()))?;
        let mut w = self.write.lock().await;
        w.send(WsMessage::Text(msg))
            .await
            .map_err(|e| AiError::Transport {
                message: format!("xAI WebSocket send_audio failed: {e}"),
                source: Some(Box::new(e)),
            })?;
        Ok(())
    }

    async fn recv(&mut self) -> Option<RealtimeEvent> {
        use futures::StreamExt;
        let mut w = self.write.lock().await;
        loop {
            match w.next().await {
                Some(Ok(WsMessage::Text(text))) => {
                    let parsed: serde_json::Value = match serde_json::from_str(&text) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    let event_type = parsed["type"].as_str().unwrap_or("");

                    match event_type {
                        "response.text.delta" => {
                            let delta = parsed["delta"].as_str().unwrap_or("").to_string();
                            return Some(RealtimeEvent::TextDelta { delta });
                        }
                        "response.text.done" => {
                            let text = parsed["text"].as_str().unwrap_or("").to_string();
                            return Some(RealtimeEvent::TextDone { text });
                        }
                        "response.audio.delta" => {
                            let delta = parsed["delta"].as_str().and_then(|b64| {
                                base64::engine::general_purpose::STANDARD.decode(b64).ok()
                            });
                            return Some(match delta {
                                Some(delta) if !delta.is_empty() => {
                                    RealtimeEvent::AudioDelta { delta }
                                }
                                _ => RealtimeEvent::Error {
                                    message: "Invalid or empty xAI realtime audio delta".into(),
                                },
                            });
                        }
                        "response.function_call_arguments.done" | "response.tool_call" => {
                            let name = parsed["name"].as_str().unwrap_or("").to_string();
                            let arguments = match parsed["arguments"]
                                .as_str()
                                .and_then(|raw| serde_json::from_str(raw).ok())
                            {
                                Some(arguments) => arguments,
                                None => {
                                    return Some(RealtimeEvent::Error {
                                        message: "Invalid or missing xAI realtime tool arguments"
                                            .into(),
                                    })
                                }
                            };
                            return Some(RealtimeEvent::ToolCall {
                                id: parsed["call_id"].as_str().unwrap_or("").to_string(),
                                name,
                                arguments,
                            });
                        }
                        "error" => {
                            let msg = parsed["error"]["message"]
                                .as_str()
                                .unwrap_or("xAI realtime error")
                                .to_string();
                            return Some(RealtimeEvent::Error { message: msg });
                        }
                        "response.done" => {
                            return Some(RealtimeEvent::Done);
                        }
                        _ => continue,
                    }
                }
                Some(Ok(WsMessage::Ping(_))) | Some(Ok(WsMessage::Pong(_))) => continue,
                Some(Ok(_)) => continue,
                Some(Err(e)) => {
                    return Some(RealtimeEvent::Error {
                        message: format!("WebSocket error: {e}"),
                    });
                }
                None => return Some(RealtimeEvent::Done),
            }
        }
    }

    async fn close(self: Box<Self>) -> AiResult<()> {
        // Drop closes the WebSocket
        Ok(())
    }
}

/// Create an xAI Grok Voice Agent session with the given config.
pub async fn create_grok_voice_session(config: GrokVoiceConfig) -> AiResult<GrokVoiceSession> {
    GrokVoiceSession::connect(config).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn audio_events_reject_malformed_payloads() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut socket = tokio_tungstenite::accept_async(socket).await.unwrap();
            for event in [
                r#"{"type":"response.audio.delta","delta":"AQIDBA=="}"#,
                r#"{"type":"response.audio.delta","delta":"not base64!"}"#,
                r#"{"type":"response.audio.delta"}"#,
                r#"{"type":"response.audio.delta","delta":""}"#,
            ] {
                socket.send(WsMessage::Text(event.into())).await.unwrap();
            }
        });
        let (socket, _) = connect_async(format!("ws://{address}")).await.unwrap();
        let mut session = GrokVoiceSession {
            write: Arc::new(Mutex::new(socket)),
            model: "test".into(),
        };
        assert!(
            matches!(session.recv().await, Some(RealtimeEvent::AudioDelta { delta }) if delta == [1, 2, 3, 4])
        );
        assert!(matches!(
            session.recv().await,
            Some(RealtimeEvent::Error { .. })
        ));
        assert!(matches!(
            session.recv().await,
            Some(RealtimeEvent::Error { .. })
        ));
        assert!(matches!(
            session.recv().await,
            Some(RealtimeEvent::Error { .. })
        ));
        server.await.unwrap();
    }
}

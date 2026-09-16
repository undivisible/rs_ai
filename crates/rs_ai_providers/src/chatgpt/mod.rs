//!
//! ⚠️ **UNSTABLE** — This crate is in active development. APIs may change without notice.
//!
//! OpenAI ChatGPT provider for the Rusty AI SDK.
//!
//! This is a thin wrapper around [`crate::openai_compatible`] that pre-configures
//! the adapter for the official OpenAI API with well-known ChatGPT models.
//!
//! # Example
//!
//! ```rust,no_run
//! use rs_ai_providers::chatgpt::ChatGptProvider;
//! use rs_ai_core::Provider;
//!
//! let provider = ChatGptProvider::new("sk-...");
//! let model = provider.language_model("gpt-4o").unwrap();
//! ```

pub(crate) mod image;
pub mod realtime_api;
pub(crate) mod stt;
pub(crate) mod tts;
pub use image::ChatGptImageModel;
pub use realtime_api::{ChatGptRealtimeSession, RealtimeSession};
pub use stt::ChatGptSttModel;
pub use tts::ChatGptTtsModel;

use crate::openai_compatible::{
    OpenAiCompatibleConfig, OpenAiCompatibleModel, OpenAiCompatibleProvider,
};
use rs_ai_core::capability::{Capability, CapabilitySet};
use rs_ai_core::error::AiResult;
use rs_ai_core::model::{EmbeddingModel, LanguageModel, SpeechToTextModel, TextToSpeechModel};
use rs_ai_core::provider::Provider;
use rs_ai_core::types::ModelInfo;

// ── Latest model aliases ──

/// Convenience alias. Use `fetch_models()` or pass any model ID string to `model()`.
pub const GPT_4O_LATEST: &str = "gpt-4o";
/// Convenience alias. Use `fetch_models()` or pass any model ID string to `model()`.
pub const GPT_4O_MINI_LATEST: &str = "gpt-4o-mini";
/// Convenience alias. Use `fetch_models()` or pass any model ID string to `model()`.
pub const O3_MINI_LATEST: &str = "o3-mini";
/// Convenience alias. Use `fetch_models()` or pass any model ID string to `model()`.
pub const GPT_5_4_LATEST: &str = "gpt-5.4";
/// Convenience alias. Use `fetch_models()` or pass any model ID string to `model()`.
pub const GPT_5_4_MINI_LATEST: &str = "gpt-5.4-mini";
/// Convenience alias. Use `fetch_models()` or pass any model ID string to `model()`.
pub const GPT_5_4_NANO_LATEST: &str = "gpt-5.4-nano";

// ── Audio / voice model identifiers ──

/// OpenAI Whisper speech-to-text.
pub const WHISPER: &str = "whisper-1";
/// OpenAI TTS.
pub const TTS: &str = "tts-1";
/// OpenAI TTS HD.
pub const TTS_HD: &str = "tts-1-hd";
/// OpenAI GPT-4o Realtime (voice).
pub const GPT_4O_REALTIME: &str = "gpt-4o-realtime-preview";
/// OpenAI GPT-4o Audio (audio modality in chat completions).
pub const GPT_4O_AUDIO: &str = "gpt-4o-audio-preview";
/// OpenAI GPT-4o Mini Realtime.
pub const GPT_4O_MINI_REALTIME: &str = "gpt-4o-mini-realtime-preview";

// ── Image / video model identifiers ──

/// OpenAI DALL-E 3.
pub const DALL_E_3: &str = "dall-e-3";
/// OpenAI DALL-E 2.
pub const DALL_E_2: &str = "dall-e-2";

/// A provider pre-configured for the official OpenAI ChatGPT API.
pub struct ChatGptProvider {
    inner: OpenAiCompatibleProvider,
    config: OpenAiCompatibleConfig,
    oauth_token: Option<String>,
}

impl ChatGptProvider {
    /// Create a new ChatGPT provider with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        let config = OpenAiCompatibleConfig::openai(api_key);
        let inner = OpenAiCompatibleProvider::new(config.clone(), "chatgpt", "ChatGPT")
            .with_model_info(ModelInfo {
                id: GPT_4O_LATEST.into(),
                provider: "chatgpt".into(),
                display_name: "GPT-4o".into(),
                capabilities: CapabilitySet::new()
                    .with(Capability::TextInput)
                    .with(Capability::TextOutput)
                    .with(Capability::ImageInput)
                    .with(Capability::Streaming)
                    .with(Capability::ToolCalling)
                    .with(Capability::StructuredOutput),
                ..Default::default()
            })
            .with_model_info(ModelInfo {
                id: GPT_4O_MINI_LATEST.into(),
                provider: "chatgpt".into(),
                display_name: "GPT-4o Mini".into(),
                capabilities: CapabilitySet::new()
                    .with(Capability::TextInput)
                    .with(Capability::TextOutput)
                    .with(Capability::ImageInput)
                    .with(Capability::Streaming)
                    .with(Capability::ToolCalling)
                    .with(Capability::StructuredOutput),
                ..Default::default()
            })
            .with_model_info(ModelInfo {
                id: O3_MINI_LATEST.into(),
                provider: "chatgpt".into(),
                display_name: "o3-mini".into(),
                capabilities: CapabilitySet::new()
                    .with(Capability::TextInput)
                    .with(Capability::TextOutput)
                    .with(Capability::Streaming)
                    .with(Capability::ToolCalling),
                ..Default::default()
            })
            .with_model_info(ModelInfo {
                id: GPT_5_4_LATEST.into(),
                provider: "chatgpt".into(),
                display_name: "GPT-5.4".into(),
                capabilities: CapabilitySet::new()
                    .with(Capability::TextInput)
                    .with(Capability::TextOutput)
                    .with(Capability::ImageInput)
                    .with(Capability::Streaming)
                    .with(Capability::ToolCalling)
                    .with(Capability::StructuredOutput)
                    .with(Capability::ExtendedThinking),
                ..Default::default()
            })
            .with_model_info(ModelInfo {
                id: GPT_5_4_MINI_LATEST.into(),
                provider: "chatgpt".into(),
                display_name: "GPT-5.4 Mini".into(),
                capabilities: CapabilitySet::new()
                    .with(Capability::TextInput)
                    .with(Capability::TextOutput)
                    .with(Capability::ImageInput)
                    .with(Capability::Streaming)
                    .with(Capability::ToolCalling)
                    .with(Capability::StructuredOutput),
                ..Default::default()
            })
            .with_model_info(ModelInfo {
                id: GPT_5_4_NANO_LATEST.into(),
                provider: "chatgpt".into(),
                display_name: "GPT-5.4 Nano".into(),
                capabilities: CapabilitySet::new()
                    .with(Capability::TextInput)
                    .with(Capability::TextOutput)
                    .with(Capability::Streaming),
                ..Default::default()
            })
            .with_model_info(ModelInfo {
                id: WHISPER.into(),
                provider: "chatgpt".into(),
                display_name: "Whisper".into(),
                capabilities: CapabilitySet::new()
                    .with(Capability::AudioInput)
                    .with(Capability::TextOutput),
                ..Default::default()
            })
            .with_model_info(ModelInfo {
                id: TTS.into(),
                provider: "chatgpt".into(),
                display_name: "TTS".into(),
                capabilities: CapabilitySet::new()
                    .with(Capability::TextInput)
                    .with(Capability::AudioOutput),
                ..Default::default()
            })
            .with_model_info(ModelInfo {
                id: TTS_HD.into(),
                provider: "chatgpt".into(),
                display_name: "TTS HD".into(),
                capabilities: CapabilitySet::new()
                    .with(Capability::TextInput)
                    .with(Capability::AudioOutput),
                ..Default::default()
            })
            .with_model_info(ModelInfo {
                id: GPT_4O_REALTIME.into(),
                provider: "chatgpt".into(),
                display_name: "GPT-4o Realtime".into(),
                capabilities: CapabilitySet::new()
                    .with(Capability::TextInput)
                    .with(Capability::TextOutput)
                    .with(Capability::AudioInput)
                    .with(Capability::AudioOutput)
                    .with(Capability::Streaming),
                ..Default::default()
            })
            .with_model_info(ModelInfo {
                id: GPT_4O_AUDIO.into(),
                provider: "chatgpt".into(),
                display_name: "GPT-4o Audio".into(),
                capabilities: CapabilitySet::new()
                    .with(Capability::TextInput)
                    .with(Capability::TextOutput)
                    .with(Capability::AudioInput)
                    .with(Capability::AudioOutput)
                    .with(Capability::Streaming),
                ..Default::default()
            })
            .with_model_info(ModelInfo {
                id: GPT_4O_MINI_REALTIME.into(),
                provider: "chatgpt".into(),
                display_name: "GPT-4o Mini Realtime".into(),
                capabilities: CapabilitySet::new()
                    .with(Capability::TextInput)
                    .with(Capability::TextOutput)
                    .with(Capability::AudioInput)
                    .with(Capability::AudioOutput)
                    .with(Capability::Streaming),
                ..Default::default()
            })
            .with_model_info(ModelInfo {
                id: DALL_E_3.into(),
                provider: "chatgpt".into(),
                display_name: "DALL-E 3".into(),
                capabilities: CapabilitySet::new().with(Capability::ImageGeneration),
                ..Default::default()
            })
            .with_model_info(ModelInfo {
                id: DALL_E_2.into(),
                provider: "chatgpt".into(),
                display_name: "DALL-E 2".into(),
                capabilities: CapabilitySet::new().with(Capability::ImageGeneration),
                ..Default::default()
            });
        Self {
            inner,
            config,
            oauth_token: None,
        }
    }

    /// Override the API key with an OAuth bearer token.
    pub fn with_oauth_token(mut self, token: impl Into<String>) -> Self {
        self.oauth_token = Some(token.into());
        self
    }

    fn effective_api_key(&self) -> secrecy::SecretString {
        if let Some(ref token) = self.oauth_token {
            secrecy::SecretString::new(token.clone().into())
        } else {
            self.config.api_key().clone()
        }
    }

    fn effective_config(&self) -> OpenAiCompatibleConfig {
        use secrecy::ExposeSecret;
        if self.oauth_token.is_some() {
            self.config
                .clone()
                .with_api_key(self.effective_api_key().expose_secret().to_string())
        } else {
            self.config.clone()
        }
    }

    /// Set an optional OpenAI organization ID.
    pub fn with_org(mut self, org_id: impl Into<String>) -> Self {
        self.config = self.config.with_org(org_id);
        // Rebuild inner provider with the updated config so that models
        // created via the Provider trait pick up the org header.
        let models: Vec<ModelInfo> = self.inner.models().to_vec();
        let mut new_inner =
            OpenAiCompatibleProvider::new(self.config.clone(), "chatgpt", "ChatGPT");
        for info in models {
            new_inner = new_inner.with_model_info(info);
        }
        self.inner = new_inner;
        self
    }

    /// Get a specific model by ID, looking up known capabilities.
    /// Any valid OpenAI model ID is accepted.
    pub fn model(&self, model_id: &str) -> OpenAiCompatibleModel {
        let caps = self
            .inner
            .models()
            .iter()
            .find(|m| m.id == model_id)
            .map(|m| m.capabilities.clone())
            .unwrap_or_else(|| {
                CapabilitySet::new()
                    .with(Capability::TextInput)
                    .with(Capability::TextOutput)
                    .with(Capability::Streaming)
            });
        OpenAiCompatibleModel::new(self.effective_config(), model_id, "chatgpt")
            .with_capabilities(caps)
    }

    pub fn gpt4o(&self) -> OpenAiCompatibleModel {
        self.model(GPT_4O_LATEST)
    }

    pub fn gpt4o_mini(&self) -> OpenAiCompatibleModel {
        self.model(GPT_4O_MINI_LATEST)
    }

    pub fn gpt54(&self) -> OpenAiCompatibleModel {
        self.model(GPT_5_4_LATEST)
    }

    pub fn gpt54_mini(&self) -> OpenAiCompatibleModel {
        self.model(GPT_5_4_MINI_LATEST)
    }

    pub fn gpt54_nano(&self) -> OpenAiCompatibleModel {
        self.model(GPT_5_4_NANO_LATEST)
    }

    /// Open a Realtime API WebSocket session for voice/audio streaming.
    ///
    /// # Example
    /// ```no_run
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let provider = rs_ai_providers::chatgpt::ChatGptProvider::new(std::env::var("OPENAI_API_KEY")?);
    /// let mut session = provider.realtime_session(rs_ai_providers::chatgpt::GPT_4O_REALTIME).await?;
    /// session.send_text("Say hello").await?;
    /// while let Some(ev) = session.recv().await {
    ///     if let Ok(rs_ai_providers::chatgpt::realtime_api::ServerEvent::TextDelta { delta, .. }) = ev {
    ///         print!("{delta}");
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::result_large_err)]
    pub async fn realtime_session(
        &self,
        model: &str,
    ) -> Result<RealtimeSession, realtime_api::RealtimeError> {
        use secrecy::ExposeSecret;
        RealtimeSession::connect(self.effective_api_key().expose_secret(), model).await
    }

    /// Create an image generation model (DALL-E 3 or DALL-E 2).
    pub fn image_model(&self, model_id: &str) -> ChatGptImageModel {
        use secrecy::ExposeSecret;
        let mut m = ChatGptImageModel::new(
            model_id,
            self.effective_api_key().expose_secret().to_string(),
            self.config.base_url(),
        );
        if let Some(ref org) = self.config.org_id {
            m = m.with_org(org);
        }
        m
    }

    /// Create a speech-to-text model (Whisper).
    pub fn stt_model(&self, model_id: &str) -> ChatGptSttModel {
        use secrecy::ExposeSecret;
        ChatGptSttModel::new(
            self.effective_api_key().expose_secret().to_string(),
            model_id.to_string(),
            self.config.org_id.clone(),
        )
    }

    /// Create a text-to-speech model (OpenAI TTS).
    pub fn tts_model(&self, model_id: &str) -> ChatGptTtsModel {
        use secrecy::ExposeSecret;
        ChatGptTtsModel::new(
            self.effective_api_key().expose_secret().to_string(),
            model_id.to_string(),
            self.config.org_id.clone(),
        )
    }

    /// Open a unified realtime session by wrapping the provider-specific
    /// [`RealtimeSession`] behind [`rs_ai_core::RealtimeSession`].
    pub async fn unified_realtime_session(
        &self,
        model: &str,
    ) -> rs_ai_core::AiResult<Box<dyn rs_ai_core::RealtimeSession>> {
        let session =
            self.realtime_session(model)
                .await
                .map_err(|e| rs_ai_core::AiError::BridgeError {
                    bridge: "chatgpt_realtime".into(),
                    message: e.to_string(),
                })?;
        Ok(Box::new(ChatGptRealtimeSession::new(session, model)))
    }

    /// Fetch the list of models from the OpenAI API.
    pub async fn list_remote_models(&self) -> rs_ai_core::AiResult<Vec<String>> {
        use secrecy::ExposeSecret;
        let client = reqwest::Client::new();
        let resp = client
            .get("https://api.openai.com/v1/models")
            .header(
                "Authorization",
                format!("Bearer {}", self.effective_api_key().expose_secret()),
            )
            .send()
            .await
            .map_err(|e| rs_ai_core::AiError::Transport {
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        let status = resp.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let body = match resp.text().await {
                Ok(body) => body,
                Err(e) => {
                    tracing::warn!(status = status_code, error = %e, "Failed to read ChatGPT error response body");
                    format!("<failed to read response body: {e}>")
                }
            };
            return Err(rs_ai_core::AiError::ProviderError {
                provider: "chatgpt".into(),
                status: Some(status_code),
                message: body,
            });
        }

        #[derive(serde::Deserialize)]
        struct ListModelsResponse {
            data: Vec<ModelEntry>,
        }
        #[derive(serde::Deserialize)]
        struct ModelEntry {
            id: String,
        }

        let list: ListModelsResponse = resp
            .json()
            .await
            .map_err(|e| rs_ai_core::AiError::Serialization(e.to_string()))?;

        Ok(list.data.into_iter().map(|m| m.id).collect())
    }
}

impl Provider for ChatGptProvider {
    fn id(&self) -> &str {
        self.inner.id()
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn language_model(&self, model_id: &str) -> AiResult<Box<dyn LanguageModel>> {
        Ok(Box::new(self.model(model_id)))
    }

    fn embedding_model(&self, _model_id: &str) -> AiResult<Box<dyn EmbeddingModel>> {
        Err(rs_ai_core::AiError::ModelUnavailable {
            model: "ChatGPT does not support embedding models".into(),
        })
    }

    fn available_models(&self) -> Vec<ModelInfo> {
        self.inner.models().to_vec()
    }

    fn speech_to_text_model(&self, model_id: &str) -> AiResult<Box<dyn SpeechToTextModel>> {
        Ok(Box::new(self.stt_model(model_id)))
    }

    fn text_to_speech_model(&self, model_id: &str) -> AiResult<Box<dyn TextToSpeechModel>> {
        Ok(Box::new(self.tts_model(model_id)))
    }
}

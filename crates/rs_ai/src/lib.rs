//!
//! ⚠️ **UNSTABLE** — This crate is in active development. APIs may change without notice.
//!
//! **rs_ai** - A fluent, ergonomic Rust SDK for AI with 15+ cloud and local providers.
//!
//! # Quick Start
//!
//! ```ignore
//! use rs_ai::gemini;
//!
//! let response = gemini()
//!     .api_key("your-api-key")
//!     .model("gemini-2.5-flash")
//!     .generate("What is 2+2?")
//!     .await?;
//!
//! println!("{}", response);
//! ```
//!
//! # Supported Providers
//!
//! - **Claude** - `rs_ai::claude()`
//! - **ChatGPT** - `rs_ai::chatgpt()`
//! - **Gemini** - `rs_ai::gemini()`
//! - **xAI Grok** - `rs_ai::xai()`
//! - **OpenAI Compatible** - `rs_ai::compatible(base_url)`

use base64::Engine as _;
use futures::stream::BoxStream;
use rs_ai_core::tool::{ToolChoice, ToolDefinition, ToolSet};
use rs_ai_core::{
    agent_loop, AiError, AiResult, CacheConfig, ContentPart, EmbeddingResult, FileData,
    GenerateOptions, ImageData, ImageGenerationOptions, ImageModel, ImageResult, LanguageModel,
    LifecycleCallbacks, Message, OnFinish, OnStepFinish, OnToolCall, Prompt, RealtimeSession,
    RerankResult, SpeechToTextModel, StreamEvent, TextToSpeechModel, TtsOptions,
    VideoGenerationOptions, VideoModel, VideoResult,
};
use rs_ai_providers::chatgpt::ChatGptProvider;
use rs_ai_providers::claude::ClaudeProvider;
use rs_ai_providers::cloudflare::CloudflareProvider;
use rs_ai_providers::gemini::GeminiProvider;
use rs_ai_providers::openai_compatible::{OpenAiCompatibleConfig, OpenAiCompatibleProvider};
use rs_ai_providers::xai::XaiProvider;

pub use rs_ai_providers::xai::XAI_OAUTH_MODEL_IDS;

/// Fluent builder for creating and configuring AI clients.
#[allow(dead_code)]
pub struct ClientBuilder {
    provider_type: ProviderType,
    api_key: Option<String>,
    model_id: Option<String>,
    /// Accumulated image URLs or paths added via [`ClientBuilder::with_image`].
    images: Vec<String>,
    /// Cloudflare AI Gateway path — either `account_id/gateway_id` or a full
    /// `https://gateway.ai.cloudflare.com/v1/...` URL prefix.
    cf_gateway: Option<String>,
    /// Cache configuration (prompt caching, conversation routing, etc.)
    cache_config: Option<CacheConfig>,
    /// Maximum tool-calling steps in the agent loop (default: 1).
    max_steps: u32,
    tools: Option<Vec<ToolDefinition>>,
    tool_choice: Option<ToolChoice>,
    /// Callback after each agent-loop step.
    on_step_finish: Option<OnStepFinish>,
    /// Callback before each tool execution.
    on_tool_call: Option<OnToolCall>,
    /// Callback after final result.
    on_finish: Option<OnFinish>,
    /// OAuth bearer token (overrides API key when set).
    oauth_token: Option<String>,
    /// Rerank model ID for Cohere/Voyage reranking.
    rerank_model_id: Option<String>,
}

#[derive(Debug, Clone)]
enum ProviderType {
    Claude,
    ChatGpt,
    Gemini,
    Xai,
    Cloudflare { account_id: String },
    Compatible { base_url: String },
}

impl ClientBuilder {
    /// Set the API key for the provider.
    ///
    /// # Examples
    /// ```ignore
    /// let response = rs_ai::claude()
    ///     .api_key("sk-ant-...")
    ///     .model("claude-sonnet-4-6")
    ///     .generate("Hello!")
    ///     .await?;
    /// ```
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set the model ID to use.
    ///
    /// # Examples
    /// ```ignore
    /// let response = rs_ai::gemini()
    ///     .model("gemini-2.5-flash")
    ///     .api_key("...")
    ///     .generate("Hello!")
    ///     .await?;
    /// ```
    pub fn model(mut self, id: impl Into<String>) -> Self {
        self.model_id = Some(id.into());
        self
    }

    /// Attach an image to the next request by URL or local file path.
    ///
    /// Multiple calls accumulate images — all of them will be included when
    /// [`generate`](ClientBuilder::generate) or [`stream`](ClientBuilder::stream) is called.
    ///
    /// If the value looks like a URL (starts with `http://` or `https://`) it is
    /// passed as an image URL reference. Gemini does not support URL references
    /// through this API; use a local file path instead. A local file is read,
    /// base64-encoded, and sent inline.
    ///
    /// # Examples
    /// ```ignore
    /// let response = rs_ai::claude()
    ///     .api_key("sk-ant-...")
    ///     .model("claude-sonnet-4-6")
    ///     .with_image("https://example.com/photo.jpg")
    ///     .generate("Describe this image.")
    ///     .await?;
    /// ```
    pub fn with_image(mut self, url_or_path: impl Into<String>) -> Self {
        self.images.push(url_or_path.into());
        self
    }

    /// Route all requests through a Cloudflare AI Gateway.
    ///
    /// Pass either `"account_id/gateway_id"` or a full gateway URL prefix.
    /// The correct provider-specific path is appended automatically.
    ///
    /// # Examples
    /// ```ignore
    /// // Using account_id/gateway_id shorthand
    /// let response = rs_ai::claude()
    ///     .api_key("sk-ant-...")
    ///     .model("claude-sonnet-4-6")
    ///     .cf_ai_gateway("fc62a6e6528bec6d3d81c3bf8967ceeb/my-gateway")
    ///     .generate("Hello!")
    ///     .await?;
    ///
    /// // Same with a full URL prefix
    /// let response = rs_ai::gemini()
    ///     .api_key("AIzaSy...")
    ///     .model("gemini-2.5-flash")
    ///     .cf_ai_gateway("https://gateway.ai.cloudflare.com/v1/abc123/prod")
    ///     .generate("Hello!")
    ///     .await?;
    /// ```
    pub fn cf_ai_gateway(mut self, gateway: impl Into<String>) -> Self {
        self.cf_gateway = Some(gateway.into());
        self
    }

    /// Enable prompt caching for this request.
    ///
    /// Enables provider-specific caching mechanisms:
    /// - **Claude**: Explicit cache_control on message blocks
    /// - **Gemini**: Cached content reuse
    /// - **OpenAI**: Automatic routing-based caching
    /// - **xAI**: Conversation routing + prompt caching
    ///
    /// # Examples
    /// ```ignore
    /// let response = rs_ai::claude()
    ///     .api_key("sk-ant-...")
    ///     .model("claude-sonnet-4-6")
    ///     .with_cache(rai_cache::CacheConfig::new())
    ///     .generate("Long prompt with cached content...")
    ///     .await?;
    /// ```
    pub fn with_cache(mut self, config: CacheConfig) -> Self {
        self.cache_config = Some(config);
        self
    }

    /// Enable caching with default settings.
    ///
    /// Equivalent to `.with_cache(CacheConfig::new())`.
    pub fn enable_cache(self) -> Self {
        self.with_cache(CacheConfig::new())
    }

    /// Set xAI conversation ID for server routing.
    ///
    /// Routes requests with the same conversation ID to the same server,
    /// improving cache hit rates for xAI/Grok models.
    pub fn with_xai_conv_id(mut self, conv_id: impl Into<String>) -> Self {
        let mut cache = self.cache_config.unwrap_or_default();
        cache.xai_conv_id = Some(conv_id.into());
        self.cache_config = Some(cache);
        self
    }

    /// Override the API key with an OAuth bearer token.
    ///
    /// When set, the token is used as the Bearer credential for the provider,
    /// overriding any API key.
    pub fn with_oauth_token(mut self, token: impl Into<String>) -> Self {
        self.oauth_token = Some(token.into());
        self
    }

    /// Set a prompt cache key for improved cache locality.
    ///
    /// Works with OpenAI and xAI models to route requests to the same backend.
    pub fn with_prompt_cache_key(mut self, key: impl Into<String>) -> Self {
        let mut cache = self.cache_config.unwrap_or_default();
        cache.prompt_cache_key = Some(key.into());
        self.cache_config = Some(cache);
        self
    }

    /// Set the maximum number of tool-calling steps in the agent loop.
    /// When > 1, tool calls are automatically executed and fed back to the model.
    /// Default: 1 (single-shot, no auto-loop).
    pub fn max_steps(mut self, n: u32) -> Self {
        self.max_steps = n;
        self
    }

    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Attach a callback invoked after each step in the agent loop.
    pub fn with_on_step_finish(mut self, cb: OnStepFinish) -> Self {
        self.on_step_finish = Some(cb);
        self
    }

    /// Attach a callback invoked before each tool execution.
    /// Return `Err(msg)` from the callback to abort the tool call.
    pub fn with_on_tool_call(mut self, cb: OnToolCall) -> Self {
        self.on_tool_call = Some(cb);
        self
    }

    /// Attach a callback invoked with the final result before returning.
    pub fn with_on_finish(mut self, cb: OnFinish) -> Self {
        self.on_finish = Some(cb);
        self
    }

    /// Generate text from the configured model.
    ///
    /// When images have been attached via [`with_image`](ClientBuilder::with_image) the
    /// prompt is sent as a `Prompt::Messages` containing those images alongside the
    /// text, enabling vision/multimodal requests.
    ///
    /// # Errors
    ///
    /// Returns an error if the API key or model ID is not set, or if the request fails.
    ///
    /// # Examples
    /// ```ignore
    /// let response = rs_ai::claude()
    ///     .api_key("sk-ant-...")
    ///     .model("claude-sonnet-4-6")
    ///     .generate("What is 2+2?")
    ///     .await?;
    /// ```
    pub async fn generate(self, prompt: impl Into<String>) -> AiResult<String> {
        let text = prompt.into();
        let images = self.images.clone();
        let client = self.build().await?;
        if images.is_empty() {
            client.generate(text).await
        } else {
            client.generate_with_images(text, images).await
        }
    }

    pub async fn generate_prompt(self, prompt: Prompt) -> AiResult<String> {
        if !self.images.is_empty() {
            return Err(AiError::ProviderError {
                provider: "rs_ai".to_string(),
                status: None,
                message: "Images cannot be combined with a structured prompt".to_string(),
            });
        }
        self.build().await?.generate_prompt(prompt).await
    }

    /// Stream text generation from the configured model.
    ///
    /// When images have been attached via [`with_image`](ClientBuilder::with_image) the
    /// prompt is sent as a `Prompt::Messages` containing those images, enabling
    /// vision/multimodal streaming requests.
    ///
    /// # Errors
    ///
    /// Returns an error if the API key or model ID is not set, or if the request fails.
    ///
    /// # Examples
    /// ```ignore
    /// use futures::StreamExt;
    ///
    /// let mut stream = rs_ai::claude()
    ///     .api_key("sk-ant-...")
    ///     .model("claude-sonnet-4-6")
    ///     .stream("Hello!")
    ///     .await?;
    ///
    /// while let Some(event) = stream.next().await {
    ///     match event? {
    ///         StreamEvent::TextDelta { text } => print!("{}", text),
    ///         _ => {}
    ///     }
    /// }
    /// ```
    pub async fn stream(
        self,
        prompt: impl Into<String>,
    ) -> AiResult<BoxStream<'static, AiResult<StreamEvent>>> {
        let text = prompt.into();
        let images = self.images.clone();
        let client = self.build().await?;
        if images.is_empty() {
            client.stream(text).await
        } else {
            client.stream_with_images(text, images).await
        }
    }

    pub async fn stream_prompt(
        self,
        prompt: Prompt,
    ) -> AiResult<BoxStream<'static, AiResult<StreamEvent>>> {
        if !self.images.is_empty() {
            return Err(AiError::ProviderError {
                provider: "rs_ai".to_string(),
                status: None,
                message: "Images cannot be combined with a structured prompt".to_string(),
            });
        }
        self.build().await?.stream_prompt(prompt).await
    }

    /// Synthesize speech from text, returning audio bytes.
    ///
    /// For providers that support native TTS this will eventually delegate to the
    /// provider's TTS endpoint. As a sensible degraded path (when the underlying
    /// `LanguageModel` does not expose native audio synthesis) the text is returned
    /// as UTF-8 bytes so callers always receive a `Vec<u8>`.
    ///
    /// # Errors
    ///
    /// Returns an error if the API key or model ID is not set, or if the request fails.
    ///
    /// # Examples
    /// ```ignore
    /// let audio_bytes = rs_ai::chatgpt()
    ///     .api_key("sk-...")
    ///     .model("gpt-4o-audio-preview")
    ///     .speak("Hello, world!")
    ///     .await?;
    /// ```
    pub async fn speak(self, text: impl Into<String>) -> AiResult<Vec<u8>> {
        let text_str = text.into();
        let api_key = self.api_key.clone();
        let model_id = self.model_id.clone();
        let provider_type = self.provider_type.clone();

        match provider_type {
            ProviderType::ChatGpt => {
                let key = api_key.ok_or_else(|| AiError::AuthError {
                    message: "API key not set. Use .api_key() to specify credentials.".to_string(),
                })?;
                let provider = ChatGptProvider::new(key);
                let model = provider.tts_model(&model_id.unwrap_or_default());
                let result = model
                    .synthesize(&text_str, "alloy", TtsOptions::default())
                    .await?;
                Ok(result.audio)
            }
            ProviderType::Xai => {
                let key = api_key.ok_or_else(|| AiError::AuthError {
                    message: "API key not set. Use .api_key() to specify credentials.".to_string(),
                })?;
                let provider = XaiProvider::new(key);
                let model = provider.speech_model(&model_id.unwrap_or_default());
                let result = model
                    .synthesize(&text_str, "alloy", TtsOptions::default())
                    .await?;
                Ok(result.audio)
            }
            ProviderType::Gemini => {
                let key = api_key.ok_or_else(|| AiError::AuthError {
                    message: "API key not set. Use .api_key() to specify credentials.".to_string(),
                })?;
                let provider = GeminiProvider::new(key);
                let model = provider.speech_model(&model_id.unwrap_or_default());
                let result = model
                    .synthesize(&text_str, "en-US-Standard-A", TtsOptions::default())
                    .await?;
                Ok(result.audio)
            }
            _ => {
                let options = GenerateOptions::default();
                let model_id_hint = model_id.unwrap_or_default();
                let client = self.build().await?;
                let result = client
                    .model
                    .as_ref()
                    .as_ref()
                    .generate(Prompt::Text(text_str.clone()), options)
                    .await?;
                if let Some(audio_b64) = result.metadata.extra.get("audio_bytes") {
                    if let Some(encoded) = audio_b64.as_str() {
                        let bytes = base64::engine::general_purpose::STANDARD
                            .decode(encoded)
                            .map_err(|e| {
                                AiError::Serialization(format!(
                                    "Failed to decode audio_bytes from model `{}` metadata: {e}",
                                    model_id_hint
                                ))
                            })?;
                        return Ok(bytes);
                    }
                }
                let spoken_text = result.text.unwrap_or(text_str);
                Ok(spoken_text.into_bytes())
            }
        }
    }

    /// Generate an image from a text prompt.
    ///
    /// Uses the provider's image model (DALL-E, Grok Imagine, Imagen, etc.).
    ///
    /// # Examples
    /// ```ignore
    /// let result = rs_ai::chatgpt()
    ///     .api_key("sk-...")
    ///     .model("dall-e-3")
    ///     .generate_image("A cat wearing a hat", rs_ai_core::ImageGenerationOptions::default())
    ///     .await?;
    /// println!("Got {} image(s)", result.images.len());
    /// ```
    pub async fn generate_image(
        self,
        prompt: impl Into<String>,
        options: ImageGenerationOptions,
    ) -> AiResult<ImageResult> {
        let text = prompt.into();
        let api_key = self.api_key.clone().ok_or_else(|| AiError::AuthError {
            message: "API key not set. Use .api_key() to specify credentials.".to_string(),
        })?;

        match self.provider_type {
            ProviderType::ChatGpt => {
                let provider = ChatGptProvider::new(api_key);
                let model = provider.image_model(&self.model_id.unwrap_or_default());
                model.generate_image(&text, options).await
            }
            ProviderType::Xai => {
                let provider = XaiProvider::new(api_key);
                let model = provider.image_model(&self.model_id.unwrap_or_default());
                model.generate_image(&text, options).await
            }
            ProviderType::Gemini => {
                let provider = GeminiProvider::new(api_key);
                let model = provider.image_model(&self.model_id.unwrap_or_default());
                model.generate_image(&text, options).await
            }
            _ => Err(AiError::UnsupportedCapability {
                capability: "image_generation".to_string(),
                provider: format!("{:?}", self.provider_type),
            }),
        }
    }

    /// Generate a video from a text prompt.
    ///
    /// Uses the provider's video model (Sora, Veo, etc.).
    ///
    /// # Examples
    /// ```ignore
    /// let result = rs_ai::gemini()
    ///     .api_key("AIzaSy...")
    ///     .model("veo-3.0-generate-001")
    ///     .generate_video("A dog playing in a park", rs_ai_core::VideoGenerationOptions::default())
    ///     .await?;
    /// ```
    pub async fn generate_video(
        self,
        prompt: impl Into<String>,
        options: VideoGenerationOptions,
    ) -> AiResult<VideoResult> {
        let text = prompt.into();
        let api_key = self.api_key.clone().ok_or_else(|| AiError::AuthError {
            message: "API key not set. Use .api_key() to specify credentials.".to_string(),
        })?;

        match self.provider_type {
            ProviderType::ChatGpt => Err(AiError::UnsupportedCapability {
                capability: "video_generation".to_string(),
                provider: "chatgpt".to_string(),
            }),
            ProviderType::Xai => {
                let provider = XaiProvider::new(api_key);
                let model = provider.video_model(&self.model_id.unwrap_or_default());
                model.generate_video(&text, options).await
            }
            ProviderType::Gemini => {
                let provider = GeminiProvider::new(api_key);
                let model = provider.video_model(&self.model_id.unwrap_or_default());
                model.generate_video(&text, options).await
            }
            _ => Err(AiError::UnsupportedCapability {
                capability: "video_generation".to_string(),
                provider: format!("{:?}", self.provider_type),
            }),
        }
    }

    /// Open a realtime voice/text session.
    ///
    /// # Examples
    /// ```ignore
    /// let mut session = rs_ai::chatgpt()
    ///     .api_key("sk-...")
    ///     .model("gpt-4o-realtime-preview")
    ///     .realtime_session()
    ///     .await?;
    /// session.send_text("Hello!").await?;
    /// while let Some(event) = session.recv().await {
    ///     match event {
    ///         RealtimeEvent::TextDelta { delta } => print!("{delta}"),
    ///         _ => {}
    ///     }
    /// }
    /// ```
    pub async fn realtime_session(self) -> AiResult<Box<dyn RealtimeSession>> {
        let api_key = self.api_key.clone().ok_or_else(|| AiError::AuthError {
            message: "API key not set. Use .api_key() to specify credentials.".to_string(),
        })?;
        let model_id = self.model_id.clone().unwrap_or_default();

        match self.provider_type {
            ProviderType::ChatGpt => {
                let provider = ChatGptProvider::new(api_key);
                provider.unified_realtime_session(&model_id).await
            }
            ProviderType::Gemini => {
                let provider = GeminiProvider::new(api_key);
                provider.unified_realtime_session(&model_id).await
            }
            _ => Err(AiError::UnsupportedCapability {
                capability: "realtime_session".to_string(),
                provider: format!("{:?}", self.provider_type),
            }),
        }
    }

    /// Generate embeddings for a list of texts.
    ///
    /// Uses the provider's embedding model. Currently supported:
    /// - **ChatGPT** (OpenAI `text-embedding-3-small`, `text-embedding-3-large`, etc.)
    ///
    /// # Examples
    /// ```ignore
    /// let embeddings = rs_ai::chatgpt()
    ///     .api_key("sk-...")
    ///     .model("text-embedding-3-small")
    ///     .embed(vec!["Hello world".into(), "Foo bar".into()])
    ///     .await?;
    /// ```
    pub async fn embed(self, texts: Vec<String>) -> AiResult<EmbeddingResult> {
        let api_key = self.api_key.clone().ok_or_else(|| AiError::AuthError {
            message: "API key not set. Use .api_key() to specify credentials.".to_string(),
        })?;
        let model_id = self.model_id.clone().unwrap_or_default();

        match self.provider_type {
            ProviderType::ChatGpt => {
                use rs_ai_core::Provider as _;
                let provider = ChatGptProvider::new(api_key);
                let model = provider.embedding_model(&model_id)?;
                model.embed(texts).await
            }
            _ => Err(AiError::UnsupportedCapability {
                capability: "embedding".to_string(),
                provider: format!("{:?}", self.provider_type),
            }),
        }
    }

    /// Embed multiple texts (batch).
    pub async fn embed_many(self, texts: Vec<String>) -> AiResult<EmbeddingResult> {
        self.embed(texts).await
    }

    /// Rerank documents by relevance to a query (Vercel AI SDK `rerank()`).
    ///
    /// Currently no provider implements reranking. This is a placeholder.
    pub async fn rerank(
        self,
        _query: impl Into<String>,
        _documents: Vec<String>,
    ) -> AiResult<RerankResult> {
        Err(AiError::UnsupportedCapability {
            capability: "reranking".to_string(),
            provider: format!(
                "{:?} — enable the `cohere` or `voyage` feature on rs_ai_providers and use the provider directly",
                self.provider_type
            ),
        })
    }

    /// Transcribe audio bytes into text.
    ///
    /// Builds a `Prompt::Messages` that embeds the audio as a base64-encoded
    /// `ContentPart::File`. Providers that support audio input (e.g. Gemini,
    /// GPT-4o-audio) will transcribe the content; providers that do not will
    /// return [`AiError::UnsupportedCapability`].
    ///
    /// # Arguments
    ///
    /// * `audio` - Raw audio bytes.
    /// * `mime_type` - MIME type of the audio, e.g. `"audio/wav"` or `"audio/mp3"`.
    ///
    /// # Errors
    ///
    /// Returns [`AiError::UnsupportedCapability`] when the provider cannot process
    /// audio input, or other errors if the API key / model ID are missing or the
    /// request fails.
    ///
    /// # Examples
    /// ```ignore
    /// let transcript = rs_ai::gemini()
    ///     .api_key("AIzaSy...")
    ///     .model("gemini-2.5-flash")
    ///     .transcribe(audio_bytes, "audio/wav")
    ///     .await?;
    /// ```
    pub async fn transcribe(self, audio: Vec<u8>, mime_type: &str) -> AiResult<String> {
        let api_key = self.api_key.clone();
        let model_id = self.model_id.clone();
        let provider_type = self.provider_type.clone();

        match provider_type {
            ProviderType::ChatGpt => {
                let key = api_key.ok_or_else(|| AiError::AuthError {
                    message: "API key not set. Use .api_key() to specify credentials.".to_string(),
                })?;
                let provider = ChatGptProvider::new(key);
                let model = provider.stt_model(&model_id.unwrap_or_default());
                let result = model.transcribe(audio, mime_type, None).await?;
                Ok(result.text)
            }
            ProviderType::Xai => {
                let key = api_key.ok_or_else(|| AiError::AuthError {
                    message: "API key not set. Use .api_key() to specify credentials.".to_string(),
                })?;
                let provider = XaiProvider::new(key);
                let model = provider.stt_model(&model_id.unwrap_or_default());
                let result = model.transcribe(audio, mime_type, None).await?;
                Ok(result.text)
            }
            _ => {
                let encoded = base64::engine::general_purpose::STANDARD.encode(&audio);
                let audio_part = ContentPart::File {
                    data: FileData::Base64 {
                        media_type: mime_type.to_string(),
                        data: encoded,
                    },
                };
                let instruction_part = ContentPart::Text {
                    text: "Transcribe the audio in the attached file. Return only the transcription text, no commentary.".to_string(),
                };
                let message = Message {
                    role: rs_ai_core::Role::User,
                    content: vec![instruction_part, audio_part],
                    name: None,
                    metadata: std::collections::HashMap::new(),
                };
                let prompt = Prompt::Messages(vec![message]);
                let model_id_hint = model_id.unwrap_or_default();
                let client = self.build().await?;
                let result = client
                    .model
                    .as_ref()
                    .as_ref()
                    .generate(prompt, GenerateOptions::default())
                    .await
                    .map_err(|e| match &e {
                        AiError::ProviderError { message, .. }
                            if message.to_lowercase().contains("audio")
                                || message.to_lowercase().contains("unsupported")
                                || message.to_lowercase().contains("media type") =>
                        {
                            AiError::UnsupportedCapability {
                                capability: "audio_transcription".to_string(),
                                provider: model_id_hint.clone(),
                            }
                        }
                        _ => e,
                    })?;
                result.text.ok_or_else(|| AiError::UnsupportedCapability {
                    capability: "audio_transcription".to_string(),
                    provider: model_id_hint,
                })
            }
        }
    }

    async fn build(self) -> AiResult<Client> {
        let model_id = self.model_id.ok_or_else(|| AiError::ProviderError {
            provider: "rs_ai".to_string(),
            status: None,
            message: "Model ID not set. Use .model() to specify a model.".to_string(),
        })?;

        let api_key = self
            .api_key
            .or_else(|| self.oauth_token.clone())
            .ok_or_else(|| AiError::AuthError {
                message: "API key not set. Use .api_key() to specify credentials.".to_string(),
            })?;

        // Compute the CF AI Gateway base URL if configured.
        // Accepts either "account_id/gateway_id" or a full URL prefix.
        let cf_base = self.cf_gateway.as_deref().map(|gw| {
            if gw.starts_with("https://") || gw.starts_with("http://") {
                gw.trim_end_matches('/').to_string()
            } else {
                format!(
                    "https://gateway.ai.cloudflare.com/v1/{}",
                    gw.trim_end_matches('/')
                )
            }
        });

        let model: Box<dyn LanguageModel> = match self.provider_type {
            ProviderType::Claude => {
                let claude_key = self.oauth_token.clone().unwrap_or(api_key);
                let mut provider = ClaudeProvider::new(claude_key);
                if let Some(gw) = &cf_base {
                    provider = provider.with_base_url(format!("{}/anthropic", gw));
                }
                let mut m = provider.model(&model_id);
                if let Some(cache_cfg) = &self.cache_config {
                    m.set_cache(cache_cfg.clone());
                }
                Box::new(m)
            }
            ProviderType::ChatGpt => {
                if let Some(gw) = &cf_base {
                    let config = OpenAiCompatibleConfig::new(format!("{}/openai", gw), api_key);
                    let provider = OpenAiCompatibleProvider::new(config, "chatgpt", "ChatGPT");
                    provider.language_model(&model_id)
                } else {
                    let mut provider = ChatGptProvider::new(api_key);
                    if let Some(ref token) = self.oauth_token {
                        provider = provider.with_oauth_token(token.clone());
                    }
                    let mut m = provider.model(&model_id);
                    if let Some(cache_cfg) = &self.cache_config {
                        m.set_cache(cache_cfg.clone());
                    }
                    Box::new(m)
                }
            }
            ProviderType::Gemini => {
                let provider = GeminiProvider::new(api_key);
                if let Some(gw) = &cf_base {
                    let mut m = provider.model_with_base_url(
                        &model_id,
                        format!("{}/google-ai-studio/v1/models", gw),
                    );
                    if let Some(cache_cfg) = &self.cache_config {
                        m.set_cache(cache_cfg.clone());
                    }
                    Box::new(m)
                } else {
                    let mut m = provider.model(&model_id);
                    if let Some(cache_cfg) = &self.cache_config {
                        m.set_cache(cache_cfg.clone());
                    }
                    Box::new(m)
                }
            }
            ProviderType::Xai => {
                if let Some(gw) = &cf_base {
                    let config = OpenAiCompatibleConfig::new(format!("{}/grok", gw), api_key);
                    let provider = OpenAiCompatibleProvider::new(config, "xai", "xAI Grok");
                    provider.language_model(&model_id)
                } else {
                    let mut provider = XaiProvider::new(api_key);
                    if let Some(ref token) = self.oauth_token {
                        provider = provider.with_oauth_token(token.clone());
                    }
                    let mut m = provider.model(&model_id);
                    if let Some(cache_cfg) = &self.cache_config {
                        m.set_cache(cache_cfg.clone());
                    }
                    Box::new(m)
                }
            }
            ProviderType::Cloudflare { account_id } => {
                let provider = CloudflareProvider::new(account_id, api_key);
                Box::new(provider.model(&model_id))
            }
            ProviderType::Compatible { base_url } => {
                let config = OpenAiCompatibleConfig::new(&base_url, &api_key);
                let provider = OpenAiCompatibleProvider::new(config, "custom", "OpenAI Compatible");
                provider.language_model(&model_id)
            }
        };

        let mut options = GenerateOptions::default().with_max_steps(self.max_steps);

        if let Some(tools) = self.tools {
            options = options.with_tools(tools);
        }
        if let Some(tool_choice) = self.tool_choice {
            options = options.with_tool_choice(tool_choice);
        }

        let callbacks = LifecycleCallbacks {
            on_step_finish: self.on_step_finish,
            on_tool_call: self.on_tool_call,
            on_finish: self.on_finish,
        };
        if callbacks.on_step_finish.is_some()
            || callbacks.on_tool_call.is_some()
            || callbacks.on_finish.is_some()
        {
            options = options.with_callbacks(callbacks);
        }

        Ok(Client {
            model: std::sync::Arc::new(model),
            options,
        })
    }
}

/// Build an [`ImageData`] from a URL string or a local file path.
///
/// - If the string starts with `http://` or `https://` it is used as a URL reference.
/// - Otherwise it is assumed to be a file path; the file is read and base64-encoded.
async fn image_data_from_url_or_path(url_or_path: &str) -> AiResult<ImageData> {
    if url_or_path.starts_with("http://") || url_or_path.starts_with("https://") {
        Ok(ImageData::Url {
            url: url_or_path.to_string(),
            detail: None,
        })
    } else {
        // Treat as a local file path.
        let bytes = tokio::fs::read(url_or_path)
            .await
            .map_err(|e| AiError::Transport {
                message: format!("Failed to read image file `{url_or_path}`: {e}"),
                source: None,
            })?;
        let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
        // Infer a basic media type from the file extension.
        let media_type = if url_or_path.ends_with(".png") {
            "image/png"
        } else if url_or_path.ends_with(".gif") {
            "image/gif"
        } else if url_or_path.ends_with(".webp") {
            "image/webp"
        } else {
            // Default to JPEG for unknown extensions.
            "image/jpeg"
        };
        Ok(ImageData::Base64 {
            media_type: media_type.to_string(),
            data: encoded,
        })
    }
}

/// Build a user `Message` that contains a text prompt and one or more images.
async fn build_vision_message(text: String, images: Vec<String>) -> AiResult<Message> {
    let mut content = vec![ContentPart::Text { text }];
    for img in images {
        let data = image_data_from_url_or_path(&img).await?;
        content.push(ContentPart::Image { data });
    }
    Ok(Message {
        role: rs_ai_core::Role::User,
        content,
        name: None,
        metadata: std::collections::HashMap::new(),
    })
}

/// A configured AI client ready for operations.
///
/// Can be cloned and reused to make multiple requests with the same configuration.
///
/// # Examples
///
/// Pre-configure and reuse:
/// ```ignore
/// let ai = rs_ai::claude()
///     .api_key("sk-ant-...")
///     .model("claude-sonnet-4-6");
///
/// let response1 = ai.generate("Hello").await?;
/// let response2 = ai.generate("Hi again").await?;
/// ```
#[derive(Clone)]
pub struct Client {
    model: std::sync::Arc<Box<dyn LanguageModel>>,
    options: GenerateOptions,
}

impl Client {
    /// Generate text from the model.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let response = ai.generate("What is 2+2?").await?;
    /// println!("{}", response);
    /// ```
    pub async fn generate(&self, prompt: impl Into<String>) -> AiResult<String> {
        self.generate_prompt(Prompt::Text(prompt.into())).await
    }

    pub async fn generate_prompt(&self, prompt: Prompt) -> AiResult<String> {
        let result = if self.options.max_steps.unwrap_or(1) > 1 {
            let tools = ToolSet::new();
            agent_loop(
                self.model.as_ref().as_ref(),
                prompt,
                self.options.clone(),
                &tools,
            )
            .await?
        } else {
            self.model
                .as_ref()
                .as_ref()
                .generate(prompt, self.options.clone())
                .await?
        };

        result.text.ok_or_else(|| AiError::ProviderError {
            provider: self.model.as_ref().as_ref().provider_id().to_string(),
            status: None,
            message: "No text in response (model returned only tool calls)".to_string(),
        })
    }

    /// Generate text from the model using a vision prompt that includes images.
    ///
    /// # Errors
    ///
    /// Returns an error if any image cannot be loaded or the request fails.
    pub async fn generate_with_images(
        &self,
        prompt: impl Into<String>,
        images: Vec<String>,
    ) -> AiResult<String> {
        let message = build_vision_message(prompt.into(), images).await?;
        let result = self
            .model
            .as_ref()
            .as_ref()
            .generate(Prompt::Messages(vec![message]), GenerateOptions::default())
            .await?;

        result.text.ok_or_else(|| AiError::ProviderError {
            provider: self.model.as_ref().as_ref().provider_id().to_string(),
            status: None,
            message: "No text in response (model returned only tool calls)".to_string(),
        })
    }

    /// Stream text generation from the model.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use futures::StreamExt;
    ///
    /// let mut stream = ai.stream("Hello").await?;
    /// while let Some(event) = stream.next().await {
    ///     match event? {
    ///         StreamEvent::TextDelta { text } => print!("{}", text),
    ///         _ => {}
    ///     }
    /// }
    /// ```
    pub async fn stream(
        &self,
        prompt: impl Into<String>,
    ) -> AiResult<BoxStream<'static, AiResult<StreamEvent>>> {
        self.stream_prompt(Prompt::Text(prompt.into())).await
    }

    pub async fn stream_prompt(
        &self,
        prompt: Prompt,
    ) -> AiResult<BoxStream<'static, AiResult<StreamEvent>>> {
        self.model
            .as_ref()
            .as_ref()
            .stream(prompt, self.options.clone())
            .await
    }

    /// Stream text generation from the model using a vision prompt that includes images.
    ///
    /// # Errors
    ///
    /// Returns an error if any image cannot be loaded or the request fails.
    pub async fn stream_with_images(
        &self,
        prompt: impl Into<String>,
        images: Vec<String>,
    ) -> AiResult<BoxStream<'static, AiResult<StreamEvent>>> {
        let message = build_vision_message(prompt.into(), images).await?;
        self.model
            .as_ref()
            .as_ref()
            .stream(Prompt::Messages(vec![message]), GenerateOptions::default())
            .await
    }

    /// Get a reference to the underlying language model.
    pub fn model(&self) -> &dyn LanguageModel {
        self.model.as_ref().as_ref()
    }
}

/// Create a client builder for Claude.
///
/// # Examples
/// ```ignore
/// let response = rs_ai::claude()
///     .api_key("sk-ant-...")
///     .model("claude-sonnet-4-6")
///     .generate("Hello!")
///     .await?;
/// ```
pub fn claude() -> ClientBuilder {
    ClientBuilder {
        provider_type: ProviderType::Claude,
        api_key: None,
        model_id: None,
        images: Vec::new(),
        cf_gateway: None,
        cache_config: None,
        max_steps: 1,
        tools: None,
        tool_choice: None,
        on_step_finish: None,
        on_tool_call: None,
        on_finish: None,
        oauth_token: None,
        rerank_model_id: None,
    }
}

/// Create a client builder for ChatGPT.
///
/// # Examples
/// ```ignore
/// let response = rs_ai::chatgpt()
///     .api_key("sk-...")
///     .model("gpt-4o")
///     .generate("Hello!")
///     .await?;
/// ```
pub fn chatgpt() -> ClientBuilder {
    ClientBuilder {
        provider_type: ProviderType::ChatGpt,
        api_key: None,
        model_id: None,
        images: Vec::new(),
        cf_gateway: None,
        cache_config: None,
        max_steps: 1,
        tools: None,
        tool_choice: None,
        on_step_finish: None,
        on_tool_call: None,
        on_finish: None,
        oauth_token: None,
        rerank_model_id: None,
    }
}

/// Create a client builder for Gemini.
///
/// # Examples
/// ```ignore
/// let response = rs_ai::gemini()
///     .api_key("AIzaSy...")
///     .model("gemini-2.5-flash")
///     .generate("Hello!")
///     .await?;
/// ```
pub fn gemini() -> ClientBuilder {
    ClientBuilder {
        provider_type: ProviderType::Gemini,
        api_key: None,
        model_id: None,
        images: Vec::new(),
        cf_gateway: None,
        cache_config: None,
        max_steps: 1,
        tools: None,
        tool_choice: None,
        on_step_finish: None,
        on_tool_call: None,
        on_finish: None,
        oauth_token: None,
        rerank_model_id: None,
    }
}

/// Create a client builder for xAI Grok.
///
/// # Examples
/// ```ignore
/// let response = rs_ai::xai()
///     .api_key("xai-...")
///     .model("grok-4.20-reasoning")
///     .generate("Hello!")
///     .await?;
/// ```
pub fn xai() -> ClientBuilder {
    ClientBuilder {
        provider_type: ProviderType::Xai,
        api_key: None,
        model_id: None,
        images: Vec::new(),
        cf_gateway: None,
        cache_config: None,
        max_steps: 1,
        tools: None,
        tool_choice: None,
        on_step_finish: None,
        on_tool_call: None,
        on_finish: None,
        oauth_token: None,
        rerank_model_id: None,
    }
}

/// Create a client builder for Cloudflare Workers AI.
///
/// Cloudflare acts as a provider gateway with built-in analytics, rate limiting,
/// and access to 100+ open-source models at the edge.
///
/// # Examples
/// ```ignore
/// let response = rs_ai::cloudflare("your-account-id")
///     .api_key("your-cf-api-token")
///     .model("@cf/meta/llama-3.1-8b-instruct")
///     .generate("Hello!")
///     .await?;
/// ```
pub fn cloudflare(account_id: impl Into<String>) -> ClientBuilder {
    ClientBuilder {
        provider_type: ProviderType::Cloudflare {
            account_id: account_id.into(),
        },
        api_key: None,
        model_id: None,
        images: Vec::new(),
        cf_gateway: None,
        cache_config: None,
        max_steps: 1,
        tools: None,
        tool_choice: None,
        on_step_finish: None,
        on_tool_call: None,
        on_finish: None,
        oauth_token: None,
        rerank_model_id: None,
    }
}

/// Create a client builder for an OpenAI-compatible endpoint.
///
/// # Examples
/// ```ignore
/// let response = rs_ai::compatible("https://api.openrouter.ai/api/v1")
///     .api_key("sk-...")
///     .model("meta-llama/llama-2-70b-chat")
///     .generate("Hello!")
///     .await?;
/// ```
pub fn compatible(base_url: impl Into<String>) -> ClientBuilder {
    ClientBuilder {
        provider_type: ProviderType::Compatible {
            base_url: base_url.into(),
        },
        api_key: None,
        model_id: None,
        images: Vec::new(),
        cf_gateway: None,
        cache_config: None,
        max_steps: 1,
        tools: None,
        tool_choice: None,
        on_step_finish: None,
        on_tool_call: None,
        on_finish: None,
        oauth_token: None,
        rerank_model_id: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn builder_preserves_tool_configuration() {
        let tool = ToolDefinition {
            name: "click".to_string(),
            description: "Click a point".to_string(),
            parameters: serde_json::json!({"type": "object"}),
            examples: None,
        };
        let client = compatible("https://example.com")
            .api_key("test")
            .model("test")
            .with_tools(vec![tool])
            .with_tool_choice(ToolChoice::Required)
            .build()
            .await
            .unwrap();

        let tools = client.options.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "click");
        assert!(matches!(
            client.options.tool_choice,
            Some(ToolChoice::Required)
        ));
    }
}

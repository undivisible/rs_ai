use async_trait::async_trait;
use futures::StreamExt;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use secrecy::ExposeSecret;

use rs_ai_core::{
    AiError, AiResult, AiStream, CapabilitySet, GenerateOptions, GenerateResult, LanguageModel,
    Prompt,
};

use super::api_types::{ApiErrorResponse, ChatCompletionRequest, ChatCompletionResponse};
use super::config::OpenAiCompatibleConfig;
use super::convert;
use super::stream_parser;

/// A language model backed by an OpenAI-compatible HTTP API.
pub struct OpenAiCompatibleModel {
    config: OpenAiCompatibleConfig,
    model_id: String,
    provider_id: String,
    capabilities: CapabilitySet,
    client: reqwest::Client,
    cache_config: Option<rs_ai_core::CacheConfig>,
}

impl OpenAiCompatibleModel {
    /// Create a new model instance.
    ///
    /// # Panics
    ///
    /// Panics if the system TLS stack cannot be initialized. This is a
    /// system-level failure (e.g. missing TLS libraries). Use [`try_new`] to
    /// handle this case without panicking.
    ///
    /// [`try_new`]: OpenAiCompatibleModel::try_new
    pub fn new(config: OpenAiCompatibleConfig, model_id: &str, provider_id: &str) -> Self {
        Self::try_new(config, model_id, provider_id)
            .expect("Failed to initialize HTTP client (TLS unavailable or system misconfigured)")
    }

    /// Create a new model instance, returning an error if the HTTP client
    /// cannot be initialized (e.g. TLS is unavailable on the system).
    pub fn try_new(
        config: OpenAiCompatibleConfig,
        model_id: &str,
        provider_id: &str,
    ) -> AiResult<Self> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let auth_value = format!("Bearer {}", config.api_key.expose_secret());
        if let Ok(v) = HeaderValue::from_str(&auth_value) {
            headers.insert(AUTHORIZATION, v);
        }

        if let Some(ref org) = config.org_id {
            if let Ok(v) = HeaderValue::from_str(org) {
                headers.insert("OpenAI-Organization", v);
            }
        }

        for (name, value) in &config.default_headers {
            if let (Ok(n), Ok(v)) = (
                HeaderName::try_from(name.as_str()),
                HeaderValue::from_str(value),
            ) {
                headers.insert(n, v);
            }
        }

        let client = reqwest::Client::builder()
            .default_headers(headers)
            .build()
            .map_err(|e| AiError::Transport {
                message: format!("Failed to build HTTP client: {e}"),
                source: Some(Box::new(e)),
            })?;

        Ok(Self {
            config,
            model_id: model_id.to_string(),
            provider_id: provider_id.to_string(),
            capabilities: CapabilitySet::new(),
            client,
            cache_config: None,
        })
    }

    /// Builder-style setter for capabilities.
    pub fn with_capabilities(mut self, caps: CapabilitySet) -> Self {
        self.capabilities = caps;
        self
    }

    /// Set cache configuration.
    pub fn set_cache(&mut self, config: rs_ai_core::CacheConfig) -> &mut Self {
        self.cache_config = Some(config);
        self
    }

    /// Set cache configuration with a cache key.
    pub fn with_cache_key(mut self, key: impl Into<String>) -> Self {
        let mut config = rs_ai_core::CacheConfig::new();
        config = config.with_openai_cache_key(key);
        self.cache_config = Some(config);
        self
    }

    /// The chat completions endpoint URL.
    fn endpoint(&self) -> String {
        let base = self.config.base_url.as_str().trim_end_matches('/');
        format!("{}/chat/completions", base)
    }

    /// Execute a non-streaming request and return the parsed response.
    async fn do_request(&self, request: ChatCompletionRequest) -> AiResult<ChatCompletionResponse> {
        let url = self.endpoint();
        tracing::debug!(url = %url, model = %request.model, "sending chat completion request");

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| AiError::Transport {
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        let status = response.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let body = match response.text().await {
                Ok(body) => body,
                Err(e) => {
                    tracing::warn!(status = status_code, error = %e, "Failed to read error response body");
                    format!("<failed to read response body: {e}>")
                }
            };

            // Attempt to parse structured error.
            if let Ok(api_err) = serde_json::from_str::<ApiErrorResponse>(&body) {
                return Err(map_api_error(status_code, &api_err.error.message));
            }

            return Err(map_api_error(status_code, &body));
        }

        let body = response.text().await.map_err(|e| AiError::Transport {
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;

        serde_json::from_str::<ChatCompletionResponse>(&body).map_err(|e| {
            tracing::error!(body = %body, error = %e, "failed to deserialize response");
            AiError::Serialization(e.to_string())
        })
    }

    /// Execute a streaming request and return a boxed stream.
    async fn do_stream_request(&self, request: ChatCompletionRequest) -> AiResult<AiStream> {
        let url = self.endpoint();
        tracing::debug!(url = %url, model = %request.model, "sending streaming chat completion request");

        let req = self.client.post(&url).json(&request);

        let mut es =
            reqwest_eventsource::EventSource::new(req).map_err(|e| AiError::Transport {
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        // Spawn the EventSource into a channel-based stream to avoid lifetime
        // issues. EventSource must be polled to completion.
        let (tx, rx) = tokio::sync::mpsc::channel(64);

        tokio::spawn(async move {
            while let Some(event) = es.next().await {
                match &event {
                    Ok(reqwest_eventsource::Event::Open) => {}
                    Ok(reqwest_eventsource::Event::Message(msg)) => {
                        if msg.data.trim() == "[DONE]" {
                            let _ = tx.send(event).await;
                            es.close();
                            break;
                        }
                    }
                    Err(_) => {
                        let _ = tx.send(event).await;
                        es.close();
                        break;
                    }
                }
                if tx.send(event).await.is_err() {
                    es.close();
                    break;
                }
            }
        });

        let receiver_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        let parsed = stream_parser::parse_sse_stream(receiver_stream);
        Ok(Box::pin(parsed))
    }
}

#[async_trait]
impl LanguageModel for OpenAiCompatibleModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn capabilities(&self) -> &CapabilitySet {
        &self.capabilities
    }

    async fn generate(&self, prompt: Prompt, options: GenerateOptions) -> AiResult<GenerateResult> {
        let mut request = convert::options_to_request(
            &self.model_id,
            &self.provider_id,
            &prompt,
            &options,
            false,
        )?;
        if let Some(config) = &self.cache_config {
            if let Some(cache_key) = &config.openai_cache_key {
                request.prompt_cache_key = Some(cache_key.clone());
            }
            if config.enabled {
                request.prompt_cache_retention = Some("24h".to_string());
            }
        }
        let response = self.do_request(request).await?;
        convert::response_to_result(response, &self.provider_id)
    }

    async fn stream(&self, prompt: Prompt, options: GenerateOptions) -> AiResult<AiStream> {
        let mut request = convert::options_to_request(
            &self.model_id,
            &self.provider_id,
            &prompt,
            &options,
            true,
        )?;
        if let Some(config) = &self.cache_config {
            if let Some(cache_key) = &config.openai_cache_key {
                request.prompt_cache_key = Some(cache_key.clone());
            }
            if config.enabled {
                request.prompt_cache_retention = Some("24h".to_string());
            }
        }
        self.do_stream_request(request).await
    }
}

/// Map an HTTP status code and message to the appropriate `AiError`.
fn map_api_error(status: u16, message: &str) -> AiError {
    match status {
        401 => AiError::AuthError {
            message: message.to_string(),
        },
        429 => AiError::RateLimit { retry_after: None },
        408 | 504 => AiError::Timeout,
        _ => AiError::ProviderError {
            provider: "openai_compatible".to_string(),
            status: Some(status),
            message: message.to_string(),
        },
    }
}

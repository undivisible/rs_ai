use async_trait::async_trait;
use secrecy::{ExposeSecret, SecretString};

use rs_ai_core::{
    AiError, AiResult, AiStream, Capability, CapabilitySet, GenerateOptions, GenerateResult,
    LanguageModel, Prompt,
};

use super::api_types::GenerateContentRequest;
use super::convert::{build_request, response_to_result, GeminiRequestParts};
use super::stream_parser;

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models";

/// A Google Gemini language model.
pub struct GeminiModel {
    api_key: SecretString,
    model_id: String,
    capabilities: CapabilitySet,
    client: reqwest::Client,
    base_url: String,
    cache_config: Option<rs_ai_core::CacheConfig>,
}

impl GeminiModel {
    /// Create a new Gemini model instance.
    pub fn new(api_key: impl Into<String>, model_id: &str) -> Self {
        let capabilities = CapabilitySet::new()
            .with(Capability::TextInput)
            .with(Capability::TextOutput)
            .with(Capability::ImageInput)
            .with(Capability::Streaming)
            .with(Capability::ToolCalling)
            .with(Capability::StructuredOutput)
            .with(Capability::ExtendedThinking)
            .with(Capability::VideoInput)
            .with(Capability::AudioInput);

        Self {
            api_key: SecretString::from(api_key.into()),
            model_id: model_id.to_string(),
            capabilities,
            client: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.to_string(),
            cache_config: None,
        }
    }

    /// Override the base URL (e.g. for Cloudflare AI Gateway).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
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
        config = config.with_gemini_cache_key(key);
        self.cache_config = Some(config);
        self
    }

    fn generate_url(&self) -> String {
        let mut url = format!(
            "{}/{}:generateContent?key={}",
            self.base_url,
            self.model_id,
            self.api_key.expose_secret()
        );

        if let Some(config) = &self.cache_config {
            if let Some(cache_key) = &config.gemini_cache_key {
                url.push_str(&format!("&cacheKey={}", urlencoding::encode(cache_key)));
            }
        }

        url
    }

    fn stream_url(&self) -> String {
        let mut url = format!(
            "{}/{}:streamGenerateContent?key={}&alt=sse",
            self.base_url,
            self.model_id,
            self.api_key.expose_secret()
        );

        if let Some(config) = &self.cache_config {
            if let Some(cache_key) = &config.gemini_cache_key {
                url.push_str(&format!("&cacheKey={}", urlencoding::encode(cache_key)));
            }
        }

        url
    }

    fn build_api_request(
        &self,
        prompt: Prompt,
        options: &GenerateOptions,
    ) -> AiResult<GenerateContentRequest> {
        let GeminiRequestParts {
            contents,
            system_instruction,
            generation_config,
            tools,
            tool_config,
        } = build_request(prompt, options)?;

        Ok(GenerateContentRequest {
            contents,
            system_instruction,
            generation_config,
            tools,
            tool_config,
        })
    }
}

#[async_trait]
impl LanguageModel for GeminiModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn provider_id(&self) -> &str {
        "gemini"
    }

    fn capabilities(&self) -> &CapabilitySet {
        &self.capabilities
    }

    async fn generate(&self, prompt: Prompt, options: GenerateOptions) -> AiResult<GenerateResult> {
        let request_body = self.build_api_request(prompt, &options)?;

        let response = self
            .client
            .post(self.generate_url())
            .header("Content-Type", "application/json")
            .json(&request_body)
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
                    tracing::warn!(status = status_code, error = %e, "Failed to read Gemini error response body");
                    format!("<failed to read response body: {e}>")
                }
            };
            return Err(AiError::ProviderError {
                provider: "gemini".to_string(),
                status: Some(status_code),
                message: body,
            });
        }

        let api_response: super::api_types::GenerateContentResponse = response
            .json()
            .await
            .map_err(|e| AiError::Serialization(e.to_string()))?;

        Ok(response_to_result(api_response, &self.model_id))
    }

    async fn stream(&self, prompt: Prompt, options: GenerateOptions) -> AiResult<AiStream> {
        let request_body = self.build_api_request(prompt, &options)?;

        let response = self
            .client
            .post(self.stream_url())
            .header("Content-Type", "application/json")
            .json(&request_body)
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
                    tracing::warn!(status = status_code, error = %e, "Failed to read Gemini error response body");
                    format!("<failed to read response body: {e}>")
                }
            };
            return Err(AiError::ProviderError {
                provider: "gemini".to_string(),
                status: Some(status_code),
                message: body,
            });
        }

        Ok(stream_parser::parse_stream(response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rs_ai_core::{ContentPart, FileData, ImageData, Message};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    async fn text_only_server() -> (String, tokio::task::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = [0; 8192];
            let mut request = Vec::new();
            loop {
                let size = stream.read(&mut buf).await.unwrap();
                assert!(size > 0);
                request.extend_from_slice(&buf[..size]);
                if let Some(header_end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&request[..header_end]);
                    let length: usize = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse().unwrap())
                        })
                        .unwrap();
                    if request.len() >= header_end + 4 + length {
                        break;
                    }
                }
            }
            let body = r#"{"candidates":[{"content":{"parts":[{"text":"plausible answer"}]},"finishReason":"STOP"}]}"#;
            let response = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body);
            stream.write_all(response.as_bytes()).await.unwrap();
            String::from_utf8_lossy(&request).into_owned()
        });
        (url, task)
    }

    #[tokio::test]
    async fn rejects_attachments_that_would_be_discarded() {
        for part in [
            ContentPart::Image {
                data: ImageData::Url {
                    url: "https://example.com/image.png".into(),
                    detail: None,
                },
            },
            ContentPart::File {
                data: FileData::Base64 {
                    media_type: "audio/wav".into(),
                    data: "AQID".into(),
                },
            },
        ] {
            let (url, server) = text_only_server().await;
            let model = GeminiModel::new("test", "test").with_base_url(url);
            let mut message = Message::user("describe this");
            message.content.push(part);
            let result = model
                .generate(
                    Prompt::Messages(vec![message.clone()]),
                    GenerateOptions::default(),
                )
                .await;
            assert!(matches!(result, Err(AiError::UnsupportedCapability { .. })));
            let stream = model
                .stream(Prompt::Messages(vec![message]), GenerateOptions::default())
                .await;
            server.abort();
            assert!(matches!(stream, Err(AiError::UnsupportedCapability { .. })));
        }
    }

    #[tokio::test]
    async fn accepts_inline_images() {
        let (url, server) = text_only_server().await;
        let model = GeminiModel::new("test", "test").with_base_url(url);
        let mut message = Message::user("describe this");
        message.content.push(ContentPart::Image {
            data: ImageData::Base64 {
                media_type: "image/png".into(),
                data: "AQID".into(),
            },
        });
        let result = model
            .generate(Prompt::Messages(vec![message]), GenerateOptions::default())
            .await;
        let request = server.await.unwrap();
        assert!(
            matches!(result, Ok(response) if response.text.as_deref() == Some("plausible answer"))
        );
        let body = request.split_once("\r\n\r\n").unwrap().1;
        let json: serde_json::Value = serde_json::from_str(body).unwrap();
        assert_eq!(
            json["contents"][0]["parts"][1]["inline_data"]["data"],
            "AQID"
        );
    }
}

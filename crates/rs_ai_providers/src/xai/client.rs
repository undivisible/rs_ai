use reqwest::Client as HttpClient;
use rs_ai_core::CacheConfig;
use serde::{Deserialize, Serialize};

use super::error::{XaiError, XaiResult};

/// Model ID for image generation: Grok Imagine
pub const GROK_4_IMAGINE: &str = "grok-4-imagine";
/// Model ID for image generation: Aurora
pub const AURORA: &str = "aurora";

const XAI_API_BASE: &str = "https://api.x.ai/v1";

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Choice {
    pub index: i32,
    pub message: Message,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct StreamChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct StreamChoice {
    pub index: i32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct Delta {
    pub role: Option<String>,
    pub content: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageGenerationRequest {
    pub model: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct ImageGenerationResponse {
    pub created: i64,
    pub data: Vec<ImageData>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct ImageData {
    pub url: Option<String>,
    pub b64_json: Option<String>,
}

pub struct XaiClient {
    http_client: HttpClient,
    api_key: String,
    #[cfg(test)]
    pub(super) image_base_url: Option<String>,
}

impl XaiClient {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http_client: HttpClient::new(),
            api_key: api_key.into(),
            #[cfg(test)]
            image_base_url: None,
        }
    }

    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    pub async fn create_chat_completion(
        &self,
        request: ChatCompletionRequest,
        cache_config: Option<&CacheConfig>,
    ) -> XaiResult<ChatCompletionResponse> {
        let mut req_builder = self
            .http_client
            .post(format!("{}/chat/completions", XAI_API_BASE))
            .bearer_auth(&self.api_key);

        if let Some(config) = cache_config {
            if let Some(conv_id) = &config.xai_conv_id {
                req_builder = req_builder.header("x-grok-conv-id", conv_id);
            }
        }

        let response = req_builder.json(&request).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(XaiError::ApiError {
                message: format!("HTTP {}: {}", status, text),
            });
        }

        response.json().await.map_err(Into::into)
    }

    pub async fn create_chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        cache_config: Option<&CacheConfig>,
    ) -> XaiResult<reqwest::Response> {
        let mut req = ChatCompletionRequest {
            stream: Some(true),
            ..request
        };
        req.stream = Some(true);

        let mut req_builder = self
            .http_client
            .post(format!("{}/chat/completions", XAI_API_BASE))
            .bearer_auth(&self.api_key);

        if let Some(config) = cache_config {
            if let Some(conv_id) = &config.xai_conv_id {
                req_builder = req_builder.header("x-grok-conv-id", conv_id);
            }
        }

        let response = req_builder.json(&req).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(XaiError::ApiError {
                message: format!("HTTP {}: {}", status, text),
            });
        }

        Ok(response)
    }

    pub async fn create_image_generation(
        &self,
        request: ImageGenerationRequest,
    ) -> XaiResult<ImageGenerationResponse> {
        #[cfg(test)]
        let base_url = self.image_base_url.as_deref().unwrap_or(XAI_API_BASE);
        #[cfg(not(test))]
        let base_url = XAI_API_BASE;
        let response = self
            .http_client
            .post(format!("{}/images/generations", base_url))
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(XaiError::ApiError {
                message: format!("HTTP {}: {}", status, text),
            });
        }

        response.json().await.map_err(Into::into)
    }
}

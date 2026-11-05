use async_trait::async_trait;
use secrecy::{ExposeSecret, SecretString};
use serde::Deserialize;
use serde::Serialize;

use rs_ai_core::{
    AiError, AiResult, GeneratedFile, ImageGenerationOptions, ImageModel, ImageResult,
    ResponseMetadata, Usage,
};

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1";

/// Model constants for Google Imagen.
pub const IMAGEN_3: &str = "imagen-3.0-generate-001";
pub const IMAGEN_3_FAST: &str = "imagen-3.0-fast-001";
pub const IMAGEN_4: &str = "imagen-4.0-generate-001";
pub const IMAGEN_4_ULTRA: &str = "imagen-4.0-ultra-generate-001";
pub const IMAGEN_4_FAST: &str = "imagen-4.0-fast-generate-001";
pub const GEMINI_2_5_FLASH_IMAGE: &str = "gemini-2.5-flash-image";
pub const GEMINI_3_PRO_IMAGE: &str = "gemini-3-pro-image-preview";
pub const GEMINI_3_1_FLASH_IMAGE: &str = "gemini-3.1-flash-image-preview";

// ── Request types ──

#[derive(Serialize)]
struct PredictRequest {
    instances: Vec<Instance>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<Parameters>,
}

#[derive(Serialize)]
struct Instance {
    prompt: String,
}

#[derive(Serialize)]
#[allow(non_snake_case)]
struct Parameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    sampleCount: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    aspectRatio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    personGeneration: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    safetyFilterLevel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    addWatermark: Option<bool>,
    #[serde(flatten)]
    extra: std::collections::HashMap<String, serde_json::Value>,
}

// ── Response types ──

#[derive(Deserialize)]
struct PredictResponse {
    predictions: Vec<PredictResult>,
}

#[derive(Deserialize)]
struct PredictResult {
    #[serde(rename = "bytesBase64Encoded")]
    bytes_base64_encoded: String,
    #[serde(rename = "mimeType")]
    mime_type: String,
}

#[derive(Deserialize)]
struct GenerateContentResponse {
    candidates: Vec<Candidate>,
}

#[derive(Deserialize)]
struct Candidate {
    content: Content,
}

#[derive(Deserialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Deserialize)]
#[serde(untagged)]
#[allow(dead_code)]
enum Part {
    Text { text: String },
    InlineData { inline_data: InlineData },
}

#[derive(Deserialize)]
struct InlineData {
    #[serde(rename = "mimeType")]
    mime_type: String,
    data: String,
}

/// A Gemini Imagen image generation model.
pub struct GeminiImageModel {
    api_key: SecretString,
    model_id: String,
    client: reqwest::Client,
    base_url: String,
}

impl GeminiImageModel {
    /// Create a new Gemini Imagen model.
    pub fn new(api_key: impl Into<String>, model_id: &str) -> Self {
        Self {
            api_key: SecretString::from(api_key.into()),
            model_id: model_id.to_string(),
            client: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }

    /// Override the base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    fn predict_url(&self) -> String {
        format!(
            "{}/models/{}:predict?key={}",
            self.base_url,
            self.model_id,
            self.api_key.expose_secret()
        )
    }

    fn generate_content_url(&self) -> String {
        format!(
            "{}/models/{}:generateContent?key={}",
            self.base_url,
            self.model_id,
            self.api_key.expose_secret()
        )
    }

    fn is_gemini_multimodal(&self) -> bool {
        self.model_id.starts_with("gemini-") || self.model_id.starts_with("gemini/")
    }
}

#[async_trait]
impl ImageModel for GeminiImageModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn provider_id(&self) -> &str {
        "gemini"
    }

    async fn generate_image(
        &self,
        prompt: &str,
        options: ImageGenerationOptions,
    ) -> AiResult<ImageResult> {
        if self.is_gemini_multimodal() {
            self.generate_gemini_image(prompt, options).await
        } else {
            self.generate_imagen_image(prompt, options).await
        }
    }
}

impl GeminiImageModel {
    async fn generate_imagen_image(
        &self,
        prompt: &str,
        options: ImageGenerationOptions,
    ) -> AiResult<ImageResult> {
        let n = options.n.unwrap_or(1);

        let aspect_ratio: Option<String> = options.aspect_ratio.clone().or_else(|| {
            options.size.as_deref().and_then(|s| {
                let parts: Vec<&str> = s.split('x').collect();
                if parts.len() == 2 {
                    Some(format!("{}:{}", parts[0], parts[1]))
                } else {
                    None
                }
            })
        });

        let mut extra = options.provider_options.clone().unwrap_or_default();

        let person_generation = extra
            .remove("personGeneration")
            .and_then(|v| v.as_str().map(|s| s.to_string()));
        let safety_filter_level = extra
            .remove("safetyFilterLevel")
            .and_then(|v| v.as_str().map(|s| s.to_string()));
        let add_watermark = extra.remove("addWatermark").and_then(|v| v.as_bool());

        let request = PredictRequest {
            instances: vec![Instance {
                prompt: prompt.to_string(),
            }],
            parameters: Some(Parameters {
                sampleCount: Some(n),
                aspectRatio: aspect_ratio,
                personGeneration: person_generation,
                safetyFilterLevel: safety_filter_level,
                addWatermark: add_watermark,
                extra,
            }),
        };

        let response = self
            .client
            .post(self.predict_url())
            .header("Content-Type", "application/json")
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
                    tracing::warn!(status = status_code, error = %e, "Failed to read Imagen error response body");
                    format!("<failed to read response body: {e}>")
                }
            };
            return Err(AiError::ProviderError {
                provider: "gemini".to_string(),
                status: Some(status_code),
                message: body,
            });
        }

        let api_response: PredictResponse = response
            .json()
            .await
            .map_err(|e| AiError::Serialization(e.to_string()))?;

        let images: Vec<GeneratedFile> = api_response
            .predictions
            .into_iter()
            .map(|p| {
                let bytes = super::decode_media(&p.bytes_base64_encoded)?;
                Ok(GeneratedFile {
                    base64: p.bytes_base64_encoded,
                    bytes,
                    media_type: p.mime_type,
                })
            })
            .collect::<AiResult<_>>()?;

        let image = images
            .first()
            .cloned()
            .ok_or_else(|| AiError::ProviderError {
                provider: "gemini".to_string(),
                status: None,
                message: "Imagen returned no predictions".to_string(),
            })?;

        Ok(ImageResult {
            image,
            images,
            usage: Usage::default(),
            metadata: ResponseMetadata::default(),
        })
    }

    async fn generate_gemini_image(
        &self,
        prompt: &str,
        _options: ImageGenerationOptions,
    ) -> AiResult<ImageResult> {
        let body = serde_json::json!({
            "contents": [{
                "parts": [{ "text": prompt }]
            }]
        });

        let response = self
            .client
            .post(self.generate_content_url())
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| AiError::Transport {
                message: format!("Gemini image request failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        let status = response.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let body = response.text().await.unwrap_or_default();
            return Err(AiError::ProviderError {
                provider: "gemini".to_string(),
                status: Some(status_code),
                message: body,
            });
        }

        let api_response: GenerateContentResponse = response
            .json()
            .await
            .map_err(|e| AiError::Serialization(e.to_string()))?;

        let images: Vec<GeneratedFile> = api_response
            .candidates
            .into_iter()
            .flat_map(|c| c.content.parts)
            .filter_map(|part| match part {
                Part::InlineData { inline_data } => Some(
                    super::decode_media(&inline_data.data).map(|bytes| GeneratedFile {
                        base64: inline_data.data,
                        bytes,
                        media_type: inline_data.mime_type,
                    }),
                ),
                _ => None,
            })
            .collect::<AiResult<_>>()?;

        let image = images
            .first()
            .cloned()
            .ok_or_else(|| AiError::ProviderError {
                provider: "gemini".to_string(),
                status: None,
                message: "Gemini returned no image data".to_string(),
            })?;

        Ok(ImageResult {
            image,
            images,
            usage: Usage::default(),
            metadata: ResponseMetadata::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn rejects_invalid_image_data_in_both_response_formats() {
        for (model_id, body) in [
            (
                "imagen-test",
                r#"{"predictions":[{"bytesBase64Encoded":"%%%","mimeType":"image/png"}]}"#,
            ),
            (
                "gemini-test",
                r#"{"candidates":[{"content":{"parts":[{"inline_data":{"data":"%%%","mimeType":"image/png"}}]}}]}"#,
            ),
        ] {
            let url = super::super::test_response(body).await;
            let model = GeminiImageModel::new("test", model_id).with_base_url(url);
            let result = model
                .generate_image("hello", ImageGenerationOptions::default())
                .await;
            assert!(result.is_err(), "{model_id}");
        }
        for (model_id, body) in [
            (
                "imagen-test",
                r#"{"predictions":[{"bytesBase64Encoded":"AQID","mimeType":"image/png"}]}"#,
            ),
            (
                "gemini-test",
                r#"{"candidates":[{"content":{"parts":[{"inline_data":{"data":"AQID","mimeType":"image/png"}}]}}]}"#,
            ),
        ] {
            let url = super::super::test_response(body).await;
            let model = GeminiImageModel::new("test", model_id).with_base_url(url);
            let Ok(result) = model
                .generate_image("hello", ImageGenerationOptions::default())
                .await
            else {
                panic!("valid image response failed");
            };
            assert_eq!(result.image.bytes, [1, 2, 3]);
        }
    }
}

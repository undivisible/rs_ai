use async_trait::async_trait;
use rs_ai_core::{
    AiError, AiResult, GeneratedFile, ResponseMetadata, Usage, VideoGenerationOptions, VideoModel,
    VideoResult,
};

/// Google Gemini video generation model.
pub struct GeminiVideoModel {
    api_key: String,
    model_id: String,
    #[cfg(test)]
    base_url: Option<String>,
}

impl GeminiVideoModel {
    pub fn new(api_key: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model_id: model_id.into(),
            #[cfg(test)]
            base_url: None,
        }
    }
}

#[async_trait]
impl VideoModel for GeminiVideoModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn provider_id(&self) -> &str {
        "gemini"
    }

    async fn generate_video(
        &self,
        prompt: &str,
        options: VideoGenerationOptions,
    ) -> AiResult<VideoResult> {
        let client = reqwest::Client::new();

        let mut params = serde_json::json!({});
        if let Some(n) = options.n {
            params["sampleCount"] = serde_json::json!(n);
        }
        if let Some(r) = &options.resolution {
            params["resolution"] = serde_json::json!(r);
        }
        if let Some(d) = options.duration {
            params["duration"] = serde_json::json!(d);
        }

        let body = serde_json::json!({
            "instances": [{ "prompt": prompt }],
            "parameters": params,
        });

        #[cfg(test)]
        let base_url = self
            .base_url
            .as_deref()
            .unwrap_or("https://generativelanguage.googleapis.com/v1");
        #[cfg(not(test))]
        let base_url = "https://generativelanguage.googleapis.com/v1";
        let url = format!(
            "{base_url}/models/{}:predict?key={}",
            self.model_id, self.api_key
        );

        let resp = client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| AiError::Transport {
                message: format!("Gemini video request failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        let status = resp.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let body_text = resp.text().await.unwrap_or_default();
            return Err(AiError::ProviderError {
                provider: "gemini".into(),
                status: Some(status_code),
                message: body_text,
            });
        }

        let data: serde_json::Value = resp.json().await.map_err(|e| AiError::Transport {
            message: format!("Gemini video response parse failed: {e}"),
            source: Some(Box::new(e)),
        })?;

        let video_b64 = data["predictions"][0]["bytesBase64Encoded"]
            .as_str()
            .unwrap_or("");
        let file = GeneratedFile {
            base64: video_b64.to_string(),
            bytes: super::decode_media(video_b64)?,
            media_type: "video/mp4".into(),
        };

        Ok(VideoResult {
            video: file.clone(),
            videos: vec![file],
            usage: Usage::default(),
            metadata: ResponseMetadata::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn rejects_missing_and_invalid_video_data() {
        for body in [
            r#"{"predictions":[{}]}"#,
            r#"{"predictions":[{"bytesBase64Encoded":"%%%"}]}"#,
            r#"{"predictions":[{"bytesBase64Encoded":""}]}"#,
        ] {
            let mut model = GeminiVideoModel::new("test", "test");
            model.base_url = Some(super::super::test_response(body).await);
            let result = model
                .generate_video("hello", VideoGenerationOptions::default())
                .await;
            assert!(result.is_err());
        }
        let mut model = GeminiVideoModel::new("test", "test");
        model.base_url = Some(
            super::super::test_response(r#"{"predictions":[{"bytesBase64Encoded":"AQID"}]}"#).await,
        );
        let Ok(result) = model
            .generate_video("hello", VideoGenerationOptions::default())
            .await
        else {
            panic!("valid video response failed");
        };
        assert_eq!(result.video.bytes, [1, 2, 3]);
    }
}

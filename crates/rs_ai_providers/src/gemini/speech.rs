use async_trait::async_trait;
use rs_ai_core::{AiError, AiResult, AudioResult, TextToSpeechModel, TtsOptions, Usage};

/// Google Gemini text-to-speech model.
pub struct GeminiSpeechModel {
    api_key: String,
    model_id: String,
    #[cfg(test)]
    base_url: Option<String>,
}

impl GeminiSpeechModel {
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
impl TextToSpeechModel for GeminiSpeechModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn provider_id(&self) -> &str {
        "gemini"
    }

    async fn synthesize(
        &self,
        text: &str,
        voice: &str,
        _options: TtsOptions,
    ) -> AiResult<AudioResult> {
        let client = reqwest::Client::new();

        let body = serde_json::json!({
            "instances": [{
                "text": text,
            }],
            "parameters": {
                "voice": voice,
            }
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
                message: format!("Gemini TTS request failed: {e}"),
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
            message: format!("Gemini TTS response parse failed: {e}"),
            source: Some(Box::new(e)),
        })?;

        let audio_b64 = data["predictions"][0]["bytesBase64Encoded"]
            .as_str()
            .unwrap_or("");
        let audio = super::decode_media(audio_b64)?;

        Ok(AudioResult {
            audio,
            mime_type: "audio/pcm".into(),
            usage: Usage::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn rejects_missing_and_invalid_audio_data() {
        for body in [
            r#"{"predictions":[{}]}"#,
            r#"{"predictions":[{"bytesBase64Encoded":"%%%"}]}"#,
            r#"{"predictions":[{"bytesBase64Encoded":""}]}"#,
        ] {
            let mut model = GeminiSpeechModel::new("test", "test");
            model.base_url = Some(super::super::test_response(body).await);
            let result = model
                .synthesize("hello", "test", TtsOptions::default())
                .await;
            assert!(result.is_err());
        }
        let mut model = GeminiSpeechModel::new("test", "test");
        model.base_url = Some(
            super::super::test_response(r#"{"predictions":[{"bytesBase64Encoded":"AQID"}]}"#).await,
        );
        let Ok(result) = model
            .synthesize("hello", "test", TtsOptions::default())
            .await
        else {
            panic!("valid speech response failed");
        };
        assert_eq!(result.audio, [1, 2, 3]);
    }
}

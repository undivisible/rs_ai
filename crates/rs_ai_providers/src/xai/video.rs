use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD, Engine};
use rs_ai_core::types::ResponseMetadata;
use rs_ai_core::Usage;
use rs_ai_core::{
    AiError, AiResult, GeneratedFile, VideoGenerationOptions, VideoModel, VideoResult,
};

/// xAI video generation model (async poll-based API).
pub struct XaiVideoModel {
    api_key: String,
    model_id: String,
    base_url: String,
}

impl XaiVideoModel {
    pub fn new(api_key: String, model_id: String) -> Self {
        Self {
            api_key,
            model_id,
            base_url: "https://api.x.ai/v1".into(),
        }
    }
}

#[async_trait]
impl VideoModel for XaiVideoModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn provider_id(&self) -> &str {
        "xai"
    }

    async fn generate_video(
        &self,
        prompt: &str,
        options: VideoGenerationOptions,
    ) -> AiResult<VideoResult> {
        let client = reqwest::Client::new();

        let mut body = serde_json::json!({
            "model": self.model_id,
            "prompt": prompt,
        });
        if options.n.is_some_and(|n| n != 1) || options.fps.is_some() || options.seed.is_some() {
            return Err(AiError::UnsupportedCapability {
                capability: "multiple videos, fps, or seed".into(),
                provider: "xai".into(),
            });
        }
        if let Some(r) = &options.resolution {
            body["resolution"] = serde_json::json!(r);
        }
        if let Some(d) = options.duration {
            if !d.is_finite() || !(1.0..=15.0).contains(&d) || d.fract() != 0.0 {
                return Err(AiError::ProviderError {
                    provider: "xai".into(),
                    status: None,
                    message: "Video duration must be a whole number of seconds from 1 to 15".into(),
                });
            }
            body["duration"] = serde_json::json!(d as u32);
        }
        if let Some(aspect_ratio) = &options.aspect_ratio {
            body["aspect_ratio"] = serde_json::json!(aspect_ratio);
        }
        if let Some(generate_audio) = options.generate_audio {
            body["generate_audio"] = serde_json::json!(generate_audio);
        }

        // Create generation
        let create_resp = client
            .post(format!("{}/videos/generations", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| AiError::Transport {
                message: format!("xAI video create failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        let status = create_resp.status();
        if !status.is_success() {
            let body_text = create_resp.text().await.unwrap_or_default();
            return Err(AiError::ProviderError {
                provider: "xai".into(),
                status: Some(status.as_u16()),
                message: body_text,
            });
        }

        let create_data: serde_json::Value =
            create_resp.json().await.map_err(|e| AiError::Transport {
                message: format!("xAI video parse failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        let generation_id = create_data["request_id"]
            .as_str()
            .ok_or_else(|| AiError::ProviderError {
                provider: "xai".into(),
                status: None,
                message: "No generation ID in response".into(),
            })?
            .to_string();

        // Poll until complete
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;

            let poll_resp = client
                .get(format!("{}/videos/{}", self.base_url, generation_id))
                .header("Authorization", format!("Bearer {}", self.api_key))
                .send()
                .await
                .map_err(|e| AiError::Transport {
                    message: format!("xAI video poll failed: {e}"),
                    source: Some(Box::new(e)),
                })?;

            let poll_status = poll_resp.status();
            if !poll_status.is_success() {
                let body_text = poll_resp.text().await.unwrap_or_default();
                return Err(AiError::ProviderError {
                    provider: "xai".into(),
                    status: Some(poll_status.as_u16()),
                    message: body_text,
                });
            }

            let poll_data: serde_json::Value =
                poll_resp.json().await.map_err(|e| AiError::Transport {
                    message: format!("xAI video poll parse failed: {e}"),
                    source: Some(Box::new(e)),
                })?;

            let status = poll_data["status"].as_str().unwrap_or("");
            match status {
                "done" => {
                    let video_url = poll_data["video"]["url"]
                        .as_str()
                        .filter(|url| !url.is_empty())
                        .ok_or_else(|| AiError::ProviderError {
                            provider: "xai".into(),
                            status: None,
                            message: "Completed video response has no URL".into(),
                        })?;

                    let response =
                        client
                            .get(video_url)
                            .send()
                            .await
                            .map_err(|e| AiError::Transport {
                                message: format!("xAI video download failed: {e}"),
                                source: Some(Box::new(e)),
                            })?;
                    if !response.status().is_success() {
                        return Err(AiError::ProviderError {
                            provider: "xai".into(),
                            status: Some(response.status().as_u16()),
                            message: "Video download failed".into(),
                        });
                    }
                    let bytes = response
                        .bytes()
                        .await
                        .map_err(|e| AiError::Transport {
                            message: format!("Failed to read xAI video: {e}"),
                            source: Some(Box::new(e)),
                        })?
                        .to_vec();
                    if bytes.get(4..8) != Some(b"ftyp") {
                        return Err(AiError::ProviderError {
                            provider: "xai".into(),
                            status: None,
                            message: "Video download is not an MP4 file".into(),
                        });
                    }

                    let file = GeneratedFile {
                        base64: STANDARD.encode(&bytes),
                        bytes,
                        media_type: "video/mp4".into(),
                    };

                    return Ok(VideoResult {
                        video: file.clone(),
                        videos: vec![file],
                        usage: Usage::default(),
                        metadata: ResponseMetadata::default(),
                    });
                }
                "failed" => {
                    let err = poll_data["error"]["message"]
                        .as_str()
                        .unwrap_or("unknown error");
                    return Err(AiError::ProviderError {
                        provider: "xai".into(),
                        status: None,
                        message: err.into(),
                    });
                }
                "pending" => {}
                _ => {
                    return Err(AiError::ProviderError {
                        provider: "xai".into(),
                        status: None,
                        message: format!("Unexpected video generation status: {status:?}"),
                    })
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    async fn generate_with_download(
        status: u16,
        download: Vec<u8>,
        include_url: bool,
    ) -> AiResult<VideoResult> {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let video_url = format!("{base_url}/video.mp4");
        let server = tokio::spawn(async move {
            loop {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut request = vec![0; 8192];
                let size = stream.read(&mut request).await.unwrap();
                let request = String::from_utf8_lossy(&request[..size]);
                let first_line = request.lines().next().unwrap();
                let (status, body) = if first_line.starts_with("POST ") {
                    assert!(first_line.starts_with("POST /videos/generations "));
                    (200, br#"{"request_id":"job-1"}"#.to_vec())
                } else if first_line.contains("/video.mp4") {
                    assert!(!request.to_lowercase().contains("authorization:"));
                    (status, download.clone())
                } else {
                    assert!(first_line.starts_with("GET /videos/job-1 "));
                    let video = if include_url {
                        serde_json::json!({"url":video_url})
                    } else {
                        serde_json::json!({})
                    };
                    (
                        200,
                        serde_json::json!({"status":"done","video":video})
                            .to_string()
                            .into_bytes(),
                    )
                };
                let headers = format!(
                    "HTTP/1.1 {status} test\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                stream.write_all(headers.as_bytes()).await.unwrap();
                stream.write_all(&body).await.unwrap();
            }
        });
        let mut model = XaiVideoModel::new("test-key".into(), "test-model".into());
        model.base_url = base_url;
        let result = model
            .generate_video("a red kite", VideoGenerationOptions::default())
            .await;
        server.abort();
        result
    }

    #[tokio::test]
    async fn video_result_contains_downloaded_bytes() {
        let expected = b"\0\0\0\x18ftypisom\0\0\0\0videodata".to_vec();
        let result = generate_with_download(200, expected.clone(), true)
            .await
            .unwrap();
        assert_eq!(result.video.bytes, expected);
        assert_eq!(STANDARD.decode(&result.video.base64).unwrap(), expected);
        assert_eq!(result.video.media_type, "video/mp4");
    }

    #[tokio::test]
    async fn video_download_failures_are_errors() {
        let (missing, failed, html) = tokio::join!(
            generate_with_download(200, vec![], false),
            generate_with_download(403, b"access denied".to_vec(), true),
            generate_with_download(200, b"<html>not a video</html>".to_vec(), true),
        );
        assert!(missing.unwrap_err().to_string().contains("no URL"));
        assert!(failed
            .unwrap_err()
            .to_string()
            .contains("Video download failed"));
        assert!(html.unwrap_err().to_string().contains("not an MP4"));
    }
}

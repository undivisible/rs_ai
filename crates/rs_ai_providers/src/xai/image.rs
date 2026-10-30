use async_trait::async_trait;
use base64::Engine as _;
use rs_ai_core::{
    AiError, AiResult, GeneratedFile, ImageGenerationOptions, ImageModel, ImageResult, Usage,
};

use super::client::{ImageGenerationRequest, XaiClient};

/// xAI image generation model (Grok Imagine / Aurora).
#[derive(Clone)]
pub struct XaiImageModel {
    model_id: String,
    client: std::sync::Arc<XaiClient>,
}

impl XaiImageModel {
    pub fn new(model_id: String, client: std::sync::Arc<XaiClient>) -> Self {
        Self { model_id, client }
    }
}

#[async_trait]
impl ImageModel for XaiImageModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn provider_id(&self) -> &str {
        "xai"
    }

    async fn generate_image(
        &self,
        prompt: &str,
        options: ImageGenerationOptions,
    ) -> AiResult<ImageResult> {
        let request = ImageGenerationRequest {
            model: self.model_id.clone(),
            prompt: prompt.to_string(),
            n: options.n,
            size: options.size,
            seed: options.seed,
        };

        let response = self.client.create_image_generation(request).await?;

        if response.data.is_empty() {
            return Err(AiError::ProviderError {
                provider: "xai".to_string(),
                status: None,
                message: "No image data in response".to_string(),
            });
        }

        let mut images: Vec<GeneratedFile> = Vec::new();
        for img in &response.data {
            let bytes = if let Some(b64) = &img.b64_json {
                base64::engine::general_purpose::STANDARD
                    .decode(b64)
                    .map_err(|e| AiError::ProviderError {
                        provider: "xai".to_string(),
                        status: None,
                        message: format!("Failed to decode base64 image: {e}"),
                    })?
            } else if let Some(url) = &img.url {
                let resp = reqwest::get(url)
                    .await
                    .map_err(|e| AiError::ProviderError {
                        provider: "xai".to_string(),
                        status: None,
                        message: format!("Failed to fetch image from URL: {e}"),
                    })?;
                if !resp.status().is_success() {
                    return Err(AiError::ProviderError {
                        provider: "xai".to_string(),
                        status: Some(resp.status().as_u16()),
                        message: "Image download failed".to_string(),
                    });
                }
                resp.bytes()
                    .await
                    .map_err(|e| AiError::ProviderError {
                        provider: "xai".to_string(),
                        status: None,
                        message: format!("Failed to read image bytes: {e}"),
                    })?
                    .to_vec()
            } else {
                return Err(AiError::ProviderError {
                    provider: "xai".to_string(),
                    status: None,
                    message: "Image response has no data or URL".to_string(),
                });
            };
            let media_type = detect_image_type(&bytes).ok_or_else(|| AiError::ProviderError {
                provider: "xai".to_string(),
                status: None,
                message: "Image response is not a supported image format".to_string(),
            })?;
            images.push(GeneratedFile {
                base64: base64::engine::general_purpose::STANDARD.encode(&bytes),
                bytes,
                media_type: media_type.to_string(),
            });
        }

        let first = images[0].clone();
        Ok(ImageResult {
            image: first,
            images,
            usage: Usage::default(),
            metadata: Default::default(),
        })
    }
}

fn detect_image_type(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if bytes.starts_with(b"\xff\xd8\xff") {
        Some("image/jpeg")
    } else if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") {
        Some("image/webp")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    async fn generate_with_download(status: u16, download: &[u8]) -> AiResult<ImageResult> {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server_url = url.clone();
        let download = download.to_vec();
        let server = tokio::spawn(async move {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut request = [0; 8192];
                let size = stream.read(&mut request).await.unwrap();
                let request = String::from_utf8_lossy(&request[..size]);
                let (code, body) = if request.starts_with("POST ") {
                    (
                        200,
                        format!(r#"{{"created":1,"data":[{{"url":"{server_url}/image.png"}}]}}"#)
                            .into_bytes(),
                    )
                } else {
                    assert!(!request.to_lowercase().contains("authorization:"));
                    (status, download.clone())
                };
                let headers = format!(
                    "HTTP/1.1 {code} test\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                stream.write_all(headers.as_bytes()).await.unwrap();
                stream.write_all(&body).await.unwrap();
            }
        });
        let mut client = XaiClient::new("test-key");
        client.image_base_url = Some(url);
        let model = XaiImageModel::new("test".into(), std::sync::Arc::new(client));
        let result = model
            .generate_image("test", ImageGenerationOptions::default())
            .await;
        server.abort();
        result
    }

    #[tokio::test]
    async fn rejects_error_pages_instead_of_returning_images() {
        let (failed, html) = tokio::join!(
            generate_with_download(403, b"access denied"),
            generate_with_download(200, b"<html>not an image</html>"),
        );
        assert!(
            matches!(
                failed,
                Err(AiError::ProviderError {
                    status: Some(403),
                    ..
                })
            ),
            "{failed:?}"
        );
        assert!(html
            .unwrap_err()
            .to_string()
            .contains("not a supported image"));
    }

    #[tokio::test]
    async fn returns_downloaded_image_bytes() {
        let jpeg = b"\xff\xd8\xffimage-data";
        let result = generate_with_download(200, jpeg).await.unwrap();
        assert_eq!(result.image.bytes, jpeg);
        assert_eq!(result.image.media_type, "image/jpeg");
        assert_eq!(result.images.len(), 1);
        assert_eq!(result.images[0].bytes, jpeg);
        assert_eq!(
            base64::engine::general_purpose::STANDARD
                .decode(&result.image.base64)
                .unwrap(),
            jpeg
        );
    }
}

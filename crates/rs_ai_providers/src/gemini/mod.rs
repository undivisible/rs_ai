//!
//! ⚠️ **UNSTABLE** — This crate is in active development. APIs may change without notice.
//!
//! Google Gemini provider for the Rusty AI SDK.

mod api_types;
mod convert;
pub(crate) mod image;
pub mod live_api;
mod model;
mod provider;
mod stream_parser;

pub(crate) mod speech;
pub(crate) mod video;

pub use image::{
    GeminiImageModel, GEMINI_2_5_FLASH_IMAGE, GEMINI_3_1_FLASH_IMAGE, GEMINI_3_PRO_IMAGE, IMAGEN_3,
    IMAGEN_3_FAST, IMAGEN_4, IMAGEN_4_FAST, IMAGEN_4_ULTRA,
};
pub use live_api::{GeminiLiveSessionAdapter, LiveSession};
pub use model::*;
pub use provider::*;
pub use speech::GeminiSpeechModel;
pub use video::GeminiVideoModel;

fn decode_media(encoded: &str) -> rs_ai_core::AiResult<Vec<u8>> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|e| {
            rs_ai_core::AiError::Serialization(format!("Invalid Gemini media data: {e}"))
        })?;
    if bytes.is_empty() {
        return Err(rs_ai_core::AiError::ProviderError {
            provider: "gemini".into(),
            status: None,
            message: "Gemini returned empty media data".into(),
        });
    }
    Ok(bytes)
}

#[cfg(test)]
async fn test_response(body: &'static str) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut request = [0; 8192];
        let _size = stream.read(&mut request).await.unwrap();
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(), body
        );
        stream.write_all(response.as_bytes()).await.unwrap();
    });
    url
}

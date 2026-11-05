#![cfg(feature = "ollama")]

use rs_ai_core::{
    AiError, ContentPart, FileData, GenerateOptions, ImageData, LanguageModel, Message, Prompt,
};
use rs_ai_providers::ollama::OllamaModel;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::test]
async fn unsupported_attachments_are_not_silently_dropped() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            let mut buffer = [0; 8192];
            let _ = socket.read(&mut buffer).await;
            let body = r#"{"model":"test","message":{"role":"assistant","content":"plausible answer"},"done":true}"#;
            let response = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body);
            let _ = socket.write_all(response.as_bytes()).await;
        }
    });
    let model = OllamaModel::new("test").with_base_url(&format!("http://{address}"));
    let mut file_message = Message::user("read the file");
    file_message.content.push(ContentPart::File {
        data: FileData::Base64 {
            media_type: "text/plain".into(),
            data: "c2VjcmV0".into(),
        },
    });
    let image_message = Message::user("describe the image").with_image(ImageData::Url {
        url: "https://example.org/image.png".into(),
        detail: None,
    });
    for message in [image_message, file_message] {
        let prompt = Prompt::Messages(vec![message]);
        assert!(matches!(
            model
                .generate(prompt.clone(), GenerateOptions::default())
                .await,
            Err(AiError::UnsupportedCapability { .. })
        ));
        assert!(matches!(
            model.stream(prompt, GenerateOptions::default()).await,
            Err(AiError::UnsupportedCapability { .. })
        ));
    }
    server.abort();
}

#[tokio::test]
async fn inline_images_reach_ollama() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        let mut buffer = [0; 4096];
        loop {
            let count = socket.read(&mut buffer).await.unwrap();
            assert!(count > 0);
            request.extend_from_slice(&buffer[..count]);
            if let Some(header_end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                let headers = std::str::from_utf8(&request[..header_end]).unwrap();
                let length: usize = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse().unwrap())
                    })
                    .unwrap();
                if request.len() >= header_end + 4 + length {
                    let body: serde_json::Value =
                        serde_json::from_slice(&request[header_end + 4..header_end + 4 + length])
                            .unwrap();
                    assert_eq!(body["messages"][0]["images"][0], "AQIDBA==");
                    break;
                }
            }
        }
        let body = r#"{"model":"test","message":{"role":"assistant","content":"image received"},"done":true}"#;
        socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).as_bytes()).await.unwrap();
    });
    let model = OllamaModel::new("test").with_base_url(&format!("http://{address}"));
    let prompt = Prompt::Messages(vec![Message::user("describe").with_image(
        ImageData::Base64 {
            media_type: "image/png".into(),
            data: "AQIDBA==".into(),
        },
    )]);
    let result = model
        .generate(prompt, GenerateOptions::default())
        .await
        .unwrap();
    assert_eq!(result.text.as_deref(), Some("image received"));
    server.await.unwrap();
}

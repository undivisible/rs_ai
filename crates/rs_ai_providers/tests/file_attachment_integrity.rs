#![cfg(any(feature = "claude", feature = "openai-compatible"))]

use rs_ai_core::{AiError, ContentPart, FileData, GenerateOptions, LanguageModel, Message, Prompt};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn file_prompt() -> Prompt {
    let mut message = Message::user("summarize the attached document");
    message.content.push(ContentPart::File {
        data: FileData::Base64 {
            media_type: "text/plain".into(),
            data: "c2VjcmV0".into(),
        },
    });
    Prompt::Messages(vec![message])
}

async fn fixture(body: &'static str) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            let mut buffer = [0; 8192];
            let _ = socket.read(&mut buffer).await;
            let response = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body);
            let _ = socket.write_all(response.as_bytes()).await;
        }
    });
    (format!("http://{address}"), server)
}

async fn check_rejection(model: &dyn LanguageModel) {
    let prompt = file_prompt();
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
    let text = model
        .generate(Prompt::Text("hello".into()), GenerateOptions::default())
        .await;
    assert!(matches!(text, Ok(result) if result.text.as_deref() == Some("plausible answer")));
}

#[cfg(feature = "claude")]
#[tokio::test]
async fn claude_rejects_file_instead_of_omitting_it() {
    let (url, server) = fixture(r#"{"id":"test","model":"test","content":[{"type":"text","text":"plausible answer"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}"#).await;
    let model = rs_ai_providers::claude::ClaudeModel::new("test", "test").with_base_url(url);
    check_rejection(&model).await;
    server.abort();
}

#[cfg(feature = "openai-compatible")]
#[tokio::test]
async fn compatible_rejects_file_instead_of_omitting_it() {
    let (url, server) = fixture(r#"{"id":"test","model":"test","choices":[{"index":0,"message":{"role":"assistant","content":"plausible answer"},"finish_reason":"stop"}]}"#).await;
    let config = rs_ai_providers::openai_compatible::OpenAiCompatibleConfig::new(url, "test");
    let model = rs_ai_providers::openai_compatible::OpenAiCompatibleModel::new(
        config,
        "test",
        "test-provider",
    );
    check_rejection(&model).await;
    server.abort();
}

#![cfg(feature = "openai-compatible")]

use rs_ai_core::{AiError, GenerateOptions, LanguageModel, Prompt};
use rs_ai_providers::openai_compatible::{OpenAiCompatibleConfig, OpenAiCompatibleModel};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::test]
async fn malformed_tool_arguments_fail_generation() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        for arguments in ["{bad", "{\"amount\":17}"] {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = [0; 8192];
            let _ = socket.read(&mut buffer).await;
            let body = serde_json::json!({"id":"test","model":"test","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":arguments}}]},"finish_reason":"tool_calls"}]}).to_string();
            let response = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body);
            socket.write_all(response.as_bytes()).await.unwrap();
        }
    });
    let model = OpenAiCompatibleModel::new(
        OpenAiCompatibleConfig::new(format!("http://{address}"), "test"),
        "test",
        "test-provider",
    );
    assert!(matches!(
        model
            .generate(Prompt::Text("hello".into()), GenerateOptions::default())
            .await,
        Err(AiError::Serialization(_))
    ));
    let result = model
        .generate(Prompt::Text("hello".into()), GenerateOptions::default())
        .await;
    assert!(
        matches!(result, Ok(result) if result.tool_calls.len() == 1 && result.tool_calls[0].arguments == serde_json::json!({"amount":17}))
    );
    server.await.unwrap();
}

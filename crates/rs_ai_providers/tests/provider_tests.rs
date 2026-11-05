#![cfg(feature = "portkey")]
use rs_ai_core::LanguageModel;
use rs_ai_providers::portkey::PortkeyProvider;

#[test]
fn test_provider_creation() {
    let provider = PortkeyProvider::new("test-key");
    let model = provider.model("gpt-4");

    assert_eq!(model.model_id(), "gpt-4");
    assert_eq!(model.provider_id(), "portkey");
}

#[test]
fn test_provider_with_custom_url() {
    let provider = PortkeyProvider::new("test-key").with_base_url("https://custom.portkey.ai/v1");

    let model = provider.model("gpt-4");
    assert_eq!(model.model_id(), "gpt-4");
}

#[test]
fn test_model_capabilities() {
    let provider = PortkeyProvider::new("test-key");
    let model = provider.model("gpt-4");

    let capabilities = model.capabilities();
    assert!(capabilities.supports_all(&[rs_ai_core::Capability::TextInput]));
    assert!(capabilities.supports_all(&[rs_ai_core::Capability::TextOutput]));
    assert!(capabilities.supports_all(&[rs_ai_core::Capability::Streaming]));
    assert!(!capabilities.supports_all(&[rs_ai_core::Capability::ToolCalling]));
}

#[test]
fn test_multiple_models_same_provider() {
    let provider = PortkeyProvider::new("test-key");

    let model1 = provider.model("gpt-4");
    let model2 = provider.model("gpt-4o");
    let model3 = provider.model("claude-3-sonnet");

    assert_eq!(model1.model_id(), "gpt-4");
    assert_eq!(model2.model_id(), "gpt-4o");
    assert_eq!(model3.model_id(), "claude-3-sonnet");
    assert_eq!(model1.provider_id(), model2.provider_id());
    assert_eq!(model1.provider_id(), model3.provider_id());
}

#[test]
fn test_provider_is_cloneable() {
    let provider1 = PortkeyProvider::new("test-key");
    let provider2 = provider1.clone();

    let model1 = provider1.model("gpt-4");
    let model2 = provider2.model("gpt-4");

    assert_eq!(model1.model_id(), model2.model_id());
}

#[tokio::test]
async fn tool_requests_are_not_silently_ignored() {
    use rs_ai_core::{AiError, GenerateOptions, Prompt, ToolChoice, ToolDefinition};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        while let Ok((mut connection, _)) = listener.accept().await {
            let mut buffer = [0; 4096];
            let _ = connection.read(&mut buffer).await;
            let body = r#"{"id":"id","object":"chat.completion","created":1,"model":"gpt-4","choices":[{"index":0,"message":{"role":"assistant","content":"ordinary response"},"finish_reason":"stop"}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = connection.write_all(response.as_bytes()).await;
        }
    });

    let model = PortkeyProvider::new("test-key")
        .with_base_url(format!("http://{address}"))
        .model("gpt-4");
    let ordinary = model
        .generate(Prompt::Text("hello".into()), GenerateOptions::default())
        .await
        .unwrap();
    assert_eq!(ordinary.text.as_deref(), Some("ordinary response"));

    for options in [
        GenerateOptions::default().with_tool_choice(ToolChoice::Required),
        GenerateOptions::default().with_tools(vec![ToolDefinition {
            name: "lookup".into(),
            description: "Look up a value".into(),
            parameters: serde_json::json!({"type": "object"}),
            examples: None,
        }]),
    ] {
        assert!(matches!(
            model
                .generate(Prompt::Text("use a tool".into()), options.clone())
                .await,
            Err(AiError::UnsupportedCapability { .. })
        ));
        assert!(matches!(
            model
                .stream(Prompt::Text("use a tool".into()), options)
                .await,
            Err(AiError::UnsupportedCapability { .. })
        ));
    }
}

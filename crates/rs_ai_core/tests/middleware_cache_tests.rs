use std::time::Duration;

use rs_ai_core::{
    middleware::{CacheMiddleware, MiddlewareChain},
    tool::ToolDefinition,
    GenerateOptions, Prompt,
};
use rs_ai_testing::{MockLanguageModel, MockResponse};

fn text_response(s: &str) -> MockResponse {
    MockResponse::Text(s.to_owned())
}

#[tokio::test]
async fn same_prompt_and_options_hits_cache() {
    let model = MockLanguageModel::new("test").with_response(text_response("response-1"));

    let chain = MiddlewareChain::new(model).with(CacheMiddleware::new(Duration::from_secs(60)));

    let prompt = Prompt::from("hello");
    let opts = GenerateOptions::default();

    let r1 = chain.generate(prompt.clone(), opts.clone()).await.unwrap();
    let r2 = chain.generate(prompt.clone(), opts.clone()).await.unwrap();

    // Both should return the same cached text; the model was only called once.
    assert_eq!(r1.text, r2.text);
    assert_eq!(r1.text.as_deref(), Some("response-1"));
}

#[tokio::test]
async fn different_temperature_is_a_cache_miss() {
    let model = MockLanguageModel::new("test")
        .with_response(text_response("response-cold"))
        .with_response(text_response("response-hot"));

    let chain = MiddlewareChain::new(model).with(CacheMiddleware::new(Duration::from_secs(60)));

    let prompt = Prompt::from("same prompt");
    let cold = chain
        .generate(
            prompt.clone(),
            GenerateOptions::default().with_temperature(0.0),
        )
        .await
        .unwrap();
    let hot = chain
        .generate(
            prompt.clone(),
            GenerateOptions::default().with_temperature(1.0),
        )
        .await
        .unwrap();

    assert_eq!(cold.text.as_deref(), Some("response-cold"));
    assert_eq!(hot.text.as_deref(), Some("response-hot"));
}

#[tokio::test]
async fn different_tools_is_a_cache_miss() {
    let model = MockLanguageModel::new("test")
        .with_response(text_response("no-tools"))
        .with_response(text_response("with-tools"));

    let chain = MiddlewareChain::new(model).with(CacheMiddleware::new(Duration::from_secs(60)));

    let prompt = Prompt::from("same prompt");

    let r_plain = chain
        .generate(prompt.clone(), GenerateOptions::default())
        .await
        .unwrap();
    let r_tools = chain
        .generate(
            prompt.clone(),
            GenerateOptions::default().with_tools(vec![ToolDefinition {
                name: "search".into(),
                description: "search the web".into(),
                parameters: serde_json::json!({}),
                examples: None,
            }]),
        )
        .await
        .unwrap();

    assert_eq!(r_plain.text.as_deref(), Some("no-tools"));
    assert_eq!(r_tools.text.as_deref(), Some("with-tools"));
}

#[tokio::test]
async fn different_max_steps_is_a_cache_miss() {
    let model = MockLanguageModel::new("test")
        .with_response(text_response("one-step"))
        .with_response(text_response("five-steps"));

    let chain = MiddlewareChain::new(model).with(CacheMiddleware::new(Duration::from_secs(60)));

    let prompt = Prompt::from("same prompt");
    let one = chain
        .generate(prompt.clone(), GenerateOptions::default().with_max_steps(1))
        .await
        .unwrap();
    let five = chain
        .generate(prompt.clone(), GenerateOptions::default().with_max_steps(5))
        .await
        .unwrap();

    assert_eq!(one.text.as_deref(), Some("one-step"));
    assert_eq!(five.text.as_deref(), Some("five-steps"));
}

#[tokio::test]
async fn expired_entry_is_not_returned() {
    let model = MockLanguageModel::new("test")
        .with_response(text_response("first"))
        .with_response(text_response("second"));

    // TTL of 1ms so the entry expires immediately.
    let chain = MiddlewareChain::new(model).with(CacheMiddleware::new(Duration::from_millis(1)));

    let prompt = Prompt::from("hello");
    let opts = GenerateOptions::default();

    let r1 = chain.generate(prompt.clone(), opts.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(5)).await;
    let r2 = chain.generate(prompt.clone(), opts.clone()).await.unwrap();

    assert_eq!(r1.text.as_deref(), Some("first"));
    assert_eq!(
        r2.text.as_deref(),
        Some("second"),
        "expired entry should trigger a fresh model call"
    );
}

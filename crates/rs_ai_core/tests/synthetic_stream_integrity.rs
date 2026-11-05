use rs_ai_core::{
    wrap_with_streaming, AiResult, AiStream, CapabilitySet, FinishReason, GenerateOptions,
    GenerateResult, LanguageModel, Prompt, StreamCollector, StreamEvent, SyntheticStreamer,
    ToolCallRequest, Usage,
};
use rs_ai_testing::MockLanguageModel;

#[tokio::test]
async fn synthetic_stream_preserves_multibyte_text_with_small_chunks() {
    let result = StreamCollector::collect(SyntheticStreamer::stream("a🦀b".into(), 2))
        .await
        .unwrap();
    assert_eq!(result.text.as_deref(), Some("a🦀b"));
}

#[tokio::test]
async fn wrapped_model_stream_preserves_tool_calls() {
    let call = ToolCallRequest {
        id: "call-1".into(),
        name: "lookup".into(),
        arguments: serde_json::json!({"amount":17}),
    };
    let model = wrap_with_streaming(Box::new(
        MockLanguageModel::new("test").with_tool_calls(vec![call]),
    ));
    let stream = model
        .stream(Prompt::Text("hello".into()), GenerateOptions::default())
        .await
        .unwrap();
    let result = StreamCollector::collect(stream).await.unwrap();
    assert_eq!(result.finish_reason, FinishReason::ToolCall);
    assert_eq!(result.tool_calls.len(), 1);
    assert_eq!(result.tool_calls[0].id, "call-1");
    assert_eq!(result.tool_calls[0].name, "lookup");
    assert_eq!(
        result.tool_calls[0].arguments,
        serde_json::json!({"amount":17})
    );
}

#[tokio::test]
async fn wrapped_tool_arguments_reach_ui_stream() {
    use futures::StreamExt;
    use rs_ai_core::ui_stream::NdjsonEncoder;

    let model = wrap_with_streaming(Box::new(MockLanguageModel::new("test").with_tool_calls(
        vec![ToolCallRequest {
            id: "call-1".into(),
            name: "lookup".into(),
            arguments: serde_json::json!({"amount":17}),
        }],
    )));
    let stream = model
        .stream(Prompt::Text("hello".into()), GenerateOptions::default())
        .await
        .unwrap();
    let events: Vec<_> = NdjsonEncoder::encode_stream(stream).collect().await;
    assert!(events.into_iter().any(|event| {
        let event: serde_json::Value = serde_json::from_slice(&event.unwrap()).unwrap();
        event["type"] == "tool_call_args"
            && event["call_id"] == "call-1"
            && serde_json::from_str::<serde_json::Value>(event["delta"].as_str().unwrap()).unwrap()
                == serde_json::json!({"amount":17})
    }));
}

#[tokio::test]
async fn collected_stream_preserves_reasoning() {
    let events = vec![
        Ok(StreamEvent::ThinkingDelta {
            delta: "first ".into(),
        }),
        Ok(StreamEvent::ThinkingDelta {
            delta: "second".into(),
        }),
        Ok(StreamEvent::MessageEnd {
            finish_reason: FinishReason::Stop,
            usage: None,
        }),
    ];
    let result = StreamCollector::collect(Box::pin(futures::stream::iter(events)))
        .await
        .unwrap();
    assert_eq!(result.reasoning.as_deref(), Some("first second"));
}

struct DetailedModel(CapabilitySet);

#[async_trait::async_trait]
impl LanguageModel for DetailedModel {
    fn model_id(&self) -> &str {
        "test"
    }
    fn provider_id(&self) -> &str {
        "test"
    }
    fn capabilities(&self) -> &CapabilitySet {
        &self.0
    }
    async fn generate(&self, _: Prompt, _: GenerateOptions) -> AiResult<GenerateResult> {
        Ok(GenerateResult {
            text: Some("partial".into()),
            reasoning: Some("thought".into()),
            finish_reason: FinishReason::Length,
            usage: Usage {
                prompt_tokens: Some(7),
                completion_tokens: Some(3),
                total_tokens: Some(10),
            },
            ..Default::default()
        })
    }
    async fn stream(&self, _: Prompt, _: GenerateOptions) -> AiResult<AiStream> {
        unreachable!("the wrapper uses generate")
    }
}

#[tokio::test]
async fn wrapped_stream_preserves_reasoning_finish_and_usage() {
    let model = wrap_with_streaming(Box::new(DetailedModel(CapabilitySet::new())));
    let stream = model
        .stream(Prompt::Text("hello".into()), GenerateOptions::default())
        .await
        .unwrap();
    let result = StreamCollector::collect(stream).await.unwrap();
    assert_eq!(result.text.as_deref(), Some("partial"));
    assert_eq!(result.reasoning.as_deref(), Some("thought"));
    assert_eq!(result.finish_reason, FinishReason::Length);
    assert_eq!(result.usage.total_tokens, Some(10));
}

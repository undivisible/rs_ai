use std::sync::Arc;

use async_trait::async_trait;
use rs_ai_core::{
    AiError, AiResult, AiStream, Capability, CapabilitySet, FinishReason, GenerateOptions,
    GenerateResult, LanguageModel, Prompt, ResponseMetadata, SyntheticStreamer, Usage,
};

use super::bridge::BrowserAiBridge;
use super::capabilities::BrowserAiOptions;

/// A [`LanguageModel`] backed by the browser's built-in AI.
pub struct BrowserAiModel {
    bridge: Arc<dyn BrowserAiBridge>,
    capabilities: CapabilitySet,
}

impl BrowserAiModel {
    pub(crate) fn new(bridge: Arc<dyn BrowserAiBridge>) -> Self {
        let capabilities = CapabilitySet::new()
            .with(Capability::TextInput)
            .with(Capability::TextOutput)
            .with(Capability::LocalExecution)
            .with(Capability::PlatformNative);
        Self {
            bridge,
            capabilities,
        }
    }
}

#[async_trait]
impl LanguageModel for BrowserAiModel {
    fn model_id(&self) -> &str {
        "browser-ai"
    }

    fn provider_id(&self) -> &str {
        "browser"
    }

    fn capabilities(&self) -> &CapabilitySet {
        &self.capabilities
    }

    async fn generate(&self, prompt: Prompt, options: GenerateOptions) -> AiResult<GenerateResult> {
        let caps = self.bridge.detect().await;
        if !caps.available {
            return Err(AiError::PlatformUnavailable {
                platform: "Browser AI (no built-in AI detected)".into(),
            });
        }

        let browser_options = BrowserAiOptions {
            temperature: options.temperature,
            top_k: options.top_k,
            ..Default::default()
        };

        let text = crate::text_prompt(&prompt, "browser")?;
        let response = self
            .bridge
            .generate(&text, &browser_options)
            .await
            .map_err(|e| AiError::BridgeError {
                bridge: "browser".into(),
                message: e,
            })?;

        Ok(GenerateResult {
            text: Some(response),
            tool_calls: vec![],
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
            metadata: ResponseMetadata {
                provider: "browser".into(),
                model: "browser-ai".into(),
                ..Default::default()
            },
            steps: Vec::new(),
            reasoning: None,
        })
    }

    async fn stream(&self, prompt: Prompt, options: GenerateOptions) -> AiResult<AiStream> {
        let result = self.generate(prompt, options).await?;
        let text = result.text.unwrap_or_default();
        Ok(SyntheticStreamer::stream(text, 20))
    }
}

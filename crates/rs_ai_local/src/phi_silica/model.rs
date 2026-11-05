use std::sync::Arc;

use async_trait::async_trait;
use rs_ai_core::{
    AiError, AiResult, AiStream, Capability, CapabilitySet, FinishReason, GenerateOptions,
    GenerateResult, LanguageModel, Prompt, ResponseMetadata, StreamEvent, Usage,
};

use super::bridge::PhiSilicaBridge;
use super::types::PhiSilicaAvailability;

/// A [`LanguageModel`] backed by Windows Phi Silica.
pub struct PhiSilicaModel {
    bridge: Arc<dyn PhiSilicaBridge>,
    capabilities: CapabilitySet,
}

impl PhiSilicaModel {
    pub(crate) fn new(bridge: Arc<dyn PhiSilicaBridge>) -> Self {
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
impl LanguageModel for PhiSilicaModel {
    fn model_id(&self) -> &str {
        "phi-silica"
    }

    fn provider_id(&self) -> &str {
        "phi_silica"
    }

    fn capabilities(&self) -> &CapabilitySet {
        &self.capabilities
    }

    async fn generate(&self, prompt: Prompt, options: GenerateOptions) -> AiResult<GenerateResult> {
        match self.bridge.availability().await {
            PhiSilicaAvailability::Available => {}
            other => {
                return Err(AiError::PlatformUnavailable {
                    platform: format!("Windows Phi Silica: {other:?}"),
                });
            }
        }

        let text = crate::text_prompt(&prompt, "phi_silica")?;
        let response = self
            .bridge
            .generate(&text, options.max_tokens)
            .await
            .map_err(|e| AiError::BridgeError {
                bridge: "phi_silica".into(),
                message: e,
            })?;

        Ok(GenerateResult {
            text: Some(response),
            tool_calls: vec![],
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
            metadata: ResponseMetadata {
                provider: "phi_silica".into(),
                model: "phi-silica".into(),
                ..Default::default()
            },
            steps: Vec::new(),
            reasoning: None,
        })
    }

    async fn stream(&self, prompt: Prompt, options: GenerateOptions) -> AiResult<AiStream> {
        match self.bridge.availability().await {
            PhiSilicaAvailability::Available => {}
            other => {
                return Err(AiError::PlatformUnavailable {
                    platform: format!("Windows Phi Silica: {other:?}"),
                });
            }
        }

        let text = crate::text_prompt(&prompt, "phi_silica")?;
        let chunks = self
            .bridge
            .stream_tokens(&text, options.max_tokens)
            .await
            .map_err(|e| AiError::BridgeError {
                bridge: "phi_silica".into(),
                message: e,
            })?;

        let message_id = uuid::Uuid::new_v4().to_string();
        let events: Vec<Result<StreamEvent, AiError>> = {
            let mut v = Vec::new();
            v.push(Ok(StreamEvent::MessageStart { message_id }));
            for chunk in chunks {
                if !chunk.is_empty() {
                    v.push(Ok(StreamEvent::TextDelta { delta: chunk }));
                }
            }
            v.push(Ok(StreamEvent::MessageEnd {
                finish_reason: FinishReason::Stop,
                usage: None,
            }));
            v
        };

        Ok(Box::pin(futures::stream::iter(events)))
    }
}

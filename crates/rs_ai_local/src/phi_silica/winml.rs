//! Native WinML inference is not implemented. Use a Phi Silica bridge instead.

use async_trait::async_trait;

use rs_ai_core::{
    AiError, AiResult, AiStream, CapabilitySet, GenerateOptions, GenerateResult, LanguageModel,
    Prompt,
};

use super::types::PhiSilicaAvailability;

const PHI_SILICA_MODEL_ID: &str = "phi-silica";
const UNSUPPORTED: &str = "Native WinML inference is not implemented; use a PhiSilicaBridge";

/// Native WinML provider. Inference is not implemented.
#[derive(Default)]
pub struct NativePhiSilicaProvider;

impl NativePhiSilicaProvider {
    /// Create a new native Phi Silica provider.
    pub fn new() -> Self {
        Self
    }

    /// Check availability.
    pub async fn availability(&self) -> PhiSilicaAvailability {
        PhiSilicaAvailability::Unavailable
    }
}

/// Native WinML model. Inference is not implemented.
pub struct NativePhiSilicaModel {
    model_id: String,
    capabilities: CapabilitySet,
}

impl NativePhiSilicaModel {
    /// Create a new native model.
    pub fn new() -> Self {
        Self {
            model_id: PHI_SILICA_MODEL_ID.to_string(),
            capabilities: CapabilitySet::new(),
        }
    }

    /// Return an error because native WinML inference is not implemented.
    pub async fn generate(&self, _prompt: &str) -> Result<String, String> {
        Err(UNSUPPORTED.into())
    }
}

impl Default for NativePhiSilicaModel {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LanguageModel for NativePhiSilicaModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn provider_id(&self) -> &str {
        "phi_silica_native"
    }

    fn capabilities(&self) -> &CapabilitySet {
        &self.capabilities
    }

    async fn generate(
        &self,
        _prompt: Prompt,
        _options: GenerateOptions,
    ) -> AiResult<GenerateResult> {
        Err(AiError::UnsupportedCapability {
            capability: "native WinML inference (not implemented)".into(),
            provider: self.provider_id().into(),
        })
    }

    async fn stream(&self, _prompt: Prompt, _options: GenerateOptions) -> AiResult<AiStream> {
        Err(AiError::UnsupportedCapability {
            capability: "native WinML inference (not implemented)".into(),
            provider: self.provider_id().into(),
        })
    }
}

// ─── uniffi exports for C# consumer ─────────────────────────────────────────────

/// Availability result for uniffi.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum UniffiPhiSilicaAvailability {
    /// Inference is available.
    Available,
    /// Inference is unavailable.
    Unavailable,
    /// The Windows version is too old.
    WindowsVersionTooOld,
    /// No NPU was detected.
    NpuNotDetected,
}

/// Get Phi Silica availability (uniffi export).
pub fn get_availability() -> UniffiPhiSilicaAvailability {
    UniffiPhiSilicaAvailability::Unavailable
}

/// Return an error because native WinML inference is not implemented.
pub fn generate_text(_prompt: &str) -> Result<String, String> {
    Err(UNSUPPORTED.into())
}

/// Provider ID for uniffi.
pub fn get_provider_id() -> String {
    "phi_silica".into()
}

/// Model ID for uniffi.
pub fn get_model_id() -> String {
    PHI_SILICA_MODEL_ID.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn unsupported_native_inference_returns_errors() {
        let model = NativePhiSilicaModel::new();
        assert!(model
            .generate("a real question")
            .await
            .unwrap_err()
            .contains("not implemented"));
        assert!(matches!(
            LanguageModel::generate(&model, "a real question".into(), GenerateOptions::default())
                .await,
            Err(AiError::UnsupportedCapability { .. })
        ));
        assert!(matches!(
            LanguageModel::stream(&model, "a real question".into(), GenerateOptions::default())
                .await,
            Err(AiError::UnsupportedCapability { .. })
        ));
        assert!(generate_text("a real question")
            .unwrap_err()
            .contains("not implemented"));
        assert_eq!(
            NativePhiSilicaProvider::new().availability().await,
            PhiSilicaAvailability::Unavailable
        );
        assert!(matches!(
            get_availability(),
            UniffiPhiSilicaAvailability::Unavailable
        ));
    }
}

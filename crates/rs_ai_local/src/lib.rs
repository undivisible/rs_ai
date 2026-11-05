//!
//! ⚠️ **UNSTABLE** — This crate is in active development. APIs may change without notice.
//!
//! Platform-specific local AI runtimes for the Rusty AI SDK.
//!
//! Each runtime is gated behind a Cargo feature flag:
//! - `browser` — WASM browser AI (Chrome/Edge built-in AI)
//! - `gemini-nano` — Android Gemini Nano (Prompt API via JNI)
//! - `foundationmodels` — Apple Foundation Models (uniffi Swift bindings)
//! - `phi-silica` — Windows Phi Silica (C# bridge via build.rs)

#[cfg(feature = "browser")]
pub mod browser;
#[cfg(feature = "foundationmodels")]
pub mod foundationmodels;
#[cfg(feature = "gemini-nano")]
pub mod gemini_nano;
#[cfg(feature = "phi-silica")]
pub mod phi_silica;

// Re-exports for convenience when the corresponding feature is enabled.
#[cfg(feature = "browser")]
pub use browser::*;
#[cfg(feature = "foundationmodels")]
pub use foundationmodels::*;
#[cfg(feature = "gemini-nano")]
pub use gemini_nano::*;
#[cfg(feature = "phi-silica")]
pub use phi_silica::*;

#[cfg(any(
    feature = "browser",
    feature = "foundationmodels",
    feature = "phi-silica"
))]
fn text_prompt(prompt: &rs_ai_core::Prompt, provider: &str) -> rs_ai_core::AiResult<String> {
    use rs_ai_core::{AiError, ContentPart, Prompt};

    match prompt {
        Prompt::Text(text) => Ok(text.clone()),
        Prompt::Messages(messages) => messages
            .iter()
            .flat_map(|message| &message.content)
            .map(|part| match part {
                ContentPart::Text { text } => Ok(text.clone()),
                _ => Err(AiError::UnsupportedCapability {
                    capability: "non-text prompt parts".into(),
                    provider: provider.into(),
                }),
            })
            .collect::<rs_ai_core::AiResult<Vec<_>>>()
            .map(|parts| parts.join("\n")),
    }
}

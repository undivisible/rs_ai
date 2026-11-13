//!
//! ⚠️ **UNSTABLE** — This crate is in active development. APIs may change without notice.
//!
//! Generic adapter for any OpenAI-compatible chat-completions API.
//!
//! This crate provides [`OpenAiCompatibleProvider`] and [`OpenAiCompatibleModel`]
//! which implement the core `rs_ai_core` traits and can be pointed at any API that
//! follows the OpenAI chat-completions wire format (OpenAI, Azure, Together,
//! Groq, local vLLM, etc.).

pub(crate) mod api_types;
#[cfg(all(feature = "openai-compatible", not(target_arch = "wasm32")))]
mod config;
pub(crate) mod convert;
#[cfg(all(feature = "openai-compatible", not(target_arch = "wasm32")))]
mod model;
#[cfg(all(feature = "openai-compatible", not(target_arch = "wasm32")))]
mod provider;
#[cfg(all(feature = "openai-compatible", not(target_arch = "wasm32")))]
mod stream_parser;

#[cfg(all(feature = "openai-compatible", not(target_arch = "wasm32")))]
pub mod presets;

#[cfg(all(feature = "openai-compatible", not(target_arch = "wasm32")))]
pub use config::OpenAiCompatibleConfig;
#[cfg(all(feature = "openai-compatible", not(target_arch = "wasm32")))]
pub use model::OpenAiCompatibleModel;
#[cfg(all(feature = "openai-compatible", not(target_arch = "wasm32")))]
pub use provider::OpenAiCompatibleProvider;

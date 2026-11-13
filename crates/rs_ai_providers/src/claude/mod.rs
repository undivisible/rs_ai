//!
//! ⚠️ **UNSTABLE** — This crate is in active development. APIs may change without notice.
//!
//! Anthropic Claude provider for the Rusty AI SDK.
//!
//! This crate implements the `LanguageModel` and `Provider` traits from
//! `rs_ai_core` for the Anthropic Messages API. It is **not** a wrapper around
//! an OpenAI-compatible endpoint -- it speaks Anthropic's native API format
//! directly.
//!
//! # Quick start
//!
//! ```rust,no_run
//! # #[cfg(all(feature = "claude", not(target_arch = "wasm32")))]
//! # {
//! use rs_ai_providers::claude::ClaudeProvider;
//!
//! let provider = ClaudeProvider::new("sk-ant-...");
//! let model = provider.claude_sonnet();
//! # }
//! ```

pub(crate) mod api_types;
pub(crate) mod convert;
#[cfg(all(feature = "claude", not(target_arch = "wasm32")))]
mod model;
#[cfg(all(feature = "claude", not(target_arch = "wasm32")))]
mod provider;
#[cfg(all(feature = "claude", not(target_arch = "wasm32")))]
mod stream_parser;

#[cfg(all(feature = "claude", not(target_arch = "wasm32")))]
pub use model::*;
#[cfg(all(feature = "claude", not(target_arch = "wasm32")))]
pub use provider::*;

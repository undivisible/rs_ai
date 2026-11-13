//!
//! ⚠️ **UNSTABLE** — This crate is in active development. APIs may change without notice.
//!
//! Cloud AI providers for the Rusty AI SDK.
//!
//! Each provider is gated behind a Cargo feature flag:
//! - `claude` — Anthropic Claude
//! - `chatgpt` — OpenAI ChatGPT
//! - `gemini` — Google Gemini
//! - `openai-compatible` — Generic OpenAI-compatible adapter
//! - `xai` — xAI Grok
//! - `cloudflare` — Cloudflare Workers AI
//! - `ollama` — Local Ollama
//! - `portkey` — Portkey AI Gateway
//! - `langfuse` — Langfuse observability wrapper
//!
//! The `browser` feature provides non-streaming Anthropic and OpenAI-compatible
//! API-key clients on Wasm and native targets. The existing `claude` and
//! `openai-compatible` LanguageModel implementations are native-only; other
//! provider features have not been ported to Wasm. Browser APIs require provider
//! CORS permission and expose the key to the calling page.
//!
//! The `catalog` module (always on) provides static metadata for
//! API-key-based providers — id, name, env vars, base URLs, models.

pub mod catalog;

#[cfg(feature = "browser")]
pub mod browser;

#[cfg(feature = "chatgpt")]
pub mod chatgpt;
#[cfg(any(feature = "claude", feature = "browser"))]
pub mod claude;
#[cfg(feature = "cloudflare")]
pub mod cloudflare;
#[cfg(feature = "cohere")]
pub mod cohere;
#[cfg(feature = "gemini")]
pub mod gemini;
#[cfg(feature = "langfuse")]
pub mod langfuse;
#[cfg(feature = "ollama")]
pub mod ollama;
#[cfg(any(feature = "openai-compatible", feature = "browser"))]
pub mod openai_compatible;
#[cfg(feature = "portkey")]
pub mod portkey;
#[cfg(feature = "voyage")]
pub mod voyage;
#[cfg(feature = "xai")]
pub mod xai;

// Re-exports for convenience when the corresponding feature is enabled.
#[cfg(feature = "chatgpt")]
pub use chatgpt::*;
#[cfg(all(feature = "claude", not(target_arch = "wasm32")))]
pub use claude::*;
#[cfg(feature = "cloudflare")]
pub use cloudflare::*;
#[cfg(feature = "cohere")]
pub use cohere::{CohereProvider, CohereRerankingModel, RERANK_MODEL as COHERE_RERANK_MODEL};
#[cfg(feature = "gemini")]
pub use gemini::*;
#[cfg(feature = "langfuse")]
pub use langfuse::*;
#[cfg(feature = "ollama")]
pub use ollama::*;
#[cfg(all(feature = "openai-compatible", not(target_arch = "wasm32")))]
pub use openai_compatible::*;
#[cfg(feature = "portkey")]
pub use portkey::*;
#[cfg(feature = "voyage")]
pub use voyage::{VoyageRerankingModel, RERANK_LITE, RERANK_MODEL as VOYAGE_RERANK_MODEL};
#[cfg(feature = "xai")]
pub use xai::*;

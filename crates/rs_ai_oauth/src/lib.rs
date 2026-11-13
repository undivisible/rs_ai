//! OAuth flows and runtime model discovery for AI providers.
//!
//! Native builds support xAI Grok and the Codex-style OpenAI flow, plus runtime
//! model discovery. [`browser_auth`] provides portable PKCE and state primitives;
//! it does not itself implement provider authorization, token exchange, or OIDC
//! validation. Browser access depends on the provider's client and transport rules.
//! Native listener, credential storage, and model discovery require the default
//! `native` feature and are unavailable on wasm32.
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(all(feature = "native", not(target_arch = "wasm32")))]
//! # {
//! use rs_ai_oauth::{start_oauth_flow, OAuthProvider, fetch_models};
//!
//! // 1. Login — opens browser, waits for callback
//! let tokens = start_oauth_flow(OAuthProvider::Xai).unwrap();
//! println!("access token: {}", tokens.access_token);
//!
//! // 2. Fetch available models at runtime
//! let models = fetch_models(OAuthProvider::Xai, &tokens.access_token).unwrap();
//! for m in &models {
//!     println!("  {}", m.id);
//! }
//! # }
//! ```
//!
//! For a long-running model picker, keep a catalog and refresh it periodically:
//!
//! ```no_run
//! # #[cfg(all(feature = "native", not(target_arch = "wasm32")))]
//! # {
//! # use rs_ai_oauth::{ModelCatalog, OAuthProvider};
//! # let token = String::new();
//! let mut catalog = ModelCatalog::new(OAuthProvider::Xai);
//! let update = catalog.refresh(&token).unwrap();
//! println!("new models: {}", update.added.len());
//! # }
//! ```

/// Platform-independent PKCE and OAuth state primitives. These do not select
/// a provider, redirect URI, or perform authorization or token exchange.
pub mod browser_auth;

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
pub mod claude_code;
#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
mod fetch;
#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
mod flow;

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
pub mod codex;
#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
pub mod credentials;

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
pub use fetch::{
    fetch_logged_in_models, fetch_logged_in_models_async, fetch_models, fetch_models_async,
    ModelCatalog, ModelCatalogUpdate, ModelInfo, ModelLimits, ModelPricing, ProviderModels,
};
#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
pub use flow::{refresh_oauth_token, start_oauth_flow, OAuthError, OAuthProvider, OAuthTokens};

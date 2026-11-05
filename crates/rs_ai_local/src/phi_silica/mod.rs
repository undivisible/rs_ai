//! ⚠️ **UNSTABLE** — This crate is in active development. APIs may change without notice.
//!
//! Windows Phi Silica local runtime for the Rusty AI SDK.
//!
//! This crate provides integration with Microsoft's on-device AI models on Windows,
//! including Phi Silica (via Windows App SDK) and generic ONNX models (via Windows ML).
//!
//! # Architecture
//!
//! ```text
//! Rust (rs_ai_local::phi_silica)
//!   │
//!   ├── Native WinML path ───────► unsupported
//!   │
//!   └── With custom bridge ─────► C# / Windows App SDK (Phi Silica APIs)
//! ```
//!
//! # Phi Silica (Windows App SDK)
//!
//! Phi Silica requires the Windows App SDK and is accessed through C# APIs:
//! - `Microsoft.Windows.AI.Text.LanguageModel`
//!
//! To use Phi Silica specifically, implement the [`PhiSilicaBridge`] trait in your
//! host application or use the C# bridge example in `csharp/`.
//!
//! # Requirements
//!
//! | Feature | Requirement |
//! |---|---|
//! | Phi Silica | Windows 11 24H2+, Copilot+ PC, Windows App SDK |

#![deny(missing_docs)]

mod bridge;
mod model;
mod native_bridge;
mod provider;
mod types;

#[cfg(any(windows, test))]
mod winml;

pub use bridge::*;
pub use model::*;
pub use native_bridge::*;
pub use provider::*;
pub use types::*;

#[cfg(any(windows, test))]
pub use winml::*;

// Re-export rs_ai_core for convenience
pub use rs_ai_core;

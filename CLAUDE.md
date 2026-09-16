# rs_ai — Codebase Guide (v0.2.36)

## Overview

`rs_ai` is a Rust workspace SDK for AI applications. The top-level `rs_ai` crate exposes a fluent builder API (`rs_ai_claude()`, `rs_ai_gemini()`, etc.) over consolidated provider and core crates.

## Workspace Layout

```
rs_ai/
├── Cargo.toml               # version = "0.2.36"
├── Makefile                 # make {fmt,check,clippy,test,doc,ci}
├── crates/
│   ├── rs_ai_core/          # Core traits, types, middleware, cache, observability
│   ├── rs_ai_providers/     # Cloud AI providers (feature-gated)
│   ├── rs_ai_local/         # Platform-specific local runtimes (feature-gated)
│   ├── rs_ai_testing/       # MockLanguageModel, MockProvider
│   └── rs_ai/               # Fluent top-level API
└── examples/ (17 examples)
```

## Core Crate: `rs_ai_core`

Defines the traits every provider implements, plus middleware, cache config, tools, streaming, and observability:

```rust
pub trait LanguageModel: Send + Sync {
    fn model_id(&self) -> &str;
    fn provider_id(&self) -> &str;
    fn capabilities(&self) -> &CapabilitySet;
    async fn generate(&self, prompt: Prompt, options: GenerateOptions) -> AiResult<GenerateResult>;
    async fn stream(&self, prompt: Prompt, options: GenerateOptions) -> AiResult<AiStream>;
}

pub trait ImageModel: Send + Sync {
    fn model_id(&self) -> &str;
    fn provider_id(&self) -> &str;
    async fn generate_image(&self, prompt: &str, options: ImageGenerationOptions) -> AiResult<ImageResult>;
}

pub trait VideoModel: Send + Sync {
    fn model_id(&self) -> &str;
    fn provider_id(&self) -> &str;
    async fn generate_video(&self, prompt: &str, options: VideoGenerationOptions) -> AiResult<VideoResult>;
}

pub trait RealtimeSession: Send + Sync {
    fn model_id(&self) -> &str;
    fn provider_id(&self) -> &str;
    async fn send_text(&mut self, text: &str) -> AiResult<()>;
    async fn send_audio(&mut self, audio: Vec<u8>, mime_type: &str) -> AiResult<()>;
    async fn recv(&mut self) -> Option<RealtimeEvent>;
    async fn close(self: Box<Self>) -> AiResult<()>;
}

pub trait RerankingModel: Send + Sync {
    fn model_id(&self) -> &str;
    fn provider_id(&self) -> &str;
    async fn rerank(&self, query: &str, documents: Vec<String>, options: RerankOptions) -> AiResult<RerankResult>;
}

pub trait EmbeddingModel: Send + Sync {
    fn model_id(&self) -> &str;
    fn provider_id(&self) -> &str;
    async fn embed(&self, texts: Vec<String>) -> AiResult<Vec<Vec<f64>>>;
}

pub trait Provider {
    fn id(&self) -> &str;
    fn name(&self) -> &str;
    fn language_model(&self, model_id: &str) -> AiResult<Box<dyn LanguageModel>>;
}
```

Agent loop feature — automatically executes tool calls up to `max_steps`:

```rust
pub async fn agent_loop(
    model: &dyn LanguageModel,
    prompt: Prompt,
    options: GenerateOptions,
    tools: Option<&ToolSet>,
) -> AiResult<GenerateResult>
```

### Key types

- `Prompt`, `GenerateResult` (with `steps: Vec<StepResult>`, `reasoning: Option<String>`)
- `StreamEvent`, `AiStream`
- `AiError`, `Usage`, `FinishReason`
- `CapabilitySet`, `ToolDefinition`, `ToolSet`, `ToolContext`, `ToolExecutionOptions`
- `StepResult`, `RerankResult`, `RerankedDocument`
- `ImageResult`, `VideoResult`, `GeneratedFile`
- `Output` enum: `Enum`, `Array`, `Object`, `NoSchema`
- `RealtimeEvent` enum: `TextDelta`, `TextDone`, `AudioDelta`, `ToolCall`, `ToolResult`, `Error`, `Done`

### Re-exports

- `cache::{CacheConfig, CacheTTL}`
- `middleware::{CacheMiddleware, RetryMiddleware, MiddlewareChain, ExtractReasoningMiddleware, extract_reasoning_middleware}`
- `observability::{with_observability, ObservableModel}`
- `registry::{ProviderRegistry, create_provider_registry}`
- `model::{agent_loop, RerankingModel, RerankOptions, ImageModel, VideoModel, RealtimeSession, ...}`
- `structured::{GenerateResult, StepResult, ImageResult, VideoResult, RerankResult, ...}`
- `schema::{OutputSchema, Output}`

## Providers Crate: `rs_ai_providers`

Feature flags:

```toml
[features]
default = []
claude = []
chatgpt = []
gemini = []
openai-compatible = []
xai = []
cloudflare = []
ollama = []
portkey = []
langfuse = []
```

| Module | Provider | Image | Video | Realtime |
|--------|----------|-------|-------|----------|
| `chatgpt` | OpenAI ChatGPT | DALL-E 3/2 | Sora (stub) | `ChatGptRealtimeSession` |
| `gemini` | Google Gemini | Imagen 3 | Veo (stub) | `GeminiLiveSessionAdapter` |
| `xai` | xAI Grok | Grok Imagine/Aurora | — | — |
| `claude` | Anthropic Claude | — | — | — |
| `openai_compatible` | Any OpenAI-compatible | — | — | — |
| `cloudflare` | Cloudflare Workers AI | — | — | — |
| `ollama` | Local Ollama | — | — | — |
| `portkey` | Portkey AI Gateway | — | — | — |
| `langfuse` | Langfuse observability | — | — | — |

## Local Crate: `rs_ai_local`

```toml
[features]
default = []
browser = ["dep:wasm-bindgen", "dep:wasm-bindgen-futures", "dep:js-sys", "dep:web-sys"]
gemini-nano = ["dep:jni", "dep:uniffi"]
foundationmodels = ["dep:uniffi"]
phi-silica = []
```

- `browser` — WASM browser AI (Chrome/Edge built-in AI)
- `gemini_nano` — Android Gemini Nano (JNI, 1-line Kotlin init)
- `foundationmodels` — Apple Foundation Models (macOS)
- `phi_silica` — Windows Phi Silica (C# bridge via `build.rs`)

## Top-Level API: `rs_ai`

### Entry points

| Function | Provider | Default env var |
|---|---|---|
| `rs_ai_claude()` | Anthropic Claude | `ANTHROPIC_API_KEY` |
| `rs_ai_chatgpt()` | OpenAI ChatGPT | `OPENAI_API_KEY` |
| `rs_ai_gemini()` | Google Gemini | `GOOGLE_API_KEY` |
| `rs_ai_xai()` | xAI Grok | `XAI_API_KEY` |
| `rs_ai_cloudflare(account_id)` | Cloudflare Workers AI | `CLOUDFLARE_API_TOKEN` |
| `rs_ai_compatible(base_url)` | Any OpenAI-compatible | `OPENAI_API_KEY` |

### Builder methods

| Method | Description |
|---|---|
| `.api_key(key)` | Override env-var key |
| `.model(id)` | Required — model identifier string |
| `.with_image(url_or_path)` | Attach image (URL or local file, base64-encoded) |
| `.cf_ai_gateway(gw)` | Route through Cloudflare AI Gateway |
| `.with_cache(config)` | Apply CacheConfig |
| `.enable_cache()` | Enable default ephemeral cache |
| `.max_steps(n)` | Agent loop max tool-call iterations |
| `.generate(prompt)` → `AiResult<String>` | Text generation |
| `.stream(prompt)` → `AiResult<BoxStream<AiResult<String>>>` | Streaming |
| `.generate_image(prompt, opts)` → `AiResult<ImageResult>` | DALL-E / Imagen / Grok Imagine |
| `.generate_video(prompt, opts)` → `AiResult<VideoResult>` | Sora / Veo |
| `.realtime_session()` → `AiResult<Box<dyn RealtimeSession>>` | Voice/text session |
| `.speak(text)` → `AiResult<Vec<u8>>` | TTS (ChatGPT only) |
| `.transcribe(audio)` → `AiResult<String>` | STT (ChatGPT only) |

### Provider Registry (Vercel-style)

```rust
use rs_ai_core::{create_provider_registry, ProviderRegistry};
use rs_ai_providers::chatgpt::ChatGptProvider;
use rs_ai_providers::claude::ClaudeProvider;

let registry = create_provider_registry()
    .register("openai", ChatGptProvider::new("sk-..."))
    .register("anthropic", ClaudeProvider::new("sk-ant-..."));
let model = registry.model("openai/gpt-4o")?;
```

## Middleware

Chain-able middleware over any `LanguageModel`:

- `RetryMiddleware` — exponential backoff on transient errors
- `CacheMiddleware` — in-memory response cache with TTL + key hash
- `ExtractReasoningMiddleware` — strips `<thinking>` and ` ```reasoning ` blocks from model output, stores extracted reasoning in `result.reasoning`
- `GuardrailMiddleware` — content filter with `FilterAction::Allow / Block / Replace`

```rust
use rs_ai_core::middleware::{
    ExtractReasoningMiddleware, GuardrailMiddleware, GuardrailConfig, FilterAction,
    extract_reasoning_middleware,
};
```

### wrapLanguageModel

```rust
use rs_ai_core::wrap_language_model;

let wrapped = wrap_language_model(model, vec![
    Box::new(GuardrailMiddleware::new(config)),
    Box::new(ExtractReasoningMiddleware::default()),
]);
```

## Agent Loop

Automatically executes tool calls up to `max_steps` iterations:

```rust
use rs_ai_core::agent_loop;
use rs_ai_core::tool::ToolSet;

let mut tools = ToolSet::new();
tools.add(MyTool);

let result = agent_loop(&model, Prompt::text("What's the weather?"), 
    GenerateOptions::default().max_steps(5),
    Some(&tools)).await?;
// result.steps contains each intermediate step
```

## Testing

```bash
make ci                    # fmt + check + clippy + test + doc
cargo test --workspace     # all workspace tests
cargo test -p rs_ai_core   # core crate tests (64 unit tests)
```

`rs_ai_testing` provides mock implementations:

```rust
use rs_ai_testing::{MockLanguageModel, MockResponse};

let mock = MockLanguageModel::new("test-model")
    .with_text("hello")
    .with_error(AiError::StreamError { message: "timeout".into() })
    .with_object(serde_json::json!({ "answer": 42 }));
```

## Adding a New Provider

1. Create module in `crates/rs_ai_providers/src/<provider>/`
2. Implement `LanguageModel` (and optionally `Provider`, `ImageModel`, `VideoModel`) from `rs_ai_core`
3. Add feature flag in `crates/rs_ai_providers/Cargo.toml`
4. Add `pub mod` behind `#[cfg(feature = "...")]` in providers `lib.rs`
5. Add entry point function in `crates/rs_ai/src/lib.rs`
6. Add integration tests in `crates/rs_ai_providers/tests/`

## Reranking Providers

### Cohere
```toml
rs_ai_providers = { version = "0.2", features = ["cohere"] }
```
```rust
use rs_ai_providers::cohere::{CohereProvider, CohereRerankingModel};

let model = CohereRerankingModel::new("rerank-v3.5", api_key);
let result = model.rerank("query", docs, RerankOptions::default()).await?;
```

### Voyage
```toml
rs_ai_providers = { version = "0.2", features = ["voyage"] }
```
```rust
use rs_ai_providers::voyage::VoyageRerankingModel;

let model = VoyageRerankingModel::new("rerank-2", api_key);
let result = model.rerank("query", docs, RerankOptions::default()).await?;
```

## OpenTelemetry

```toml
rs_ai_core = { version = "0.2", features = ["telemetry"] }
```
```rust
use rs_ai_core::telemetry::{init_telemetry, TelemetryConfig};

init_telemetry(&TelemetryConfig {
    service_name: Some("my-app".into()),
    ..Default::default()
})?;
```

## OAuth

```rust
use rs_ai_core::oauth::{start_oauth_flow, OAuthProvider};

let tokens = start_oauth_flow(OAuthProvider::ChatGpt)?;
let client = rs_ai::chatgpt()
    .with_oauth_token(tokens.access_token)
    .model("gpt-4o")
    .generate("Hello?"); 
```

## Key Dependencies

- `tokio` — async runtime
- `reqwest` — HTTP client with streaming
- `serde` / `serde_json` — serialisation
- `futures` — `Stream` trait and combinators
- `async-trait` — async in trait definitions
- `thiserror` — error types
- `pin-project-lite` — safe `Stream` pinning
- `uuid` — trace IDs
- `tracing` — structured logging
- `opentelemetry` / `opentelemetry-sdk` / `opentelemetry-otlp` — OTEL export (optional)

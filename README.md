# rs_ai — Rust AI SDK (v0.2.36)

Comprehensive Rust SDK for AI applications. Cloud + local providers, streaming, agent loop, image/video generation, realtime voice, and a clean async-first API.

**For agents:** see [AGENTS.md](AGENTS.md) for workspace layout, trait definitions, provider internals, and testing conventions.

## Quick Start

Add the SDK and pick a provider. API keys are read from the environment by
default (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, …), or pass
one explicitly with `.api_key(...)`.

```toml
[dependencies]
rs_ai = "0.2"
tokio = { version = "1", features = ["full"] }
```

```rust
use rs_ai::claude;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let answer = claude()
        .model("claude-sonnet-4-6")
        .generate("What is 2+2?")
        .await?;
    println!("{}", answer);
    Ok(())
}
```

```bash
ANTHROPIC_API_KEY=sk-ant-... cargo run
```

### Streaming

```rust
use rs_ai::claude;
use futures::StreamExt;

let mut stream = claude()
    .model("claude-sonnet-4-6")
    .stream("Write a poem")
    .await?;
while let Some(chunk) = stream.next().await {
    print!("{}", chunk?);
}
```

### Agent Loop (auto tool execution)

```rust
use rs_ai_core::agent_loop;
use rs_ai_core::tool::ToolSet;

let mut tools = ToolSet::new();
tools.add(MyWeatherTool);

let result = agent_loop(&model, "What's the weather?".into(),
    Default::default().max_steps(5), Some(&tools)).await?;
// result.steps holds each intermediate step
```

### Provider Registry

```rust
use rs_ai_core::create_provider_registry;
use rs_ai_providers::chatgpt::ChatGptProvider;

let registry = create_provider_registry()
    .register("openai", ChatGptProvider::new("sk-..."));
let model = registry.model("openai/gpt-4o")?;
```

### Image Generation

```rust
let image = rs_ai::chatgpt()
    .model("dall-e-3")
    .generate_image("A cat in space", Default::default())
    .await?;
// image.data holds the bytes
```

## Providers

| Function | Provider | Env Var | Extras |
|---|---|---|---|
| `rs_ai::claude()` | Anthropic Claude | `ANTHROPIC_API_KEY` | streaming, tools, vision |
| `rs_ai::chatgpt()` | OpenAI ChatGPT | `OPENAI_API_KEY` | DALL-E, Sora, TTS, STT, realtime |
| `rs_ai::gemini()` | Google Gemini | `GOOGLE_API_KEY` | Imagen, Veo, live API |
| `rs_ai::xai()` | xAI Grok | `XAI_API_KEY` | Grok Imagine |
| `rs_ai::cloudflare(id)` | Cloudflare Workers AI | `CLOUDFLARE_API_TOKEN` | — |
| `rs_ai::compatible(url)` | Any OpenAI-compatible | `OPENAI_API_KEY` | OpenRouter, vLLM, Ollama, etc. |

## Real TTS/STT

```rust
let audio = rs_ai::chatgpt()
    .model("tts-1")
    .speak("Hello world!")  // Returns real MP3 bytes via OpenAI TTS
    .await?;

let text = rs_ai::chatgpt()
    .model("whisper-1")
    .transcribe(audio_bytes, "audio/webm")
    .await?;
```

## Reranking (Cohere / Voyage)

```rust
use rs_ai_providers::cohere::CohereRerankingModel;
use rs_ai_core::RerankOptions;

let model = CohereRerankingModel::new("rerank-v3.5", api_key);
let result = model.rerank("my query", vec!["doc1".into(), "doc2".into()], RerankOptions::default()).await?;
```

## Local Runtimes

```toml
[dependencies]
rs_ai_local = { version = "0.2", features = ["browser"] }       # WASM browser AI
rs_ai_local = { version = "0.2", features = ["gemini-nano"] }   # Android Gemini Nano
rs_ai_local = { version = "0.2", features = ["foundationmodels"] } # macOS Apple FM
rs_ai_local = { version = "0.2", features = ["phi-silica"] }    # Windows Phi Silica
```

## Testing

```bash
make ci        # fmt + check + clippy + test + doc (152 tests)
cargo test     # same
```

## License

ISC

use std::sync::Arc;

use async_trait::async_trait;

use crate::capability::CapabilitySet;
use crate::error::{AiError, AiResult};
use crate::prompt::Prompt;
use crate::schema::OutputSchema;
use crate::stream::AiStream;
use crate::structured::{
    AudioResult, EmbeddingResult, GenerateResult, ImageResult, ObjectResult, RerankResult,
    StepResult, TranscriptionResult, TtsOptions, VideoResult,
};
use crate::tool::{ToolCallRequest, ToolCallResult, ToolChoice, ToolContext, ToolDefinition};
use crate::types::RequestMetadata;

/// Callback invoked after each step in the agent loop.
pub type OnStepFinish = Arc<dyn Fn(&StepResult) + Send + Sync>;

/// Callback invoked before a tool is executed.
/// Return `Ok(())` to proceed, or `Err(msg)` to abort the tool call.
pub type OnToolCall = Arc<dyn Fn(&ToolCallRequest) -> Result<(), String> + Send + Sync>;

/// Callback invoked after the final result is available.
pub type OnFinish = Arc<dyn Fn(&GenerateResult) + Send + Sync>;

/// Lifecycle callbacks for observing and intervening in the agent loop.
///
/// Attach via [`GenerateOptions::with_callbacks`].
#[derive(Clone, Default)]
pub struct LifecycleCallbacks {
    /// Called after each step completes in the agent loop.
    pub on_step_finish: Option<OnStepFinish>,
    /// Called before each tool execution. Return `Err(msg)` to abort.
    pub on_tool_call: Option<OnToolCall>,
    /// Called with the final result before returning.
    pub on_finish: Option<OnFinish>,
}

impl std::fmt::Debug for LifecycleCallbacks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LifecycleCallbacks")
            .field("on_step_finish", &self.on_step_finish.is_some())
            .field("on_tool_call", &self.on_tool_call.is_some())
            .field("on_finish", &self.on_finish.is_some())
            .finish()
    }
}

/// Extended-thinking / reasoning configuration.
///
/// Supported by Anthropic (adaptive thinking), Gemini 2.5+ (thinking budget),
/// and Ollama reasoning models (think flag).
#[derive(Debug, Clone)]
pub enum ThinkingConfig {
    /// Enable thinking with adaptive budget (Anthropic claude-3-7-sonnet and later).
    Adaptive,
    /// Enable thinking with a fixed token budget (Gemini 2.5+).
    Budget {
        /// Maximum thinking tokens.
        tokens: u32,
    },
    /// Simple on/off flag (Ollama, Gemini 3 Flash `think: true`).
    Enabled,
}

/// Reasoning effort level for models that support it (OpenAI Responses API).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    /// No reasoning.
    None,
    /// Low reasoning effort.
    Low,
    /// Medium reasoning effort.
    Medium,
    /// High reasoning effort.
    High,
}

/// Options that control generation behaviour.
#[derive(Debug, Clone, Default)]
pub struct GenerateOptions {
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Maximum tokens to generate.
    pub max_tokens: Option<u32>,
    /// Nucleus sampling threshold.
    pub top_p: Option<f64>,
    /// Top-k sampling threshold.
    pub top_k: Option<u32>,
    /// Sequences that stop generation.
    pub stop_sequences: Vec<String>,
    /// Frequency penalty.
    pub frequency_penalty: Option<f64>,
    /// Presence penalty.
    pub presence_penalty: Option<f64>,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
    /// Tool definitions available to the model.
    pub tools: Option<Vec<ToolDefinition>>,
    /// Strategy for choosing tools.
    pub tool_choice: Option<ToolChoice>,
    /// JSON schema for structured output.
    pub output_schema: Option<OutputSchema>,
    /// Extended thinking / reasoning configuration.
    pub thinking: Option<ThinkingConfig>,
    /// Reasoning effort (OpenAI Responses API).
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Maximum number of tool-calling steps (agent loop).
    /// When > 1, the model's tool calls are automatically executed and fed back
    /// as new messages. Default: 1 (single-shot, no auto-loop).
    pub max_steps: Option<u32>,
    /// Tool context — arbitrary per-tool data passed to tool handlers.
    /// Analogous to Vercel's `toolsContext` parameter.
    pub tools_context: Option<ToolContext>,
    /// Request metadata.
    pub metadata: RequestMetadata,
    /// Lifecycle callbacks for the agent loop.
    pub callbacks: Option<LifecycleCallbacks>,
}

impl GenerateOptions {
    /// Set the temperature.
    pub fn with_temperature(mut self, t: f64) -> Self {
        self.temperature = Some(t);
        self
    }

    /// Set the maximum number of tokens to generate.
    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }

    /// Set top-p (nucleus sampling).
    pub fn with_top_p(mut self, p: f64) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Set top-k sampling.
    pub fn with_top_k(mut self, k: u32) -> Self {
        self.top_k = Some(k);
        self
    }

    /// Set stop sequences.
    pub fn with_stop_sequences(mut self, seqs: Vec<String>) -> Self {
        self.stop_sequences = seqs;
        self
    }

    /// Set frequency penalty.
    pub fn with_frequency_penalty(mut self, p: f64) -> Self {
        self.frequency_penalty = Some(p);
        self
    }

    /// Set presence penalty.
    pub fn with_presence_penalty(mut self, p: f64) -> Self {
        self.presence_penalty = Some(p);
        self
    }

    /// Set the random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Provide tool definitions.
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Set the tool choice strategy.
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Set the output schema for structured generation.
    pub fn with_output_schema(mut self, schema: OutputSchema) -> Self {
        self.output_schema = Some(schema);
        self
    }

    /// Enable extended thinking with adaptive budget.
    pub fn with_thinking(mut self, config: ThinkingConfig) -> Self {
        self.thinking = Some(config);
        self
    }

    /// Set the reasoning effort level (OpenAI Responses API).
    pub fn with_reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }

    /// Set request metadata.
    pub fn with_metadata(mut self, metadata: RequestMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Attach lifecycle callbacks for the agent loop.
    pub fn with_callbacks(mut self, cbs: LifecycleCallbacks) -> Self {
        self.callbacks = Some(cbs);
        self
    }

    /// Enable agent loop with the given max steps.
    /// When > 1, tool calls are automatically executed and fed back to the model.
    pub fn with_max_steps(mut self, max_steps: u32) -> Self {
        self.max_steps = Some(max_steps);
        self
    }

    /// Set tool context — arbitrary per-tool data passed to tool handlers.
    pub fn with_tools_context(mut self, ctx: ToolContext) -> Self {
        self.tools_context = Some(ctx);
        self
    }
}

/// Describes a provider backend.
#[derive(Debug, Clone)]
pub struct ProviderInfo {
    /// Human-readable provider name.
    pub name: String,
    /// Default base URL for API requests.
    pub default_base_url: Option<String>,
}

/// The core language model trait that all providers implement.
#[async_trait]
pub trait LanguageModel: Send + Sync {
    /// Return the model identifier (e.g. "gpt-4o").
    fn model_id(&self) -> &str;

    /// Return the provider identifier (e.g. "openai").
    fn provider_id(&self) -> &str;

    /// Return the set of capabilities this model supports.
    fn capabilities(&self) -> &CapabilitySet;

    /// Generate a complete response.
    async fn generate(&self, prompt: Prompt, options: GenerateOptions) -> AiResult<GenerateResult>;

    /// Stream a response as a series of events.
    async fn stream(&self, prompt: Prompt, options: GenerateOptions) -> AiResult<AiStream>;
}

/// Generate a structured object from a language model response.
///
/// Adds the JSON schema to the options, calls `generate`, and parses the text result.
pub async fn generate_object<T: serde::de::DeserializeOwned + schemars::JsonSchema>(
    model: &dyn LanguageModel,
    prompt: Prompt,
    options: GenerateOptions,
) -> AiResult<ObjectResult<T>> {
    let mut opts = options;
    opts.output_schema = Some(OutputSchema::from_type::<T>());
    let result = model.generate(prompt, opts).await?;
    let text = result
        .text
        .as_deref()
        .ok_or_else(|| AiError::Serialization("No text in response to parse as object".into()))?;
    let object: T = serde_json::from_str(text)
        .map_err(|e| AiError::Serialization(format!("Failed to parse response as object: {e}")))?;
    Ok(ObjectResult {
        object,
        text: text.to_owned(),
        usage: result.usage,
        metadata: result.metadata,
    })
}

/// Auto-execute tool calls in a loop (Vercel's `maxSteps` agent loop).
///
/// Equivalent to calling `generateText` with `maxSteps > 1` in Vercel AI SDK.
/// Each iteration executes tool calls via `ToolSet`, feeds results back as
/// new messages, and re-invokes the model until no more tool calls or
/// `max_steps` is reached.
///
/// Returns a `GenerateResult` containing all steps and aggregated usage.
pub async fn agent_loop(
    model: &dyn LanguageModel,
    prompt: Prompt,
    options: GenerateOptions,
    tools: &crate::tool::ToolSet,
) -> AiResult<GenerateResult> {
    use crate::message::{Message, Role};
    use crate::tool::ToolExecutionOptions;
    use std::collections::HashMap;

    let max_steps = options.max_steps.unwrap_or(1).max(1);
    let mut current_prompt = prompt;
    let mut all_steps: Vec<StepResult> = Vec::new();
    let mut aggregated_usage = crate::usage::Usage::default();
    let tool_context = options.tools_context.clone().unwrap_or_default();

    for step_num in 0..max_steps {
        // Generate with tools
        let result = model
            .generate(current_prompt.clone(), options.clone())
            .await?;

        aggregated_usage = aggregated_usage + result.usage.clone();

        let has_tool_calls = !result.tool_calls.is_empty();

        if has_tool_calls {
            // Execute tool calls
            let mut tool_results = Vec::new();
            for call in &result.tool_calls {
                // Fire on_tool_call — allows aborting
                if let Some(ref cbs) = options.callbacks {
                    if let Some(ref cb) = cbs.on_tool_call {
                        if let Err(msg) = cb(call) {
                            tool_results.push(ToolCallResult {
                                call_id: call.id.clone(),
                                content: format!("Aborted by on_tool_call: {msg}"),
                                is_error: true,
                            });
                            continue;
                        }
                    }
                }
                let exec_opts = ToolExecutionOptions {
                    call_id: call.id.clone(),
                    context: tool_context.data.get(&call.name).cloned(),
                };
                let tr = tools.execute_with_options(call, &exec_opts).await?;
                tool_results.push(tr);
            }

            // Record step
            all_steps.push(StepResult {
                step_number: step_num,
                text: result.text.clone(),
                tool_calls: result.tool_calls.clone(),
                tool_results: tool_results.clone(),
                finish_reason: result.finish_reason.clone(),
                usage: result.usage,
                reasoning: None,
            });

            // Fire on_step_finish
            if let Some(ref cbs) = options.callbacks {
                if let Some(ref cb) = cbs.on_step_finish {
                    cb(all_steps.last().unwrap());
                }
            }

            // Build tool result messages and append to prompt
            let tool_messages: Vec<Message> = tool_results
                .into_iter()
                .map(|tr| Message {
                    role: Role::Tool,
                    content: vec![crate::content::ContentPart::Text { text: tr.content }],
                    name: None,
                    metadata: HashMap::new(),
                })
                .collect();

            // Append tool messages to current prompt
            let mut msgs = current_prompt.clone().into_messages();
            msgs.extend(tool_messages);
            current_prompt = Prompt::Messages(msgs);

            // If last step, still record final state
            if step_num == max_steps - 1 {
                all_steps.push(StepResult {
                    step_number: step_num + 1,
                    text: None,
                    tool_calls: Vec::new(),
                    tool_results: Vec::new(),
                    finish_reason: result.finish_reason.clone(),
                    usage: crate::usage::Usage::default(),
                    reasoning: None,
                });
                // Fire on_step_finish
                if let Some(ref cbs) = options.callbacks {
                    if let Some(ref cb) = cbs.on_step_finish {
                        cb(all_steps.last().unwrap());
                    }
                }
            }
        } else {
            // No tool calls — done
            all_steps.push(StepResult {
                step_number: step_num,
                text: result.text.clone(),
                tool_calls: Vec::new(),
                tool_results: Vec::new(),
                finish_reason: result.finish_reason.clone(),
                usage: result.usage.clone(),
                reasoning: None,
            });

            // Fire on_step_finish
            if let Some(ref cbs) = options.callbacks {
                if let Some(ref cb) = cbs.on_step_finish {
                    cb(all_steps.last().unwrap());
                }
            }

            let res = GenerateResult {
                text: result.text,
                tool_calls: Vec::new(),
                finish_reason: result.finish_reason,
                usage: aggregated_usage,
                metadata: result.metadata,
                steps: all_steps,
                reasoning: None,
            };

            // Fire on_finish
            if let Some(ref cbs) = options.callbacks {
                if let Some(ref cb) = cbs.on_finish {
                    cb(&res);
                }
            }
            return Ok(res);
        }
    }

    // Exhausted max_steps — extract final text from last step
    let last = all_steps.last().cloned().unwrap_or(StepResult {
        step_number: 0,
        text: None,
        tool_calls: Vec::new(),
        tool_results: Vec::new(),
        finish_reason: crate::types::FinishReason::Stop,
        usage: crate::usage::Usage::default(),
        reasoning: None,
    });

    let res = GenerateResult {
        text: last.text,
        tool_calls: Vec::new(),
        finish_reason: crate::types::FinishReason::Stop,
        usage: aggregated_usage,
        metadata: crate::types::ResponseMetadata::default(),
        steps: all_steps,
        reasoning: None,
    };

    // Fire on_finish
    if let Some(ref cbs) = options.callbacks {
        if let Some(ref cb) = cbs.on_finish {
            cb(&res);
        }
    }
    Ok(res)
}

/// A model that produces vector embeddings from text.
#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    /// Return the model identifier.
    fn model_id(&self) -> &str;

    /// Return the provider identifier.
    fn provider_id(&self) -> &str;

    /// Return the dimensionality of the embeddings, if known.
    fn dimensions(&self) -> Option<usize>;

    /// Embed a batch of texts into vectors.
    async fn embed(&self, texts: Vec<String>) -> AiResult<EmbeddingResult>;
}

/// Middleware sits between the caller and the model, intercepting generate calls.
#[async_trait]
pub trait Middleware: Send + Sync {
    /// Process a generate request.
    ///
    /// Implementations should call `next.run(prompt, options).await` to
    /// continue the chain, and may inspect / modify the prompt, options,
    /// or result.
    async fn process(
        &self,
        prompt: Prompt,
        options: GenerateOptions,
        next: MiddlewareNext<'_>,
    ) -> AiResult<GenerateResult>;
}

/// A handle to the next element in a middleware chain.
///
/// Calling `run` will invoke either the next middleware or the final model.
pub struct MiddlewareNext<'a> {
    /// Remaining middlewares in the chain.
    pub middlewares: &'a [Box<dyn Middleware>],
    /// The underlying language model.
    pub model: &'a dyn LanguageModel,
}

impl<'a> MiddlewareNext<'a> {
    /// Execute the next middleware (or the model if no middleware remains).
    pub async fn run(self, prompt: Prompt, options: GenerateOptions) -> AiResult<GenerateResult> {
        if let Some((first, rest)) = self.middlewares.split_first() {
            let next = MiddlewareNext {
                middlewares: rest,
                model: self.model,
            };
            first.process(prompt, options, next).await
        } else {
            self.model.generate(prompt, options).await
        }
    }
}

/// A model that converts speech audio to text (e.g. OpenAI Whisper).
#[async_trait]
pub trait SpeechToTextModel: Send + Sync {
    /// Return the model identifier.
    fn model_id(&self) -> &str;

    /// Return the provider identifier.
    fn provider_id(&self) -> &str;

    /// Transcribe audio bytes into text.
    async fn transcribe(
        &self,
        audio: Vec<u8>,
        mime_type: &str,
        language: Option<&str>,
    ) -> AiResult<TranscriptionResult>;
}

/// A model that converts text to speech audio (e.g. OpenAI TTS).
#[async_trait]
pub trait TextToSpeechModel: Send + Sync {
    /// Return the model identifier.
    fn model_id(&self) -> &str;

    /// Return the provider identifier.
    fn provider_id(&self) -> &str;

    /// Synthesize speech from text.  Returns audio bytes.
    async fn synthesize(
        &self,
        text: &str,
        voice: &str,
        options: TtsOptions,
    ) -> AiResult<AudioResult>;
}

// ─── Image Generation ───────────────────────────────────────────────────────────

/// Options for image generation, matching Vercel AI SDK `generateImage`.
#[derive(Debug, Clone, Default)]
pub struct ImageGenerationOptions {
    /// Number of images to generate (default: 1).
    pub n: Option<u32>,
    /// Size as `{width}x{height}` (e.g. "1024x1024").
    pub size: Option<String>,
    /// Aspect ratio as `{width}:{height}` (e.g. "16:9").
    pub aspect_ratio: Option<String>,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
    /// Provider-specific options forwarded as body parameters.
    pub provider_options: Option<std::collections::HashMap<String, serde_json::Value>>,
}

/// A model that generates images from text prompts.
#[async_trait]
pub trait ImageModel: Send + Sync {
    /// Return the model identifier.
    fn model_id(&self) -> &str;

    /// Return the provider identifier.
    fn provider_id(&self) -> &str;

    /// Generate one or more images from a text prompt.
    async fn generate_image(
        &self,
        prompt: &str,
        options: ImageGenerationOptions,
    ) -> AiResult<ImageResult>;
}

// ─── Video Generation ───────────────────────────────────────────────────────────

/// Options for video generation, matching Vercel AI SDK `experimental_generateVideo`.
#[derive(Debug, Clone, Default)]
pub struct VideoGenerationOptions {
    /// Number of videos to generate (default: 1).
    pub n: Option<u32>,
    /// Aspect ratio as `{width}:{height}` (e.g. "16:9").
    pub aspect_ratio: Option<String>,
    /// Resolution as `{width}x{height}` (e.g. "1920x1080").
    pub resolution: Option<String>,
    /// Duration in seconds.
    pub duration: Option<f32>,
    /// Frames per second.
    pub fps: Option<u32>,
    /// Random seed.
    pub seed: Option<u64>,
    /// Whether to generate audio track.
    pub generate_audio: Option<bool>,
}

/// A model that generates videos from text prompts.
#[async_trait]
pub trait VideoModel: Send + Sync {
    /// Return the model identifier.
    fn model_id(&self) -> &str;

    /// Return the provider identifier.
    fn provider_id(&self) -> &str;

    /// Generate one or more videos from a text prompt.
    async fn generate_video(
        &self,
        prompt: &str,
        options: VideoGenerationOptions,
    ) -> AiResult<VideoResult>;
}

// ─── Realtime Session ───────────────────────────────────────────────────────────

/// Events emitted by a realtime session.
#[derive(Debug, Clone)]
pub enum RealtimeEvent {
    /// Partial text delta from the model.
    TextDelta { delta: String },
    /// Complete text response.
    TextDone { text: String },
    /// Audio chunk received (PCM or opus bytes).
    AudioDelta { delta: Vec<u8> },
    /// Tool call from the model.
    ToolCall {
        id: String,
        name: String,
        arguments: serde_json::Value,
    },
    /// Tool execution result.
    ToolResult { call_id: String, content: String },
    /// The session encountered an error.
    Error { message: String },
    /// Session closed.
    Done,
}

/// A bidirectional realtime session for voice/text interaction.
///
/// Analogous to Vercel's `AbstractRealtimeSession` + `RealtimeSession`.
/// Specific provider implementations (OpenAI Realtime, Gemini Live) implement this.
#[async_trait]
pub trait RealtimeSession: Send + Sync {
    /// Return the model identifier.
    fn model_id(&self) -> &str;

    /// Return the provider identifier.
    fn provider_id(&self) -> &str;

    /// Send a text message to the model.
    async fn send_text(&mut self, text: &str) -> AiResult<()>;

    /// Send audio bytes to the model.
    async fn send_audio(&mut self, audio: Vec<u8>, mime_type: &str) -> AiResult<()>;

    /// Receive the next event from the session. Blocks until an event is available.
    async fn recv(&mut self) -> Option<RealtimeEvent>;

    /// Close the session.
    async fn close(self: Box<Self>) -> AiResult<()>;
}

// ─── Reranking ────────────────────────────────────────────────────────────────────

/// Options for reranking documents by relevance to a query.
/// Matching Vercel AI SDK's `RerankOptions`.
#[derive(Debug, Clone, Default)]
pub struct RerankOptions {
    /// Maximum number of results to return.
    pub top_k: Option<usize>,
    /// Include the document text in results.
    pub return_documents: Option<bool>,
    /// Model-specific query prefix/instruction.
    pub query_instruction: Option<String>,
}

/// A model that reranks documents by relevance to a query.
/// Equivalent to Vercel AI SDK's `RerankingModel`.
#[async_trait]
pub trait RerankingModel: Send + Sync {
    /// Return the model identifier.
    fn model_id(&self) -> &str;

    /// Return the provider identifier.
    fn provider_id(&self) -> &str;

    /// Rerank documents by relevance to the query.
    async fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
        options: RerankOptions,
    ) -> AiResult<RerankResult>;
}

/// Rerank documents by relevance to a query.
/// Equivalent to Vercel AI SDK's `rerank()`.
pub async fn rerank(
    model: &dyn RerankingModel,
    query: &str,
    documents: Vec<String>,
    options: RerankOptions,
) -> AiResult<RerankResult> {
    model.rerank(query, documents, options).await
}

/// Embed multiple texts in a batch.
pub async fn embed_many(
    model: &dyn EmbeddingModel,
    texts: Vec<String>,
) -> AiResult<EmbeddingResult> {
    model.embed(texts).await
}

/// Wrap a generate-only model to add streaming support.
/// Text is generated first, then streamed as synthetic chunks.
pub fn wrap_with_streaming(inner: Box<dyn LanguageModel>) -> StreamEnabledModel {
    StreamEnabledModel { inner }
}

/// A model wrapper that adds streaming to generate-only models.
pub struct StreamEnabledModel {
    inner: Box<dyn LanguageModel>,
}

#[async_trait]
impl LanguageModel for StreamEnabledModel {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
    fn provider_id(&self) -> &str {
        self.inner.provider_id()
    }
    fn capabilities(&self) -> &CapabilitySet {
        self.inner.capabilities()
    }

    async fn generate(&self, prompt: Prompt, options: GenerateOptions) -> AiResult<GenerateResult> {
        self.inner.generate(prompt, options).await
    }

    async fn stream(&self, prompt: Prompt, options: GenerateOptions) -> AiResult<AiStream> {
        use crate::stream::{StreamEvent, SyntheticStreamer};
        use futures::StreamExt;

        let result = self.inner.generate(prompt, options).await?;
        let text = result.text.unwrap_or_default();
        let mut events = Vec::new();
        if let Some(reasoning) = result.reasoning {
            events.push(Ok(StreamEvent::ThinkingDelta { delta: reasoning }));
        }
        for call in result.tool_calls {
            events.push(Ok(StreamEvent::ToolCallStart {
                call_id: call.id.clone(),
                tool_name: call.name,
            }));
            events.push(Ok(StreamEvent::ToolCallDelta {
                call_id: call.id.clone(),
                delta: call.arguments.to_string(),
            }));
            events.push(Ok(StreamEvent::ToolCallEnd {
                call_id: call.id,
                arguments: call.arguments,
            }));
        }
        events.push(Ok(StreamEvent::MessageEnd {
            finish_reason: result.finish_reason,
            usage: Some(result.usage),
        }));
        let text_stream = SyntheticStreamer::from_text(&text)
            .stream()
            .filter(|event| {
                futures::future::ready(!matches!(event, Ok(StreamEvent::MessageEnd { .. })))
            });
        Ok(Box::pin(text_stream.chain(futures::stream::iter(events))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::CapabilitySet;
    use crate::tool::ToolSet;
    use crate::types::{FinishReason, ResponseMetadata};
    use crate::Usage;
    use async_trait::async_trait;

    struct TestModel {
        model_id: String,
        provider_id: String,
        response_text: String,
    }

    #[async_trait]
    impl LanguageModel for TestModel {
        fn model_id(&self) -> &str {
            &self.model_id
        }
        fn provider_id(&self) -> &str {
            &self.provider_id
        }
        fn capabilities(&self) -> &CapabilitySet {
            unimplemented!()
        }
        async fn generate(
            &self,
            _prompt: Prompt,
            _options: GenerateOptions,
        ) -> AiResult<GenerateResult> {
            Ok(GenerateResult {
                text: Some(self.response_text.clone()),
                tool_calls: vec![],
                finish_reason: FinishReason::Stop,
                usage: Usage::default(),
                metadata: ResponseMetadata::default(),
                steps: vec![],
                reasoning: None,
            })
        }
        async fn stream(&self, _prompt: Prompt, _options: GenerateOptions) -> AiResult<AiStream> {
            unimplemented!()
        }
    }

    #[test]
    fn test_generate_options_default() {
        let opts = GenerateOptions::default();
        assert!(opts.temperature.is_none());
        assert!(opts.stop_sequences.is_empty());
    }

    #[test]
    fn test_generate_options_builder() {
        let opts = GenerateOptions::default()
            .with_temperature(0.5)
            .with_max_tokens(100)
            .with_seed(42);
        assert_eq!(opts.temperature, Some(0.5));
        assert_eq!(opts.max_tokens, Some(100));
        assert_eq!(opts.seed, Some(42));
    }

    #[test]
    fn test_thinking_config_variants() {
        match ThinkingConfig::Adaptive {
            ThinkingConfig::Adaptive => {}
            _ => panic!("expected Adaptive"),
        }
        match (ThinkingConfig::Budget { tokens: 1000 }) {
            ThinkingConfig::Budget { tokens } => assert_eq!(tokens, 1000),
            _ => panic!("expected Budget"),
        }
        match ThinkingConfig::Enabled {
            ThinkingConfig::Enabled => {}
            _ => panic!("expected Enabled"),
        }
    }

    #[test]
    fn test_rerank_options_default() {
        let opts = RerankOptions::default();
        assert!(opts.top_k.is_none());
    }

    #[tokio::test]
    async fn test_agent_loop_no_tools() {
        let model = TestModel {
            model_id: "test".into(),
            provider_id: "test".into(),
            response_text: "Hello".into(),
        };
        let tools = ToolSet::new();
        let result = agent_loop(
            &model,
            Prompt::Text("Hi".into()),
            GenerateOptions::default(),
            &tools,
        )
        .await
        .unwrap();
        assert_eq!(result.text.as_deref(), Some("Hello"));
    }
}

use std::hash::{DefaultHasher, Hash, Hasher};

use crate::{AiResult, GenerateOptions, GenerateResult, Middleware, MiddlewareNext, Prompt};
use async_trait::async_trait;

/// In-memory caching middleware for non-streaming generate calls.
///
/// Caches responses keyed on a hash of the prompt + options.
/// Uses `moka` internally for concurrent lock-free access with TTL eviction.
pub struct CacheMiddleware {
    cache: moka::sync::Cache<u64, GenerateResult>,
}

impl CacheMiddleware {
    /// Create a new `CacheMiddleware` with the specified time-to-live.
    pub fn new(ttl: std::time::Duration) -> Self {
        Self {
            cache: moka::sync::Cache::builder().time_to_live(ttl).build(),
        }
    }

    /// Compute a deterministic cache key from the prompt and generation options.
    ///
    /// Returns `None` if the prompt cannot be serialized, in which case the
    /// caller should bypass the cache entirely to avoid hash collisions.
    ///
    /// Both the prompt content and all generation-affecting options are included
    /// so that requests with the same prompt but different parameters (temperature,
    /// tools, output schema, etc.) are treated as distinct cache entries.
    /// The request metadata (which contains a per-request UUID) is excluded.
    fn cache_key(prompt: &Prompt, options: &GenerateOptions) -> Option<u64> {
        let mut hasher = DefaultHasher::new();

        let prompt_json = serde_json::to_string(prompt)
            .map_err(|e| tracing::error!(error = %e, "Failed to serialize prompt for cache key; bypassing cache"))
            .ok()?;
        prompt_json.hash(&mut hasher);

        // Numeric options — hash the bit pattern to keep f64 deterministic.
        options.temperature.map(f64::to_bits).hash(&mut hasher);
        options.max_tokens.hash(&mut hasher);
        options.top_p.map(f64::to_bits).hash(&mut hasher);
        options.top_k.hash(&mut hasher);
        options.stop_sequences.hash(&mut hasher);
        options
            .frequency_penalty
            .map(f64::to_bits)
            .hash(&mut hasher);
        options.presence_penalty.map(f64::to_bits).hash(&mut hasher);
        options.seed.hash(&mut hasher);
        options.max_steps.hash(&mut hasher);

        // Complex types — serialize to JSON for a stable, content-based hash.
        if let Ok(json) = serde_json::to_string(&options.tools) {
            json.hash(&mut hasher);
        }
        if let Ok(json) = serde_json::to_string(&options.tool_choice) {
            json.hash(&mut hasher);
        }
        if let Ok(json) = serde_json::to_string(&options.output_schema) {
            json.hash(&mut hasher);
        }

        // Enum options without Serialize — use Debug, which is stable for owned enums.
        format!("{:?}", options.thinking).hash(&mut hasher);
        format!("{:?}", options.reasoning_effort).hash(&mut hasher);

        Some(hasher.finish())
    }
}

#[async_trait]
impl Middleware for CacheMiddleware {
    async fn process(
        &self,
        prompt: Prompt,
        options: GenerateOptions,
        next: MiddlewareNext<'_>,
    ) -> AiResult<GenerateResult> {
        let Some(key) = Self::cache_key(&prompt, &options) else {
            // Prompt could not be serialized; bypass cache to avoid collisions.
            return next.run(prompt, options).await;
        };

        // Check cache (moka is thread-safe, no lock needed)
        if let Some(cached) = self.cache.get(&key) {
            tracing::debug!(cache_key = key, "cache hit");
            return Ok(cached);
        }

        tracing::debug!(cache_key = key, "cache miss");

        // Execute the downstream chain.
        let result = next.run(prompt, options).await?;

        // Store in cache (moka handles TTL eviction automatically)
        self.cache.insert(key, result.clone());

        Ok(result)
    }
}

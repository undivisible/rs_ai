//! Provider catalog — static metadata for API-key-based AI providers.
//!
//! Mirrors the models.dev catalog from OpenCode, providing stable
//! Rust-side definitions for provider lookup, key resolution, and
//! surface rendering.

/// Protocol family a provider speaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderApi {
    /// OpenAI Chat Completions / Responses API.
    OpenAiCompatible,
    /// Anthropic Messages API.
    Anthropic,
    /// Custom protocol or multi-field auth (AWS, Azure, Vertex, etc.)
    Custom,
}

/// Static metadata for a single provider.
#[derive(Clone, Copy, Debug)]
pub struct ProviderSpec {
    pub id: &'static str,
    pub name: &'static str,
    pub env_vars: &'static [&'static str],
    pub base_url: &'static str,
    pub api: ProviderApi,
    pub default_model: &'static str,
    pub models: &'static [&'static str],
    pub aliases: &'static [&'static str],
}

/// All API-key-based providers with static metadata.
///
/// Providers needing multi-field config (Azure, Vertex, Bedrock,
/// Cloudflare, SAP, Databricks, GitLab) are included with
/// `api: Custom` and empty `base_url` — consumers handle setup.
pub const API_KEY_PROVIDERS: &[ProviderSpec] = &[
    ProviderSpec {
        id: "openai",
        name: "OpenAI",
        env_vars: &["OPENAI_API_KEY"],
        base_url: "https://api.openai.com/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "gpt-5.5",
        models: &[
            "gpt-5.5",
            "gpt-5.5-pro",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            "gpt-5.4-pro",
            "gpt-5.4",
        ],
        aliases: &["gpt"],
    },
    ProviderSpec {
        id: "anthropic",
        name: "Anthropic",
        env_vars: &["ANTHROPIC_API_KEY"],
        base_url: "https://api.anthropic.com/v1",
        api: ProviderApi::Anthropic,
        default_model: "claude-opus-4-7",
        models: &[
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "claude-opus-4-6",
            "claude-opus-4-5",
            "claude-opus-4-5-20251101",
            "claude-haiku-4-5",
        ],
        aliases: &["claude"],
    },
    ProviderSpec {
        id: "google",
        name: "Google Gemini",
        env_vars: &["GOOGLE_GENERATIVE_AI_API_KEY", "GEMINI_API_KEY"],
        base_url: "https://generativelanguage.googleapis.com/v1beta",
        api: ProviderApi::OpenAiCompatible,
        default_model: "gemini-3.1-flash-lite",
        models: &[
            "gemini-3.1-flash-lite",
            "gemma-4-31b-it",
            "gemma-4-26b-a4b-it",
            "gemini-3.1-flash-lite-preview",
            "gemini-3.1-pro-preview-customtools",
            "gemini-3.1-pro-preview",
        ],
        aliases: &["gemini"],
    },
    ProviderSpec {
        id: "xai",
        name: "xAI Grok",
        env_vars: &["XAI_API_KEY"],
        base_url: "https://api.x.ai/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "grok-4.3",
        models: &[
            "grok-4.3",
            "grok-4.20-0309-reasoning",
            "grok-4.20-0309-non-reasoning",
            "grok-4.20-multi-agent-0309",
            "grok-4-1-fast-non-reasoning",
            "grok-4-1-fast",
        ],
        aliases: &["grok"],
    },
    ProviderSpec {
        id: "deepseek",
        name: "DeepSeek",
        env_vars: &["DEEPSEEK_API_KEY"],
        base_url: "https://api.deepseek.com",
        api: ProviderApi::OpenAiCompatible,
        default_model: "deepseek-v4-flash",
        models: &[
            "deepseek-v4-flash",
            "deepseek-v4-pro",
            "deepseek-chat",
            "deepseek-reasoner",
        ],
        aliases: &["ds"],
    },
    ProviderSpec {
        id: "groq",
        name: "Groq",
        env_vars: &["GROQ_API_KEY"],
        base_url: "https://api.groq.com/openai/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "canopylabs/orpheus-v1-english",
        models: &[
            "canopylabs/orpheus-v1-english",
            "canopylabs/orpheus-arabic-saudi",
            "moonshotai/kimi-k2-instruct-0905",
            "groq/compound",
            "groq/compound-mini",
            "openai/gpt-oss-20b",
        ],
        aliases: &[],
    },
    ProviderSpec {
        id: "togetherai",
        name: "Together AI",
        env_vars: &["TOGETHER_API_KEY"],
        base_url: "https://api.together.xyz/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "Qwen/Qwen3.6-Plus",
        models: &[
            "Qwen/Qwen3.6-Plus",
            "deepseek-ai/DeepSeek-V4-Pro",
            "moonshotai/Kimi-K2.6",
            "zai-org/GLM-5.1",
            "google/gemma-4-31B-it",
            "MiniMaxAI/MiniMax-M2.7",
        ],
        aliases: &["together"],
    },
    ProviderSpec {
        id: "mistral",
        name: "Mistral AI",
        env_vars: &["MISTRAL_API_KEY"],
        base_url: "https://api.mistral.ai/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "mistral-medium-2604",
        models: &[
            "mistral-medium-2604",
            "mistral-medium-latest",
            "mistral-small-latest",
            "mistral-small-2603",
            "labs-devstral-small-2512",
            "devstral-2512",
        ],
        aliases: &[],
    },
    ProviderSpec {
        id: "fireworks-ai",
        name: "Fireworks AI",
        env_vars: &["FIREWORKS_API_KEY"],
        base_url: "https://api.fireworks.ai/inference/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "accounts/fireworks/models/deepseek-v4-pro",
        models: &[
            "accounts/fireworks/models/deepseek-v4-pro",
            "accounts/fireworks/models/kimi-k2p6",
            "accounts/fireworks/models/minimax-m2p7",
            "accounts/fireworks/models/qwen3p6-plus",
            "accounts/fireworks/models/glm-5p1",
            "accounts/fireworks/models/minimax-m2p5",
        ],
        aliases: &["fireworks"],
    },
    ProviderSpec {
        id: "nvidia",
        name: "NVIDIA NIM",
        env_vars: &["NVIDIA_API_KEY"],
        base_url: "https://integrate.api.nvidia.com/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "deepseek-ai/deepseek-v4-flash",
        models: &[
            "deepseek-ai/deepseek-v4-flash",
            "deepseek-ai/deepseek-v4-pro",
            "moonshotai/kimi-k2.6",
            "nvidia/nemotron-3-content-safety",
            "google/gemma-4-31b-it",
            "z-ai/glm-5.1",
        ],
        aliases: &["nim"],
    },
    ProviderSpec {
        id: "huggingface",
        name: "Hugging Face",
        env_vars: &["HF_TOKEN"],
        base_url: "https://router.huggingface.co/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "deepseek-ai/DeepSeek-V4-Pro",
        models: &[
            "deepseek-ai/DeepSeek-V4-Pro",
            "moonshotai/Kimi-K2.6",
            "zai-org/GLM-5.1",
            "MiniMaxAI/MiniMax-M2.7",
            "MiniMaxAI/MiniMax-M2.5",
            "zai-org/GLM-5",
        ],
        aliases: &["hf"],
    },
    ProviderSpec {
        id: "openrouter",
        name: "OpenRouter",
        env_vars: &["OPENROUTER_API_KEY"],
        base_url: "https://openrouter.ai/api/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "x-ai/grok-4.3",
        models: &[
            "x-ai/grok-4.3",
            "openrouter/owl-alpha",
            "poolside/laguna-m.1:free",
            "poolside/laguna-xs.2:free",
            "deepseek/deepseek-v4-flash",
            "deepseek/deepseek-v4-pro",
        ],
        aliases: &["router"],
    },
    ProviderSpec {
        id: "perplexity",
        name: "Perplexity",
        env_vars: &["PERPLEXITY_API_KEY"],
        base_url: "https://api.perplexity.ai",
        api: ProviderApi::OpenAiCompatible,
        default_model: "sonar-deep-research",
        models: &[
            "sonar-deep-research",
            "sonar-pro",
            "sonar",
            "sonar-reasoning-pro",
        ],
        aliases: &["pplx"],
    },
    ProviderSpec {
        id: "cohere",
        name: "Cohere",
        env_vars: &["COHERE_API_KEY", "CO_API_KEY"],
        base_url: "https://api.cohere.ai/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "command-a-translate-08-2025",
        models: &[
            "command-a-translate-08-2025",
            "command-a-reasoning-08-2025",
            "command-a-03-2025",
            "command-r7b-arabic-02-2025",
            "c4ai-aya-expanse-8b",
            "c4ai-aya-expanse-32b",
        ],
        aliases: &[],
    },
    ProviderSpec {
        id: "deepinfra",
        name: "Deep Infra",
        env_vars: &["DEEPINFRA_API_KEY"],
        base_url: "https://api.deepinfra.com/v1/openai",
        api: ProviderApi::OpenAiCompatible,
        default_model: "deepseek-ai/DeepSeek-V4-Flash",
        models: &[
            "deepseek-ai/DeepSeek-V4-Flash",
            "deepseek-ai/DeepSeek-V4-Pro",
            "xiaomi/mimo-v2.5-pro",
            "xiaomi/mimo-v2.5",
            "moonshotai/Kimi-K2.6",
            "zai-org/GLM-5.1",
        ],
        aliases: &["di"],
    },
    ProviderSpec {
        id: "cerebras",
        name: "Cerebras",
        env_vars: &["CEREBRAS_API_KEY"],
        base_url: "https://api.cerebras.ai/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "zai-glm-4.7",
        models: &[
            "zai-glm-4.7",
            "gpt-oss-120b",
            "qwen-3-235b-a22b-instruct-2507",
            "llama3.1-8b",
        ],
        aliases: &[],
    },
    ProviderSpec {
        id: "opencode-go",
        name: "OpenCode Zen Go",
        env_vars: &["OPENCODE_API_KEY"],
        base_url: "https://opencode.ai/zen/go/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "muse-spark-1.3-contributor",
        models: &[
            "muse-spark-1.3-contributor",
            "muse-spark-1.2-contributor",
            "deepseek-v4-flash",
            "deepseek-v4-pro",
            "mimo-v2.5",
            "glm-5.1",
        ],
        aliases: &["opencode", "go", "zen"],
    },
    ProviderSpec {
        id: "opencode",
        name: "OpenCode Zen",
        env_vars: &["OPENCODE_API_KEY"],
        base_url: "https://opencode.ai/zen/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "muse-spark-1.3-contributor-free",
        models: &[
            "muse-spark-1.3-contributor-free",
            "muse-spark-1.2-contributor-free",
            "deepseek-v4-flash-free",
            "mimo-v2.5-free",
            "ling-3.0-flash-fin-free",
            "nemotron-3.5-lightning-free",
        ],
        aliases: &["opencode-free"],
    },
    ProviderSpec {
        id: "venice",
        name: "Venice AI",
        env_vars: &["VENICE_API_KEY"],
        base_url: "https://api.venice.ai/api/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "openai-gpt-55-pro",
        models: &[
            "openai-gpt-55-pro",
            "deepseek-v4-flash",
            "qwen3-6-27b",
            "deepseek-v4-pro",
            "openai-gpt-55",
            "kimi-k2-6",
        ],
        aliases: &[],
    },
    ProviderSpec {
        id: "clinepass",
        name: "Cline-pass",
        env_vars: &["CLINE_API_KEY"],
        base_url: "https://api.cline.bot/api/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "cline-pass/deepseek-v4-flash",
        models: &[
            "cline-pass/deepseek-v4-flash",
            "cline-pass/qwen3.7-max",
            "cline-pass/glm-5.2",
        ],
        aliases: &["cline-pass", "cline"],
    },
    // ── Alibaba / Qwen ──
    ProviderSpec {
        id: "alibaba",
        name: "Alibaba (Qwen)",
        env_vars: &["DASHSCOPE_API_KEY"],
        base_url: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "qwen3.6-27b",
        models: &[
            "qwen3.6-27b",
            "qwen3.6-max-preview",
            "qwen3.6-35b-a3b",
            "qwen3.6-plus",
            "qwen3.5-122b-a10b",
            "qwen3.5-27b",
        ],
        aliases: &["qwen", "dashscope"],
    },
    ProviderSpec {
        id: "alibaba-cn",
        name: "Alibaba China",
        env_vars: &["DASHSCOPE_API_KEY"],
        base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "deepseek-v4-flash",
        models: &[
            "deepseek-v4-flash",
            "deepseek-v4-pro",
            "kimi-k2.6",
            "qwen3.6-max-preview",
            "glm-5.1",
            "qwen3.6-plus",
        ],
        aliases: &["qwen-cn"],
    },
    ProviderSpec {
        id: "alibaba-coding-plan",
        name: "Alibaba Coding Plan",
        env_vars: &["ALIBABA_CODING_PLAN_API_KEY"],
        base_url: "https://coding-intl.dashscope.aliyuncs.com/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "qwen3.6-plus",
        models: &["qwen3-coder-plus", "qwen3-235b-a22b"],
        aliases: &["ali-code"],
    },
    ProviderSpec {
        id: "alibaba-coding-plan-cn",
        name: "Alibaba Coding Plan China",
        env_vars: &["ALIBABA_CODING_PLAN_API_KEY"],
        base_url: "https://coding.dashscope.aliyuncs.com/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "qwen3.6-plus",
        models: &["qwen3-coder-plus", "qwen3-235b-a22b"],
        aliases: &["ali-code-cn"],
    },
    // ── Moonshot / Kimi ──
    ProviderSpec {
        id: "moonshotai",
        name: "Moonshot AI (Kimi)",
        env_vars: &["MOONSHOT_API_KEY"],
        base_url: "https://api.moonshot.ai/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "kimi-k2.6",
        models: &[
            "kimi-k2.6",
            "kimi-k2.5",
            "kimi-k2-thinking-turbo",
            "kimi-k2-thinking",
            "kimi-k2-0905-preview",
            "kimi-k2-turbo-preview",
        ],
        aliases: &["kimi", "moonshot"],
    },
    ProviderSpec {
        id: "moonshotai-cn",
        name: "Moonshot AI China",
        env_vars: &["MOONSHOT_API_KEY"],
        base_url: "https://api.moonshot.cn/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "kimi-k2.6",
        models: &[
            "kimi-k2.6",
            "kimi-k2.5",
            "kimi-k2-thinking",
            "kimi-k2-thinking-turbo",
            "kimi-k2-turbo-preview",
            "kimi-k2-0905-preview",
        ],
        aliases: &["kimi-cn"],
    },
    ProviderSpec {
        id: "kimi-for-coding",
        name: "Kimi For Coding",
        env_vars: &["KIMI_API_KEY"],
        base_url: "https://api.kimi.com/coding/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "k2p6",
        models: &["k2p6", "kimi-k2-turbo"],
        aliases: &["kimi-code"],
    },
    // ── MiniMax ──
    ProviderSpec {
        id: "minimax",
        name: "MiniMax",
        env_vars: &["MINIMAX_API_KEY"],
        base_url: "https://api.minimax.io/anthropic/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "MiniMax-M2.7",
        models: &[
            "MiniMax-M2.7",
            "MiniMax-M2.7-highspeed",
            "MiniMax-M2.5-highspeed",
            "MiniMax-M2.5",
            "MiniMax-M2.1",
            "MiniMax-M2",
        ],
        aliases: &["mmax"],
    },
    ProviderSpec {
        id: "minimax-cn",
        name: "MiniMax China",
        env_vars: &["MINIMAX_API_KEY"],
        base_url: "https://api.minimaxi.com/anthropic/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "MiniMax-M2.7",
        models: &[
            "MiniMax-M2.7",
            "MiniMax-M2.7-highspeed",
            "MiniMax-M2.5-highspeed",
            "MiniMax-M2.5",
            "MiniMax-M2.1",
            "MiniMax-M2",
        ],
        aliases: &["mmax-cn"],
    },
    ProviderSpec {
        id: "minimax-coding-plan",
        name: "MiniMax Coding Plan",
        env_vars: &["MINIMAX_API_KEY"],
        base_url: "https://api.minimax.io/anthropic/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "MiniMax-M2.7",
        models: &["MiniMax-M2.7", "MiniMax-M2.5"],
        aliases: &["mmax-code"],
    },
    ProviderSpec {
        id: "minimax-cn-coding-plan",
        name: "MiniMax Coding Plan China",
        env_vars: &["MINIMAX_API_KEY"],
        base_url: "https://api.minimaxi.com/anthropic/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "MiniMax-M2.7",
        models: &["MiniMax-M2.7", "MiniMax-M2.5"],
        aliases: &["mmax-code-cn"],
    },
    // ── Zhipu / Z.AI ──
    ProviderSpec {
        id: "zhipuai",
        name: "Zhipu AI (GLM)",
        env_vars: &["ZHIPU_API_KEY"],
        base_url: "https://open.bigmodel.cn/api/paas/v4",
        api: ProviderApi::OpenAiCompatible,
        default_model: "glm-5v-turbo",
        models: &[
            "glm-5v-turbo",
            "glm-5.1",
            "glm-5",
            "glm-4.7-flash",
            "glm-4.7-flashx",
            "glm-4.7",
        ],
        aliases: &["glm-cn"],
    },
    ProviderSpec {
        id: "zhipuai-coding-plan",
        name: "Zhipu AI Coding Plan",
        env_vars: &["ZHIPU_API_KEY"],
        base_url: "https://open.bigmodel.cn/api/coding/paas/v4",
        api: ProviderApi::OpenAiCompatible,
        default_model: "glm-5v-turbo",
        models: &[
            "glm-5v-turbo",
            "glm-5.1",
            "glm-5-turbo",
            "glm-4.7",
            "glm-4.5-air",
        ],
        aliases: &["glm-code-cn"],
    },
    ProviderSpec {
        id: "zai",
        name: "Z.AI (GLM)",
        env_vars: &["ZHIPU_API_KEY"],
        base_url: "https://api.z.ai/api/paas/v4",
        api: ProviderApi::OpenAiCompatible,
        default_model: "glm-5v-turbo",
        models: &[
            "glm-5v-turbo",
            "glm-5.1",
            "glm-5-turbo",
            "glm-5",
            "glm-4.7-flashx",
            "glm-4.7-flash",
        ],
        aliases: &["z.ai", "glm"],
    },
    ProviderSpec {
        id: "zai-coding-plan",
        name: "Z.AI Coding Plan",
        env_vars: &["ZHIPU_API_KEY"],
        base_url: "https://api.z.ai/api/coding/paas/v4",
        api: ProviderApi::OpenAiCompatible,
        default_model: "glm-5v-turbo",
        models: &[
            "glm-5v-turbo",
            "glm-5.1",
            "glm-5-turbo",
            "glm-4.7",
            "glm-4.5-air",
        ],
        aliases: &["zai-code"],
    },
    // ── Xiaomi MiMo ──
    ProviderSpec {
        id: "xiaomi",
        name: "Xiaomi MiMo",
        env_vars: &["XIAOMI_API_KEY"],
        base_url: "https://api.xiaomimimo.com/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "mimo-v2.5-pro",
        models: &["mimo-v2.5-pro", "mimo-v2.5", "mimo-v2-pro", "mimo-v2-flash"],
        aliases: &["mimo"],
    },
    ProviderSpec {
        id: "xiaomi-token-plan-cn",
        name: "Xiaomi Token Plan CN",
        env_vars: &["XIAOMI_API_KEY"],
        base_url: "https://token-plan-cn.xiaomimimo.com/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "mimo-v2.5-pro",
        models: &["mimo-v2.5-pro", "mimo-v2.5", "mimo-v2-pro", "mimo-v2-flash"],
        aliases: &["mimo-cn"],
    },
    ProviderSpec {
        id: "xiaomi-token-plan-ams",
        name: "Xiaomi Token Plan AMS",
        env_vars: &["XIAOMI_API_KEY"],
        base_url: "https://token-plan-ams.xiaomimimo.com/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "mimo-v2.5",
        models: &["mimo-v2.5", "mimo-v2.5-pro", "mimo-v2-pro", "mimo-v2-flash"],
        aliases: &["mimo-ams"],
    },
    ProviderSpec {
        id: "xiaomi-token-plan-sgp",
        name: "Xiaomi Token Plan SGP",
        env_vars: &["XIAOMI_API_KEY"],
        base_url: "https://token-plan-sgp.xiaomimimo.com/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "mimo-v2.5",
        models: &["mimo-v2.5", "mimo-v2.5-pro", "mimo-v2-pro", "mimo-v2-flash"],
        aliases: &["mimo-sgp"],
    },
    // ── StepFun ──
    ProviderSpec {
        id: "stepfun",
        name: "StepFun",
        env_vars: &["STEPFUN_API_KEY"],
        base_url: "https://api.stepfun.com/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "step-3.5-flash-2603",
        models: &[
            "step-3.5-flash-2603",
            "step-3.5-flash",
            "step-1-32k",
            "step-2-16k",
        ],
        aliases: &["step"],
    },
    // ── SiliconFlow ──
    ProviderSpec {
        id: "siliconflow",
        name: "SiliconFlow",
        env_vars: &["SILICONFLOW_API_KEY"],
        base_url: "https://api.siliconflow.com/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "moonshotai/Kimi-K2.6",
        models: &[
            "moonshotai/Kimi-K2.6",
            "zai-org/GLM-5.1",
            "zai-org/GLM-5V-Turbo",
            "MiniMaxAI/MiniMax-M2.5",
            "zai-org/GLM-5",
            "stepfun-ai/Step-3.5-Flash",
        ],
        aliases: &["sf"],
    },
    ProviderSpec {
        id: "siliconflow-cn",
        name: "SiliconFlow China",
        env_vars: &["SILICONFLOW_CN_API_KEY"],
        base_url: "https://api.siliconflow.cn/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "Pro/moonshotai/Kimi-K2.6",
        models: &[
            "Pro/moonshotai/Kimi-K2.6",
            "Qwen/Qwen3.6-35B-A3B",
            "Pro/zai-org/GLM-5.1",
            "Qwen/Qwen3.5-9B",
            "Qwen/Qwen3.5-4B",
            "Qwen/Qwen3.5-122B-A10B",
        ],
        aliases: &["sf-cn"],
    },
    // ── Gateway / Router providers ──
    ProviderSpec {
        id: "kilo",
        name: "Kilo Gateway",
        env_vars: &["KILO_API_KEY"],
        base_url: "https://api.kilo.ai/api/gateway",
        api: ProviderApi::OpenAiCompatible,
        default_model: "baidu/cobuddy:free",
        models: &[
            "baidu/cobuddy:free",
            "openai/gpt-chat-latest",
            "x-ai/grok-4.3",
            "ibm-granite/granite-4.1-8b",
            "mistralai/mistral-medium-3-5",
            "openrouter/owl-alpha",
        ],
        aliases: &[],
    },
    ProviderSpec {
        id: "llmgateway",
        name: "LLM Gateway",
        env_vars: &["LLMGATEWAY_API_KEY"],
        base_url: "https://api.llmgateway.io/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "gemini-3.1-flash-lite",
        models: &["gpt-4o-mini", "gpt-4o", "claude-sonnet-4-6"],
        aliases: &[],
    },
    ProviderSpec {
        id: "zenmux",
        name: "ZenMux",
        env_vars: &["ZENMUX_API_KEY"],
        base_url: "https://zenmux.ai/api/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "deepseek/deepseek-v4-flash",
        models: &[
            "deepseek/deepseek-v4-flash",
            "deepseek/deepseek-v4-pro",
            "openai/gpt-5.5",
            "openai/gpt-5.5-pro",
            "xiaomi/mimo-v2.5-pro",
            "xiaomi/mimo-v2.5",
        ],
        aliases: &["zmux"],
    },
    ProviderSpec {
        id: "helicone",
        name: "Helicone",
        env_vars: &["HELICONE_API_KEY"],
        base_url: "https://ai-gateway.helicone.ai/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "claude-4.5-opus",
        models: &[
            "claude-4.5-opus",
            "gemini-3-pro-preview",
            "grok-4-1-fast-reasoning",
            "grok-4-1-fast-non-reasoning",
            "kimi-k2-thinking",
            "claude-4.5-haiku",
        ],
        aliases: &[],
    },
    ProviderSpec {
        id: "requesty",
        name: "Requesty",
        env_vars: &["REQUESTY_API_KEY"],
        base_url: "https://router.requesty.ai/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "openai/gpt-5.4-pro",
        models: &[
            "openai/gpt-5.4-pro",
            "openai/gpt-5.4",
            "openai/gpt-5.3-codex",
            "anthropic/claude-sonnet-4-6",
            "anthropic/claude-opus-4-6",
            "openai/gpt-5.2-codex",
        ],
        aliases: &[],
    },
    ProviderSpec {
        id: "vercel",
        name: "Vercel AI Gateway",
        env_vars: &["AI_GATEWAY_API_KEY"],
        base_url: "",
        api: ProviderApi::Custom,
        default_model: "google/gemini-3.1-flash-lite",
        models: &[],
        aliases: &["ai-gateway"],
    },
    // ── Misc API-key providers ──
    ProviderSpec {
        id: "baseten",
        name: "Baseten",
        env_vars: &["BASETEN_API_KEY"],
        base_url: "https://inference.baseten.co/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "deepseek-ai/DeepSeek-V4-Pro",
        models: &[
            "deepseek-ai/DeepSeek-V4-Pro",
            "moonshotai/Kimi-K2.6",
            "nvidia/Nemotron-120B-A12B",
            "zai-org/GLM-5",
            "MiniMaxAI/MiniMax-M2.5",
            "moonshotai/Kimi-K2.5",
        ],
        aliases: &[],
    },
    ProviderSpec {
        id: "novita-ai",
        name: "Novita AI",
        env_vars: &["NOVITA_API_KEY"],
        base_url: "https://api.novita.ai/openai",
        api: ProviderApi::OpenAiCompatible,
        default_model: "inclusionai/ling-2.6-flash",
        models: &[
            "inclusionai/ling-2.6-flash",
            "deepseek/deepseek-v4-flash",
            "deepseek/deepseek-v4-pro",
            "inclusionai/ling-2.6-1t",
            "moonshotai/kimi-k2.6",
            "google/gemma-4-31b-it",
        ],
        aliases: &["novita"],
    },
    ProviderSpec {
        id: "nebius",
        name: "Nebius Token Factory",
        env_vars: &["NEBIUS_API_KEY"],
        base_url: "https://api.tokenfactory.nebius.com/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "deepseek-ai/DeepSeek-V4-Pro",
        models: &[
            "deepseek-ai/DeepSeek-V4-Pro",
            "nvidia/nemotron-3-super-120b-a12b",
            "zai-org/GLM-5",
            "NousResearch/Hermes-4-70B",
            "NousResearch/Hermes-4-405B",
            "Qwen/Qwen3-30B-A3B-Instruct-2507",
        ],
        aliases: &[],
    },
    ProviderSpec {
        id: "github-copilot",
        name: "GitHub Copilot",
        env_vars: &["GITHUB_TOKEN"],
        base_url: "https://api.githubcopilot.com",
        api: ProviderApi::OpenAiCompatible,
        default_model: "gpt-5.5",
        models: &[
            "gpt-5.5",
            "claude-opus-4.7",
            "gpt-5.4-mini",
            "gpt-5.4",
            "gpt-5.3-codex",
            "gemini-3.1-pro-preview",
        ],
        aliases: &["copilot"],
    },
    ProviderSpec {
        id: "github-models",
        name: "GitHub Models",
        env_vars: &["GITHUB_TOKEN"],
        base_url: "https://models.github.ai/inference",
        api: ProviderApi::OpenAiCompatible,
        default_model: "deepseek/deepseek-r1-0528",
        models: &[
            "deepseek/deepseek-r1-0528",
            "mistral-ai/mistral-medium-2505",
            "openai/gpt-4.1-nano",
            "openai/gpt-4.1",
            "openai/gpt-4.1-mini",
            "deepseek/deepseek-v3-0324",
        ],
        aliases: &["gh-models"],
    },
    ProviderSpec {
        id: "perplexity-agent",
        name: "Perplexity Agent",
        env_vars: &["PERPLEXITY_API_KEY"],
        base_url: "https://api.perplexity.ai/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "openai/gpt-5.5",
        models: &[
            "openai/gpt-5.5",
            "anthropic/claude-opus-4-7",
            "nvidia/nemotron-3-super-120b-a12b",
            "openai/gpt-5.4",
            "google/gemini-3.1-pro-preview",
            "anthropic/claude-sonnet-4-6",
        ],
        aliases: &["pplx-agent"],
    },
    ProviderSpec {
        id: "firepass",
        name: "Fireworks Firepass",
        env_vars: &["FIREPASS_API_KEY"],
        base_url: "https://api.fireworks.ai/inference/v1/",
        api: ProviderApi::OpenAiCompatible,
        default_model: "accounts/fireworks/routers/kimi-k2p6-turbo",
        models: &["accounts/fireworks/routers/kimi-k2p6-turbo"],
        aliases: &[],
    },
    ProviderSpec {
        id: "ollama-cloud",
        name: "Ollama Cloud",
        env_vars: &["OLLAMA_API_KEY"],
        base_url: "https://ollama.com/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "deepseek-v4-flash",
        models: &[
            "deepseek-v4-flash",
            "deepseek-v4-pro",
            "kimi-k2.6",
            "gemma4:31b",
            "glm-5.1",
            "minimax-m2.7",
        ],
        aliases: &[],
    },
    // ── Chinese ecosystem ──
    ProviderSpec {
        id: "bailing",
        name: "Bailing",
        env_vars: &["BAILING_API_TOKEN"],
        base_url: "https://api.tbox.cn/api/llm/v1/chat/completions",
        api: ProviderApi::OpenAiCompatible,
        default_model: "Ring-1T",
        models: &["Ring-1T", "DeepSeek-R1"],
        aliases: &[],
    },
    ProviderSpec {
        id: "qiniu-ai",
        name: "Qiniu AI",
        env_vars: &["QINIU_API_KEY"],
        base_url: "https://api.qnaigc.com/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "qwen3.5-397b-a17b",
        models: &["deepseek-chat", "deepseek-reasoner"],
        aliases: &[],
    },
    ProviderSpec {
        id: "qihang-ai",
        name: "QiHang AI",
        env_vars: &["QIHANG_API_KEY"],
        base_url: "https://api.qhaigc.net/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "gemini-3-flash-preview",
        models: &["deepseek-chat"],
        aliases: &[],
    },
    ProviderSpec {
        id: "siliconflow-cn",
        name: "SiliconFlow China",
        env_vars: &["SILICONFLOW_CN_API_KEY"],
        base_url: "https://api.siliconflow.cn/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "Pro/moonshotai/Kimi-K2.6",
        models: &[
            "Pro/moonshotai/Kimi-K2.6",
            "Qwen/Qwen3.6-35B-A3B",
            "Pro/zai-org/GLM-5.1",
            "Qwen/Qwen3.5-9B",
            "Qwen/Qwen3.5-4B",
            "Qwen/Qwen3.5-122B-A10B",
        ],
        aliases: &["sf-cn"],
    },
    ProviderSpec {
        id: "iflowcn",
        name: "iFlow AI",
        env_vars: &["IFLOW_API_KEY"],
        base_url: "https://apis.iflow.cn/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "kimi-k2-0905",
        models: &["qwen3-coder-plus", "gpt-5.4"],
        aliases: &["iflow"],
    },
    ProviderSpec {
        id: "kuae-cloud-coding-plan",
        name: "KUAE Cloud Coding Plan",
        env_vars: &["KUAE_API_KEY"],
        base_url: "https://coding-plan-endpoint.kuaecloud.net/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "GLM-4.7",
        models: &["GLM-4.7", "GLM-5"],
        aliases: &["kuae"],
    },
    ProviderSpec {
        id: "tencent-coding-plan",
        name: "Tencent Coding Plan",
        env_vars: &["TENCENT_CODING_PLAN_API_KEY"],
        base_url: "https://api.lkeap.cloud.tencent.com/coding/v3",
        api: ProviderApi::OpenAiCompatible,
        default_model: "hunyuan-turbos",
        models: &["deepseek-v4", "hunyuan-turbo"],
        aliases: &["tencent-code"],
    },
    ProviderSpec {
        id: "tencent-tokenhub",
        name: "Tencent TokenHub",
        env_vars: &["TENCENT_TOKENHUB_API_KEY"],
        base_url: "https://tokenhub.tencentmaas.com/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "hy3-preview",
        models: &["hunyuan-turbo", "deepseek-chat"],
        aliases: &[],
    },
    // ── Cloud / Edge ──
    ProviderSpec {
        id: "scaleway",
        name: "Scaleway",
        env_vars: &["SCALEWAY_API_KEY"],
        base_url: "https://api.scaleway.ai/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "qwen3.5-397b-a17b",
        models: &["deepseek-chat", "llama-3.3-70b"],
        aliases: &[],
    },
    ProviderSpec {
        id: "ovhcloud",
        name: "OVHcloud AI Endpoints",
        env_vars: &["OVHCLOUD_API_KEY"],
        base_url: "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "qwen3.5-9b",
        models: &["deepseek-chat", "llama-3.3-70b"],
        aliases: &["ovh"],
    },
    ProviderSpec {
        id: "stackit",
        name: "STACKIT",
        env_vars: &["STACKIT_API_KEY"],
        base_url: "https://api.openai-compat.model-serving.eu01.onstackit.cloud/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "openai/gpt-oss-120b",
        models: &["deepseek-chat"],
        aliases: &[],
    },
    ProviderSpec {
        id: "digitalocean",
        name: "DigitalOcean",
        env_vars: &["DIGITALOCEAN_ACCESS_TOKEN"],
        base_url: "https://inference.do-ai.run/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "deepseek-v4-pro",
        models: &["gpt-4o-mini", "deepseek-chat"],
        aliases: &["do"],
    },
    // ── Special: needs account ID ──
    ProviderSpec {
        id: "cloudflare-workers-ai",
        name: "Cloudflare Workers AI",
        env_vars: &["CLOUDFLARE_API_KEY", "CLOUDFLARE_ACCOUNT_ID"],
        base_url: "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        api: ProviderApi::Custom,
        default_model: "@cf/moonshotai/kimi-k2.6",
        models: &["@cf/meta/llama-3.3-70b-instruct"],
        aliases: &["workers-ai", "cf-workers"],
    },
    ProviderSpec {
        id: "cloudflare-ai-gateway",
        name: "Cloudflare AI Gateway",
        env_vars: &[
            "CLOUDFLARE_API_TOKEN",
            "CLOUDFLARE_ACCOUNT_ID",
            "CLOUDFLARE_GATEWAY_ID",
        ],
        base_url: "",
        api: ProviderApi::Custom,
        default_model: "openai/gpt-5.5",
        models: &[],
        aliases: &["cf-gateway", "cf-aig"],
    },
    // ── Complex auth providers ──
    ProviderSpec {
        id: "amazon-bedrock",
        name: "Amazon Bedrock",
        env_vars: &["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
        base_url: "",
        api: ProviderApi::Custom,
        default_model: "global.anthropic.claude-opus-4-7",
        models: &["anthropic.claude-sonnet-4-20250514"],
        aliases: &["bedrock"],
    },
    ProviderSpec {
        id: "azure",
        name: "Azure OpenAI",
        env_vars: &["AZURE_RESOURCE_NAME", "AZURE_API_KEY"],
        base_url: "",
        api: ProviderApi::Custom,
        default_model: "gpt-5.5",
        models: &[
            "gpt-5.5",
            "kimi-k2.6",
            "grok-4-20-non-reasoning",
            "grok-4-20-reasoning",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
        ],
        aliases: &[],
    },
    ProviderSpec {
        id: "azure-cognitive-services",
        name: "Azure Cognitive Services",
        env_vars: &[
            "AZURE_COGNITIVE_SERVICES_RESOURCE_NAME",
            "AZURE_COGNITIVE_SERVICES_API_KEY",
        ],
        base_url: "",
        api: ProviderApi::Custom,
        default_model: "gpt-5.5",
        models: &[
            "gpt-5.5",
            "kimi-k2.6",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            "gpt-5.4-pro",
            "gpt-5.4",
        ],
        aliases: &["azure-cog"],
    },
    ProviderSpec {
        id: "google-vertex",
        name: "Vertex AI",
        env_vars: &[
            "GOOGLE_VERTEX_PROJECT",
            "GOOGLE_VERTEX_LOCATION",
            "GOOGLE_APPLICATION_CREDENTIALS",
        ],
        base_url: "",
        api: ProviderApi::Custom,
        default_model: "gemini-3.1-flash-lite",
        models: &[
            "gemini-3.1-flash-lite",
            "claude-opus-4-7@default",
            "gemini-3.1-flash-lite-preview",
            "gemini-3.1-pro-preview-customtools",
            "gemini-3.1-pro-preview",
            "claude-sonnet-4-6@default",
        ],
        aliases: &["vertex"],
    },
    ProviderSpec {
        id: "google-vertex-anthropic",
        name: "Vertex (Anthropic)",
        env_vars: &[
            "GOOGLE_VERTEX_PROJECT",
            "GOOGLE_VERTEX_LOCATION",
            "GOOGLE_APPLICATION_CREDENTIALS",
        ],
        base_url: "",
        api: ProviderApi::Custom,
        default_model: "claude-opus-4-7@default",
        models: &[
            "claude-opus-4-7@default",
            "claude-sonnet-4-6@default",
            "claude-opus-4-6@default",
            "claude-opus-4-5@20251101",
            "claude-haiku-4-5@20251001",
            "claude-sonnet-4-5@20250929",
        ],
        aliases: &["vertex-claude"],
    },
    ProviderSpec {
        id: "gitlab",
        name: "GitLab Duo",
        env_vars: &["GITLAB_TOKEN"],
        base_url: "",
        api: ProviderApi::Custom,
        default_model: "duo-chat-opus-4-7",
        models: &["duo-chat"],
        aliases: &["gitlab-duo"],
    },
    ProviderSpec {
        id: "sap-ai-core",
        name: "SAP AI Core",
        env_vars: &["AICORE_SERVICE_KEY"],
        base_url: "",
        api: ProviderApi::Custom,
        default_model: "anthropic--claude-4.6-sonnet",
        models: &[],
        aliases: &["sap"],
    },
    // ── Long-tail API-key providers ──
    ProviderSpec {
        id: "302ai",
        name: "302.AI",
        env_vars: &["302AI_API_KEY"],
        base_url: "https://api.302.ai/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "claude-opus-4-7",
        models: &["qwen3-235b-a22b", "grok-4.1", "gemini-2.5-flash"],
        aliases: &["302"],
    },
    ProviderSpec {
        id: "abacus",
        name: "Abacus AI",
        env_vars: &["ABACUS_API_KEY"],
        base_url: "https://routellm.abacus.ai/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "gpt-5.4",
        models: &[
            "gpt-5.4",
            "gemini-3.1-flash-lite-preview",
            "gpt-5.3-chat-latest",
            "gemini-3.1-pro-preview",
            "claude-sonnet-4-6",
            "zai-org/glm-5",
        ],
        aliases: &[],
    },
    ProviderSpec {
        id: "berget",
        name: "Berget.AI",
        env_vars: &["BERGET_API_KEY"],
        base_url: "https://api.berget.ai/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "mistralai/Mistral-Medium-3.5-128B",
        models: &["zai-org/GLM-4.7"],
        aliases: &[],
    },
    ProviderSpec {
        id: "chutes",
        name: "Chutes AI",
        env_vars: &["CHUTES_API_KEY"],
        base_url: "https://llm.chutes.ai/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "Qwen/Qwen3-32B-TEE",
        models: &["deepseek-chat", "llama-3.3-70b"],
        aliases: &[],
    },
    ProviderSpec {
        id: "clarifai",
        name: "Clarifai",
        env_vars: &["CLARIFAI_PAT"],
        base_url: "https://api.clarifai.com/v2/ext/openai/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "moonshotai/chat-completion/models/Kimi-K2_6",
        models: &["deepseek-chat"],
        aliases: &[],
    },
    ProviderSpec {
        id: "cortecs",
        name: "Cortecs AI",
        env_vars: &["CORTECS_API_KEY"],
        base_url: "https://api.cortecs.ai/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "deepseek-v4-flash",
        models: &["deepseek-chat"],
        aliases: &[],
    },
    ProviderSpec {
        id: "evroc",
        name: "Evroc AI",
        env_vars: &["EVROC_API_KEY"],
        base_url: "https://models.think.evroc.com/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "moonshotai/Kimi-K2.5",
        models: &["Qwen/Qwen3-VL-30B-A3B-Instruct"],
        aliases: &[],
    },
    ProviderSpec {
        id: "fastrouter",
        name: "FastRouter",
        env_vars: &["FASTROUTER_API_KEY"],
        base_url: "https://go.fastrouter.ai/api/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "z-ai/glm-5",
        models: &["x-ai/grok-4", "gpt-5.4"],
        aliases: &[],
    },
    ProviderSpec {
        id: "friendli",
        name: "Friendli",
        env_vars: &["FRIENDLI_TOKEN"],
        base_url: "https://api.friendli.ai/serverless/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "zai-org/GLM-5.1",
        models: &["Qwen/Qwen3-235B-A22B-Instruct-2507"],
        aliases: &[],
    },
    ProviderSpec {
        id: "frogbot",
        name: "FrogBot",
        env_vars: &["FROGBOT_API_KEY"],
        base_url: "https://app.frogbot.ai/api/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "grok-4-3",
        models: &["grok-4-1-fast-reasoning", "gpt-5.4"],
        aliases: &[],
    },
    ProviderSpec {
        id: "hpc-ai",
        name: "HPC-AI",
        env_vars: &["HPC_AI_API_KEY"],
        base_url: "https://api.hpc-ai.com/inference/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "zai-org/glm-5.1",
        models: &["zai-org/glm-5.1"],
        aliases: &[],
    },
    ProviderSpec {
        id: "inception",
        name: "Inception Labs",
        env_vars: &["INCEPTION_API_KEY"],
        base_url: "https://api.inceptionlabs.ai/v1/",
        api: ProviderApi::OpenAiCompatible,
        default_model: "mercury-edit-2",
        models: &["mercury-edit-2"],
        aliases: &[],
    },
    ProviderSpec {
        id: "io-net",
        name: "IO.NET",
        env_vars: &["IOINTELLIGENCE_API_KEY"],
        base_url: "https://api.intelligence.io.solutions/api/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "Qwen/Qwen3-235B-A22B-Thinking-2507",
        models: &["deepseek-chat", "Qwen3-Coder-480B-A35B"],
        aliases: &["io"],
    },
    ProviderSpec {
        id: "jiekou",
        name: "Jiekou.AI",
        env_vars: &["JIEKOU_API_KEY"],
        base_url: "https://api.jiekou.ai/openai",
        api: ProviderApi::OpenAiCompatible,
        default_model: "gpt-5.1",
        models: &["gpt-5.1-codex-max", "deepseek-chat"],
        aliases: &[],
    },
    ProviderSpec {
        id: "modelscope",
        name: "ModelScope",
        env_vars: &["MODELSCOPE_API_KEY"],
        base_url: "https://api-inference.modelscope.cn/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "ZhipuAI/GLM-4.6",
        models: &["Qwen/Qwen3-30B-A3B-Thinking-2507"],
        aliases: &["ms"],
    },
    ProviderSpec {
        id: "moark",
        name: "Moark AI",
        env_vars: &["MOARK_API_KEY"],
        base_url: "https://moark.com/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "MiniMax-M2.1",
        models: &["GLM-4.7", "deepseek-chat"],
        aliases: &[],
    },
    ProviderSpec {
        id: "morph",
        name: "Morph AI",
        env_vars: &["MORPH_API_KEY"],
        base_url: "https://api.morphllm.com/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "morph-v3-fast",
        models: &["morph-v3-fast", "morph-v3-large", "auto"],
        aliases: &[],
    },
    ProviderSpec {
        id: "nano-gpt",
        name: "NanoGPT",
        env_vars: &["NANO_GPT_API_KEY"],
        base_url: "https://nano-gpt.com/api/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "qwen3.6-max-preview",
        models: &["glm-4-flash", "deepseek-chat"],
        aliases: &[],
    },
    ProviderSpec {
        id: "neuralwatt",
        name: "Neuralwatt",
        env_vars: &["NEURALWATT_API_KEY"],
        base_url: "https://api.neuralwatt.com/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "kimi-k2.6-fast",
        models: &["glm-5-fast", "deepseek-chat"],
        aliases: &[],
    },
    ProviderSpec {
        id: "vultr",
        name: "Vultr",
        env_vars: &["VULTR_API_KEY"],
        base_url: "https://api.vultrinference.com/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "GLM-5-FP8",
        models: &["deepseek-chat", "llama-3.3-70b"],
        aliases: &[],
    },
    ProviderSpec {
        id: "vivgrid",
        name: "Vivgrid",
        env_vars: &["VIVGRID_API_KEY"],
        base_url: "https://api.vivgrid.com/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "deepseek-v4-pro",
        models: &["deepseek-chat"],
        aliases: &[],
    },
    ProviderSpec {
        id: "wandb",
        name: "Weights & Biases",
        env_vars: &["WANDB_API_KEY"],
        base_url: "https://api.inference.wandb.ai/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "zai-org/GLM-5.1",
        models: &["deepseek-chat", "gpt-5.4"],
        aliases: &["w&b", "wb"],
    },
    ProviderSpec {
        id: "synthetic",
        name: "Synthetic",
        env_vars: &["SYNTHETIC_API_KEY"],
        base_url: "https://api.synthetic.new/openai/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "hf:moonshotai/Kimi-K2.6",
        models: &["deepseek-chat"],
        aliases: &[],
    },
    ProviderSpec {
        id: "v0",
        name: "V0 AI",
        env_vars: &["V0_API_KEY"],
        base_url: "https://api.v0.ai/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "v0-1.5-lg",
        models: &["v0-1.5-lg", "v0-1.5-md", "v0-1.0-md"],
        aliases: &[],
    },
    // ── Local / Self-hosted ──
    ProviderSpec {
        id: "lmstudio",
        name: "LM Studio",
        env_vars: &["LMSTUDIO_API_KEY"],
        base_url: "http://127.0.0.1:1234/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "openai/gpt-oss-20b",
        models: &["local-model"],
        aliases: &[],
    },
    ProviderSpec {
        id: "ollama",
        name: "Ollama (Local)",
        env_vars: &[],
        base_url: "http://localhost:11434/v1",
        api: ProviderApi::OpenAiCompatible,
        default_model: "llama3",
        models: &["llama3", "mistral", "codellama"],
        aliases: &[],
    },
];

// ─── Lookup helpers ───────────────────────────────────────────

/// Find provider by id, name, or alias (case-insensitive).
pub fn find(query: &str) -> Option<&'static ProviderSpec> {
    let q = query.trim().to_ascii_lowercase();
    API_KEY_PROVIDERS.iter().find(|p| {
        p.id == q || p.name.to_ascii_lowercase() == q || p.aliases.iter().any(|a| *a == q)
    })
}

/// Find provider by exact id.
pub fn by_id(id: &str) -> Option<&'static ProviderSpec> {
    API_KEY_PROVIDERS.iter().find(|p| p.id == id)
}

/// Try each env var in order; return first non-empty value.
pub fn env_key(spec: &ProviderSpec) -> Option<String> {
    for var in spec.env_vars {
        if let Ok(val) = std::env::var(var) {
            if !val.trim().is_empty() {
                return Some(val);
            }
        }
    }
    None
}

/// Parse a `provider/model` string to extract provider.
pub fn infer_from_model(model: &str) -> Option<&'static ProviderSpec> {
    let prefix = model.trim().split_once('/')?.0;
    find(prefix)
}

/// Normalize a model string for a given provider (e.g. clinepass slug).
pub fn normalize_model(spec: &ProviderSpec, model: &str) -> String {
    let model = model.trim();
    match spec.id {
        "clinepass" => {
            if let Some(rest) = model.strip_prefix("clinepass/") {
                format!("cline-pass/{rest}")
            } else if model.starts_with("cline-pass/") {
                model.to_string()
            } else {
                format!("cline-pass/{model}")
            }
        }
        _ => model.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_providers_by_id() {
        assert_eq!(find("openai").map(|p| p.id), Some("openai"));
        assert_eq!(find("anthropic").map(|p| p.id), Some("anthropic"));
        assert_eq!(find("xai").map(|p| p.id), Some("xai"));
        assert_eq!(find("deepseek").map(|p| p.id), Some("deepseek"));
    }

    #[test]
    fn finds_providers_by_alias() {
        assert_eq!(find("gpt").map(|p| p.id), Some("openai"));
        assert_eq!(find("claude").map(|p| p.id), Some("anthropic"));
        assert_eq!(find("grok").map(|p| p.id), Some("xai"));
        assert_eq!(find("gemini").map(|p| p.id), Some("google"));
    }

    #[test]
    fn finds_providers_by_name() {
        assert_eq!(find("OpenAI").map(|p| p.id), Some("openai"));
        assert_eq!(find("anThropic").map(|p| p.id), Some("anthropic"));
    }

    #[test]
    fn by_id_exact() {
        assert!(by_id("openai").is_some());
        assert!(by_id("nonexistent").is_none());
    }

    #[test]
    fn infer_from_model_prefix() {
        assert_eq!(
            infer_from_model("anthropic/claude-sonnet-4-6").map(|p| p.id),
            Some("anthropic")
        );
        assert_eq!(infer_from_model("xai/grok-4.5").map(|p| p.id), Some("xai"));
        assert!(infer_from_model("no-prefix-model").is_none());
    }

    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn env_key_returns_none_when_unset() {
        let _guard = ENV_LOCK.lock().unwrap();
        let spec = by_id("openai").unwrap();
        let old = std::env::var_os("OPENAI_API_KEY");
        std::env::remove_var("OPENAI_API_KEY");
        assert_eq!(env_key(spec), None);
        if let Some(key) = old {
            std::env::set_var("OPENAI_API_KEY", key);
        }
    }

    #[test]
    fn env_key_returns_some_when_set() {
        let _guard = ENV_LOCK.lock().unwrap();
        let spec = by_id("openai").unwrap();
        let old = std::env::var_os("OPENAI_API_KEY");
        std::env::set_var("OPENAI_API_KEY", "sk-test");
        assert_eq!(env_key(spec).as_deref(), Some("sk-test"));
        if let Some(key) = old {
            std::env::set_var("OPENAI_API_KEY", key);
        } else {
            std::env::remove_var("OPENAI_API_KEY");
        }
    }

    #[test]
    fn clinepass_aliases() {
        let spec = find("clinepass").expect("clinepass id");
        assert_eq!(spec.id, "clinepass");
        assert_eq!(find("cline-pass").map(|p| p.id), Some("clinepass"));
    }

    #[test]
    fn normalizes_clinepass_slugs() {
        let spec = find("clinepass").unwrap();
        assert_eq!(
            normalize_model(spec, "deepseek-v4-flash"),
            "cline-pass/deepseek-v4-flash"
        );
        assert_eq!(
            normalize_model(spec, "cline-pass/qwen3.7-max"),
            "cline-pass/qwen3.7-max"
        );
        assert_eq!(
            normalize_model(spec, "clinepass/glm-5.2"),
            "cline-pass/glm-5.2"
        );
    }

    #[test]
    fn non_zero_providers() {
        assert!(API_KEY_PROVIDERS.len() > 50);
    }
}

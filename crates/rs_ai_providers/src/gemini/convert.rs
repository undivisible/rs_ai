use rs_ai_core::{
    AiError, AiResult, ContentPart, FinishReason, GenerateOptions, GenerateResult, ImageData,
    Prompt, ResponseMetadata, Role, ThinkingConfig as CoreThinkingConfig, ToolCallRequest,
    ToolChoice, Usage,
};

use super::api_types::*;

/// The parts of a Gemini API request built from a `Prompt` and `GenerateOptions`.
pub(crate) struct GeminiRequestParts {
    pub contents: Vec<GeminiContent>,
    pub system_instruction: Option<GeminiContent>,
    pub generation_config: Option<GenerationConfig>,
    pub tools: Option<Vec<GeminiTool>>,
    pub tool_config: Option<ToolConfig>,
}

/// Separate system messages from conversation messages and build the Gemini request parts.
pub(crate) fn build_request(
    prompt: Prompt,
    options: &GenerateOptions,
) -> AiResult<GeminiRequestParts> {
    let messages = prompt.into_messages();

    let mut system_parts: Vec<GeminiPart> = Vec::new();
    let mut contents: Vec<GeminiContent> = Vec::new();

    for msg in messages {
        match msg.role {
            Role::System => {
                for part in &msg.content {
                    system_parts.push(content_part_to_gemini(part)?);
                }
            }
            Role::User => {
                let parts = msg
                    .content
                    .iter()
                    .map(content_part_to_gemini)
                    .collect::<AiResult<_>>()?;
                contents.push(GeminiContent {
                    role: Some("user".to_string()),
                    parts,
                });
            }
            Role::Assistant => {
                let parts = msg
                    .content
                    .iter()
                    .map(content_part_to_gemini)
                    .collect::<AiResult<_>>()?;
                contents.push(GeminiContent {
                    role: Some("model".to_string()),
                    parts,
                });
            }
            Role::Tool => {
                let parts = msg
                    .content
                    .iter()
                    .map(content_part_to_gemini)
                    .collect::<AiResult<_>>()?;
                contents.push(GeminiContent {
                    role: Some("user".to_string()),
                    parts,
                });
            }
        }
    }

    let system_instruction = if system_parts.is_empty() {
        None
    } else {
        Some(GeminiContent {
            role: None,
            parts: system_parts,
        })
    };

    let generation_config = build_generation_config(options);
    let (tools, tool_config) = build_tools(options);

    Ok(GeminiRequestParts {
        contents,
        system_instruction,
        generation_config,
        tools,
        tool_config,
    })
}

fn content_part_to_gemini(part: &ContentPart) -> AiResult<GeminiPart> {
    match part {
        ContentPart::Text { text } => Ok(GeminiPart::Text { text: text.clone() }),
        ContentPart::Image { data } => match data {
            ImageData::Base64 { media_type, data } => Ok(GeminiPart::InlineData {
                inline_data: InlineData {
                    mime_type: media_type.clone(),
                    data: data.clone(),
                },
            }),
            ImageData::Url { .. } => Err(AiError::UnsupportedCapability {
                capability: "image URLs (use base64 image data)".into(),
                provider: "gemini".into(),
            }),
        },
        ContentPart::ToolCall { call } => Ok(GeminiPart::FunctionCall {
            function_call: FunctionCall {
                id: None,
                name: call.name.clone(),
                args: call.arguments.clone(),
            },
        }),
        ContentPart::ToolResult { result } => {
            let response = serde_json::json!({
                "content": result.content,
                "is_error": result.is_error,
            });
            Ok(GeminiPart::FunctionResponse {
                function_response: FunctionResponse {
                    id: None,
                    name: result.call_id.clone(),
                    response,
                },
            })
        }
        ContentPart::File { .. } => Err(AiError::UnsupportedCapability {
            capability: "file attachments".into(),
            provider: "gemini".into(),
        }),
    }
}

fn build_generation_config(options: &GenerateOptions) -> Option<GenerationConfig> {
    let max_tokens = options.max_tokens.or(Some(8192));

    let stop_sequences = if options.stop_sequences.is_empty() {
        None
    } else {
        Some(options.stop_sequences.clone())
    };

    let (response_mime_type, response_schema) = if let Some(schema) = &options.output_schema {
        (
            Some("application/json".to_string()),
            Some(schema.as_value().clone()),
        )
    } else {
        (None, None)
    };

    let thinking_config = options.thinking.as_ref().map(|t| match t {
        CoreThinkingConfig::Budget { tokens } => ThinkingConfig {
            thinking_budget: Some(*tokens),
            thinking_level: None,
        },
        CoreThinkingConfig::Adaptive | CoreThinkingConfig::Enabled => ThinkingConfig {
            thinking_budget: Some(8192),
            thinking_level: None,
        },
    });

    Some(GenerationConfig {
        temperature: options.temperature,
        max_output_tokens: max_tokens,
        top_p: options.top_p,
        top_k: options.top_k,
        stop_sequences,
        response_mime_type,
        response_schema,
        thinking_config,
    })
}

fn build_tools(options: &GenerateOptions) -> (Option<Vec<GeminiTool>>, Option<ToolConfig>) {
    let tool_defs = match &options.tools {
        Some(tools) if !tools.is_empty() => tools,
        _ => return (None, None),
    };

    let declarations: Vec<FunctionDeclaration> = tool_defs
        .iter()
        .map(|t| FunctionDeclaration {
            name: t.name.clone(),
            description: t.description.clone(),
            parameters: t.parameters.clone(),
        })
        .collect();

    let tools = vec![GeminiTool {
        function_declarations: declarations,
    }];

    let mode = match &options.tool_choice {
        Some(ToolChoice::None) => "NONE",
        Some(ToolChoice::Required) => "ANY",
        Some(ToolChoice::Auto) | None => "AUTO",
        Some(ToolChoice::Specific(_)) => "ANY",
    };

    let tool_config = ToolConfig {
        function_calling_config: FunctionCallingConfig {
            mode: mode.to_string(),
        },
    };

    (Some(tools), Some(tool_config))
}

/// Convert a Gemini API response into our GenerateResult.
pub(crate) fn response_to_result(
    response: GenerateContentResponse,
    model_id: &str,
) -> GenerateResult {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCallRequest> = Vec::new();
    let mut finish_reason = FinishReason::Unknown;

    if let Some(candidates) = &response.candidates {
        if let Some(candidate) = candidates.first() {
            if let Some(ref reason) = candidate.finish_reason {
                finish_reason = map_finish_reason(reason);
            }

            if let Some(ref content) = candidate.content {
                for part in &content.parts {
                    match part {
                        GeminiPart::Text { text } => {
                            text_parts.push(text.clone());
                        }
                        GeminiPart::FunctionCall { function_call } => {
                            tool_calls.push(ToolCallRequest {
                                id: uuid::Uuid::new_v4().to_string(),
                                name: function_call.name.clone(),
                                arguments: function_call.args.clone(),
                            });
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    let usage = map_usage(response.usage_metadata.as_ref());

    let combined_text = if text_parts.is_empty() {
        None
    } else {
        Some(text_parts.join(""))
    };

    if !tool_calls.is_empty() && finish_reason == FinishReason::Stop {
        finish_reason = FinishReason::ToolCall;
    }

    GenerateResult {
        text: combined_text,
        tool_calls,
        finish_reason,
        usage,
        metadata: ResponseMetadata {
            request_id: uuid::Uuid::new_v4(),
            provider: "gemini".to_string(),
            model: model_id.to_string(),
            latency_ms: None,
            extra: Default::default(),
        },
        steps: Vec::new(),
        reasoning: None,
    }
}

pub(crate) fn map_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "STOP" => FinishReason::Stop,
        "MAX_TOKENS" => FinishReason::Length,
        "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" => FinishReason::ContentFilter,
        _ => FinishReason::Unknown,
    }
}

pub(crate) fn map_usage(meta: Option<&UsageMetadata>) -> Usage {
    match meta {
        Some(m) => Usage {
            prompt_tokens: m.prompt_token_count,
            completion_tokens: m.candidates_token_count,
            total_tokens: m.total_token_count,
        },
        None => Usage::default(),
    }
}

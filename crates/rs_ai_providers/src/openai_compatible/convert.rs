use rs_ai_core::{
    ContentPart, FinishReason, GenerateOptions, GenerateResult, ImageData, Message, Prompt,
    ResponseMetadata, Role, ToolCallRequest, ToolChoice, Usage,
};

use super::api_types::*;

// ---------------------------------------------------------------------------
// Prompt -> API messages
// ---------------------------------------------------------------------------

pub(crate) fn prompt_to_messages(prompt: &Prompt) -> Vec<ChatMessage> {
    match prompt {
        Prompt::Text(text) => vec![ChatMessage {
            role: "user".to_string(),
            content: Some(serde_json::Value::String(text.clone())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        Prompt::Messages(msgs) => msgs.iter().map(message_to_chat).collect(),
    }
}

fn message_to_chat(msg: &Message) -> ChatMessage {
    let role = match msg.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };

    // Collect tool calls from content parts (for assistant messages).
    let mut tool_calls: Vec<ChatToolCall> = Vec::new();
    // For tool-result messages, extract the call_id.
    let mut tool_call_id: Option<String> = None;

    let content_parts: Vec<serde_json::Value> = msg
        .content
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text { text } => Some(serde_json::json!({ "type": "text", "text": text })),
            ContentPart::Image { data } => Some(image_to_json(data)),
            ContentPart::ToolCall { call } => {
                tool_calls.push(ChatToolCall {
                    id: call.id.clone(),
                    index: None,
                    call_type: "function".to_string(),
                    function: ChatFunctionCall {
                        name: call.name.clone(),
                        arguments: call.arguments.to_string(),
                    },
                });
                None
            }
            ContentPart::ToolResult { result } => {
                tool_call_id = Some(result.call_id.clone());
                Some(serde_json::Value::String(result.content.clone()))
            }
            ContentPart::File { .. } => None, // files not supported by OpenAI chat API
        })
        .collect();

    // Build the content field.
    let content = if msg.role == Role::Tool {
        // Tool messages use a plain string content.
        content_parts.into_iter().next()
    } else if content_parts.len() == 1 {
        // Single text part -> use a plain string for compatibility.
        if let Some(serde_json::Value::Object(ref obj)) = content_parts.first() {
            if obj.get("type").and_then(|v| v.as_str()) == Some("text") {
                obj.get("text").cloned()
            } else {
                Some(serde_json::Value::Array(content_parts))
            }
        } else {
            Some(serde_json::Value::Array(content_parts))
        }
    } else if content_parts.is_empty() {
        None
    } else {
        Some(serde_json::Value::Array(content_parts))
    };

    ChatMessage {
        role: role.to_string(),
        content,
        name: msg.name.clone(),
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
        tool_call_id,
    }
}

fn image_to_json(data: &ImageData) -> serde_json::Value {
    match data {
        ImageData::Url { url, detail } => {
            let mut image_url = serde_json::json!({ "url": url });
            if let Some(d) = detail {
                let detail_str = match d {
                    rs_ai_core::ImageDetail::Auto => "auto",
                    rs_ai_core::ImageDetail::Low => "low",
                    rs_ai_core::ImageDetail::High => "high",
                };
                image_url["detail"] = serde_json::Value::String(detail_str.to_string());
            }
            serde_json::json!({
                "type": "image_url",
                "image_url": image_url,
            })
        }
        ImageData::Base64 { media_type, data } => {
            let data_url = format!("data:{};base64,{}", media_type, data);
            serde_json::json!({
                "type": "image_url",
                "image_url": { "url": data_url },
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Build the full API request
// ---------------------------------------------------------------------------

pub(crate) fn options_to_request(
    model_id: &str,
    prompt: &Prompt,
    options: &GenerateOptions,
    stream: bool,
) -> ChatCompletionRequest {
    let messages = prompt_to_messages(prompt);

    let tools: Option<Vec<ChatTool>> = options.tools.as_ref().map(|tool_defs| {
        tool_defs
            .iter()
            .map(|td| ChatTool {
                tool_type: "function".to_string(),
                function: ChatFunction {
                    name: td.name.clone(),
                    description: if td.description.is_empty() {
                        None
                    } else {
                        Some(td.description.clone())
                    },
                    parameters: td.parameters.clone(),
                },
            })
            .collect()
    });

    let tool_choice = options.tool_choice.as_ref().map(|tc| match tc {
        ToolChoice::Auto => serde_json::json!("auto"),
        ToolChoice::None => serde_json::json!("none"),
        ToolChoice::Required => serde_json::json!("required"),
        ToolChoice::Specific(name) => serde_json::json!({
            "type": "function",
            "function": { "name": name }
        }),
    });

    let response_format = options.output_schema.as_ref().map(|schema| ResponseFormat {
        format_type: "json_schema".to_string(),
        json_schema: Some(JsonSchemaFormat {
            name: schema.name.clone(),
            schema: schema.schema.clone(),
            strict: Some(true),
        }),
    });

    let stop = if options.stop_sequences.is_empty() {
        None
    } else {
        Some(options.stop_sequences.clone())
    };

    let stream_options = if stream {
        Some(StreamOptions {
            include_usage: true,
        })
    } else {
        None
    };

    ChatCompletionRequest {
        model: model_id.to_string(),
        messages,
        temperature: options.temperature,
        max_tokens: options.max_tokens,
        top_p: options.top_p,
        frequency_penalty: options.frequency_penalty,
        presence_penalty: options.presence_penalty,
        seed: options.seed,
        stop,
        tools,
        tool_choice,
        response_format,
        stream,
        stream_options,
        prompt_cache_key: None,
        prompt_cache_retention: None,
    }
}

// ---------------------------------------------------------------------------
// API response -> GenerateResult
// ---------------------------------------------------------------------------

pub(crate) fn response_to_result(
    response: ChatCompletionResponse,
    provider: &str,
) -> GenerateResult {
    let choice = response.choices.into_iter().next();

    let (text, tool_calls, finish_reason) = match choice {
        Some(c) => {
            let msg = c.message.unwrap_or(ChatMessage {
                role: "assistant".to_string(),
                content: None,
                name: None,
                tool_calls: None,
                tool_call_id: None,
            });

            let text = msg.content.and_then(|v| match v {
                serde_json::Value::String(s) => Some(s),
                _ => None,
            });

            let tool_calls: Vec<ToolCallRequest> = msg
                .tool_calls
                .unwrap_or_default()
                .into_iter()
                .map(|tc| {
                    let args: serde_json::Value =
                        serde_json::from_str(&tc.function.arguments).unwrap_or_default();
                    ToolCallRequest {
                        id: tc.id,
                        name: tc.function.name,
                        arguments: args,
                    }
                })
                .collect();

            let finish_reason = parse_finish_reason(c.finish_reason.as_deref());
            (text, tool_calls, finish_reason)
        }
        None => (None, Vec::new(), FinishReason::Unknown),
    };

    let usage = response
        .usage
        .map(|u| Usage {
            prompt_tokens: Some(u.prompt_tokens),
            completion_tokens: Some(u.completion_tokens),
            total_tokens: Some(u.total_tokens),
        })
        .unwrap_or_default();

    GenerateResult {
        text,
        tool_calls,
        finish_reason,
        usage,
        metadata: ResponseMetadata {
            request_id: uuid::Uuid::new_v4(),
            provider: provider.to_string(),
            model: response.model,
            latency_ms: None,
            extra: Default::default(),
        },
        steps: Vec::new(),
        reasoning: None,
    }
}

// ---------------------------------------------------------------------------
// Finish reason mapping
// ---------------------------------------------------------------------------

pub(crate) fn parse_finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("stop") => FinishReason::Stop,
        Some("length") => FinishReason::Length,
        Some("tool_calls") => FinishReason::ToolCall,
        Some("content_filter") => FinishReason::ContentFilter,
        Some("error") => FinishReason::Error,
        _ => FinishReason::Unknown,
    }
}

//! Portable, non-streaming HTTP generation using the providers' normal wire conversions.
//!
//! These concrete APIs do not implement `LanguageModel`: browser fetch futures are
//! not `Send`. Browser requests are subject to CORS and expose credentials to the
//! user; use a trusted proxy when credentials must remain secret. Fetch rejects
//! redirects and omits ambient credentials. Explicit API-key headers are still sent.

use rs_ai_core::{AiError, AiResult, GenerateOptions, GenerateResult, Prompt};
use secrecy::{ExposeSecret, SecretString};

/// Anthropic Messages API client. Only API-key authentication is supported.
pub struct Anthropic {
    api_key: SecretString,
    base_url: String,
    model: String,
}

impl Anthropic {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: SecretString::from(api_key.into()),
            base_url: "https://api.anthropic.com".into(),
            model: model.into(),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub async fn generate(
        &self,
        prompt: Prompt,
        options: GenerateOptions,
    ) -> AiResult<GenerateResult> {
        let body =
            crate::claude::convert::build_request(&self.model, prompt, &options, false, None)?;
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));
        let headers = [
            ("anthropic-version", "2023-06-01"),
            ("anthropic-dangerous-direct-browser-access", "true"),
            ("x-api-key", self.api_key.expose_secret()),
        ];
        let response: crate::claude::api_types::MessagesResponse =
            send(&url, &headers, &body, "anthropic").await?;
        Ok(crate::claude::convert::convert_response(response))
    }
}

/// OpenAI-compatible chat-completions API client; base URL includes `/v1` if required.
pub struct OpenAiCompatible {
    api_key: SecretString,
    base_url: String,
    model: String,
}

impl OpenAiCompatible {
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            api_key: SecretString::from(api_key.into()),
            base_url: base_url.into(),
            model: model.into(),
        }
    }

    pub async fn generate(
        &self,
        prompt: Prompt,
        options: GenerateOptions,
    ) -> AiResult<GenerateResult> {
        let body = crate::openai_compatible::convert::options_to_request(
            &self.model,
            "openai_compatible",
            &prompt,
            &options,
            false,
        )?;
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let auth = format!("Bearer {}", self.api_key.expose_secret());
        let response: crate::openai_compatible::api_types::ChatCompletionResponse = send(
            &url,
            &[("authorization", &auth)],
            &body,
            "openai_compatible",
        )
        .await?;
        crate::openai_compatible::convert::response_to_result(response, "openai_compatible")
    }
}

async fn send<T: serde::de::DeserializeOwned>(
    url: &str,
    headers: &[(&str, &str)],
    body: &impl serde::Serialize,
    provider: &str,
) -> AiResult<T> {
    let body = serde_json::to_string(body).map_err(|e| AiError::Serialization(e.to_string()))?;
    let (status, body) = post(url, headers, &body).await?;
    if !(200..300).contains(&status) {
        let message = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|value| value["error"]["message"].as_str().map(str::to_owned))
            .unwrap_or(body);
        return Err(match status {
            401 => AiError::AuthError { message },
            429 => AiError::RateLimit { retry_after: None },
            408 | 504 => AiError::Timeout,
            _ => AiError::ProviderError {
                provider: provider.into(),
                status: Some(status),
                message,
            },
        });
    }
    serde_json::from_str(&body).map_err(|e| AiError::Serialization(e.to_string()))
}

#[cfg(not(target_arch = "wasm32"))]
async fn post(url: &str, headers: &[(&str, &str)], body: &str) -> AiResult<(u16, String)> {
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| AiError::Transport {
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;
    let mut request = client
        .post(url)
        .header("content-type", "application/json")
        .body(body.to_owned());
    for (name, value) in headers {
        request = request.header(*name, *value);
    }
    let response = request.send().await.map_err(|e| AiError::Transport {
        message: e.to_string(),
        source: Some(Box::new(e)),
    })?;
    let status = response.status().as_u16();
    let body = response.text().await.map_err(|e| AiError::Transport {
        message: e.to_string(),
        source: Some(Box::new(e)),
    })?;
    Ok((status, body))
}

#[cfg(target_arch = "wasm32")]
async fn post(url: &str, headers: &[(&str, &str)], body: &str) -> AiResult<(u16, String)> {
    use wasm_bindgen::{JsCast, JsValue};
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Headers, Request, RequestCredentials, RequestInit, RequestRedirect, Response};

    fn transport(_: JsValue) -> AiError {
        AiError::Transport {
            message: "Browser fetch failed".into(),
            source: None,
        }
    }
    let request_headers = Headers::new().map_err(transport)?;
    request_headers
        .set("content-type", "application/json")
        .map_err(transport)?;
    for (name, value) in headers {
        request_headers.set(name, value).map_err(transport)?;
    }
    let init = RequestInit::new();
    init.set_method("POST");
    init.set_headers(&request_headers);
    init.set_body(&JsValue::from_str(body));
    init.set_credentials(RequestCredentials::Omit);
    init.set_redirect(RequestRedirect::Error);
    let request = Request::new_with_str_and_init(url, &init).map_err(transport)?;
    let global = js_sys::global();
    let fetch: js_sys::Function = js_sys::Reflect::get(&global, &JsValue::from_str("fetch"))
        .map_err(transport)?
        .dyn_into()
        .map_err(transport)?;
    let promise: js_sys::Promise = fetch
        .call1(&global, &request)
        .map_err(transport)?
        .dyn_into()
        .map_err(transport)?;
    let response: Response = JsFuture::from(promise)
        .await
        .map_err(transport)?
        .dyn_into()
        .map_err(transport)?;
    let status = response.status();
    let body = JsFuture::from(response.text().map_err(transport)?)
        .await
        .map_err(transport)?
        .as_string()
        .ok_or_else(|| AiError::Transport {
            message: "Browser response was not text".into(),
            source: None,
        })?;
    Ok((status, body))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    fn server(response: &'static str, status: &str) -> (String, std::thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let reply = format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}",
            response.len()
        );
        let handle = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            let mut data = Vec::new();
            let mut buffer = [0; 4096];
            loop {
                let n = socket.read(&mut buffer).unwrap();
                assert!(n > 0);
                data.extend_from_slice(&buffer[..n]);
                if let Some(end) = data.windows(4).position(|w| w == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&data[..end]);
                    let length: usize = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse().unwrap())
                        })
                        .unwrap();
                    if data.len() >= end + 4 + length {
                        break;
                    }
                }
            }
            socket.write_all(reply.as_bytes()).unwrap();
            String::from_utf8(data).unwrap()
        });
        (url, handle)
    }

    #[tokio::test]
    async fn anthropic_http_generation() {
        let (url, request) = server(
            r#"{"id":"id","model":"claude-test","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":1}}"#,
            "200 OK",
        );
        let result = Anthropic::new("test-key", "claude-test")
            .with_base_url(url)
            .generate(Prompt::Text("hi".into()), GenerateOptions::default())
            .await
            .unwrap();
        assert_eq!(result.text.as_deref(), Some("hello"));
        let request = request.join().unwrap();
        assert!(request.starts_with("POST /v1/messages "));
        assert!(request.to_lowercase().contains("x-api-key: test-key"));
        let body: serde_json::Value =
            serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(body["model"], "claude-test");
        assert_eq!(body["messages"][0]["content"], "hi");
    }

    #[tokio::test]
    async fn openai_http_generation_and_errors() {
        let (url, request) = server(
            r#"{"id":"id","model":"model","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}]}"#,
            "200 OK",
        );
        let result = OpenAiCompatible::new(format!("{url}/v1"), "test-key", "model")
            .generate(Prompt::Text("hi".into()), GenerateOptions::default())
            .await
            .unwrap();
        assert_eq!(result.text.as_deref(), Some("hello"));
        let request = request.join().unwrap();
        assert!(request.starts_with("POST /v1/chat/completions "));
        assert!(request
            .to_lowercase()
            .contains("authorization: bearer test-key"));
        let body: serde_json::Value =
            serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(body["messages"][0]["content"], "hi");
        let (url, request) = server(r#"{"error":{"message":"bad key"}}"#, "401 Unauthorized");
        let err = OpenAiCompatible::new(url, "bad", "model")
            .generate(Prompt::Text("hi".into()), GenerateOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(err, AiError::AuthError { message } if message == "bad key"));
        request.join().unwrap();
    }

    #[tokio::test]
    async fn anthropic_multi_tool_continuation_is_one_user_message() {
        use rs_ai_core::{ContentPart, Message};
        let (url, request) = server(
            r#"{"id":"id","model":"claude-test","content":[{"type":"tool_use","id":"a","name":"one","input":{}},{"type":"tool_use","id":"b","name":"two","input":{}}],"stop_reason":"tool_use","usage":{"input_tokens":2,"output_tokens":1}}"#,
            "200 OK",
        );
        let first = Anthropic::new("test-key", "claude-test")
            .with_base_url(url)
            .generate(Prompt::Text("hi".into()), GenerateOptions::default())
            .await
            .unwrap();
        request.join().unwrap();
        assert_eq!(first.tool_calls.len(), 2);
        let mut assistant = Message::assistant("");
        assistant.content = first
            .tool_calls
            .into_iter()
            .map(|call| ContentPart::ToolCall { call })
            .collect();
        let prompt = Prompt::Messages(vec![
            Message::user("hi"),
            assistant,
            Message::tool_result("a", "first"),
            Message::tool_result("b", "second"),
        ]);
        let (url, request) = server(
            r#"{"id":"id2","model":"claude-test","content":[{"type":"text","text":"done"}],"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":1}}"#,
            "200 OK",
        );
        let result = Anthropic::new("test-key", "claude-test")
            .with_base_url(url)
            .generate(prompt, GenerateOptions::default())
            .await
            .unwrap();
        assert_eq!(result.text.as_deref(), Some("done"));
        let request = request.join().unwrap();
        let body: serde_json::Value =
            serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1]["content"].as_array().unwrap().len(), 2);
        assert_eq!(messages[2]["role"], "user");
        assert_eq!(messages[2]["content"].as_array().unwrap().len(), 2);
        assert_eq!(messages[2]["content"][0]["tool_use_id"], "a");
        assert_eq!(messages[2]["content"][1]["tool_use_id"], "b");
    }

    #[tokio::test]
    async fn redirects_are_not_followed() {
        let (url, request) = server(
            "redirect",
            "307 Temporary Redirect\r\nLocation: http://127.0.0.1:1/elsewhere",
        );
        let error = OpenAiCompatible::new(url, "secret", "model")
            .generate(Prompt::Text("hi".into()), GenerateOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            AiError::ProviderError {
                status: Some(307),
                ..
            }
        ));
        request.join().unwrap();
    }
}

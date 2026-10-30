//! ⚠️ **UNSTABLE** — This crate is in active development. APIs may change without notice.
//!
//! Apple Foundation Models bridge for the Rusty AI SDK.
//!
//! This crate provides safe Rust bindings for Apple's [FoundationModels] on-device AI
//! framework (Apple Intelligence), integrated with the [`rs_ai_core`] ecosystem.
//!
//! # Overview
//!
//! FoundationModels gives access to an on-device ≈3B-parameter language model that runs
//! entirely locally — no network requests, no API keys, no data leaving the device.
//!
//! This crate exposes:
//! - **Availability checking** — [`is_available`] / [`availability`]
//! - **Single-shot generation** — [`respond`] / [`respond_with_options`]
//! - **Session-based multi-turn** — [`Session::respond`]
//! - **Streaming** — [`Session::stream`] returns a [`ResponseStream`]
//! - **Structured generation** — [`Session::respond_as`]
//! - **Tool calling** — [`Session::with_tools`]
//! - **LanguageModel trait** — [`FoundationModel`] implements [`rs_ai_core::LanguageModel`]
//! - **Provider trait** — [`FoundationModelProvider`] implements [`rs_ai_core::Provider`]
//!
//! # Requirements
//!
//! | Requirement | Value |
//! |---|---|
//! | macOS | 26 (Tahoe) or later |
//! | Hardware | Apple Silicon (M1 or later) |
//! | Setting | Apple Intelligence enabled in System Settings |
//! | Build tool | Xcode with the macOS 26 SDK |
//!
//! On unsupported hardware or older macOS the crate still compiles and links; all APIs
//! return [`Error::Unavailable`] at runtime.
//!
//! # Quick start
//!
//! ```no_run
//! # async fn example() -> Result<(), rs_ai_local::foundationmodels::Error> {
//! use rs_ai_local::foundationmodels::{is_available, respond};
//!
//! if !is_available() {
//!     eprintln!("Apple Intelligence not available on this device");
//!     return Ok(());
//! }
//!
//! let answer = respond("What is the capital of France?").await?;
//! println!("{answer}");
//! # Ok(()) }
//! ```
//!
//! # Using with the rs_ai ecosystem
//!
//! ```no_run
//! use rs_ai_local::foundationmodels::FoundationModelProvider;
//! use rs_ai_core::Provider;
//!
//! let provider = FoundationModelProvider::new();
//! let model = provider.language_model("apple-foundation-model").unwrap();
//! ```
//!
//! [FoundationModels]: https://developer.apple.com/documentation/foundationmodels

#![deny(missing_docs)]

use std::pin::Pin;
use std::task::{Context as StdContext, Poll};

use futures_core::Stream;

#[cfg(foundation_models_bridge)]
use std::ffi::{c_char, c_void, CStr, CString};

#[cfg(foundation_models_bridge)]
use std::ptr::null;

#[cfg(foundation_models_bridge)]
use std::sync::Arc;

use futures_channel::mpsc;

#[cfg(foundation_models_bridge)]
use futures_channel::oneshot;

// Re-export rs_ai_core types used in public API
// ─── FFI declarations ──────────────────────────────────────────────────────────

#[cfg(foundation_models_bridge)]
unsafe extern "C" {
    fn fm_availability_reason() -> i32;
    fn fm_session_create(instructions: *const c_char) -> *mut c_void;
    fn fm_session_create_with_tools(
        instructions: *const c_char,
        tools_json: *const c_char,
        tool_ctx: *mut c_void,
        tool_dispatch: extern "C" fn(
            *mut c_void,
            *const c_char,
            *const c_char,
            *mut c_void,
            extern "C" fn(*mut c_void, *const c_char, *const c_char),
        ),
    ) -> *mut c_void;
    fn fm_session_destroy(handle: *mut c_void);
    fn fm_session_respond(
        handle: *mut c_void,
        prompt: *const c_char,
        temperature: f64,
        max_tokens: i64,
        ctx: *mut c_void,
        callback: extern "C" fn(*mut c_void, *const c_char, *const c_char),
    );
    fn fm_session_respond_structured(
        handle: *mut c_void,
        prompt: *const c_char,
        schema_json: *const c_char,
        temperature: f64,
        max_tokens: i64,
        ctx: *mut c_void,
        callback: extern "C" fn(*mut c_void, *const c_char, *const c_char),
    );
    fn fm_session_stream(
        handle: *mut c_void,
        prompt: *const c_char,
        temperature: f64,
        max_tokens: i64,
        ctx: *mut c_void,
        on_token: extern "C" fn(*mut c_void, *const c_char),
        on_done: extern "C" fn(*mut c_void, *const c_char),
    );
    fn fm_session_update_profile(handle: *mut c_void, instructions: *const c_char);
    fn fm_session_respond_with_attachment(
        handle: *mut c_void,
        prompt: *const c_char,
        image_bytes: *const u8,
        image_len: usize,
        image_mime: *const c_char,
        temperature: f64,
        max_tokens: i64,
        ctx: *mut c_void,
        callback: extern "C" fn(*mut c_void, *const c_char, *const c_char),
    );
    fn fm_session_stream_structured(
        handle: *mut c_void,
        prompt: *const c_char,
        schema_json: *const c_char,
        temperature: f64,
        max_tokens: i64,
        ctx: *mut c_void,
        on_token: extern "C" fn(*mut c_void, *const c_char),
        on_done: extern "C" fn(*mut c_void, *const c_char),
    );
}

// ─── Error ─────────────────────────────────────────────────────────────────────

/// Reasons why Apple Intelligence is not available on the current device.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum UnavailabilityReason {
    /// The device does not have compatible hardware (requires Apple Silicon M1 or later).
    #[error("device is not eligible (requires Apple Silicon M1 or later)")]
    DeviceNotEligible,
    /// Apple Intelligence is supported but has not been enabled in System Settings.
    #[error("Apple Intelligence is not enabled in System Settings")]
    NotEnabled,
    /// The on-device model is still downloading or is otherwise not ready.
    #[error("the on-device model is not ready yet")]
    ModelNotReady,
    /// An unrecognized availability state was returned by the framework.
    #[error("unknown availability state")]
    Unknown,
}

/// Errors returned by this crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Apple Intelligence is not available on this device.
    #[error("Apple Intelligence unavailable: {0}")]
    Unavailable(#[source] UnavailabilityReason),

    /// The model produced an error during text generation.
    #[error("generation error: {0}")]
    Generation(String),

    /// An argument contained a null byte and could not be converted to a C string.
    #[error("argument contains a null byte: {0}")]
    NullByte(#[from] std::ffi::NulError),

    /// A `temperature` value outside the valid range [0.0, 2.0] was supplied.
    #[error("temperature {0} is out of range; expected 0.0 – 2.0")]
    InvalidTemperature(f64),

    /// JSON serialisation or deserialisation failed.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// A tool invoked by the model returned an error.
    #[error("tool '{name}' failed: {message}")]
    ToolError {
        /// Name of the tool that failed.
        name: String,
        /// Error message from the tool.
        message: String,
    },
}

// ─── GenerationOptions ─────────────────────────────────────────────────────────

/// Tuning parameters for a single generation request.
#[derive(Debug, Default, Clone)]
pub struct GenerationOptions {
    /// Controls output randomness. Range: 0.0 to 2.0.
    pub temperature: Option<f64>,
    /// Maximum number of tokens to generate.
    pub max_tokens: Option<usize>,
}

impl GenerationOptions {
    /// Returns an error if any field contains an out-of-range value.
    pub fn validate(&self) -> Result<(), Error> {
        if let Some(t) = self.temperature {
            if !(0.0..=2.0).contains(&t) {
                return Err(Error::InvalidTemperature(t));
            }
        }
        Ok(())
    }

    #[cfg_attr(not(foundation_models_bridge), allow(dead_code))]
    fn ffi_temperature(&self) -> f64 {
        self.temperature.unwrap_or(-1.0)
    }

    #[cfg_attr(not(foundation_models_bridge), allow(dead_code))]
    fn ffi_max_tokens(&self) -> i64 {
        self.max_tokens.map(|n| n as i64).unwrap_or(-1)
    }
}

// ─── Attachment types for multimodal prompts ───────────────────────────────────

/// A media attachment to include in a prompt.
///
/// Currently supports images passed as raw bytes with a MIME type hint.
/// The Swift bridge decodes these into `NSImage` / `UIImage` and wraps
/// them as `Prompt.Attachment.image(_:)`.
#[derive(Debug, Clone)]
pub struct Attachment {
    /// Raw image file bytes (JPEG, PNG, etc.).
    pub data: Vec<u8>,
    /// MIME type, e.g. `"image/jpeg"` or `"image/png"`.
    pub mime: String,
}

impl Attachment {
    /// Creates a new image attachment from raw bytes and a MIME type.
    pub fn image(data: impl Into<Vec<u8>>, mime: impl Into<String>) -> Self {
        Self {
            data: data.into(),
            mime: mime.into(),
        }
    }
}

// ─── Schema types for structured generation ────────────────────────────────────

/// The type of a single property in a [`Schema`].
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SchemaPropertyType {
    /// UTF-8 text.
    String,
    /// Whole number.
    Integer,
    /// Floating-point number.
    Double,
    /// Boolean.
    Bool,
}

/// A single property within a [`Schema`].
#[derive(Debug, Clone, serde::Serialize)]
pub struct SchemaProperty {
    /// Property name.
    pub name: String,
    /// Optional description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The expected type.
    #[serde(rename = "type")]
    pub property_type: SchemaPropertyType,
    /// Whether the property is optional.
    #[serde(default)]
    pub optional: bool,
}

impl SchemaProperty {
    /// Creates a required property.
    pub fn new(name: impl Into<String>, property_type: SchemaPropertyType) -> Self {
        Self {
            name: name.into(),
            description: None,
            property_type,
            optional: false,
        }
    }

    /// Attaches a description.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Marks as optional.
    pub fn optional(mut self) -> Self {
        self.optional = true;
        self
    }
}

/// Describes the JSON object shape for structured generation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Schema {
    /// Type name.
    pub name: String,
    /// Optional description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Properties.
    pub properties: Vec<SchemaProperty>,
}

impl Schema {
    /// Creates a new schema.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            properties: Vec::new(),
        }
    }

    /// Attaches a description.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Adds a property.
    pub fn property(mut self, property: SchemaProperty) -> Self {
        self.properties.push(property);
        self
    }
}

// ─── Tool calling ──────────────────────────────────────────────────────────────

/// A function that the model can invoke.
pub struct ToolDefinition {
    /// Tool name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Parameter schema.
    pub parameters: Schema,
    #[cfg_attr(not(foundation_models_bridge), allow(dead_code))]
    pub(crate) handler: Box<dyn Fn(serde_json::Value) -> Result<String, String> + Send + Sync>,
}

impl ToolDefinition {
    /// Creates a new tool definition.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Schema,
        handler: impl Fn(serde_json::Value) -> Result<String, String> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            handler: Box::new(handler),
        }
    }
}

impl std::fmt::Debug for ToolDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolDefinition")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

#[cfg(foundation_models_bridge)]
#[allow(clippy::type_complexity)]
struct ToolsContext {
    tools: Vec<(
        String,
        Box<dyn Fn(serde_json::Value) -> Result<String, String> + Send + Sync>,
    )>,
}

// ─── Availability ──────────────────────────────────────────────────────────────

#[cfg_attr(not(foundation_models_bridge), allow(dead_code))]
const FM_AVAILABLE: i32 = 0;
#[cfg_attr(not(foundation_models_bridge), allow(dead_code))]
const FM_DEVICE_NOT_ELIGIBLE: i32 = 1;
#[cfg_attr(not(foundation_models_bridge), allow(dead_code))]
const FM_NOT_ENABLED: i32 = 2;
#[cfg_attr(not(foundation_models_bridge), allow(dead_code))]
const FM_MODEL_NOT_READY: i32 = 3;

/// Returns `true` if Apple Intelligence is available and ready.
pub fn is_available() -> bool {
    availability().is_ok()
}

/// Returns availability status.
pub fn availability() -> Result<(), UnavailabilityReason> {
    #[cfg(foundation_models_bridge)]
    {
        let code = unsafe { fm_availability_reason() };
        match code {
            FM_AVAILABLE => Ok(()),
            FM_DEVICE_NOT_ELIGIBLE => Err(UnavailabilityReason::DeviceNotEligible),
            FM_NOT_ENABLED => Err(UnavailabilityReason::NotEnabled),
            FM_MODEL_NOT_READY => Err(UnavailabilityReason::ModelNotReady),
            _ => Err(UnavailabilityReason::Unknown),
        }
    }
    #[cfg(not(foundation_models_bridge))]
    Err(UnavailabilityReason::DeviceNotEligible)
}

// ─── Convenience top-level functions ──────────────────────────────────────────

/// Sends a single prompt and returns the response text.
pub async fn respond(prompt: &str) -> Result<String, Error> {
    respond_with_options(prompt, &GenerationOptions::default()).await
}

/// Like [`respond`] but with options.
pub async fn respond_with_options(
    prompt: &str,
    options: &GenerationOptions,
) -> Result<String, Error> {
    let session = Session::new()?;
    session.respond_with_options(prompt, options).await
}

// ─── Session ───────────────────────────────────────────────────────────────────

/// A stateful conversation session backed by a `LanguageModelSession`.
pub struct Session {
    #[cfg(foundation_models_bridge)]
    handle: *mut c_void,
    #[cfg(foundation_models_bridge)]
    _tools: Option<Arc<ToolsContext>>,
}

// Safety: The underlying Swift LanguageModelSession is designed for concurrent
// use via Swift's async/await actor system.
unsafe impl Send for Session {}
unsafe impl Sync for Session {}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session").finish_non_exhaustive()
    }
}

impl Session {
    /// Creates a new session with no system instructions.
    pub fn new() -> Result<Self, Error> {
        Self::with_instructions("")
    }

    /// Creates a new session with system instructions.
    pub fn with_instructions(instructions: &str) -> Result<Self, Error> {
        availability().map_err(Error::Unavailable)?;

        #[cfg(foundation_models_bridge)]
        {
            let c_instructions = CString::new(instructions)?;
            let handle = unsafe { fm_session_create(c_instructions.as_ptr()) };
            if handle.is_null() {
                return Err(Error::Unavailable(UnavailabilityReason::Unknown));
            }
            Ok(Self {
                handle,
                _tools: None,
            })
        }
        #[cfg(not(foundation_models_bridge))]
        {
            let _ = instructions;
            Err(Error::Unavailable(UnavailabilityReason::DeviceNotEligible))
        }
    }

    /// Creates a session pre-loaded with tools.
    pub fn with_tools(instructions: &str, tools: Vec<ToolDefinition>) -> Result<Self, Error> {
        availability().map_err(Error::Unavailable)?;

        #[cfg(foundation_models_bridge)]
        {
            let tool_descs: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.name,
                        "description": t.description,
                        "properties": t.parameters.properties,
                    })
                })
                .collect();
            let tools_json = serde_json::to_string(&tool_descs)?;

            let tools_ctx = Arc::new(ToolsContext {
                tools: tools.into_iter().map(|t| (t.name, t.handler)).collect(),
            });
            let tool_ctx_ptr = Arc::as_ptr(&tools_ctx) as *mut c_void;

            let c_instructions = CString::new(instructions)?;
            let c_tools_json = CString::new(tools_json)?;

            let handle = unsafe {
                fm_session_create_with_tools(
                    c_instructions.as_ptr(),
                    c_tools_json.as_ptr(),
                    tool_ctx_ptr,
                    tool_dispatch,
                )
            };
            if handle.is_null() {
                return Err(Error::Unavailable(UnavailabilityReason::Unknown));
            }
            Ok(Self {
                handle,
                _tools: Some(tools_ctx),
            })
        }
        #[cfg(not(foundation_models_bridge))]
        {
            let _ = (instructions, tools);
            Err(Error::Unavailable(UnavailabilityReason::DeviceNotEligible))
        }
    }

    /// Dynamically updates the session's system instructions without tearing down state.
    ///
    /// Replaces the previous instructions and affects all subsequent responses.
    /// Useful for changing the model's role or behaviour mid-conversation.
    pub fn update_profile(&self, instructions: &str) -> Result<(), Error> {
        #[cfg(foundation_models_bridge)]
        {
            let c_instructions = CString::new(instructions)?;
            unsafe {
                fm_session_update_profile(self.handle, c_instructions.as_ptr());
            }
            Ok(())
        }
        #[cfg(not(foundation_models_bridge))]
        {
            let _ = instructions;
            Err(Error::Unavailable(UnavailabilityReason::DeviceNotEligible))
        }
    }

    /// Sends a prompt and returns the full response text.
    pub async fn respond(&self, prompt: &str) -> Result<String, Error> {
        self.respond_with_options(prompt, &GenerationOptions::default())
            .await
    }

    /// Like [`respond`] but with options.
    pub async fn respond_with_options(
        &self,
        prompt: &str,
        options: &GenerationOptions,
    ) -> Result<String, Error> {
        options.validate()?;

        #[cfg(foundation_models_bridge)]
        {
            let (tx, rx) = oneshot::channel::<Result<String, String>>();
            let ctx = Box::into_raw(Box::new(tx)) as *mut c_void;
            let c_prompt = CString::new(prompt)?;

            unsafe {
                fm_session_respond(
                    self.handle,
                    c_prompt.as_ptr(),
                    options.ffi_temperature(),
                    options.ffi_max_tokens(),
                    ctx,
                    respond_callback,
                );
            }

            rx.await
                .map_err(|_| Error::Generation("session was dropped before responding".into()))?
                .map_err(Error::Generation)
        }
        #[cfg(not(foundation_models_bridge))]
        {
            let _ = (prompt, options);
            Err(Error::Unavailable(UnavailabilityReason::DeviceNotEligible))
        }
    }

    /// Sends a prompt and deserialises the response into `T`.
    pub async fn respond_as<T: serde::de::DeserializeOwned>(
        &self,
        prompt: &str,
        schema: &Schema,
    ) -> Result<T, Error> {
        self.respond_as_with_options(prompt, schema, &GenerationOptions::default())
            .await
    }

    /// Like [`Self::respond_as`] but with options.
    pub async fn respond_as_with_options<T: serde::de::DeserializeOwned>(
        &self,
        prompt: &str,
        schema: &Schema,
        options: &GenerationOptions,
    ) -> Result<T, Error> {
        options.validate()?;

        #[cfg(foundation_models_bridge)]
        {
            let (tx, rx) = oneshot::channel::<Result<String, String>>();
            let ctx = Box::into_raw(Box::new(tx)) as *mut c_void;
            let c_prompt = CString::new(prompt)?;
            let schema_json = serde_json::to_string(schema)?;
            let c_schema_json = CString::new(schema_json)?;

            unsafe {
                fm_session_respond_structured(
                    self.handle,
                    c_prompt.as_ptr(),
                    c_schema_json.as_ptr(),
                    options.ffi_temperature(),
                    options.ffi_max_tokens(),
                    ctx,
                    respond_callback,
                );
            }

            let json = rx
                .await
                .map_err(|_| Error::Generation("session was dropped before responding".into()))?
                .map_err(Error::Generation)?;
            Ok(serde_json::from_str(&json)?)
        }
        #[cfg(not(foundation_models_bridge))]
        {
            let _ = (prompt, schema, options);
            Err(Error::Unavailable(UnavailabilityReason::DeviceNotEligible))
        }
    }

    /// Sends a prompt with an image attachment and returns the response text.
    ///
    /// The model sees both the text and the image, enabling visual reasoning
    /// over diagrams, screenshots, or photos.
    pub async fn respond_with_attachment(
        &self,
        prompt: &str,
        attachment: &Attachment,
        options: &GenerationOptions,
    ) -> Result<String, Error> {
        options.validate()?;

        #[cfg(foundation_models_bridge)]
        {
            let (tx, rx) = oneshot::channel::<Result<String, String>>();
            let ctx = Box::into_raw(Box::new(tx)) as *mut c_void;
            let c_prompt = CString::new(prompt)?;
            let c_mime = CString::new(&*attachment.mime)?;

            unsafe {
                fm_session_respond_with_attachment(
                    self.handle,
                    c_prompt.as_ptr(),
                    attachment.data.as_ptr(),
                    attachment.data.len(),
                    c_mime.as_ptr(),
                    options.ffi_temperature(),
                    options.ffi_max_tokens(),
                    ctx,
                    respond_callback,
                );
            }

            rx.await
                .map_err(|_| Error::Generation("session was dropped before responding".into()))?
                .map_err(Error::Generation)
        }
        #[cfg(not(foundation_models_bridge))]
        {
            let _ = (prompt, attachment, options);
            Err(Error::Unavailable(UnavailabilityReason::DeviceNotEligible))
        }
    }

    /// Returns a [`ResponseStream`] that yields text chunks.
    pub fn stream(&self, prompt: &str) -> Result<ResponseStream, Error> {
        self.stream_with_options(prompt, &GenerationOptions::default())
    }

    /// Like [`Self::stream`] but with options.
    pub fn stream_with_options(
        &self,
        prompt: &str,
        options: &GenerationOptions,
    ) -> Result<ResponseStream, Error> {
        options.validate()?;

        #[cfg(foundation_models_bridge)]
        {
            let (tx, rx) = mpsc::unbounded::<Result<String, String>>();
            let ctx = Box::into_raw(Box::new(StreamContext { tx })) as *mut c_void;
            let c_prompt = CString::new(prompt)?;

            unsafe {
                fm_session_stream(
                    self.handle,
                    c_prompt.as_ptr(),
                    options.ffi_temperature(),
                    options.ffi_max_tokens(),
                    ctx,
                    stream_token_callback,
                    stream_done_callback,
                );
            }

            Ok(ResponseStream { rx })
        }
        #[cfg(not(foundation_models_bridge))]
        {
            let _ = (prompt, options);
            Err(Error::Unavailable(UnavailabilityReason::DeviceNotEligible))
        }
    }

    /// Returns a [`ResponseStructuredStream`] that yields JSON chunks conforming to
    /// the given schema as the model generates them.
    ///
    /// Each chunk is a JSON string representing a partially-populated view of the
    /// structured type defined by `schema`. The stream ends when generation finishes.
    pub fn stream_structured(
        &self,
        prompt: &str,
        schema: &Schema,
        options: &GenerationOptions,
    ) -> Result<ResponseStructuredStream, Error> {
        options.validate()?;

        #[cfg(foundation_models_bridge)]
        {
            let (tx, rx) = mpsc::unbounded::<Result<String, String>>();
            let ctx = Box::into_raw(Box::new(StreamContext { tx })) as *mut c_void;
            let c_prompt = CString::new(prompt)?;
            let schema_json = serde_json::to_string(schema)?;
            let c_schema_json = CString::new(schema_json)?;

            unsafe {
                fm_session_stream_structured(
                    self.handle,
                    c_prompt.as_ptr(),
                    c_schema_json.as_ptr(),
                    options.ffi_temperature(),
                    options.ffi_max_tokens(),
                    ctx,
                    stream_token_callback,
                    stream_done_callback,
                );
            }

            Ok(ResponseStructuredStream { rx })
        }
        #[cfg(not(foundation_models_bridge))]
        {
            let _ = (prompt, schema, options);
            Err(Error::Unavailable(UnavailabilityReason::DeviceNotEligible))
        }
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            // On non-macOS platforms, create a dummy session that will error on use
            #[cfg(foundation_models_bridge)]
            {
                panic!("Session::new should not fail on macOS when bridge is available");
            }
            #[cfg(not(foundation_models_bridge))]
            {
                // This is a placeholder that will never be used in practice
                // because all methods check availability first
                Session {
                    #[cfg(foundation_models_bridge)]
                    handle: std::ptr::null_mut(),
                    #[cfg(foundation_models_bridge)]
                    _tools: None,
                }
            }
        })
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        #[cfg(foundation_models_bridge)]
        unsafe {
            fm_session_destroy(self.handle);
        }
    }
}

// ─── ResponseStream ────────────────────────────────────────────────────────────

/// An async stream of text chunks produced by [`Session::stream`].
pub struct ResponseStream {
    rx: mpsc::UnboundedReceiver<Result<String, String>>,
}

impl Stream for ResponseStream {
    type Item = Result<String, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut StdContext<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.rx)
            .poll_next(cx)
            .map(|opt| opt.map(|r| r.map_err(Error::Generation)))
    }
}

/// An async stream of JSON chunks produced by [`Session::stream_structured`].
///
/// Each item is `Ok(String)` containing a JSON fragment of the structured output,
/// or `Err(Error)` if generation failed. The stream ends when the model finishes.
///
/// Implements [`futures_core::Stream`]; use with `.next()` from `StreamExt`.
pub struct ResponseStructuredStream {
    rx: mpsc::UnboundedReceiver<Result<String, String>>,
}

impl Stream for ResponseStructuredStream {
    type Item = Result<String, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut StdContext<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.rx)
            .poll_next(cx)
            .map(|opt| opt.map(|r| r.map_err(Error::Generation)))
    }
}

// ─── FFI callbacks ─────────────────────────────────────────────────────────────

#[cfg(foundation_models_bridge)]
extern "C" fn respond_callback(ctx: *mut c_void, result: *const c_char, error: *const c_char) {
    let tx = unsafe { Box::from_raw(ctx as *mut oneshot::Sender<Result<String, String>>) };

    if !error.is_null() {
        let msg = unsafe { CStr::from_ptr(error).to_string_lossy().into_owned() };
        tx.send(Err(msg)).ok();
    } else if !result.is_null() {
        let text = unsafe { CStr::from_ptr(result).to_string_lossy().into_owned() };
        tx.send(Ok(text)).ok();
    }
}

#[cfg(foundation_models_bridge)]
struct StreamContext {
    tx: mpsc::UnboundedSender<Result<String, String>>,
}

#[cfg(foundation_models_bridge)]
extern "C" fn stream_token_callback(ctx: *mut c_void, token: *const c_char) {
    let stream_ctx = unsafe { &*(ctx as *const StreamContext) };
    let text = unsafe { CStr::from_ptr(token).to_string_lossy().into_owned() };
    stream_ctx.tx.unbounded_send(Ok(text)).ok();
}

#[cfg(foundation_models_bridge)]
extern "C" fn stream_done_callback(ctx: *mut c_void, error: *const c_char) {
    let stream_ctx = unsafe { Box::from_raw(ctx as *mut StreamContext) };
    if !error.is_null() {
        let msg = unsafe { CStr::from_ptr(error).to_string_lossy().into_owned() };
        stream_ctx.tx.unbounded_send(Err(msg)).ok();
    }
}

#[cfg(foundation_models_bridge)]
extern "C" fn tool_dispatch(
    ctx: *mut c_void,
    name_ptr: *const c_char,
    args_ptr: *const c_char,
    result_ctx: *mut c_void,
    result_cb: extern "C" fn(*mut c_void, *const c_char, *const c_char),
) {
    let tools = unsafe { &*(ctx as *const ToolsContext) };
    let name = unsafe { CStr::from_ptr(name_ptr).to_string_lossy() };
    let args_str = unsafe { CStr::from_ptr(args_ptr).to_string_lossy() };

    let args: serde_json::Value = match serde_json::from_str(&args_str) {
        Ok(v) => v,
        Err(e) => {
            let msg = format!("invalid tool args JSON: {e}");
            if let Ok(c) = CString::new(msg) {
                result_cb(result_ctx, null(), c.as_ptr());
            }
            return;
        }
    };

    match tools.tools.iter().find(|(n, _)| n == name.as_ref()) {
        Some((_, handler)) => match handler(args) {
            Ok(result) => {
                if let Ok(c) = CString::new(result) {
                    result_cb(result_ctx, c.as_ptr(), null());
                }
            }
            Err(err) => {
                if let Ok(c) = CString::new(err) {
                    result_cb(result_ctx, null(), c.as_ptr());
                }
            }
        },
        None => {
            let msg = format!("unknown tool: {name}");
            if let Ok(c) = CString::new(msg) {
                result_cb(result_ctx, null(), c.as_ptr());
            }
        }
    }
}

// ─── LanguageModel integration ─────────────────────────────────────────────────

use async_trait::async_trait;
use rs_ai_core::{
    AiError, AiResult, AiStream, Capability, CapabilitySet, FinishReason,
    GenerateOptions as TraitGenerateOptions, GenerateResult, LanguageModel, Prompt,
    ResponseMetadata as TraitResponseMetadata, SyntheticStreamer, Usage,
};

// ─── LanguageModel integration ─────────────────────────────────────────────────

/// A [`LanguageModel`] backed by Apple Foundation Models running on-device.
pub struct FoundationModel {
    capabilities: CapabilitySet,
}

impl FoundationModel {
    /// Creates a new Foundation Model.
    pub fn new() -> Self {
        let capabilities = CapabilitySet::new()
            .with(Capability::TextInput)
            .with(Capability::TextOutput)
            .with(Capability::LocalExecution)
            .with(Capability::PlatformNative);
        Self { capabilities }
    }

    fn build_config(options: &TraitGenerateOptions) -> GenerationOptions {
        GenerationOptions {
            temperature: options.temperature,
            max_tokens: options.max_tokens.map(|n| n as usize),
        }
    }

    async fn ensure_available() -> AiResult<()> {
        match availability() {
            Ok(()) => Ok(()),
            Err(reason) => Err(AiError::PlatformUnavailable {
                platform: format!("apple/foundation_models: {reason}"),
            }),
        }
    }
}

impl Default for FoundationModel {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LanguageModel for FoundationModel {
    fn model_id(&self) -> &str {
        "apple-foundation-model"
    }

    fn provider_id(&self) -> &str {
        "foundationmodels"
    }

    fn capabilities(&self) -> &CapabilitySet {
        &self.capabilities
    }

    async fn generate(
        &self,
        prompt: Prompt,
        options: TraitGenerateOptions,
    ) -> AiResult<GenerateResult> {
        let prompt_text = crate::text_prompt(&prompt, "foundationmodels")?;
        Self::ensure_available().await?;

        let config = Self::build_config(&options);
        let start = std::time::Instant::now();

        let session = Session::new().map_err(|e| AiError::BridgeError {
            bridge: "foundationmodels".into(),
            message: e.to_string(),
        })?;

        let response = session
            .respond_with_options(&prompt_text, &config)
            .await
            .map_err(|e| AiError::BridgeError {
                bridge: "foundationmodels".into(),
                message: e.to_string(),
            })?;

        let latency_ms = start.elapsed().as_millis() as u64;

        Ok(GenerateResult {
            text: Some(response),
            tool_calls: vec![],
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
            metadata: TraitResponseMetadata {
                provider: "foundationmodels".into(),
                model: "apple-foundation-model".into(),
                latency_ms: Some(latency_ms),
                ..Default::default()
            },
            steps: Vec::new(),
            reasoning: None,
        })
    }

    async fn stream(&self, prompt: Prompt, options: TraitGenerateOptions) -> AiResult<AiStream> {
        let prompt_text = crate::text_prompt(&prompt, "foundationmodels")?;
        Self::ensure_available().await?;

        let config = Self::build_config(&options);

        let session = Session::new().map_err(|e| AiError::BridgeError {
            bridge: "foundationmodels".into(),
            message: e.to_string(),
        })?;

        let mut stream = session
            .stream_with_options(&prompt_text, &config)
            .map_err(|e| AiError::BridgeError {
                bridge: "foundationmodels".into(),
                message: e.to_string(),
            })?;

        // Collect all chunks from the stream
        let chunks: Vec<String> = {
            let mut v = Vec::new();
            use futures::StreamExt;
            while let Some(item) = stream.next().await {
                match item {
                    Ok(text) => v.push(text),
                    Err(e) => {
                        return Err(AiError::BridgeError {
                            bridge: "foundationmodels".into(),
                            message: e.to_string(),
                        });
                    }
                }
            }
            v
        };

        let text = chunks.join("");
        Ok(SyntheticStreamer::stream(text, 20))
    }
}

// ─── Provider integration ──────────────────────────────────────────────────────

use rs_ai_core::{EmbeddingModel, ModelInfo, Provider};

/// Provider for Apple Foundation Models on-device inference.
pub struct FoundationModelProvider;

impl FoundationModelProvider {
    /// Creates a new provider.
    pub fn new() -> Self {
        Self
    }

    /// Gets the Foundation Model.
    pub fn model(&self) -> FoundationModel {
        FoundationModel::new()
    }
}

impl Default for FoundationModelProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Provider for FoundationModelProvider {
    fn id(&self) -> &str {
        "foundationmodels"
    }

    fn name(&self) -> &str {
        "Apple Foundation Models"
    }

    fn language_model(&self, _model_id: &str) -> AiResult<Box<dyn LanguageModel>> {
        Ok(Box::new(self.model()))
    }

    fn embedding_model(&self, model_id: &str) -> AiResult<Box<dyn EmbeddingModel>> {
        Err(AiError::UnsupportedCapability {
            capability: "embeddings".into(),
            provider: format!("foundationmodels/{model_id}"),
        })
    }

    fn available_models(&self) -> Vec<ModelInfo> {
        vec![ModelInfo {
            id: "apple-foundation-model".into(),
            provider: "foundationmodels".into(),
            display_name: "Apple Foundation Model (On-Device)".into(),
            capabilities: CapabilitySet::new()
                .with(Capability::TextInput)
                .with(Capability::TextOutput)
                .with(Capability::LocalExecution)
                .with(Capability::PlatformNative),
            ..Default::default()
        }]
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_available_returns_without_panic() {
        let _ = is_available();
    }

    #[test]
    fn test_options_default_is_valid() {
        let opts = GenerationOptions::default();
        assert!(opts.validate().is_ok());
    }

    #[test]
    fn test_session_creation_fails_gracefully_when_unavailable() {
        if is_available() {
            return;
        }
        let err = Session::new().unwrap_err();
        assert!(matches!(err, Error::Unavailable(_)));
    }

    #[test]
    fn test_schema_builder() {
        let schema = Schema::new("Point")
            .property(SchemaProperty::new("x", SchemaPropertyType::Double))
            .property(SchemaProperty::new("y", SchemaPropertyType::Double));
        assert_eq!(schema.name, "Point");
        assert_eq!(schema.properties.len(), 2);
    }

    #[test]
    fn test_provider_available_models() {
        let provider = FoundationModelProvider::new();
        let models = provider.available_models();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "apple-foundation-model");
    }

    #[test]
    fn test_language_model_traits() {
        let model = FoundationModel::new();
        assert_eq!(model.model_id(), "apple-foundation-model");
        assert_eq!(model.provider_id(), "foundationmodels");
        assert!(model.capabilities().has(&Capability::TextInput));
        assert!(model.capabilities().has(&Capability::TextOutput));
        assert!(model.capabilities().has(&Capability::LocalExecution));
        assert!(model.capabilities().has(&Capability::PlatformNative));
    }
}

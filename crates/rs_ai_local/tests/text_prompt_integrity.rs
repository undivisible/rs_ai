#![cfg(any(
    feature = "browser",
    feature = "phi-silica",
    feature = "foundationmodels"
))]

use rs_ai_core::{
    AiError, ContentPart, FileData, GenerateOptions, ImageData, LanguageModel, Message, Prompt,
};

async fn rejects_non_text(model: &dyn LanguageModel) {
    let image = Message::user("describe").with_image(ImageData::Base64 {
        media_type: "image/png".into(),
        data: "AQIDBA==".into(),
    });
    let mut file = Message::user("read");
    file.content.push(ContentPart::File {
        data: FileData::Url {
            url: "https://example.org/file.txt".into(),
            media_type: None,
        },
    });
    for message in [
        image,
        file,
        Message::tool_result("call-1", "important result"),
    ] {
        let prompt = Prompt::Messages(vec![message]);
        assert!(matches!(
            model
                .generate(prompt.clone(), GenerateOptions::default())
                .await,
            Err(AiError::UnsupportedCapability { .. })
        ));
        assert!(matches!(
            model.stream(prompt, GenerateOptions::default()).await,
            Err(AiError::UnsupportedCapability { .. })
        ));
    }
}

#[cfg(feature = "phi-silica")]
#[tokio::test]
async fn phi_silica_rejects_non_text_prompt_parts() {
    let model = rs_ai_local::phi_silica::PhiSilicaProvider::new(
        rs_ai_local::phi_silica::MockPhiSilicaBridge,
    )
    .model();
    rejects_non_text(&model).await;
    let result = model
        .generate(Prompt::Text("hello".into()), GenerateOptions::default())
        .await;
    assert!(
        matches!(result, Ok(result) if result.text.as_deref().is_some_and(|text| text.contains("hello")))
    );
}

#[cfg(feature = "browser")]
struct AvailableBrowser;

#[cfg(feature = "browser")]
#[async_trait::async_trait]
impl rs_ai_local::browser::BrowserAiBridge for AvailableBrowser {
    async fn detect(&self) -> rs_ai_local::browser::BrowserAiCapabilities {
        rs_ai_local::browser::BrowserAiCapabilities {
            available: true,
            ..Default::default()
        }
    }
    async fn generate(
        &self,
        prompt: &str,
        _: &rs_ai_local::browser::BrowserAiOptions,
    ) -> Result<String, String> {
        Ok(prompt.into())
    }
    async fn stream(
        &self,
        prompt: &str,
        _: &rs_ai_local::browser::BrowserAiOptions,
    ) -> Result<Vec<String>, String> {
        Ok(vec![prompt.into()])
    }
}

#[cfg(feature = "browser")]
#[tokio::test]
async fn browser_rejects_non_text_prompt_parts() {
    let model = rs_ai_local::browser::BrowserAiProvider::new(AvailableBrowser).model();
    rejects_non_text(&model).await;
    let result = model
        .generate(Prompt::Text("hello".into()), GenerateOptions::default())
        .await;
    assert!(matches!(result, Ok(result) if result.text.as_deref() == Some("hello")));
}

#[cfg(feature = "foundationmodels")]
#[tokio::test]
async fn foundation_model_rejects_non_text_prompt_parts() {
    rejects_non_text(&rs_ai_local::foundationmodels::FoundationModel::new()).await;
}

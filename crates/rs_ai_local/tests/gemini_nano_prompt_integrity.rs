#![cfg(feature = "gemini-nano")]

use std::sync::Arc;

use rs_ai_core::{
    AiError, ContentPart, FileData, GenerateOptions, ImageData, LanguageModel, Message, Prompt,
};
use rs_ai_local::gemini_nano::{GeminiNanoModel, MockNanoBridge};

#[tokio::test]
async fn nano_rejects_parts_it_cannot_send_to_the_bridge() {
    let model = GeminiNanoModel::new(Arc::new(MockNanoBridge));
    let mut file = Message::user("read the file");
    file.content.push(ContentPart::File {
        data: FileData::Url {
            url: "https://example.org/file.txt".into(),
            media_type: None,
        },
    });
    let image = Message::user("describe the image").with_image(ImageData::Url {
        url: "https://example.org/image.png".into(),
        detail: None,
    });
    let tool_result = Message::tool_result("call-1", "important result");
    for message in [image, file, tool_result] {
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
    let supported = Prompt::Messages(vec![Message::user("describe").with_image(
        ImageData::Base64 {
            media_type: "image/png".into(),
            data: "AQIDBA==".into(),
        },
    )]);
    let result = model.generate(supported, GenerateOptions::default()).await;
    assert!(
        matches!(result, Ok(result) if result.text.as_deref().is_some_and(|text| text.contains("[image]")))
    );
}

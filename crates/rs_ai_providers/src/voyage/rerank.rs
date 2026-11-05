use async_trait::async_trait;
use rs_ai_core::{
    AiError, AiResult, RerankOptions, RerankResult, RerankedDocument, RerankingModel, Usage,
};

/// Voyage AI reranking model implementing the `RerankingModel` trait.
///
/// Voyage's API is simpler than Cohere's — it only supports reranking and
/// embeddings, not chat/completions.
pub struct VoyageRerankingModel {
    model_id: String,
    api_key: String,
}

impl VoyageRerankingModel {
    /// Create a new Voyage reranking model.
    ///
    /// * `model_id` — e.g. `"rerank-2"` or `"rerank-2-lite"`
    /// * `api_key` — Voyage API key (from `VOYAGE_API_KEY` env var)
    pub fn new(model_id: &str, api_key: &str) -> Self {
        Self {
            model_id: model_id.to_string(),
            api_key: api_key.to_string(),
        }
    }
}

#[async_trait]
impl RerankingModel for VoyageRerankingModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn provider_id(&self) -> &str {
        "voyage"
    }

    async fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
        options: RerankOptions,
    ) -> AiResult<RerankResult> {
        let client = reqwest::Client::new();

        let mut body = serde_json::json!({
            "model": self.model_id,
            "query": query,
            "documents": documents,
        });

        if let Some(top_k) = options.top_k {
            body["top_k"] = serde_json::json!(top_k);
        }

        let resp = client
            .post("https://api.voyageai.com/v1/rerank")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| AiError::Transport {
                message: format!("Voyage rerank request failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        let data: serde_json::Value = resp.json().await.map_err(|e| AiError::Transport {
            message: format!("Voyage rerank response parse failed: {e}"),
            source: Some(Box::new(e)),
        })?;

        let results = parse_results(&data)?;

        // Voyage API doesn't return token usage; use defaults.
        Ok(RerankResult {
            results,
            usage: Usage::default(),
        })
    }
}

fn parse_results(data: &serde_json::Value) -> AiResult<Vec<RerankedDocument>> {
    let results = data["results"]
        .as_array()
        .ok_or_else(|| AiError::Transport {
            message: "Voyage rerank: missing results array in response".into(),
            source: None,
        })?
        .iter()
        .map(|r| {
            // Voyage returns `document` as a plain string, not an object with `text`.
            let doc = r["document"].as_str().map(String::from);
            let index = r["index"]
                .as_u64()
                .and_then(|index| usize::try_from(index).ok())
                .ok_or_else(|| {
                    AiError::Serialization("Voyage rerank: invalid or missing index".into())
                })?;
            let score = r["relevance_score"].as_f64().ok_or_else(|| {
                AiError::Serialization("Voyage rerank: invalid or missing relevance score".into())
            })?;
            Ok(RerankedDocument {
                index,
                score,
                document: doc,
            })
        })
        .collect::<AiResult<_>>()?;
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voyage_model_identity() {
        let model = VoyageRerankingModel::new("rerank-2", "test-key");
        assert_eq!(model.provider_id(), "voyage");
        assert_eq!(model.model_id(), "rerank-2");
    }

    #[test]
    fn test_voyage_model_lite() {
        let model = VoyageRerankingModel::new("rerank-2-lite", "test-key");
        assert_eq!(model.model_id(), "rerank-2-lite");
    }

    #[test]
    fn missing_rerank_values_are_errors() {
        for response in [
            serde_json::json!({"results":[{"relevance_score":0.72}]}),
            serde_json::json!({"results":[{"index":1}]}),
        ] {
            assert!(parse_results(&response).is_err());
        }
        let response =
            serde_json::json!({"results":[{"index":1,"relevance_score":0.72,"document":"second"}]});
        let results = parse_results(&response).unwrap();
        assert_eq!(results[0].index, 1);
        assert_eq!(results[0].score, 0.72);
        assert_eq!(results[0].document.as_deref(), Some("second"));
    }
}

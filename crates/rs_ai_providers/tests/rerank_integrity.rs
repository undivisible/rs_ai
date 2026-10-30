#![cfg(feature = "cohere")]

use rs_ai_core::{RerankOptions, RerankingModel};
use rs_ai_providers::cohere::CohereRerankingModel;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::test]
async fn cohere_does_not_default_missing_scores() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        for body in [
            r#"{"results":[{"index":1}]}"#,
            r#"{"results":[{"index":1,"relevance_score":0.72}]}"#,
        ] {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = [0; 8192];
            let _ = socket.read(&mut buffer).await;
            let response = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body);
            socket.write_all(response.as_bytes()).await.unwrap();
        }
    });
    let model = CohereRerankingModel::new("test", "test", &format!("http://{address}"));
    let docs = vec!["first".into(), "second".into()];
    assert!(model
        .rerank("query", docs.clone(), RerankOptions::default())
        .await
        .is_err());
    let result = model.rerank("query", docs, RerankOptions::default()).await;
    assert!(
        matches!(result, Ok(result) if result.results[0].index == 1 && result.results[0].score == 0.72)
    );
    server.await.unwrap();
}

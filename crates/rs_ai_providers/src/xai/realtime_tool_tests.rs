use super::*;

#[tokio::test]
async fn malformed_realtime_tool_arguments_are_errors() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(socket).await.unwrap();
        for event in [
            r#"{"type":"response.tool_call","call_id":"call-1","name":"lookup","arguments":"{\"amount\":17}"}"#,
            r#"{"type":"response.tool_call","call_id":"call-2","name":"lookup","arguments":"{bad"}"#,
            r#"{"type":"response.tool_call","call_id":"call-3","name":"lookup"}"#,
        ] {
            socket.send(WsMessage::Text(event.into())).await.unwrap();
        }
    });
    let (socket, _) = connect_async(format!("ws://{address}")).await.unwrap();
    let mut session = GrokVoiceSession {
        write: Arc::new(Mutex::new(socket)),
        model: "test".into(),
    };
    assert!(
        matches!(session.recv().await, Some(RealtimeEvent::ToolCall { arguments, .. }) if arguments == serde_json::json!({"amount":17}))
    );
    assert!(matches!(
        session.recv().await,
        Some(RealtimeEvent::Error { .. })
    ));
    assert!(matches!(
        session.recv().await,
        Some(RealtimeEvent::Error { .. })
    ));
    server.await.unwrap();
}

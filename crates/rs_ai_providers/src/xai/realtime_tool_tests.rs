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
        state: ConnectionState::Open,
        response_has_tool_calls: false,
        response_has_invalid_tool_calls: false,
        turn_complete_pending: false,
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

#[tokio::test]
async fn tool_outputs_resume_and_lifecycle() {
    use futures::StreamExt;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(socket).await.unwrap();
        for id in ["one", "two"] {
            let message = socket.next().await.unwrap().unwrap().into_text().unwrap();
            let event: serde_json::Value = serde_json::from_str(&message).unwrap();
            assert_eq!(
                event,
                serde_json::json!({"type":"conversation.item.create","item":{"type":"function_call_output","call_id":id,"output":"{\"ok\":true}"}})
            );
        }
        let message = socket.next().await.unwrap().unwrap().into_text().unwrap();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&message).unwrap(),
            serde_json::json!({"type":"response.create"})
        );
        for event in [
            serde_json::json!({"type":"response.output_text.delta","delta":"hi"}),
            serde_json::json!({"type":"response.output_audio.delta","delta":"AQI="}),
            serde_json::json!({"type":"input_audio_buffer.speech_started"}),
            serde_json::json!({"type":"response.done","response":{"status":"cancelled"}}),
            serde_json::json!({"type":"response.done","response":{"status":"completed"}}),
            serde_json::json!({"type":"response.done","response":{"status":"failed"}}),
            serde_json::json!({"type":"response.done","response":{"status":"incomplete"}}),
            serde_json::json!({"type":"response.done","response":{}}),
        ] {
            socket
                .send(WsMessage::Text(event.to_string()))
                .await
                .unwrap();
        }
        socket.close(None).await.unwrap();
    });
    let (socket, _) = connect_async(format!("ws://{address}")).await.unwrap();
    let mut session = GrokVoiceSession {
        write: Arc::new(Mutex::new(socket)),
        model: "test".into(),
        state: ConnectionState::Open,
        response_has_tool_calls: false,
        response_has_invalid_tool_calls: false,
        turn_complete_pending: false,
    };
    for id in ["one", "two"] {
        session
            .send_tool_result(id, "lookup", serde_json::json!({"ok":true}))
            .await
            .unwrap();
    }
    session.resume().await.unwrap();
    assert!(
        matches!(session.recv().await, Some(RealtimeEvent::TextDelta { delta }) if delta == "hi")
    );
    assert!(
        matches!(session.recv().await, Some(RealtimeEvent::AudioDelta { delta }) if delta == [1,2])
    );
    assert!(matches!(
        session.recv().await,
        Some(RealtimeEvent::SpeechStarted)
    ));
    assert!(matches!(
        session.recv().await,
        Some(RealtimeEvent::Interrupted)
    ));
    assert!(matches!(
        session.recv().await,
        Some(RealtimeEvent::TurnComplete)
    ));
    for _ in 0..3 {
        assert!(matches!(
            session.recv().await,
            Some(RealtimeEvent::Error { .. })
        ));
    }
    assert!(matches!(
        session.recv().await,
        Some(RealtimeEvent::SessionClosed)
    ));
    assert!(session.recv().await.is_none());
    server.await.unwrap();
}

#[tokio::test]
async fn batch_boundary_follows_completed_response_only() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(socket).await.unwrap();
        for event in [
            serde_json::json!({"type":"response.function_call_arguments.done","call_id":"one","name":"lookup","arguments":"{}"}),
            serde_json::json!({"type":"response.function_call_arguments.done","call_id":"two","name":"lookup","arguments":"{}"}),
            serde_json::json!({"type":"response.done","response":{"status":"completed"}}),
            serde_json::json!({"type":"response.function_call_arguments.done","call_id":"three","name":"lookup","arguments":"{}"}),
            serde_json::json!({"type":"response.done","response":{"status":"cancelled"}}),
            serde_json::json!({"type":"response.done","response":{"status":"completed"}}),
        ] {
            socket
                .send(WsMessage::Text(event.to_string()))
                .await
                .unwrap();
        }
        socket.close(None).await.unwrap();
    });
    let (socket, _) = connect_async(format!("ws://{address}")).await.unwrap();
    let mut session = GrokVoiceSession {
        write: Arc::new(Mutex::new(socket)),
        model: "test".into(),
        state: ConnectionState::Open,
        response_has_tool_calls: false,
        response_has_invalid_tool_calls: false,
        turn_complete_pending: false,
    };
    for expected in ["one", "two"] {
        assert!(
            matches!(session.recv().await, Some(RealtimeEvent::ToolCall { id, .. }) if id == expected)
        );
    }
    assert!(matches!(
        session.recv().await,
        Some(RealtimeEvent::ToolCallBatchComplete)
    ));
    assert!(matches!(
        session.recv().await,
        Some(RealtimeEvent::TurnComplete)
    ));
    assert!(
        matches!(session.recv().await, Some(RealtimeEvent::ToolCall { id, .. }) if id == "three")
    );
    assert!(matches!(
        session.recv().await,
        Some(RealtimeEvent::Interrupted)
    ));
    assert!(matches!(
        session.recv().await,
        Some(RealtimeEvent::TurnComplete)
    ));
    assert!(matches!(
        session.recv().await,
        Some(RealtimeEvent::SessionClosed)
    ));
    assert!(session.recv().await.is_none());
    server.await.unwrap();
}

#[tokio::test]
async fn malformed_call_suppresses_batch_boundary_until_next_response() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(socket).await.unwrap();
        for event in [
            serde_json::json!({"type":"response.function_call_arguments.done","call_id":"one","name":"lookup","arguments":"{}"}),
            serde_json::json!({"type":"response.function_call_arguments.done","call_id":"two","name":"lookup","arguments":"{bad"}),
            serde_json::json!({"type":"response.function_call_arguments.done","call_id":"three","name":"lookup","arguments":"{}"}),
            serde_json::json!({"type":"response.done","response":{"status":"completed"}}),
            serde_json::json!({"type":"response.function_call_arguments.done","call_id":"four","name":"lookup","arguments":"{}"}),
            serde_json::json!({"type":"response.done","response":{"status":"completed"}}),
        ] {
            socket
                .send(WsMessage::Text(event.to_string()))
                .await
                .unwrap();
        }
        socket.close(None).await.unwrap();
    });
    let (socket, _) = connect_async(format!("ws://{address}")).await.unwrap();
    let mut session = GrokVoiceSession {
        write: Arc::new(Mutex::new(socket)),
        model: "test".into(),
        state: ConnectionState::Open,
        response_has_tool_calls: false,
        response_has_invalid_tool_calls: false,
        turn_complete_pending: false,
    };
    assert!(
        matches!(session.recv().await, Some(RealtimeEvent::ToolCall { id, .. }) if id == "one")
    );
    assert!(matches!(
        session.recv().await,
        Some(RealtimeEvent::Error { .. })
    ));
    assert!(
        matches!(session.recv().await, Some(RealtimeEvent::ToolCall { id, .. }) if id == "three")
    );
    assert!(matches!(
        session.recv().await,
        Some(RealtimeEvent::TurnComplete)
    ));
    assert!(
        matches!(session.recv().await, Some(RealtimeEvent::ToolCall { id, .. }) if id == "four")
    );
    assert!(matches!(
        session.recv().await,
        Some(RealtimeEvent::ToolCallBatchComplete)
    ));
    assert!(matches!(
        session.recv().await,
        Some(RealtimeEvent::TurnComplete)
    ));
    assert!(matches!(
        session.recv().await,
        Some(RealtimeEvent::SessionClosed)
    ));
    server.await.unwrap();
}

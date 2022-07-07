use std::sync::{Arc, Mutex};

use axum::{http::StatusCode, response::IntoResponse, Extension, Json};
use serde_json::json;

use crate::ServerState;

use super::Session;

pub async fn create_session(state: Extension<Arc<Mutex<ServerState>>>) -> impl IntoResponse {
    let session = Session::new();
    let id = session.id.clone();
    state.lock().unwrap().sessions.push(session);
    (StatusCode::CREATED, Json(json!({ "id": id })))
}

use std::sync::{Arc, Mutex};

use axum::{
    routing::{get, post},
    Extension, Router,
};
use fights_srv::{session::handlers::create_session, ServerState};

#[tokio::main]
async fn main() {
    let state = Arc::new(Mutex::new(ServerState::new()));

    let app = Router::new()
        .route("/", get(|| async { "Hello, World!" }))
        .route("/sessions", post(create_session))
        .layer(Extension(state));

    // run it with hyper on localhost:3000
    axum::Server::bind(&"0.0.0.0:3000".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}

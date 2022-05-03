use std::io;

use actix_web::{web, App, HttpServer};
use fights_srv::room::config_routes;

#[actix_web::main]
async fn main() -> io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .route("/", web::get().to(|| async { "Hello World!" }))
            .configure(config_routes)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}

use actix_web::{get, web, HttpResponse, Responder};

use crate::user::User;

#[get("/rooms")]
async fn list() -> impl Responder {
    HttpResponse::Ok().json(vec![
        User {
            id: "1".to_string(),
        },
        User {
            id: "2".to_string(),
        },
    ])
}

pub fn config_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(list);
}

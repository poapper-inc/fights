use actix_web::{get, web, HttpResponse, Responder};

use crate::agent::Agent;

#[get("/rooms")]
async fn list() -> impl Responder {
    HttpResponse::Ok().json(vec![
        Agent {
            id: "1".to_string(),
        },
        Agent {
            id: "2".to_string(),
        },
    ])
}

pub fn config_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(list);
}

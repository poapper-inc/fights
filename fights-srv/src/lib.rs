pub mod game;
pub mod room;
pub mod user;

use pyo3::prelude::*;

use actix_web::{web, App, HttpServer};
use user::User;

#[pyo3::pyfunction]
fn return_one() -> i32 {
    1
}

#[actix_web::main]
#[pyo3::pyfunction]
async fn start_server() -> PyResult<()> {
    HttpServer::new(|| App::new().route("/", web::get().to(|| async { "Hello World!" })))
        .bind(("127.0.0.1", 8080))?
        .run()
        .await?;
    Ok(())
}

#[pymodule]
fn fights_srv(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(start_server, m)?)?;
    m.add_function(wrap_pyfunction!(return_one, m)?)?;
    Ok(())
}

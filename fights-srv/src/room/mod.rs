use crate::User;
pub mod routes;

pub struct Room {
    pub id: String,
    pub users: Vec<User>,
}

pub use routes::config_routes;

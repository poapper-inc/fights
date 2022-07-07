pub mod agent;
pub mod env;
pub mod ndarray;
pub mod session;

use crate::session::Session;

pub struct ServerState {
    pub sessions: Vec<Session>,
}

impl ServerState {
    pub fn new() -> ServerState {
        ServerState {
            sessions: Vec::new(),
        }
    }
}

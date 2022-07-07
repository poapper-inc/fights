use uuid::Uuid;

use crate::agent::Agent;
pub mod handlers;

#[derive(Clone)]
pub struct Session {
    pub id: String,
    pub agents: Vec<Agent>,
}

impl Session {
    pub fn new() -> Session {
        Session {
            id: Uuid::new_v4().to_string(),
            agents: Vec::new(),
        }
    }
}

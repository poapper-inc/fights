use crate::agent::Agent;
pub mod routes;

pub struct Room {
    pub id: String,
    pub agents: Vec<Agent>,
}

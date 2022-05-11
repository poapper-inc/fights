use crate::agent::Agent;

pub mod gomoku;

pub trait Env {
    type Action;
    type State;
    fn step(&mut self, agent: &Agent, action: Self::Action) -> Result<Self::State>;
    fn reset(&mut self) -> Result<Self::State>;
}

#[derive(Debug, Clone)]
pub struct Result<S> {
    pub state: S,
    pub reward: f64,
    pub done: bool,
    pub info: String,
}

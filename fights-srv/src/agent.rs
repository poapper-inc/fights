use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, Clone)]
pub struct Agent {
    pub id: String,
}

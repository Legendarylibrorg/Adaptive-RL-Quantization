//! Core simulator metrics (matches Python ``SimulatorBackend`` without MoE).

pub mod evaluate;
pub mod types;

pub use evaluate::evaluate_episode;
pub use types::{EvaluateRequest, MetricsOut};

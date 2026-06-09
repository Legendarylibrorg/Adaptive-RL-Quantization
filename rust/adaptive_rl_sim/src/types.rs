use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
pub struct EvaluateRequest {
    pub hardware: HardwareIn,
    pub input_features: InputFeaturesIn,
    pub sensitivity: SensitivityIn,
    pub decision: DecisionIn,
    #[serde(default)]
    pub calibration: HashMap<String, CalibrationIn>,
}

#[derive(Debug, Deserialize)]
pub struct HardwareIn {
    pub hardware_type: String,
    pub compute_factor: f64,
    pub throughput_bias: f64,
    pub latency_bias: f64,
    pub memory_budget_mb: f64,
    pub preferred_bits: f64,
    pub kernel_uniformity_preference: f64,
}

#[derive(Debug, Deserialize)]
pub struct InputFeaturesIn {
    pub prompt_length: i64,
    pub complexity_score: f64,
}

#[derive(Debug, Deserialize)]
pub struct SensitivityIn {
    pub layer_stats: Vec<f64>,
}

#[derive(Debug, Deserialize)]
pub struct DecisionIn {
    pub mode: String,
    pub scale_factor: f64,
    pub clipping_range: f64,
    #[serde(default)]
    pub effective_layer_bits: Vec<f64>,
}

#[derive(Debug, Deserialize, Default)]
pub struct CalibrationIn {
    #[serde(default = "default_one")]
    pub latency_multiplier: f64,
    #[serde(default = "default_one")]
    pub throughput_multiplier: f64,
    #[serde(default = "default_one")]
    pub memory_multiplier: f64,
}

fn default_one() -> f64 {
    1.0
}

#[derive(Debug, Serialize)]
pub struct MetricsOut {
    pub latency_ms: f64,
    pub throughput_tps: f64,
    pub perplexity: f64,
    pub memory_mb: f64,
    pub swap_cost_ms: f64,
    pub cache_miss_count: f64,
    pub variant_churn: f64,
    pub tokens_processed: f64,
    pub latency_ms_per_token: f64,
    pub latency_source: String,
    pub throughput_source: String,
    pub memory_source: String,
    pub perplexity_source: String,
    pub simulator_engine: String,
}

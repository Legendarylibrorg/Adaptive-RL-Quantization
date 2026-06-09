use crate::types::{DecisionIn, EvaluateRequest, HardwareIn, MetricsOut};

fn clamp(value: f64, low: f64, high: f64) -> f64 {
    value.clamp(low, high)
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        0.0
    } else {
        let m = mean(values);
        values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / values.len() as f64
    }
}

fn mode_bonus(mode: &str) -> f64 {
    match mode {
        "discrete" => 0.10,
        "grouped" => 0.16,
        "per_layer" => 0.18,
        "dynamic" => 0.28,
        "learned" => 0.34,
        _ => 0.10,
    }
}

fn evaluate_core(req: &EvaluateRequest) -> MetricsOut {
    let hardware = &req.hardware;
    let decision = &req.decision;
    let avg_bits = mean(&decision.effective_layer_bits);
    let bit_variance = variance(&decision.effective_layer_bits);
    let complexity = req.input_features.complexity_score;
    let sensitivity = mean(&req.sensitivity.layer_stats);
    let prompt_length = req.input_features.prompt_length.max(8) as f64;
    let compression = ((8.0 - avg_bits) / 6.0).max(0.0);
    let bonus = mode_bonus(&decision.mode);

    let mut latency_ms = 8.5 * prompt_length * hardware.latency_bias
        / (0.35_f64).max(hardware.compute_factor + (8.0 - avg_bits) * 0.12 + bonus);
    latency_ms *= 1.0
        + complexity * 0.55
        + (bit_variance - hardware.kernel_uniformity_preference).max(0.0) * 0.18;

    let mut throughput_tps = 140.0
        * hardware.throughput_bias
        * (1.0 + (8.0 - avg_bits) * 0.10 + bonus * 0.40)
        / (1.0 + complexity * 0.80 + hardware.latency_bias * 0.08);

    if hardware.hardware_type == "gpu" {
        throughput_tps *= 1.0 - (bit_variance * 0.03).min(0.12);
    } else {
        throughput_tps *= 1.0 + (hardware.preferred_bits - avg_bits).max(0.0) * 0.02;
    }

    let mut memory_mb = 4800.0 * (avg_bits / 16.0) * (1.0 + complexity * 0.15);
    if decision.mode == "per_layer" || decision.mode == "learned" {
        memory_mb *= 1.02;
    }

    let mut perplexity = 5.6
        + complexity * 3.4
        + (5.5 - avg_bits).max(0.0) * (0.60 + complexity * 0.90 + sensitivity * 0.35)
        + (1.0 - decision.scale_factor).abs() * 0.65
        + (1.05 - decision.clipping_range).max(0.0) * 1.20
        - bonus * 0.70;

    let hardware_alignment = (avg_bits - hardware.preferred_bits).abs();
    latency_ms *= 1.0 + hardware_alignment * 0.04;
    throughput_tps *= 1.0 - hardware_alignment * 0.02;
    perplexity += hardware_alignment * 0.15;

    apply_hardware_alignment(
        hardware,
        avg_bits,
        &mut latency_ms,
        &mut throughput_tps,
        &mut memory_mb,
        &mut perplexity,
    );
    apply_mode_adjustments(
        decision,
        compression,
        complexity,
        sensitivity,
        hardware,
        &mut latency_ms,
        &mut throughput_tps,
        &mut memory_mb,
        &mut perplexity,
    );
    apply_memory_overflow(hardware, memory_mb, &mut latency_ms, &mut throughput_tps, &mut perplexity);

    let mut latency_ms = clamp(latency_ms, 5.0, 20_000.0);
    let mut throughput_tps = clamp(throughput_tps, 1.0, 10_000.0);
    let mut perplexity = clamp(perplexity, 3.0, 100.0);
    let mut memory_mb = clamp(memory_mb, 200.0, 128_000.0);

    if let Some(cal) = req.calibration.get(&hardware.hardware_type) {
        if cal.latency_multiplier > 0.0 {
            latency_ms = clamp(latency_ms * cal.latency_multiplier, 1.0, 60_000.0);
        }
        if cal.throughput_multiplier > 0.0 {
            throughput_tps = clamp(throughput_tps * cal.throughput_multiplier, 0.1, 100_000.0);
        }
        if cal.memory_multiplier > 0.0 {
            memory_mb = clamp(memory_mb * cal.memory_multiplier, 50.0, 512_000.0);
        }
    }

    let tokens_processed = prompt_length.max(1.0);
    MetricsOut {
        latency_ms,
        throughput_tps,
        perplexity,
        memory_mb,
        swap_cost_ms: 0.0,
        cache_miss_count: 0.0,
        variant_churn: 0.0,
        tokens_processed,
        latency_ms_per_token: latency_ms / tokens_processed,
        latency_source: "simulator".into(),
        throughput_source: "simulator".into(),
        memory_source: "simulator".into(),
        perplexity_source: "simulator".into(),
        simulator_engine: "rust_cli".into(),
    }
}

fn apply_hardware_alignment(
    hardware: &HardwareIn,
    avg_bits: f64,
    latency_ms: &mut f64,
    throughput_tps: &mut f64,
    memory_mb: &mut f64,
    perplexity: &mut f64,
) {
    if (hardware.hardware_type == "cpu" || hardware.hardware_type == "low_resource")
        && avg_bits > hardware.preferred_bits
    {
        let excess = avg_bits - hardware.preferred_bits;
        let cpu = hardware.hardware_type == "cpu";
        *latency_ms *= 1.0 + excess * if cpu { 0.16 } else { 0.24 };
        *throughput_tps *= (1.0 - excess * if cpu { 0.07 } else { 0.12 }).max(0.55);
        *memory_mb *= 1.0 + excess * if cpu { 0.10 } else { 0.18 };
    } else if hardware.hardware_type == "gpu" && avg_bits < hardware.preferred_bits {
        let deficit = hardware.preferred_bits - avg_bits;
        *perplexity += deficit * 0.45;
        *throughput_tps *= (1.0 - deficit * 0.03).max(0.78);
    }
}

fn apply_mode_adjustments(
    decision: &DecisionIn,
    compression: f64,
    complexity: f64,
    sensitivity: f64,
    hardware: &HardwareIn,
    latency_ms: &mut f64,
    throughput_tps: &mut f64,
    memory_mb: &mut f64,
    perplexity: &mut f64,
) {
    match decision.mode.as_str() {
        "dynamic" => {
            *latency_ms *= 0.92;
            *throughput_tps *= 1.06;
            *perplexity -= 0.25 + complexity * 0.20;
        }
        "learned" => {
            *latency_ms *= 0.82 - compression * 0.06;
            *throughput_tps *= 1.12 + compression * 0.08;
            *memory_mb *= 0.78 - compression * 0.04;
            *perplexity -= 0.38 + sensitivity * 0.22;
        }
        "grouped" if hardware.hardware_type != "gpu" => {
            *latency_ms *= 0.95;
            *throughput_tps *= 1.03;
        }
        _ => {}
    }
}

fn apply_memory_overflow(
    hardware: &HardwareIn,
    memory_mb: f64,
    latency_ms: &mut f64,
    throughput_tps: &mut f64,
    perplexity: &mut f64,
) {
    let overflow = (memory_mb - hardware.memory_budget_mb).max(0.0) / hardware.memory_budget_mb;
    if overflow > 0.0 {
        *latency_ms *= 1.0 + overflow * 2.50;
        *throughput_tps /= 1.0 + overflow * 1.8;
        *perplexity += overflow * 1.50;
    }
}

pub fn evaluate_episode(req: &EvaluateRequest) -> MetricsOut {
    evaluate_core(req)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        DecisionIn, EvaluateRequest, HardwareIn, InputFeaturesIn, SensitivityIn,
    };

    fn sample_request() -> EvaluateRequest {
        EvaluateRequest {
            hardware: HardwareIn {
                hardware_type: "gpu".into(),
                compute_factor: 1.0,
                throughput_bias: 1.0,
                latency_bias: 1.0,
                memory_budget_mb: 24_000.0,
                preferred_bits: 4.0,
                kernel_uniformity_preference: 0.5,
            },
            input_features: InputFeaturesIn {
                prompt_length: 64,
                complexity_score: 0.3,
            },
            sensitivity: SensitivityIn {
                layer_stats: vec![0.2, 0.2],
            },
            decision: DecisionIn {
                mode: "learned".into(),
                scale_factor: 1.0,
                clipping_range: 1.0,
                effective_layer_bits: vec![4.0, 4.0, 4.0],
            },
            calibration: Default::default(),
        }
    }

    #[test]
    fn metrics_are_finite_and_positive() {
        let out = evaluate_episode(&sample_request());
        assert!(out.latency_ms > 0.0);
        assert!(out.throughput_tps > 0.0);
        assert!(out.simulator_engine == "rust_cli");
    }
}

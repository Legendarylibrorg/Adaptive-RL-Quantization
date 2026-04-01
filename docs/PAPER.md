# Adaptive RL Quantization for llama.cpp
## Universal Hardware-Aware, Input-Adaptive, Learned, and MoE-Aware Policies

### Abstract

Quantization is one of the most effective tools for reducing the cost of large language model inference, but most deployments still rely on static rules: a fixed preset, a hand-written hardware heuristic, or an offline calibration pass that does not adapt to the prompt being served. This paper presents **Adaptive RL Quantization**, a simulator-first research framework for learning quantization and execution policies that are simultaneously **hardware-aware**, **input-aware**, and **learned** rather than limited to a small menu of fixed bit-width rules. The framework targets `llama.cpp`-style inference, supports a pure-Python offline research loop, includes a CUDA/PyTorch path for NVIDIA GPUs, and now extends to **Mixture-of-Experts (MoE)** models through a packed-expert-bank formulation.

The system contributes four ideas. First, it learns **one universal policy** across multiple hardware targets instead of one policy per device. Second, it allows quantization behavior to depend on **input complexity**. Third, it replaces purely discrete presets with **learned quantization functions** that control scale, clipping, and effective precision. Fourth, it introduces **MoE-aware packed expert selection**, where the policy chooses among prepacked expert variants while accounting for swap cost, cache misses, and expert sensitivity.

In the current offline simulator-backed benchmark, the universal dense policy reduces the hardware generalization gap by **16.55 reward units** relative to a GPU-only policy. Dynamic quantization improves reward from **-7.31** to **-4.44** while reducing instability from **0.0153** to **0.00229**. Learned quantization improves reward from **-7.31** to **-2.53**, reducing latency from **210.83 ms** to **121.79 ms** and memory from **1395.70 MB** to **877.09 MB**. On the MoE path, the packed-expert-bank policy improves reward over the dense adaptive baseline by **2.43** reward units, and improves reward over a single-variant MoE baseline by **2.23** reward units. However, in the current short-budget MoE benchmark, a strong static balanced-variant baseline still slightly outperforms the RL-controlled MoE variant selector by **0.38** reward units, highlighting that the MoE extension is promising but not yet fully optimized.

This paper is intentionally honest about scope. The headline numbers are currently **offline and simulator-backed**. The repository includes an explicit **RTX 4090 host universal-policy training path**, but the primary quantitative claims here still come from the reproducible offline benchmark rather than from fully hardware-grounded multi-device measurements.

### 1. Introduction

Practical large language model deployment is constrained by a simple tension: the strongest models are expensive, while the cheapest deployments are often too slow, too memory-hungry, or too degraded in quality. Quantization is one of the main ways practitioners navigate that tradeoff. In most systems, however, quantization is still treated as a static artifact. A model is exported at a chosen precision, served on a specific device, and left there.

That workflow breaks down once the deployment setting changes. A quantization choice that works well on one GPU may be suboptimal on CPU. A compression level that is acceptable for short factual prompts may degrade badly on more complex prompts. In MoE models, the problem becomes even more structured: different experts have different sensitivity, different routing frequency, and different residency costs.

This paper argues for a different framing. Instead of treating quantization as a one-time compression choice, we treat it as a **learned control problem**. A policy observes hardware context, prompt-level features, sensitivity estimates, and optionally MoE expert state, then selects a quantization or packed-expert strategy that trades off latency, throughput, memory, quality, and stability.

The repository now supports this framing in three increasingly expressive regimes:

1. dense adaptive quantization across hardware and inputs,
2. learned quantization functions with continuous control,
3. MoE-aware packed expert variant selection.

The core experimental story remains offline-first and reproducible. The project includes a CUDA path tuned for RTX 4090-class hardware and an explicit `run_4090_universal.py` entrypoint for “train on a 4090, learn a universal policy,” but the stable evidence base is still the simulator-backed offline benchmark.

### 2. Contributions

This version of the system makes four concrete contributions:

1. It formulates quantization as a **universal policy-learning problem** over multiple hardware targets rather than a device-specific export rule.
2. It introduces **input-adaptive quantization**, where prompt complexity changes the chosen precision behavior.
3. It extends fixed presets into **learned quantization functions** with continuous control over scale, clipping, and effective precision.
4. It adds an **MoE packed-expert-bank extension** in which RL chooses among prepacked expert variants under cache and swap constraints.

### 3. Problem Formulation

We consider an inference-time controller for a `llama.cpp`-style backend. For each episode, the environment samples a hardware target and a prompt. The dense-policy state is:

\[
s = [h, x, \sigma, a_{t-1}]
\]

where:

- \(h\) is a hardware encoding,
- \(x\) is a prompt feature vector,
- \(\sigma\) is a sensitivity summary,
- \(a_{t-1}\) is a compact encoding of the previous action.

Prompt features include prompt length, approximate token entropy, token variance, embedding norm, and a derived complexity score. Sensitivity features summarize attention sensitivity, FFN sensitivity, and per-layer statistics.

When MoE is enabled, the state is extended with expert-routing context:

\[
s_{\text{moe}} = [h, x, \sigma, a_{t-1}, m]
\]

where \(m\) includes:

- router entropy,
- active expert count,
- cache pressure,
- estimated swap cost,
- top-k expert descriptors including sensitivity, hotness, residency, and available packed variants.

### 4. Action Space

The framework supports the following dense-model action modes:

- `discrete`
- `grouped`
- `per_layer`
- `dynamic`
- `learned`
- `hybrid`

In learned mode, the policy emits continuous parameters:

\[
q = [\text{scale}, \text{clip}, \text{precision}]
\]

which are clamped to safe ranges and mapped into effective layer precision.

In the MoE extension, the action space is augmented with:

- packed variant choice per active expert,
- optional residual variation induced by the base quantization mode,
- safety fallback to balanced or safe variants when swap or aggressiveness constraints are exceeded.

The current packed-expert bank uses the variants:

- `safe`
- `balanced`
- `aggressive`

This design is deliberate. Fully continuous expert repacking is not practical at runtime, while a small prepacked variant bank preserves the benefits of learning with realistic systems constraints.

### 5. Reward

The dense reward is:

\[
r = -\alpha \cdot \text{latency} + \beta \cdot \text{throughput} - \gamma \cdot \text{perplexity} - \delta \cdot \text{memory} - \epsilon \cdot \text{instability}
\]

where instability is measured as perplexity variance across probe prompts.

In the MoE path, this is extended with explicit systems penalties:

\[
r_{\text{moe}} = r - \eta \cdot \text{swap\_cost} - \zeta \cdot \text{cache\_misses} - \xi \cdot \text{variant\_churn}
\]

This matters because in sparse MoE serving, memory traffic and expert movement can dominate the operational cost surface.

### 6. System Overview

The repository separates:

- environment
- backend
- quantization logic
- policy
- trainer
- benchmark suite
- analysis

The main research entrypoints are:

- `run_research.py`: canonical dense offline baseline
- `run_moe_research.py`: canonical MoE offline baseline
- `run_4090_universal.py`: explicit 4090-host universal-policy training path

The PyTorch/CUDA path exists to accelerate policy learning on a strong local device, especially an RTX 4090, while still conditioning on multiple target hardware profiles. In that sense, the 4090 is the **training host**, not the only intended deployment target.

### 7. Experimental Setup

The current results come from the offline benchmark artifacts produced by:

- `outputs/benchmarks/adaptive_universal_policy_summary.json`
- `outputs/benchmarks/adaptive_moe_policy_summary.json`
- `outputs/benchmarks/*_benchmarks.json` (the benchmark comparison suite)
- `outputs/reports/*_report.md` (a human-readable report that links figures)

The dense benchmark compares:

1. single-hardware vs multi-hardware training,
2. static vs dynamic quantization,
3. discrete vs learned quantization.

The MoE benchmark adds:

1. dense adaptive vs MoE adaptive,
2. single packed variant vs packed-expert bank,
3. static MoE policy vs RL MoE policy.

All headline numbers in this paper are simulator-backed. The simulator is not presented as a substitute for real deployment measurements; it is a structured research harness that preserves the directional tradeoffs of serving while allowing fast, reproducible iteration.

To reproduce the numbers locally:

```bash
python3 run_research.py
python3 run_moe_research.py
```

Then inspect the generated benchmark JSON and report under `outputs/benchmarks/` and `outputs/reports/`.

For more meaningful and publishable numbers (mean/std across randomness), run multi-seed aggregates:

```bash
python3 run_multiseed.py --preset dense --seeds 13,17,23,29,31
python3 run_multiseed.py --preset moe --seeds 13,17,23
```

Those write `outputs/reports/<run_name>_multiseed_report.md` plus per-seed reports and figures.

### 8. Results

#### 8.1 Universal Dense Policy

The dense universal policy improves cross-hardware robustness substantially. Relative to a GPU-only policy:

- single-policy gap: **29.22**
- multi-hardware gap: **12.67**
- improvement: **16.55**

This result supports the central framing: training one policy over hardware-conditioned state is meaningfully different from training one policy per device or overfitting to a single GPU regime.

#### 8.2 Dynamic and Learned Quantization

Dynamic quantization improves reward and stability:

| Mode | Reward | Latency (ms) | Throughput (tok/s) | Memory (MB) | Perplexity | Stability |
|---|---:|---:|---:|---:|---:|---:|
| Static | -7.31 | 210.83 | 136.49 | 1395.70 | 9.98 | 0.01530 |
| Dynamic | -4.44 | 134.24 | 164.16 | 869.36 | 11.61 | 0.00229 |

Learned quantization functions further improve the deployment objective:

| Mode | Reward | Latency (ms) | Throughput (tok/s) | Memory (MB) | Perplexity | Stability |
|---|---:|---:|---:|---:|---:|---:|
| Discrete | -7.31 | 210.83 | 136.49 | 1395.70 | 9.98 | 0.01530 |
| Learned | -2.53 | 121.79 | 178.88 | 877.09 | 10.66 | 0.01310 |

These dense results justify the first half of the framework: adaptive, learned policies can improve the composite deployment objective even when they do not optimize perplexity alone.

#### 8.3 MoE Packed-Expert Policies

The MoE extension improves over the dense adaptive baseline under the current composite reward:

| Mode | Reward | Latency (ms) | Throughput (tok/s) | Memory (MB) | Perplexity | Swap Cost (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Dense adaptive | -6.18 | 204.47 | 154.88 | 1378.72 | 10.14 | 0.00 |
| MoE adaptive | -3.75 | 136.15 | 186.07 | 868.03 | 12.04 | 2.58 |

The MoE packed-expert-bank policy also improves over a single-variant MoE baseline:

| Mode | Reward | Latency (ms) | Throughput (tok/s) | Memory (MB) | Perplexity | Variant Churn |
|---|---:|---:|---:|---:|---:|---:|
| Single balanced variant | -5.98 | 198.83 | 162.46 | 1306.88 | 10.48 | 0.00 |
| Packed expert bank | -3.75 | 136.15 | 186.07 | 868.03 | 12.04 | 0.25 |

This is an encouraging result. It suggests that the packed expert bank is adding meaningful systems flexibility rather than just complexity.

#### 8.4 A Useful Negative Result

The current RL MoE variant selector does **not yet** beat the strongest static balanced-variant baseline:

- static MoE reward: **-3.38**
- RL MoE reward: **-3.75**
- delta: **-0.38**

This is important. It means the MoE extension is not yet a clean “RL wins everywhere” result. What it does show is:

- MoE modeling itself is valuable,
- packed expert banks are valuable,
- the current RL credit assignment for MoE still has room to improve.

That is exactly the kind of result a serious systems paper should report honestly.

#### 8.5 MoE Behavior Analysis

In the current logged MoE run:

- mean router entropy is **0.999**
- mean aggressiveness is **0.252**
- variant usage is heavily concentrated in `safe` and `balanced`
- `aggressive` is selected only once

This is consistent with the safety caps in the current implementation. The policy is learning conservatively under swap and aggressiveness constraints, which is sensible in a low-risk offline benchmark, but may also be one reason the RL MoE policy has not yet surpassed the strongest static baseline.

### 9. Discussion

Three conclusions are stable across the current results.

First, **hardware conditioning matters**. Universal policies clearly outperform hardware-specialized policies when evaluated across multiple target regimes.

Second, **input adaptation matters**. Dynamic policies reduce instability and improve the multi-objective reward relative to static quantization.

Third, **MoE adds a richer control surface**. Once expert routing, residency, and swap behavior are exposed to the policy, the problem becomes more realistic and more interesting.

At the same time, the MoE results are mixed in a useful way. The packed-expert-bank idea is strong, but the current RL controller is not yet fully extracting its value. That suggests the right next step is not to abandon RL, but to improve the MoE policy parameterization, horizon, and reward shaping.

### 10. Why the RTX 4090 Matters

The repository now makes a specific workflow explicit: **train on a 4090, learn a universal policy**.

This is an important systems point. The 4090 is not presented here as the only target device. Instead, it is a strong and practical **training host** for:

- larger PPO batches,
- faster offline iteration,
- prompt-feature caching,
- CUDA preflight validation,
- repeated reproducible training runs.

The learned policy itself is still conditioned on multiple hardware targets. In other words, the system is designed to use a 4090 as the machine that performs learning, not as the only environment the learned behavior should understand.

### 11. Limitations

This draft has important limitations.

1. The reported numbers are simulator-backed rather than fully hardware-grounded.
2. The prompt set is small and synthetic relative to a real production traffic distribution.
3. The current `llama.cpp` hook exists, but it is not yet the primary source of the headline metrics.
4. The MoE extension is still short-horizon and uses a compact packed-variant abstraction rather than a full real runtime integration.
5. The RL MoE policy does not yet outperform the strongest static MoE baseline in the current benchmark.
6. The 4090-host universal-policy path exists in code, but this paper does not claim broad real-hardware transfer without additional validation.

### 12. Future Work

The next steps are clear.

#### 12.1 Real Hardware Calibration

The most important next step is to calibrate the simulator and benchmark against:

- real `llama.cpp` measurements,
- real CPU runs,
- real non-4090 GPU runs,
- real low-memory settings.

#### 12.2 Better MoE Credit Assignment

The MoE negative result suggests several concrete upgrades:

- longer-horizon credit assignment,
- better expert-specific embeddings,
- stronger cache-residency modeling,
- richer expert-routing traces,
- improved safety-aware exploration.

#### 12.3 VRAM-Budgeted Model Selection

A natural extension is to move from “best quantization policy for a chosen model” to:

- model selection,
- quantization selection,
- placement selection,
- expert residency scheduling,

all under an explicit VRAM budget.

#### 12.4 Hardware-Conditioned Transfer

The dedicated `run_4090_universal.py` path should eventually be paired with held-out real hardware evaluation so the project can make stronger claims about universal transfer.

### 13. Conclusion

Adaptive RL Quantization reframes quantization as a learned control problem rather than a fixed preset choice. In its current form, the framework supports:

- universal hardware-aware policies,
- input-adaptive quantization,
- learned quantization functions,
- MoE-aware packed expert selection,
- explicit 4090-host universal-policy training.

The dense results are strong: universal policies generalize better, dynamic quantization improves stability, and learned quantization improves the deployment objective. The MoE results are promising but more nuanced: packed expert banks are clearly useful, while the current RL MoE controller still trails a strong static baseline in the present benchmark. That combination of positive and negative results makes the project stronger, not weaker. It means the system is already interesting, but it still has meaningful headroom for real ML systems research.

### Appendix A. Reproducibility

Canonical dense offline run:

```bash
python3 run_research.py
```

Canonical MoE offline run:

```bash
python3 run_moe_research.py
```

Explicit 4090-host universal-policy run:

```bash
python3 run_4090_universal.py
```

Fixed 4090 CUDA run:

```bash
python3 run_pytorch_4090.py
```

Linux 4090 validation wrapper:

```bash
bash scripts/run_4090_pipeline.sh
```

Tests:

```bash
python3 -m unittest discover -s tests -v
```

### Appendix B. Main Implementation Files

- `adaptive_quant/environment.py`
- `adaptive_quant/quantization.py`
- `adaptive_quant/policy.py`
- `adaptive_quant/torch_policy.py`
- `adaptive_quant/torch_trainer.py`
- `adaptive_quant/moe.py`
- `adaptive_quant/benchmark.py`
- `analysis/hardware_generalization.py`
- `analysis/input_adaptation.py`
- `analysis/quant_function_behavior.py`
- `analysis/moe_expert_behavior.py`
- `analysis/moe_cache_behavior.py`

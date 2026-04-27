# Adaptive RL Quantization for llama.cpp
## Universal Hardware-Aware, Input-Adaptive, Learned, and MoE-Aware Policies

### Abstract

Quantization is one of the most effective tools for reducing the cost of large language model inference, but most deployments still rely on static rules: a fixed preset, a hand-written hardware heuristic, or an offline calibration pass that does not adapt to the prompt being served. This paper proposes **Adaptive RL Quantization**, a research framework for learning quantization and execution policies that are simultaneously **hardware-aware**, **input-aware**, and **learned** rather than limited to a small menu of fixed bit-width rules. The framework targets `llama.cpp`-style inference, supports a pure-Python offline research loop, includes a CUDA/PyTorch path for NVIDIA GPUs, and extends to **Mixture-of-Experts (MoE)** models through a packed-expert-bank formulation.

The system contributes four ideas. First, it learns **one universal policy** across multiple hardware targets instead of one policy per device. Second, it allows quantization behavior to depend on **input complexity**. Third, it replaces purely discrete presets with **learned quantization functions** that control scale, clipping, and effective precision. Fourth, it introduces **MoE-aware packed expert selection**, where the policy chooses among prepacked expert variants while accounting for swap cost, cache misses, and expert sensitivity.

**Crucially, this has not yet been proven to improve real deployments.** This draft intentionally focuses on the **research proposal** and system design, not on simulated headline numbers. The offline harness exists to accelerate iteration, but this paper treats it as **non-evidence** for deployment claims. The claims the project aims to eventually support are **real-hardware claims**: measured latency/throughput/memory on actual `llama.cpp` builds and GGUF artifacts, plus external quality metrics evaluated on real datasets.

The repository includes an explicit **RTX 4090 host universal-policy training path** and a `llama.cpp` route workflow that can measure real GGUF latency and throughput on a given machine. The core outcome of this paper is an architecture and evaluation plan that can produce publishable, hardware-grounded results once executed.

### 1. Introduction

Practical large language model deployment is constrained by a simple tension: the strongest models are expensive, while the cheapest deployments are often too slow, too memory-hungry, or too degraded in quality. Quantization is one of the main ways practitioners navigate that tradeoff. In most systems, however, quantization is still treated as a static artifact. A model is exported at a chosen precision, served on a specific device, and left there.

That workflow breaks down once the deployment setting changes. A quantization choice that works well on one GPU may be suboptimal on CPU. A compression level that is acceptable for short factual prompts may degrade badly on more complex prompts. In MoE models, the problem becomes even more structured: different experts have different sensitivity, different routing frequency, and different residency costs.

This paper argues for a different framing. Instead of treating quantization as a one-time compression choice, we treat it as a **learned control problem**. A policy observes hardware context, prompt-level features, sensitivity estimates, and optionally MoE expert state, then selects a quantization or packed-expert strategy that trades off latency, throughput, memory, quality, and stability.

The repository now supports this framing in three increasingly expressive regimes:

1. dense adaptive quantization across hardware and inputs,
2. learned quantization functions with continuous control,
3. MoE-aware packed expert variant selection.

The core experimental story in this paper is **evaluation-first**: define what constitutes evidence, define the deployment-relevant metrics, and define the architecture that makes real-hardware measurement feasible and reproducible. The project includes a CUDA path tuned for RTX 4090-class hardware and `adaptive-rl-quant-pytorch --preset 4090-universal` for “train on a 4090, learn a universal policy,” but the emphasis of this draft is how to validate transfer and systems impact on real hardware rather than on simulated aggregates.

### 1.1 Status of Evidence (Read This First)

This work is currently a **proposal + prototype research artifact**. As of this draft:

- **Not proven**: there are no peer-reviewed, real-hardware results in this paper demonstrating improvements over strong static baselines.
- **Not a deployment claim**: offline-harness reward and simulator-derived metrics are treated as internal iteration signals only.
- **What would count as proof**: the “Proposed Evaluation” section specifies the concrete measurements, baselines, datasets, and reporting standards required before making real-world claims.

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

where instability is measured as perplexity variance across probe prompts. This reward is primarily a **training objective** for policy learning and should not be interpreted as a publishable deployment metric unless each term is grounded in real measurement and the quality term is computed from external evaluation.

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

- `adaptive-rl-quant`: canonical dense offline baseline
- `adaptive-rl-quant-moe`: canonical MoE offline baseline
- `adaptive-rl-quant-pytorch --preset 4090-universal`: explicit 4090-host universal-policy training path

The PyTorch/CUDA path exists to accelerate policy learning on a strong local device, especially an RTX 4090, while still conditioning on multiple target hardware profiles. In that sense, the 4090 is the **training host**, not the only intended deployment target.

### 7. Experimental Setup

This proposal defines an evidence ladder and an experimental design that can produce publishable, hardware-grounded results. The repository already includes benchmark artifact formats produced by:

- `outputs/benchmarks/adaptive_universal_policy_summary.json`
- `outputs/benchmarks/adaptive_moe_policy_summary.json`
- `outputs/benchmarks/*_benchmarks.json` (the benchmark comparison suite)
- `outputs/reports/*_report.md` (a human-readable report that links figures)
- `outputs/paper_bundles/<run_name>/` (manifest, metric tables, telemetry export, appendix, and claims validation)

The dense benchmark compares:

1. single-hardware vs multi-hardware training,
2. static vs dynamic quantization,
3. discrete vs learned quantization.

The MoE benchmark adds:

1. dense adaptive vs MoE adaptive,
2. single packed variant vs packed-expert bank,
3. static MoE policy vs RL MoE policy.

The repo distinguishes three evidence levels. Only levels 2 and 3 should be used to support claims about real systems performance:

1. **Offline harness evidence (non-claim)**: the offline environment is used to iterate on policies and debug credit assignment. These outputs are explicitly not treated as evidence for real-hardware performance.
2. **Local `llama.cpp` evidence (claimable)**: latency, throughput, and memory are measured from a real `llama.cpp` binary and GGUF files on a real machine with pinned configs; quality is evaluated with external metrics on real datasets.
3. **Multi-device evidence (strong claimable)**: the same experiment protocol is run on a device matrix (CPU + multiple GPU classes), with consistent prompts, consistent generation settings, and transparent variance reporting.

To reproduce the baseline pipeline locally:

```bash
adaptive-rl-quant
adaptive-rl-quant-moe
```

Then inspect the generated benchmark JSON and report under `outputs/benchmarks/` and `outputs/reports/`. For citation or review, prefer the matching `outputs/paper_bundles/<run_name>/manifest.json`, `metrics_summary.csv`, and `claims_validation.md` so metric provenance and evidence level travel with the measurements.

For more meaningful and publishable numbers (mean/std across randomness), run multi-seed aggregates:

```bash
adaptive-rl-quant-multiseed --preset dense --seeds 13,17,23,29,31
adaptive-rl-quant-multiseed --preset moe --seeds 13,17,23
```

Those write `outputs/reports/<run_name>_multiseed_report.md` plus per-seed reports and figures.

To anchor the offline harness to real measurements, you can calibrate simulator coefficients against real `llama.cpp` runs:

```bash
# requires backend="llama_cpp" and valid llama_cpp_binary/llama_cpp_model paths
adaptive-rl-quant-calibrate
```

This writes a calibration JSON with per-hardware multipliers that can be copied into `sim_calibration` in your chosen config preset.

### 8. Proposed Evaluation (Real Results, Not Simulation)

This section defines the experiments required for publishable, real-hardware claims. The key goal is to evaluate whether an adaptive policy can improve the **Pareto frontier** over latency, throughput, memory footprint, and quality—under realistic serving constraints—relative to strong static baselines.

#### 8.1 Hardware Matrix and Serving Protocol

We propose a device matrix that spans meaningful regimes:

- **CPU**: one AVX2/AVX-512 class desktop/server CPU.
- **NVIDIA GPU**: one consumer GPU (e.g., RTX 4090-class) and one lower-VRAM GPU.
- **Optional**: an ARM device (Apple Silicon) to test portability of hardware-conditioned policies.

All latency/throughput measurements must use:

- pinned `llama.cpp` commit hash and build flags,
- pinned generation settings (temperature, top-p, max tokens, KV cache settings),
- fixed batch size and concurrency,
- warm-up runs and repeated trials with confidence intervals.

#### 8.2 Datasets and Quality Metrics

Quality must be assessed on real datasets rather than a proxy perplexity inside the harness. Suggested evaluation blocks:

- **Instruction following / chat**: a public instruction set with standardized prompts and deterministic decoding for scoring.
- **Reasoning / multi-step**: a reasoning benchmark with strict answer checking where applicable.
- **Long-context sensitivity**: a long-context benchmark to test degradation under aggressive quantization.

Metrics should include:

- task accuracy / exact match (where applicable),
- model-graded preference only as a secondary metric with strict controls,
- calibration (e.g., ECE) for tasks where confidence can be extracted.

The codebase now supports an external quality sidecar through `external_quality_path` and `external_quality_metric`. When provided, prompt-level scores keyed by `prompt_id` replace simulator perplexity in the backend metric dictionary and are recorded in paper-bundle provenance. This is only as credible as the scoring file used to produce it; the sidecar should therefore be generated from fixed datasets, fixed decoding settings, and versioned scoring code.

#### 8.3 Systems Metrics and Instrumentation

Primary systems metrics:

- **latency**: p50/p95/p99 end-to-end decode latency and per-token decode time,
- **throughput**: tokens/sec at fixed concurrency,
- **memory**: peak RSS / VRAM and KV cache footprint,
- **stability**: variance across runs under controlled nondeterminism, plus failure rate (OOM, errors).

MoE-specific metrics:

- expert residency hit rate,
- swap/transfer volume and churn rate,
- router entropy and expert utilization skew.

#### 8.4 Baselines and Ablations (What Must Be Beaten)

Baselines:

- static quantization presets (multiple bitwidths),
- hardware-specific tuned presets (per device),
- input-agnostic dynamic heuristics (prompt length thresholds),
- “balanced” MoE packed-variant baseline (static).

Ablations:

- remove hardware encoding (tests true “universal” transfer),
- remove input features (tests input adaptivity),
- discrete vs learned quantization parameters,
- MoE bank size and churn penalties.

#### 8.5 Claim Types and Reporting

To keep the work research-grade, claims should be stated only at the evidence level supported:

- **Local claim**: “on device X, under protocol Y, policy Z improves the Pareto frontier vs baseline B.”
- **Transfer claim**: “a policy trained on host H transfers to device D with bounded degradation vs device-tuned baseline.”
- **MoE systems claim**: “packed-expert banks reduce memory traffic / improve throughput under controlled churn constraints.”

All result reporting should include:

- the exact GGUF artifacts used,
- commit hashes and config manifests,
- the external quality sidecar hash and scoring metric, if used,
- repeated trials with mean/std and confidence intervals,
- full prompt lists and scoring code.

### 9. Discussion

This proposal is structured around three research questions.

First, **does hardware conditioning enable transfer**? A universal policy should generalize to held-out hardware targets with less degradation than a policy trained on a single regime.

Second, **does input adaptation move the Pareto frontier**? The policy should allocate precision where it matters (quality-sensitive prompts/layers) and reduce it where it does not, improving systems metrics without unacceptable quality loss.

Third, **does MoE-aware control pay off under real constraints**? The packed-expert-bank abstraction is meant to reflect realistic runtime constraints (packing cost, cache behavior). The evaluation must verify whether variant selection is worth the complexity.

The expected outcome is not “RL wins everywhere,” but a careful mapping of when adaptive control improves the deployment frontier and when static baselines remain strong.

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

1. This paper is currently a proposal and architecture description; it does not present simulator-derived performance tables as publishable results.
2. Real prompt distributions and end-to-end serving integration are out of scope for this draft; the evaluation plan targets reproducible public datasets and controlled protocols.
3. The `llama.cpp` measurement route must be treated as the source of truth for systems metrics; the offline harness is primarily an iteration tool.
4. The MoE extension uses a compact packed-variant abstraction; validating the abstraction requires careful instrumentation and churn controls.
5. Universal transfer claims require a device matrix and repeated trials; those are planned but not yet reported here.
6. The central hypothesis—that RL-based, input- and hardware-conditioned control can improve the deployment Pareto frontier—may fail under real-world constraints (kernel-level bottlenecks, cache effects, quantization artifacts, routing dynamics, or evaluation noise). This work is designed to make that failure mode measurable and honest.

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

The dedicated `adaptive-rl-quant-pytorch --preset 4090-universal` path should eventually be paired with held-out real hardware evaluation so the project can make stronger claims about universal transfer.

### 13. Conclusion

Adaptive RL Quantization reframes quantization as a learned control problem rather than a fixed preset choice. In its current form, the framework supports:

- universal hardware-aware policies,
- input-adaptive quantization,
- learned quantization functions,
- MoE-aware packed expert selection,
- explicit 4090-host universal-policy training.

This draft does not claim real-world improvements yet. It presents a research-grade architecture and a concrete evaluation plan aimed at producing **real-hardware** evidence: measured `llama.cpp` latency/throughput/memory, plus external quality metrics on real datasets, reported with transparent variance. The contribution is a controllable, extensible framework for learning adaptive quantization policies (dense and MoE) and a protocol for validating whether those policies improve the deployment Pareto frontier in practice.

### Appendix A. Reproducibility

Canonical dense offline run:

```bash
adaptive-rl-quant
```

Canonical MoE offline run:

```bash
adaptive-rl-quant-moe
```

Explicit 4090-host universal-policy run:

```bash
adaptive-rl-quant-pytorch --preset 4090-universal
```

Fixed 4090 CUDA run:

```bash
adaptive-rl-quant-pytorch --preset 4090
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
- `analysis/analyzers.py`
- `analysis/hardware_generalization.py`
- `analysis/input_adaptation.py`
- `analysis/quant_function_behavior.py`
- `analysis/moe_expert_behavior.py`
- `analysis/moe_cache_behavior.py`

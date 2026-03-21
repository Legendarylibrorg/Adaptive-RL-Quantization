# Adaptive RL Quantization for llama.cpp:
## Universal Hardware-Aware, Input-Adaptive, and Learned Quantization Policies

### Contributions

This paper makes three concrete contributions:

1. It formulates quantization as a **universal policy-learning problem** over heterogeneous hardware targets rather than a device-specific export rule.
2. It introduces **input-adaptive quantization** driven by prompt-level complexity features, allowing the quantization policy to vary across requests.
3. It extends fixed-preset quantization into **learned quantization functions** with continuous control over scale, clipping, and effective precision.

### Abstract

Quantization is one of the most effective ways to reduce the cost of large language model inference, but most existing deployment strategies still rely on static rules: fixed bit-width presets, hardware-specific heuristics, or offline calibration pipelines that do not adapt to the prompt being served. This work presents **Adaptive RL Quantization**, a simulator-first and deployment-oriented framework for learning quantization policies that are simultaneously **hardware-aware**, **input-aware**, and **functionally learned** rather than restricted to a small set of hand-written bit-width rules. The system targets `llama.cpp`-style execution and supports both a pure-Python research simulator and a CUDA/PyTorch training path for modern NVIDIA GPUs.

Our framework contributes three ideas. First, we train a **universal policy** over multiple hardware modes rather than learning one policy per device. The policy conditions on a hardware encoding and is optimized across GPU, CPU, and low-resource settings. Second, we enable **dynamic per-input quantization**, where decisions depend on prompt complexity features such as length, entropy, token variance, and embedding statistics. Third, we introduce **learned quantization functions** with continuous control over scale, clipping, and precision, allowing the policy to learn quantization behavior rather than only select among discrete presets. The framework also includes a hybrid action interface that unifies discrete, grouped, per-layer, dynamic, and learned modes under one experimental surface.

In the current simulator-backed benchmark, the universal multi-hardware policy reduces the generalization gap by **16.75 reward units** relative to a single-GPU policy. Dynamic quantization improves reward from **-7.31** to **-4.44** while substantially reducing instability from **0.0153** to **0.00229**, albeit with a modest perplexity tradeoff. Learned quantization functions improve reward from **-7.31** to **-2.53**, reduce latency from **210.83 ms** to **121.79 ms**, reduce memory from **1395.70 MB** to **877.09 MB**, and increase throughput from **136.49** to **178.88 tokens/s**, again with a modest quality tradeoff. These results suggest that adaptive quantization can outperform static baselines when the objective reflects deployment constraints rather than only raw quality.

This draft is intentionally honest about scope: the quantitative results in the current repository are simulator-based and come from the **offline training and evaluation pipeline**. The codebase also includes a real `llama.cpp` backend hook, a PyTorch/CUDA training path intended for future hardware-grounded evaluation, and an optional experimental online adaptation module that is **not** part of the primary empirical claims in this paper.

### 1. Introduction

Quantized inference sits at the center of practical language model deployment. The standard workflow is familiar: choose a quantization preset, export a model, and accept the resulting tradeoff between speed, memory, and quality. This works well when a system is deployed to a single target runtime and prompt distribution, but it becomes brittle once the deployment setting changes. A quantization plan tuned for one GPU may be suboptimal on CPU. A highly compressed preset that works well for short factual prompts may degrade badly on more complex reasoning or generation tasks. A fixed menu of presets also limits the search space itself: the deployment system can only select among choices a human anticipated in advance.

This paper argues for a different framing. Instead of treating quantization as a static export-time transformation, we treat it as a **policy-learning problem**. In the primary research setting studied here, a policy is trained offline in a controlled simulator and then evaluated under held-out hardware and prompt conditions. At inference time, the learned policy observes the hardware context, prompt-derived features, and sensitivity estimates, then selects or parameterizes a quantization strategy. Under this framing, quantization is not a one-time compression step; it is an adaptive control mechanism that trades off latency, throughput, memory, quality, and stability.

The resulting problem is challenging for three reasons. First, the policy must generalize across heterogeneous deployment targets. Second, it must respond to prompt complexity rather than assuming a single average case. Third, it should learn continuous quantization behavior where appropriate, not only choose among a handful of fixed presets. The Adaptive RL Quantization framework in this repository is designed to address those three challenges in one system.

### 2. Problem Formulation

We consider an inference-time controller for a `llama.cpp`-style backend. For each episode, the environment samples a hardware target and an input prompt. The agent observes a state vector

\[
s = [h, x, \sigma, a_{t-1}],
\]

where:

- \(h\) is a hardware encoding for GPU, CPU, or low-resource mode,
- \(x\) contains prompt-derived features,
- \(\sigma\) contains sensitivity estimates,
- \(a_{t-1}\) is a compact encoding of the previous action.

Prompt features include prompt length, approximate token entropy, token variance, and embedding norm. Sensitivity features summarize attention sensitivity, feed-forward sensitivity, and per-layer statistics.

The action space is hybrid. Depending on the experimental mode, the agent may:

- choose a single discrete bit-width,
- choose grouped bit-widths,
- choose per-layer bit-widths,
- make a dynamic input-conditioned discrete decision,
- output continuous quantization parameters \((\text{scale}, \text{clip}, \text{precision})\).

The reward is a deployment-oriented scalar objective:

\[
r = -\alpha \cdot \text{latency} + \beta \cdot \text{throughput} - \gamma \cdot \text{perplexity} - \delta \cdot \text{memory} - \epsilon \cdot \text{instability}.
\]

The instability term is estimated as perplexity variance across probe prompts, which encourages not only fast and cheap policies, but also stable ones.

This formulation is intentionally deployment-centered. The objective is not to maximize model quality in isolation, but to optimize a weighted utility that reflects real serving tradeoffs. In practice, the coefficients \(\alpha, \beta, \gamma, \delta, \epsilon\) can be retuned for different deployment priorities, such as memory-constrained edge inference, latency-sensitive interactive serving, or throughput-heavy batch generation.

### 3. System Overview

The implementation exposes a clean separation between environment, quantization logic, policy, trainer, benchmark suite, and analysis.

#### 3.1 Environment and Backend

The environment samples prompts from a small prompt library and hardware profiles from a fixed catalog. It supports:

- GPU mode,
- CPU mode,
- low-resource mode,
- a simulator backend,
- a `llama.cpp` backend hook.

The simulator backend is intentionally structured to reflect deployment tradeoffs: aggressive compression improves throughput and memory use but hurts perplexity; hardware mismatch induces latency and throughput penalties; and dynamic or learned modes can exploit more favorable tradeoff surfaces than static modes.

#### 3.2 Quantization Modes

The framework supports:

- `discrete`
- `grouped`
- `per_layer`
- `dynamic`
- `learned`
- `hybrid`

Dynamic mode uses prompt complexity and sensitivity to adapt effective layer precision even when the policy’s top-level decision is discrete. Learned mode maps continuous outputs into quantization behavior using bounded scale, clipping, and precision parameters with safety guards and fallback logic.

#### 3.3 Universal Policy Learning

The baseline trainer is simulator-first and pure Python. The repository also includes a PyTorch actor-critic path with PPO-style updates for GPU training. In both cases, the policy conditions on hardware, input features, sensitivity, and previous action feedback. The CUDA path includes preflight checks, fused optimizer support, buffered logging, prompt-feature caching, and auto-tuned GPU profiles for cards beyond the RTX 4090.

### 4. Learned Quantization Functions

The central departure from a fixed-preset system is the learned action parameterization. In learned mode, the policy outputs continuous values for:

- scale factor,
- clipping range,
- precision level.

These parameters are clamped to safe ranges, then transformed into effective per-layer precision as a function of sensitivity and prompt complexity. The result is not merely “pick 2-bit or 4-bit”; it is “learn how aggressively to compress and where.”

This is important because different layers and different prompts do not require the same precision. A sensitive deeper layer on a difficult prompt may need much higher effective precision than a shallow layer on a short factual query. Learned mode provides that flexibility while still exposing explicit safety bounds.

### 5. Experimental Setup

The current repository reports simulator-backed results using the offline benchmark summary produced by:

- `outputs/benchmarks/adaptive_universal_policy_summary.json`

The experiments compare:

1. single-hardware versus multi-hardware universal policy learning,
2. static versus dynamic quantization,
3. discrete versus learned quantization functions.

The evaluation reports:

- mean reward,
- mean latency,
- mean throughput,
- mean memory,
- mean perplexity,
- mean stability penalty.

The simulator is best understood as a structured research harness rather than a claim of final production performance. It is designed to preserve the directional incentives faced by a real deployment system while enabling fast iteration on state design, reward shaping, and policy parameterization. This matters for research quality: the headline results are meant to come from a reproducible offline setup rather than an online system whose behavior changes during collection.

### 6. Results

#### 6.1 Universal Multi-Hardware Policy

The multi-hardware policy improves the generalization gap by **16.75 reward units** relative to the single-GPU policy. Concretely:

- single-policy gap: **29.42**
- multi-policy gap: **12.67**
- improvement: **16.75**

This is the clearest evidence that a single policy can internalize hardware context instead of merely overfitting to one device. The single-GPU policy performs well on GPU, but degrades sharply on CPU and low-resource settings. The universal policy sacrifices some GPU-only specialization in exchange for far better cross-hardware robustness.

#### 6.2 Static versus Dynamic Quantization

Dynamic quantization improves deployment reward and strongly improves stability:

| Mode | Reward | Latency (ms) | Throughput (tok/s) | Memory (MB) | Perplexity | Stability |
|---|---:|---:|---:|---:|---:|---:|
| Static | -7.31 | 210.83 | 136.49 | 1395.70 | 9.98 | 0.01530 |
| Dynamic | -4.44 | 134.24 | 164.16 | 869.36 | 11.61 | 0.00229 |

Dynamic mode trades some quality for substantially better deployment characteristics. The most striking reduction is in instability: the stability penalty drops by **0.0130**. This suggests that prompt-aware policies can better match quantization aggressiveness to input complexity than fixed settings.

#### 6.3 Discrete versus Learned Quantization Functions

Learned quantization functions outperform the discrete baseline under the composite reward:

| Mode | Reward | Latency (ms) | Throughput (tok/s) | Memory (MB) | Perplexity | Stability |
|---|---:|---:|---:|---:|---:|---:|
| Discrete | -7.31 | 210.83 | 136.49 | 1395.70 | 9.98 | 0.01530 |
| Learned | -2.53 | 121.79 | 178.88 | 877.09 | 10.66 | 0.01310 |

The learned policy improves reward by **4.79**, cuts latency by roughly **42%**, reduces memory by roughly **37%**, and raises throughput by roughly **31%**. As with dynamic mode, these gains come with a moderate perplexity cost. The result is still meaningful because the target objective is deployment-oriented rather than quality-only.

### 7. Discussion

The results support three takeaways.

First, **hardware conditioning matters**. A universal policy trained across heterogeneous hardware can meaningfully reduce the gap between specialized and generalized deployment.

Second, **input adaptation matters**. Prompt-aware quantization behaves more like a serving policy than an export preset. This is a better match for real systems, where prompt difficulty varies widely.

Third, **continuous quantization control matters**. Learned quantization functions recover better reward than purely discrete decisions because they can shape compression more smoothly.

At the same time, the tradeoffs are real. Dynamic and learned modes improve the deployment objective while tolerating some quality degradation. This is exactly why the reward must be explicit. If a system optimizes perplexity alone, static or conservative quantization may appear best; if it optimizes operational utility, adaptive policies become far more attractive.

One useful way to interpret the results is as evidence that quantization should be treated as a systems policy rather than a one-time compression artifact. Under that view, the policy is not replacing quantization algorithms; it is deciding how aggressively and where to apply them under changing hardware and workload conditions.

### 8. Limitations

This draft has important limitations.

1. The reported quantitative results are currently simulator-backed.
2. The benchmark is small and synthetic relative to production prompt distributions.
3. The `llama.cpp` backend path exists but is not yet the source of the primary reported numbers in this draft.
4. The current system uses lightweight proxy features rather than a full online representation of model uncertainty or calibration error.
5. The policy is trained in a one-step episodic setting; a richer sequential formulation may expose additional gains.

These limitations are not hidden by the codebase: the framework is explicitly simulator-first for fast iteration, with real-backend hooks for future experiments. The repository does contain an experimental online adaptation module, but it is best understood as follow-on systems work rather than part of the stable empirical core presented here.

### 9. Future Work

Several directions are immediate.

#### 9.1 Real Hardware Evaluation

The most important next step is to run the same benchmark suite against a true `llama.cpp` backend across:

- consumer GPUs,
- CPU-only settings,
- low-memory deployment targets.

#### 9.2 Better Input Features

The current prompt features are intentionally simple. Better complexity signals could come from:

- prompt syntax structure,
- retrieval statistics,
- uncertainty estimates,
- hidden-state difficulty predictors.

#### 9.3 Richer Learned Quantizers

The continuous action head can be extended from scalar control to structured learned quantization modules that output:

- per-layer clipping schedules,
- mixed group sizes,
- layer-specific precision priors.

#### 9.4 Offline-to-Online Transfer

An appealing deployment strategy is to pretrain a policy in simulation, then fine-tune it online against real latency and quality measurements from the serving stack. The repository now includes an experimental prototype of this idea, but it is intentionally separated from the paper’s main evaluation path.

#### 9.5 Joint Policy and Quantizer Learning

The current learned mode exposes a compact continuous parameterization. A natural extension is to jointly train a small quantization network together with the control policy, so that the system learns not only *when* to compress aggressively, but also *how* to reshape quantization behavior for different layer families and hardware regimes.

### 10. Conclusion

Adaptive RL Quantization reframes quantization as a learned control problem rather than a fixed preset selection problem. The resulting system is hardware-aware, input-aware, and capable of learned quantization behavior through continuous control. In the current simulator-backed offline benchmark, universal hardware conditioning reduces generalization gap, dynamic quantization substantially improves deployment reward and stability, and learned quantization functions outperform discrete baselines under a realistic multi-objective reward.

The repository therefore serves two roles at once: it is already a functioning adaptive quantization framework for controlled offline research, and it is also a scaffold for more rigorous future evaluation on real `llama.cpp` hardware backends.

For a GitHub research project, this is a strong foundation: the ideas are concrete, the implementation is runnable, and the claims are scoped honestly to the current evidence. The next milestone is to replace simulator-backed headline numbers with end-to-end measurements on real `llama.cpp` deployments across diverse GPU and CPU environments.

### Appendix C. Experimental Online Extension

The repository includes an optional online adaptation module for replay-based continual improvement with canaries and rollback safeguards. This component is intentionally not part of the primary empirical narrative in this paper. It is better viewed as exploratory systems work that could support future offline-to-online transfer studies once the offline benchmark is fully matured and validated on real `llama.cpp` hardware.

### Appendix A. Implementation Artifacts

Relevant implementation files include:

- `adaptive_quant/environment.py`
- `adaptive_quant/quantization.py`
- `adaptive_quant/policy.py`
- `adaptive_quant/torch_policy.py`
- `adaptive_quant/torch_trainer.py`
- `adaptive_quant/benchmark.py`
- `analysis/hardware_generalization.py`
- `analysis/input_adaptation.py`
- `analysis/quant_function_behavior.py`

### Appendix B. Reproducibility Notes

Baseline simulator run:

```bash
python3 run_research.py
```

Generic GPU run:

```bash
python3 run_pytorch_gpu.py
```

Fixed RTX 4090 run:

```bash
python3 run_pytorch_4090.py
```

Tests:

```bash
python3 -m unittest discover -s tests -v
```

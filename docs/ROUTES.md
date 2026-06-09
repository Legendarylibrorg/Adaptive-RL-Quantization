# Route Learning: Best (Model, Quant) Per Task & Hardware

The route learning workflow learns **which Hugging Face GGUF route wins for which task on
which hardware**, then pulls the matching file via `huggingface-cli` (or the modern `hf` CLI).
A *route* is one row of the catalog: a `(repo_id, filename, quant_label)` triple plus
hardware hints, parameter count, and on-disk size. The bandit treats every route as an arm
and learns from the same simulator/llama.cpp reward stack the rest of the framework uses.

This is layered **on top of** the existing RL quantization policy — the per-layer / dynamic /
learned modes still apply to whichever model you ultimately run. Route learning answers the
*outer* question: *given a task, which model/quant should I even download?*

## Why a contextual bandit?

Route choice is a categorical action over a small (~10) set of arms with quick reward signals
and high heterogeneity across `(hardware, task domain, complexity)` cells. That is the textbook
setting for a bucketed UCB bandit, which is what `RouteBandit` implements:

- **Buckets** — `hardware × task domain × complexity bin` (low / mid / high).
- **Per-arm stats** — Welford-tracked mean and variance, plus a global prior so cold buckets
  do not collapse onto a single arm.
- **Selection** — UCB1 with a configurable exploration coefficient (`--ucb-c`) and a feasibility
  mask that excludes routes whose `hardware_hints` do not include the active hardware.
- **Persistence** — `state_dict()` / `load_state_dict()` survives JSON round trips so you can
  resume training (or just serve recommendations) from a saved bandit artifact.

## CLI cheat sheet

```bash
# 1. Seed the catalog with curated GGUF routes.
adaptive-rl-quant-route --catalog outputs/routes/catalog.json seed

# 2. (Optional) Register a new route by hand. --quant maps to effective bits via the built-in
#    table; pass --effective-bits to register a novel quant family.
adaptive-rl-quant-route register \
  --route-id mistral7b-q4km \
  --repo bartowski/Mistral-7B-Instruct-v0.3-GGUF \
  --filename Mistral-7B-Instruct-v0.3-Q4_K_M.gguf \
  --quant Q4_K_M \
  --parameters-b 7 --size-mb 4400 \
  --local-path /absolute/path/to/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf \
  --hardware-hint gpu --hardware-hint cpu \
  --notes "general-purpose 7B baseline"

# 3. Fetch the GGUF via huggingface-cli (or hf). Validated argv only — no shell.
adaptive-rl-quant-route download --route-id mistral7b-q4km --dry-run    # preview
adaptive-rl-quant-route download --route-id mistral7b-q4km              # actually fetch

# 4. Train the bandit on the simulator (or local llama.cpp GGUFs if config.backend="llama_cpp").
adaptive-rl-quant-route train --iterations 1024 --evaluate --require-local-models

# 5. Score your own prompt JSON against every route and choose the lowest-VRAM route whose
#    evaluation reward/perplexity stays within regression bounds.
adaptive-rl-quant-route evaluate-prompts \
  --prompts-json prompts.json \
  --hardware gpu \
  --max-reward-regression 0.05 \
  --max-perplexity-regression 0.02 \
  --output outputs/routes/prompt_eval.json

# 6. Ask which route to serve a given task on which hardware.
adaptive-rl-quant-route recommend \
  --prompt-text "Generate a SQL query that aggregates monthly revenue per region." \
  --domain code \
  --hardware gpu
```

`adaptive-rl-quant-route --help` lists every subcommand and option. The catalog defaults to
`outputs/routes/catalog.json`; pass `--catalog` to use multiple catalogs side by side.

## JSON prompt evaluation

`evaluate-prompts` accepts either a JSON list or an object with a `prompts` list. Each item may
be a plain string or an object with `text` (or `prompt`), optional `id` / `prompt_id`, and
optional `domain`:

```json
{
  "prompts": [
    {"id": "sql_case", "domain": "code", "text": "Generate a SQL query for monthly revenue."},
    "Summarize this incident report for an executive audience."
  ]
}
```

For every prompt and hardware mode, the command evaluates every hardware-feasible catalog
route with the configured backend. It treats the highest reward as the reference, filters out
routes whose reward or perplexity regression exceeds the requested bounds, then recommends
the remaining route with the lowest measured `memory_mb`. The JSON report includes both
`rows` (all route measurements) and `recommendations` (the selected lowest-VRAM route per
prompt/hardware pair) so quality tradeoffs stay inspectable.

## Artifact layout

**Catalog and downloads:** the default catalog path is `outputs/routes/catalog.json` (`--catalog` overrides). `download` writes GGUFs under `outputs/models/<route_id>/` unless `--local-dir` is set. Both paths sit under `outputs_dir` when you relocate artifacts (see [CONFIG.md](CONFIG.md#output-paths)).

After `train` you get two new artifacts under `outputs/benchmarks/<run_name>_*`:

- `<run_name>_route_bandit.json` — full bandit state (per-bucket arm stats) plus the catalog
  snapshot it was trained against. Reload via `load_bandit_artifact`.
- `<run_name>_route_summary.json` — training pulls, mean reward, explore rate, per-bucket
  pull counts, optional greedy evaluation sweep, and the bandit's report card.

A JSONL telemetry stream lands under `outputs/logs/<run_name>_route_telemetry.jsonl` so you
can replay every pull (including the bandit's reasoning string).

## Hugging Face CLI integration

Both binaries are supported:

- **Modern**: `hf` (huggingface_hub ≥ 0.34).
- **Legacy**: `huggingface-cli`.

Install the lightweight route-download extra if you do not already have either CLI:

```bash
python3 -m pip install -e ".[hub]"
```

The wrapper resolves whichever is on `PATH` (override with `HF_CLI=/path/to/hf`). It only
spawns subprocesses with **validated argv lists** — no shell expansion — so you can include
free-form notes / prompts in your catalog without worrying about quoting.

To download with a specific Hugging Face token, run the CLI's `login` command yourself first
(`hf auth login` or `huggingface-cli login`); we never read tokens from the catalog or echo
them to logs.

## Programmatic API

The ``adaptive_quant.routes`` namespace re-exports [`model_routes`](../src/adaptive_quant/model_routes.py),
[`route_policy`](../src/adaptive_quant/route_policy.py), and [`route_pipeline`](../src/adaptive_quant/route_pipeline.py)
so you can import route-learning symbols from one place.

Everything the CLI does is exposed through the public Python surface:

```python
from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.routes import default_route_catalog, recommend_route, train_route_bandit
from adaptive_quant.types import HardwareType

config = FrameworkConfig()
catalog = default_route_catalog()
bandit, summary = train_route_bandit(config, catalog=catalog, iterations=512)

selection = recommend_route(
    config=config,
    bandit=bandit,
    prompt_text="Plan a debugging session for distributed inference latency spikes.",
    domain="systems",
    hardware=HardwareType.GPU,
)
print(selection.route.repo_id, selection.route.quant_label, selection.score)
```

## How the reward maps to route choice

Each pull constructs a `QuantizationDecision` whose `effective_layer_bits` are pinned to the
route's effective bits-per-weight (e.g. `Q4_K_M ≈ 4.83 bpw`). The simulator scores the
decision with the existing latency / throughput / memory / perplexity formulas, then route
size is checked against `HardwareProfile.memory_budget_mb`: routes that exceed the budget
absorb a heavy linear penalty so the bandit cannot rationalize an OOM-ing arm with throughput.

Switching backends works the same way — set `config.backend = "llama_cpp"` (with
`llama_cpp_binary` and `llama_cpp_model` configured) and register each route with a
`local_path`. The bandit will pass each route's local GGUF to `llama.cpp` as the measured
model. Use `--require-local-models` for research runs so missing files fail before training.

`llama.cpp` route runs write a paper bundle under `outputs/paper_bundles/<run_name>/`.
See [LOCAL_RESEARCH.md](LOCAL_RESEARCH.md) for the evidence levels, bundle files, and
claim-interpretation rules.

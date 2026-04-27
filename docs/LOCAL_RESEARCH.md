# Local llama.cpp Research Runs

This workflow is for **local, research-grade evidence** from real `llama.cpp` measurements. It is stronger than the simulator-only path because latency and throughput come from your local binary and GGUF files, but it is still not deployment-grade multi-device validation by itself.

## Evidence Levels

| Level | What it means |
| --- | --- |
| Simulator | All latency, throughput, memory, perplexity, and reward terms come from the built-in simulator. |
| Local `llama.cpp` | Latency and throughput come from a local `llama.cpp` binary and local GGUF files. Memory is measured when parseable. Perplexity remains simulator-derived unless you add an external quality metric. |
| Deployment-grade | Multiple target devices, real prompt distributions, end-to-end serving integration, and real quality measurements. This repo now helps produce the local evidence bundle, but does not claim this level automatically. |

## Route-Based Local Measurements

`llama.cpp` measures actual GGUF files, so local model comparisons should use the route workflow:

```bash
adaptive-rl-quant-route --catalog outputs/routes/local.json register \
  --route-id llama-q4km-local \
  --repo local/llama \
  --filename llama-Q4_K_M.gguf \
  --quant Q4_K_M \
  --local-path /absolute/path/to/llama-Q4_K_M.gguf \
  --hardware-hint gpu \
  --replace
```

Use a config with `backend="llama_cpp"`, `llama_cpp_binary`, and a fallback `llama_cpp_model`. During route training, each route's `local_path` is passed to `llama.cpp` as the `-m` model path.

```bash
adaptive-rl-quant-route \
  --catalog outputs/routes/local.json \
  train --config local_llama.json --iterations 128 --evaluate --require-local-models
```

## Paper Bundle Outputs

Every successful local research path writes a paper bundle under:

```text
outputs/paper_bundles/<run_name>/
```

The bundle includes:

- `manifest.json`: Python/platform details, config digest, `llama.cpp` binary/model hashes, generation settings, and metric source labels.
- `metrics_summary.csv` / `metrics_summary.json`: paper-facing scalar metrics.
- `episodes.csv`: flattened JSONL telemetry when route or episode logs are available.
- `aggregate_stats.csv` / `aggregate_stats.json`: multi-seed mean, std, standard error, 95% CI, and effect-size-vs-zero fields.
- `appendix.md`: links to upstream artifacts and a reproducibility checklist.
- `claims_validation.json` / `claims_validation.md`: explicit evidence-level warnings and deployment-grade status.

## Interpreting Claims

For `backend="llama_cpp"`, cite latency and throughput as local measured values. Cite reward as a mixed objective unless you have added real quality measurements, because the current perplexity term remains simulator-derived. Do not cite local results as deployment-grade unless you have separately run multi-device, real-traffic, end-to-end validation.

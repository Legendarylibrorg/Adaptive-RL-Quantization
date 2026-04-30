# Secure VM and Docker Run Guide

This guide is for running Adaptive RL Quantization with a stronger host boundary than a normal local install. The safest default is:

1. run inside a dedicated Linux VM,
2. run Docker inside that VM,
3. use the hardened simulator-only container by default,
4. opt in to network downloads, GPU devices, external binaries, and model files only when needed.

Docker improves repeatability and reduces the process privileges inside the VM. It is not a kernel security boundary by itself, so keep the VM boundary when you are evaluating untrusted code, models, checkpoints, configs, or native binaries.

## Recommended VM setup

Use a dedicated Linux VM for this project instead of running directly on your daily-use host.

- Create the VM from a current Ubuntu, Debian, Fedora, or similar image.
- Snapshot the VM before installing Docker, GPU drivers, or large model tooling.
- Do not copy personal SSH keys, cloud credentials, browser profiles, or token files into the VM.
- Disable shared folders by default. If you need one, mount it read-only unless you are intentionally exporting `outputs/`.
- Keep the repository, Docker data directory, and model cache inside the VM disk.
- Revert to a clean snapshot after testing untrusted artifacts.

For GPU work, use a single-purpose Linux VM or host with NVIDIA Container Toolkit. GPU passthrough exposes driver and device interfaces to the container, so it is a weaker isolation profile than simulator-only runs.

## Safe install policy

The default Docker build installs the package from the checked-out source tree and does not copy your local `.venv` into the image.

Use these rules for safer installs:

- Do not run `curl | bash` installers inside the VM or container.
- Do not enable `ADAPTIVE_RL_ALLOW_NETWORK_PIP_BOOTSTRAP` unless you intentionally accept the `get-pip.py` bootstrap path and pin `ADAPTIVE_RL_PIP_BOOTSTRAP_SHA256`.
- Prefer pinned base images, pinned package indexes, or an internal wheelhouse for high-assurance environments.
- Build simulator images first; install PyTorch only on a GPU VM that needs it.
- Treat `HF_CLI`, Hugging Face tokens, model downloads, and external `llama.cpp` paths as trusted configuration, not casual inputs.

The default `Dockerfile` pins the Python base image by tag and digest, then installs the CI bootstrap package set with `pip --require-hashes` before installing this package with `--no-build-isolation`. For release or high-assurance images, rotate the digest intentionally during base image updates and build optional PyTorch/GPU wheels from a controlled index or wheelhouse.

For air-gapped or high-assurance builds, prebuild wheels in a controlled environment and pass pip an internal index or wheelhouse during image build.

## Build and run

Build the default simulator image:

```bash
docker compose build
```

Run the smoke configuration under the hardened Compose profile:

```bash
docker compose run --rm adaptive-rl-quant
```

Run unit tests inside the same locked-down container:

```bash
docker compose run --rm adaptive-rl-quant python -m unittest discover -s tests -q
```

Run a different simulator config:

```bash
docker compose run --rm adaptive-rl-quant adaptive-rl-quant --config config.e2e_smoke.json
```

The default Compose service uses:

- a non-root user (`10001:10001`),
- read-only root filesystem,
- all Linux capabilities dropped,
- `no-new-privileges`,
- PID, CPU, and memory limits,
- a `tmpfs` `/tmp`,
- a named `adaptive_outputs` volume mounted only at `/app/outputs`.

Do not add broad host mounts such as `$HOME`, `~/.ssh`, cloud credential directories, or `/var/run/docker.sock`.

## No-network runtime

After the image has been built and all trusted artifacts are already present in the image or mounted volumes, run simulator commands with Docker networking disabled:

```bash
docker compose run --rm --network none adaptive-rl-quant
```

Do not use `--network none` for first-time package installs, Hugging Face downloads, or other workflows that intentionally need network access.

## GPU opt-in

GPU runs are intentionally separate from the default service because they expose NVIDIA devices and host driver interfaces.

Build and run the GPU override only on a GPU VM or host with NVIDIA Container Toolkit installed:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm adaptive-rl-quant
```

By default, the override exposes GPU `0`. Set `NVIDIA_VISIBLE_DEVICES=<id>` when you intentionally want a different GPU. Use GPU mode only for trusted training workloads. Keep secrets out of the container, pin model sources and revisions, and avoid shared GPU hosts for untrusted experiments.

## External binaries and model artifacts

The simulator backend is the safer default. Extra care is required for native tools and downloaded artifacts:

- `llama.cpp` binaries execute as code. Only mount and configure binaries you built or verified.
- GGUF/model files can be large and should be mounted explicitly, preferably read-only.
- Do not enable legacy PyTorch checkpoint loading for checkpoints you did not create.
- Pin Hugging Face revisions where possible and avoid passing long-lived tokens into containers.
- For `router_feature_backend="hf"`, prefer `router_hf_allowed_models`, `router_hf_embedding_revision`, and `router_hf_local_files_only=true` after warming a vetted cache.

Example read-only model mount:

```bash
docker compose run --rm \
  --volume "$PWD/trusted-models:/models:ro" \
  adaptive-rl-quant adaptive-rl-quant-calibrate --help
```

## Cleanup

Remove stopped containers and the named output volume when you no longer need local artifacts:

```bash
docker compose down --volumes --remove-orphans
```

Inside a disposable VM, prefer reverting to a clean snapshot after testing untrusted workloads.

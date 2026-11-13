# Person 1 — Stable Diffusion 1.5

See [TODO.md](TODO.md) for the code implementation checklist and
[ASSESSMENT.md](ASSESSMENT.md) for the project audit, hardware feasibility,
and paid-compute budget.

Install diffusion dependencies with `python -m pip install -e '.[diffusion]'`.
On CUDA nodes, install the appropriate PyTorch/torchvision build for the node
before installing this extra. Data preparation uses the separate `.[data]` extra.

## LoRA training

Run from the repository root with the Python 3.12 environment activated. Install
the diffusion extra above, then obtain the pinned upstream training script once:

```bash
git clone --depth 1 --branch v0.35.1 https://github.com/huggingface/diffusers.git external/diffusers
```

Preview the command without importing PyTorch, downloading models or starting training:

```bash
genmoji diffusion train --dry-run --device cuda
```

Train on one allocated CUDA GPU (including one 2080 Ti):

```bash
genmoji diffusion train --device cuda --output outputs/diffusion/lora
```

On the M4 Pro, first run a short MPS pilot:

```bash
genmoji diffusion train --device mps --steps 50 --checkpoint-every 25 --output outputs/diffusion/mps-pilot
```

Extend a run from its latest checkpoint; `--steps` is the new total update target:

```bash
genmoji diffusion train --device cuda --resume latest --steps 2000 --output outputs/diffusion/lora
```

Defaults are 512px, rank 4, learning rate 1e-4, microbatch 1, accumulation 4,
1,000 updates and gradient checkpointing. CUDA uses FP16; MPS/CPU use FP32.
Use `--help` for overrides, including `--no-gradient-checkpointing`.
Launch the command directly in a single-GPU allocation; do not wrap it in
`accelerate launch` or `torchrun`. On a two-GPU node, select one GPU using your
cluster allocation or `CUDA_VISIBLE_DEVICES`.

The launcher validates the prepared training metadata and records a content hash
of training files. It resolves the Hub model revision to an immutable commit and
downloads only required training components to the Hugging Face cache. The initial
download requires network access and several GB of disk space. The default model
and its SD 1.5 component layout are supported; alternative architectures are not.

Outputs include:

- `pytorch_lora_weights.safetensors`: standalone adapter for loading with
  `StableDiffusionPipeline.load_lora_weights(...)`.
- `checkpoint-N/`: optimizer/model/RNG state and adapter exports for resume.
  Only the last three checkpoints are retained by default.
- `training.json`, `run-*.json`, and `dataset-manifest.json`: settings, exact
  command, resolved model revision, dataset hashes, package versions and status.
- TensorBoard events under `logs/` for training loss and learning rate.

A nonempty output directory requires explicit resume. Resume reuses the original
model commit and rejects changed data, rank, learning rate, batch/accumulation,
seed, device type or precision. Supply the same nondefault settings when resuming.
The upstream trainer may replay batches within the resumed epoch; do not assume
bitwise equivalence to an uninterrupted run. Adapter files alone cannot restore
optimizer state. No baseline images or Person 3 implementation are needed to train.

Training uses the prepared `imagefolder/train` split only. Fixed-prompt image
validation and quantitative evaluation remain separate TODOs; loss alone does not
establish emoji quality. MPS training and CUDA training still require real hardware
smoke tests before relying on a long run.

The optimization loop and adapter exports use the unmodified
[Diffusers v0.35.1 trainer](https://github.com/huggingface/diffusers/blob/v0.35.1/examples/text_to_image/train_text_to_image_lora.py).
Baseline generation and adapted comparison tooling remain to implement.

Use the shared 512px OpenMoji ImageFolder; the exact trainer arguments are in
[the dataset guide](../../docs/dataset.md#stable-diffusion-15--person-1).
Write adapters to `outputs/diffusion/lora/`, generated images and logs to
`outputs/diffusion/`. Log model revision, dataset manifest, prompt, seed, steps,
guidance, precision, resolution, elapsed time, and peak memory where available.
Coordinate model loading and scheduler choices with Person 3.

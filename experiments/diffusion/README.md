# Person 1 — Stable Diffusion 1.5 and SDXL

See [TODO.md](TODO.md) for the code implementation checklist and
[ASSESSMENT.md](ASSESSMENT.md) for the project audit, hardware feasibility,
and paid-compute budget.

Install diffusion dependencies with `python -m pip install -e '.[diffusion]'`.
On CUDA nodes, install the appropriate PyTorch/torchvision build for the node
before installing this extra. Data preparation uses the separate `.[data]` extra.

## LoRA training

For Google Colab, open [colab_training.ipynb](colab_training.ipynb) after
pushing it and the training code to the `diffusion` branch. The notebook includes
dataset backup and checkpoint recovery instructions. Set `ARCHITECTURE = "sd15"`
or `"sdxl"` in its first code cell before running the notebook. This selection
controls SVG-to-PNG conversion, dataset archives, model/trainer selection,
training, resume and baseline/adapted PNG generation. SD1.5 uses 512px and
`lora/`; SDXL uses 1024px and `lora-sdxl/`. Each family has a separate Drive
archive and output folder. Keep the same selection when resuming.

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

For SDXL, use the same architecture flag for preparation and training:

```bash
genmoji prepare --architecture sdxl
genmoji diffusion train --architecture sdxl --dry-run --device cuda
genmoji diffusion train --architecture sdxl --device cuda
# Extend the SDXL run:
genmoji diffusion train --architecture sdxl --device cuda --resume latest --steps 2000
```

SDXL defaults to `stabilityai/stable-diffusion-xl-base-1.0`, 1024px PNGs rendered
directly from SVG, `data/processed/openmoji-1024`, and
`outputs/diffusion/lora-sdxl`. It uses the unmodified
[Diffusers v0.35.1 SDXL trainer](https://github.com/huggingface/diffusers/blob/v0.35.1/examples/text_to_image/train_text_to_image_lora_sdxl.py),
including both tokenizers and text encoders. Training updates UNet LoRA layers;
text encoders stay frozen, as in the SD1.5 recipe. The SDXL trainer keeps the
base VAE in FP32 for numerical stability. Start with a short pilot on your GPU;
1024px SDXL requires substantially more memory than SD1.5, and the notebook's
pilot is not a guarantee that it fits a T4. Use a larger GPU if needed.

SD1.5 defaults remain 512px. Both families default to rank 4, learning rate 1e-4, microbatch 1, accumulation 4,
1,000 updates and gradient checkpointing. CUDA uses FP16; MPS/CPU use FP32.
Use `--help` for overrides, including `--no-gradient-checkpointing`.
Launch the command directly in a single-GPU allocation; do not wrap it in
`accelerate launch` or `torchrun`. On a two-GPU node, select one GPU using your
cluster allocation or `CUDA_VISIBLE_DEVICES`.

The launcher validates the prepared training metadata and records a content hash
of training files. It resolves the Hub model revision to an immutable commit and
downloads only required training components to the Hugging Face cache. The initial
download requires network access and several GB of disk space. The SD1.5 and SDXL base component layouts are supported. Use `--architecture`
to choose the layout; `--model` alone does not select an architecture.

Outputs include:

- `pytorch_lora_weights.safetensors`: standalone adapter for loading with
  `StableDiffusionPipeline.load_lora_weights(...)` or
  `StableDiffusionXLPipeline.load_lora_weights(...)` for SDXL.
- `checkpoint-N/`: optimizer/model/RNG state and adapter exports for resume.
  Only the last three checkpoints are retained by default.
- `training.json`, `run-*.json`, and `dataset-manifest.json`: settings, exact
  command, resolved model revision, dataset hashes, package versions and status.
- TensorBoard events under `logs/` for training loss and learning rate.

A nonempty output directory requires explicit resume. Resume reuses the original
model commit and rejects changed data, rank, learning rate, batch/accumulation,
seed, architecture, device type or precision. Supply the same nondefault settings when resuming.
The upstream trainer may replay batches within the resumed epoch; do not assume
bitwise equivalence to an uninterrupted run. Adapter files alone cannot restore
optimizer state. No baseline images or Person 3 implementation are needed to train.

Training uses the prepared `imagefolder/train` split only. Fixed-prompt image
validation and quantitative evaluation remain separate TODOs; loss alone does not
establish emoji quality. MPS training and CUDA training still require real hardware
smoke tests before relying on a long run.

The optimization loop and adapter exports use the unmodified
[Diffusers v0.35.1 trainer](https://github.com/huggingface/diffusers/blob/v0.35.1/examples/text_to_image/train_text_to_image_lora.py).
Automated comparison grids and evaluation tooling remain to implement.

## Generate baseline and adapted images

Install `.[diffusion]`, then run from the repository root. Supply the training
record for both runs so the base-model revision is identical:

```bash
python experiments/diffusion/generate.py --training-record outputs/diffusion/lora/training.json
python experiments/diffusion/generate.py --training-record outputs/diffusion/lora/training.json --adapter outputs/diffusion/lora/pytorch_lora_weights.safetensors
```

The adapter can be a downloaded `.safetensors` file or a folder containing
`pytorch_lora_weights.safetensors`. Pass `--training-record` explicitly if
`training.json` was downloaded separately. Use the same options for both runs.
By default three prompts and three seeds are generated in separate `baseline/`
and `adapted/` folders under `outputs/diffusion/images/`, with JSON metadata.
Repeat `--prompt` and `--seed` to override defaults, and pass `--output` to
choose another parent folder. Existing files are never overwritten. CUDA, MPS
and CPU are detected automatically; the first run downloads the base model.
Adapted image filenames include the training step when it can be verified from
the matching training record or checkpoint directory (otherwise `step-unknown`).
Baseline filenames end in `base`. Sampling steps are recorded separately in JSON.
Generation infers the architecture from `training.json` (older records mean
SD1.5), or accepts `--architecture sd15|sdxl` explicitly. Without a record it
defaults to SD1.5. For an SDXL baseline without a training record:

```bash
python experiments/diffusion/generate.py --architecture sdxl --output outputs/diffusion/images-sdxl
```

Use `--export-size 128` (or 64, etc.) to downscale when saving. Generation
occurs at the selected model's native size (512px or 1024px), so export size
does not reduce inference time or memory. PNG dimensions and metadata follow
the selection. Use separate image output folders when comparing model families.

Use the 512px (SD1.5) or 1024px (SDXL) OpenMoji ImageFolder; the exact trainer arguments are in
[the dataset guide](../../docs/dataset.md#stable-diffusion-15--person-1).
Write adapters to `outputs/diffusion/lora/`, generated images and logs to
`outputs/diffusion/`. Log model revision, dataset manifest, prompt, seed, steps,
guidance, precision, resolution, elapsed time, and peak memory where available.
Coordinate model loading and scheduler choices with Person 3.

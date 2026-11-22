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
training, resume and baseline/adapted PNG generation. The notebook uses separate
90/10 dataset archives and Drive output folders for SD1.5 at 512px
(`lora-90-10/`) and SDXL at 1024px (`lora-sdxl-90-10/`). Keep the same
selection when resuming.

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
directly from SVG, `data/processed/openmoji-1024-90-10`, and
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
  Save every 50 updates; keep the newest three plus every 500-step milestone
  permanently by default. The Colab notebook keeps the newest two plus milestones.
  At step 950 the notebook retains 500, 900 and 950. Use `--keep-checkpoints`
  to set the recent count and `--keep-every` to set the milestone interval
  (`0` disables milestones). The milestone interval must be a multiple of the
  checkpoint interval. Milestones add to storage usage.
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

Checkpoint retention runs after a full Accelerate save returns successfully,
using a wrapper around the unmodified upstream trainer. The upstream pre-save
deletion is disabled so it cannot remove milestones or prune before a failed save.
This confirms filesystem completion, not remote synchronization of a Drive mount.
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
The default is [Text-to-Emoji v1](../../benchmarks/text-to-emoji-v1/evaluation_protocol.md).
It generates 90 images per condition using the 30 exact prompts and seeds 42,
123, and 456. PNGs and JSON sidecars go under
`outputs/diffusion/text-to-emoji-v1/<model>/<condition>/<prompt_id>/`. The
sidecars record the exact prompt, category, expected elements, and sampling
settings. The condition is `base` or the adapter's verified training step,
with the weight included when explicitly set.

Use `--prompt-id` and `--seed` to run a smaller benchmark subset, or repeat
`--prompt` to use custom subjects with the OpenMoji training-style template.
Repeat `--raw-prompt` for custom prompt text passed verbatim to the model.
Use `--output` to select a result root. Existing files are protected. CUDA, MPS,
and CPU are detected automatically. The first run downloads the base model.
Generation infers the architecture from `training.json` (older records mean
SD1.5), or accepts `--architecture sd15|sdxl` explicitly. Without a record it
defaults to SD1.5. For an SDXL baseline without a training record:

```bash
python experiments/diffusion/generate.py --architecture sdxl
```

Benchmark exports default to 512px. SDXL generates at 1024px and resizes to
512px on export. Use `--export-size` for an exploratory comparison at a
different viewing size. Generation still uses the model's native resolution.

Use the 512px (SD1.5) or 1024px (SDXL) OpenMoji ImageFolder; the exact trainer arguments are in
[the dataset guide](../../docs/dataset.md#stable-diffusion-15--person-1).
Write adapters to `outputs/diffusion/lora/`, generated images and logs to
`outputs/diffusion/`. Log model revision, dataset manifest, prompt, seed, steps,
guidance, precision, resolution, elapsed time, and peak memory where available.
Coordinate model loading and scheduler choices with Person 3.

## SDXL adapter-strength sweep

From the repository root, activate the environment and run:

```bash
source .venv/bin/activate
bash experiments/diffusion/sweep_sdxl.sh
```

The default strength sweep renders seven images on MPS with DDIM, 30 steps,
and CFG 7.5. It compares base SDXL with the 250- and 1,000-update checkpoints
at weights 0.25, 0.5, and 1.0. All seven use benchmark prompt
`relationships_01` (a sleepy penguin drinking coffee while wearing a blue hat)
and seed 42. Outputs go to `outputs/diffusion/sdxl-strength-quick/`. Use
`--prompt-id` or `--seed` to change the selection.

The full 630-image run is opt-in:

```bash
bash experiments/diffusion/sweep_sdxl.sh --full
```

Add `--dry-run` to preview reuse and output counts. Both checkpoint folders under
`outputs/diffusion/lora-sdxl/` need the standalone
`pytorch_lora_weights.safetensors` file. If you move the sweep to a different
device, pass `--device cuda` or `--device cpu`. Existing images generated on MPS
are eligible for reuse on MPS when prompt text, seed, and generation settings match.

Full sweep results go to `outputs/diffusion/sdxl-strength-sweep/`. Pass
`--reuse-from outputs/diffusion/sdxl-strength-quick` to reuse matching images
from a completed quick sweep. Use the benchmark rubric to rate emoji style and
prompt adherence independently. If interrupted,
start with a new `--output` and add `--reuse-from` pointing to the partial output
folder to reuse completed PNG and JSON pairs.

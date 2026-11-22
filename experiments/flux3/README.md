# FLUX 3 Image Klein with MFLUX

This setup targets Apple Silicon and uses MFLUX 0.24.0 with the distilled Q4
checkpoint by default and Q8 as an option. Each image is generated at **512 × 512**
with standard memory settings, 16 steps, and the model's flow-map schedule. The runner
passes each benchmark prompt verbatim. [MFLUX model documentation](https://github.com/mflux-community/mflux/blob/main/src/mflux/models/flux3/README.md) · [Q4 checkpoint](https://huggingface.co/mflux-community/flux-3-image-klein-mflux-q4)

## Preview

From the repository root, use Python 3 to print the three smoke-test commands.
Preview reads the prompt file and needs only the Python standard library.

```bash
python3 experiments/flux3/run_benchmark.py
```

## Install and download

Keep using the repository's Python 3.12 `.venv` for GenMoji. The upstream
`uv tool install` workflow gives the MFLUX CLI its own Python environment and
exposes `mflux-generate-flux3` on `PATH`. The benchmark script calls that CLI as
a subprocess, so you can run the script from your active `.venv`.
[uv tool environments](https://docs.astral.sh/uv/concepts/tools/#tool-environments)

If `uv --version` fails on macOS, install it with `brew install uv` or use the
[official installation guide](https://docs.astral.sh/uv/getting-started/installation/).
Then install the pinned MFLUX version.

```bash
uv tool install --python 3.12 'mflux==0.24.0'
```

If `command -v mflux-generate-flux3` finds nothing, run `uv tool update-shell`
and open a new terminal. From the repository root, download the pinned checkpoint
to the ignored `downloads/` directory.

```bash
uv tool run --from huggingface_hub hf download mflux-community/flux-3-image-klein-mflux-q4 --revision 79c0034de040c460e9685bf0127f40af042a2abe --local-dir downloads/flux-3-image-klein-mflux-q4-79c0034
```

For Q8, download the [prequantized checkpoint](https://huggingface.co/mflux-community/flux-3-image-klein-mflux-q8) (about 9.9 GB) and select it with `--quantization q8`.

```bash
uv tool run --from huggingface_hub hf download mflux-community/flux-3-image-klein-mflux-q8 --revision 7e46237dea3999a9792959f5e000aeb1b9f9e7df --local-dir downloads/flux-3-image-klein-mflux-q8-7e46237
python3 experiments/flux3/run_benchmark.py --quantization q8 --run
```

## Generate images

From the repository root, use the Python available in your shell. If you
previously activated `downloads/mflux-venv`, run `deactivate` before choosing
a run below so the script finds the uv-installed MFLUX executable. Each command
starts inference only when you pass `--run`.

```bash
# One image using seed 42
python3 experiments/flux3/run_benchmark.py --run --prompt-id simple_01

# Three images using seed 42, one prompt per category
python3 experiments/flux3/run_benchmark.py --run

# Full benchmark with 90 images using seeds 42, 123, and 456
python3 experiments/flux3/run_benchmark.py --run --full
```

PNG files and JSON metadata are written to
`outputs/text-to-emoji-v1/flux-3-image-klein-mflux-<quantization>/<condition>/<prompt_id>/`.
The smoke condition is `pretrained-standard-smoke` and the full benchmark
condition is `pretrained-standard`. The earlier low-RAM runs used
`pretrained-smoke` and `pretrained`. Press Ctrl-C to stop and rerun the same
command to resume. The runner starts a fresh MFLUX process for each image. For
timing comparisons, use the [benchmark's warm-up protocol](../../benchmarks/text-to-emoji-v1/evaluation_protocol.md).

## Compare with the SDXL strength-validation run

Generate the three prompts from that run with Q8 and seed 123. The FLUX runner
generates at 512 × 512. The SDXL run generated at 1024 × 1024 and exported at
512 × 512. Compare with the SDXL `base` condition for the pretrained comparison.

```bash
python3 experiments/flux3/run_benchmark.py --quantization q8 --seed 123 --prompt-id attributes_07 --prompt-id relationships_01 --prompt-id relationships_03 --run
```

## Available checkpoints

MFLUX Community publishes Q3 (4.2 GB), Q4 (5.3 GB), Q5 (6.5 GB), Q6 (7.6 GB),
Q8 (9.9 GB), and BF16 (18.5 GB) for the distilled model. These figures are
checkpoint sizes on disk. Each prequantized checkpoint stores its quantization
level in the weights. [Checkpoint sizes and usage](https://huggingface.co/mflux-community/flux-3-image-klein-mflux-q3#flux3-image-klein--mflux-q3)

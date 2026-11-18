# GenMoji: Sana baseline (Person 2)

Reproducible pretrained text-to-emoji generation with Sana 600M at 512×512 on Apple Silicon. This baseline performs no training or sampling experiments. The original `sana-emoji/test_sana.py` is preserved.

## Setup

Run commands from the `sana-emoji/` folder (`cd sana-emoji` from the repository root). Use the existing working environment:

```bash
source .venv/bin/activate
python -c 'import torch; print(torch.backends.mps.is_available())'
```

The verified environment is Python 3.14.4 ARM64 on an Apple M4 Pro (24 GB unified memory). MPS must report `True`; run in a normal local terminal if a sandbox hides the GPU. The runner deliberately stops when MPS is unavailable, since the working fp16 configuration is intended for this device.

For a fresh environment, use the same Python/architecture and install the pinned direct dependencies (PyYAML and Pillow are already installed in the existing environment):

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-baseline.txt
```

The first model load may download the pinned Hugging Face checkpoint. Use `--local-files-only` to require cached files. Nothing needs to be installed when using the existing environment. Each generated sidecar records actual package versions; the requirements file pins direct dependencies, not the full transitive environment.

## Run

Validate inputs without loading the model:

```bash
python run_sana_baseline.py --validate-only
```

Start with three prompts and seed 42:

```bash
python run_sana_baseline.py --limit 3 --seeds 42 --output-dir results/smoke_seed42 --local-files-only
```

`--limit` selects the first N prompts in file order, so the first three are simple concepts. The default seed is 42. When ready, run the 30-prompt baseline or the complete 90-image shared evaluation explicitly:

```bash
python run_sana_baseline.py
python run_sana_baseline.py --seeds 42 123 456 --output-dir results/all_seeds
```

The model loads once per process only when outputs need generation. Every image uses 20 steps and CFG 4.5. Loading mirrors the working test: fp16 checkpoint/transformer, float32 text encoder and VAE, and an independently seeded CPU generator for each image.

Results default to `results/pretrained_baseline/{prompt_id}/seed_{seed}.png`, with a `.json` sidecar containing full prompt, model revision, settings, scheduler configuration, seed, software/device information, synchronized generation time and image hash. `contact_sheet.png` shows the selected images with prompt labels. Results are ignored by Git.

Rerun the same command to resume: valid matching image/metadata pairs are skipped without loading the model when all are complete. Missing/corrupt pairs are regenerated. Settings conflicts stop with an error; choose another output directory for another condition, or use `--overwrite` to explicitly regenerate selected images. Do not run two processes writing the same output directory simultaneously.

No extra warmup images are generated. The first generation is marked cold; exclude it from steady-state speed comparisons. Loading and file saving are excluded from synchronized inference timing. Exact pixels can vary across software/hardware versions despite fixed seeds.

## Shared evaluation

- `../shared/prompts.json`: 30 frozen prompts, ten each for simple concepts, multiple attributes and complex relationships, with a common emoji suffix and scoring checks.
- `../shared/experiment_config.yaml`: common image size and seed IDs, pinned Sana settings and logging conventions. The runner reads these settings directly.
- `../shared/evaluation_protocol.md`: comparison, human scoring and timing rules for Sana, Stable Diffusion and future custom guidance conditions.

Use original images for scoring and record all outcomes. Numerical seeds align benchmark IDs across models; they do not imply matching noise across architectures.

## Local validation (2026-11-17)

Validated JSON/YAML contents, equal category counts, required example prompts, consistent suffixes, Python compilation, completion checks, settings conflicts and corrupt-image detection. Generated only the first three prompts at seed 42 on MPS: 4.54s (cold), 3.84s and 3.60s. Inspected `results/smoke_seed42/contact_sheet.png`; all images are 512×512 RGB. A repeat invocation skipped all three with no model load. These functional smoke results do not measure cross-model quality or benchmark speed.

Diffusers emitted a `torch_dtype` deprecation warning and a VAE casting advisory with the preserved loader configuration; neither prevented generation. No full benchmark, training or sampling experiments were run.

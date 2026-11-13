# Person 1 — Code implementation TODO

## Baseline generation

- [ ] Add `src/genmoji/diffusion.py` to load SD 1.5 on CUDA, MPS or CPU,
  with a pinned model revision and configurable scheduler. Expose pipeline
  components so Person 3 can reuse them for custom guidance.
- [ ] Add `genmoji diffusion generate` for prompt JSONL files, fixed seeds,
  512px output, sampling steps and CFG settings. Support both the base model
  and a saved LoRA adapter through the same generation path.
- [ ] Save PNGs and JSONL metadata containing prompts, seeds, model/adapter
  identity, scheduler, precision, settings, runtime and device memory usage.
- [ ] Add a benchmark command with warm-up, synchronized timing and peak-memory
  measurements for the selected backend.

## LoRA training

- [x] Add a training launcher around the Diffusers v0.35.1 text-to-image LoRA
  script, consuming the existing `imagefolder/train` dataset. Save the resolved
  command, environment versions and dataset manifest with each run.
- [x] Add a configurable starting recipe: UNet LoRA rank 4, frozen VAE/text
  encoder, 512px, no flips, learning rate 1e-4, effective batch 4 and 1,000
  optimizer updates. Support CUDA FP16 and MPS FP32 with gradient checkpointing.
- [x] Support checkpoint saving/resuming and standalone safetensors adapter export.
- [ ] Add fixed-prompt validation generation at saved checkpoints.

## Comparisons and evaluation tools

- [ ] Add shared prompt/seed configuration and a paired-run command that
  generates baseline and adapted images with identical inputs and settings.
- [ ] Add labeled comparison grids and configurable sweeps over checkpoints,
  adapter strength and sampling steps.
- [ ] Export a shuffled human-rating CSV and summarize ratings by prompt
  category, alongside training time, inference time and memory measurements.
- [ ] Add integration checks for generation metadata, training command arguments,
  checkpoint resume and adapter reload. Keep model/GPU checks opt-in.

# Bounded Sana LoRA pilot plan

Prepared 2026-11-18. **Pilot not executed.** Dataset preparation and CPU-only control tests are complete. This script extends the successful corrected-schedule MPS feasibility probe; the new multi-step loop, feature caching, checkpoint/resume and validation generation have not yet been exercised end to end on Sana. Review the dataset before launching.

## Proposed experiment

Adapt the pretrained Sana 600M 512px checkpoint using 192 balanced OpenMoji images. Maximum **200 optimizer updates**, batch size 1: approximately one shuffled pass plus eight samples. Retain adapters at 50, 100 and 200. Keep the original 30-prompt pretrained benchmark untouched. This pilot tests adaptation behavior and sustained execution, not architectural superiority or optimal sampling.

`pilot_config.yaml` specifies:

- Checkpoint `Efficient-Large-Model/Sana_600M_512px_diffusers`, revision `83d7a190bfd1fd070570a793d2dab5c7a3231b9d`, existing fp16 variant loaded into float32.
- MPS float32 computation; LoRA rank 4/alpha 4, Gaussian initialization, `to_q/to_k/to_v` in self/cross attention.
- Frozen base transformer, text encoder and VAE. Only adapters enter AdamW: lr 1e-4, betas 0.9/0.999, eps 1e-8, weight decay 1e-4, foreach=False. Norm clipping 1.0; gradient checkpointing enabled; no accumulation, augmentation or prior loss.
- Explicit **FlowMatchEulerDiscreteScheduler** from the pinned scheduler subfolder. Uniform timestep index sampling, unit loss weighting, scaled VAE latents; `x_s=(1-s)*x+s*noise`, target `noise-x`, float32 mean squared error. Sigma is checked in [0,1]. This matches corrected feasibility, not the pipeline's inference scheduler.
- No AMP/scaler, CUDA-only optimization, operator CPU fallback or custom MPS watermarks. Local cached checkpoint files required; the script does not download a new model.

The sampler distribution and objective are unchanged. Multi-step RNG behavior is necessarily extended: independent CPU noise-generator seed 42 evolves across updates rather than resetting each step. LoRA initialization is seeded 42 independently of cache creation. Global training RNG is seeded 43 to separate timestep sampling from initialization. Per-epoch shuffling uses `random.Random(42 + epoch)` and step-derived positions. All relevant tensor RNG states are checkpointed. These RNG choices differ from the single-batch probe but not its timestep distribution or target.

## Data and model loading

Shared data: `datasets/openmoji-17.0.0/`; pilot manifest: `manifests/pilot_train.jsonl`. Validation images are never passed to the training optimizer.

On first launch, frozen text embeddings and image latents are precomputed **one image at a time on MPS** for the 192 training examples. Embeddings are computed first, then the text encoder is released; VAE latents follow, then the VAE is released. Features are saved as safetensors under the run directory. The transformer is subsequently moved to MPS. This extends the verified sequential-encoding approach while avoiding loading both frozen encoders onto MPS during training. Cache creation can take substantially longer than 200 transformer-only updates and uses roughly 0.5 GiB disk for text embeddings plus latent tensors. It is not performed by a dry run.

Run fingerprints bind configuration, manifest contents, trainer source and recorded package versions. Changed configuration/data/code requires a new run directory rather than an implicit incompatible resume. Features are local generated artifacts; do not edit or share a writable cache between active runs. CPU is used explicitly for preprocessing, RNG, checkpoint staging and serialization; neural compute stays on MPS.

No dependency upgrades were made. Use `sana-emoji/.venv` and the existing isolated PEFT 0.21.3 in `sana-emoji/feasibility/vendor`. If recreating the environment, install baseline requirements and PEFT with `--no-deps --target` as documented in `TRAINING_FEASIBILITY.md`. Dataset tools need only already-installed Pillow/NumPy; configuration uses already-installed PyYAML.

## Commands (from repository root)

Revalidate and inspect without training:

```bash
sana-emoji/.venv/bin/python datasets/tools/prepare_openmoji.py --validate-only
sana-emoji/.venv/bin/python sana-emoji/pilot/test_pilot.py
sana-emoji/.venv/bin/python sana-emoji/pilot/train_pilot.py --config sana-emoji/pilot/pilot_config.yaml
```

**Proposed launch command; not executed:**

```bash
sana-emoji/.venv/bin/python sana-emoji/pilot/train_pilot.py \
  --config sana-emoji/pilot/pilot_config.yaml \
  --output-dir sana-emoji/pilot/runs/openmoji_rank4_v1 \
  --train
```

Use a normal local terminal with MPS access. The script requires `--train`; otherwise it validates inputs and prints a plan without importing/loading Sana. The maximum step budget is an absolute total, not an additional count on resume. The program rejects limits above 200 and restricts all training outputs to named subdirectories of `sana-emoji/pilot/runs/`.

Resume a safely checkpointed run with the same command plus `--resume`. It restores adapter, optimizer, CPU/MPS RNG and the dedicated CPU noise generator, and derives the next shuffled image from the saved completed step. A completed 200-step run performs no more updates. Numerical equality across software/hardware is not guaranteed, and exact resumed trajectory has not been validated on MPS yet.

## Logging and crash safety

`loss.jsonl` logs sample ID, step, loss, gradient norm, synchronized forward/backward/update time, sigma and sampled peak MPS driver/live allocation. Memory is sampled every 50 ms across preprocessing and training; brief peaks may be missed. It is not a measurement of unique total unified memory, and memory counters must not be added to RSS. Finite loss, every trainable gradient, nonzero aggregate gradients and finite updated weights are required.

Every successful update is serialized into a temporary checkpoint directory, including adapter safetensors, optimizer state and RNG state, per-step metrics and file hashes. The completed directory is renamed into place only after saving. The program verifies adapter serialization and exact adapter tensor equality on resume. Latest plus steps 50/100/200 are retained; older rolling checkpoints are deleted within this new pilot run only. Extra disk writes are a deliberate tradeoff for safe resume.

A process lock prevents concurrent writers. An `update_in_flight.json` marker is created exclusively **before** optimizer.step and removed only after checkpoint commit. If interrupted between updating and committing, automatic resume refuses to repeat the uncertain update. A stale lock after a force-kill also refuses a new writer. Do not blindly delete either marker: inspect the PID, latest checkpoint, metric files and hashes first. Aborting safely is preferable to claiming an exact resume after an ambiguous update. Per-checkpoint metrics are authoritative if an interrupted log append omitted a line. Only load locally produced trusted `.pt` optimizer state; it uses PyTorch pickle serialization.

## Controlled validation generation

`validation_plan.json` freezes four held-out examples, one each from expressions, animals, food and objects. Use the same captions, seed **42**, resolution **512×512**, **20 steps**, CFG **4.5** for all three checkpoints. The validation script selects them deterministically from the pinned validation manifest. It loads the saved adapter in a **separate process after training has stopped**, using the model's original inference scheduler; training continues to use FlowMatchEuler. Generation is opt-in with `--generate`.

For example, after an actual checkpoint exists:

```bash
sana-emoji/.venv/bin/python sana-emoji/pilot/validate_pilot.py \
  --config sana-emoji/pilot/pilot_config.yaml \
  --checkpoint sana-emoji/pilot/runs/openmoji_rank4_v1/checkpoints/step_000050
```

That command prints the validation plan only. Add `--generate` to produce four images, metadata and a contact sheet under `runs/openmoji_rank4_v1/validation/step_000050/`. Repeat for `step_000100` and `step_000200` after training stops. Existing matching outputs are reused; conflicts abort. Validation refuses while the run lock is present to avoid loading a second model alongside training. No validation generation ran during preparation. These held-out prompts are separate from the unchanged shared 30-prompt evaluation benchmark.

## Review and pilot limitations

The shared dataset report documents content filtering and residual caption/duplicate uncertainty. Single-object source emojis will mainly teach visual style; they do not establish generalization to multi-object relations. Generic object concepts may occur in both splits, while detected related variants remain grouped; this is not a disjoint-concept benchmark. Do not train on the team's evaluation prompts.

The earlier measured peak MPS driver memory was about 12.46 GiB with a roughly 17.76 GiB recommended limit. This is evidence for a small pilot, not a sustained-memory guarantee. Monitor cache preprocessing, optimizer allocation and checkpoint I/O. Stop on OOMs or non-finite values without changing MPS safety limits. Review losses and fixed-prompt checkpoints before proposing larger training. This implementation is prepared and statically/control-flow tested; first-launch cache construction, multi-step MPS execution, serialization/resume and checkpoint inference need runtime validation during the subsequently authorized pilot. No pilot budget was consumed here.

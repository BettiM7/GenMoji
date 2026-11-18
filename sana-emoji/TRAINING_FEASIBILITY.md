# Sana LoRA feasibility on Apple MPS

Tested 2026-11-18 on Apple M4 Pro, 24 GB unified memory, macOS 26.3 ARM64.

**Conclusion: the corrected-schedule end-to-end optimizer smoke test succeeded on MPS. Sana 0.6B rank-4 LoRA is ready for a small, bounded pilot experiment using this configuration. Exactly one new optimizer update was performed in the corrected test; the earlier diagnostic update is retained separately.** No full training, epochs, quality evaluation or sampling experiments were performed.

## Environment and compatibility

| Component | Observed version |
| --- | --- |
| Python | 3.14.4 ARM64 |
| PyTorch | 2.15.0 |
| Diffusers | 0.43.0 |
| Transformers | 5.23.0 |
| Accelerate | 1.16.0 |
| PEFT, baseline environment | Not installed |
| PEFT, isolated test | 0.21.3 |

PEFT was installed with `--no-deps --target feasibility/vendor`. Only the test adds this directory to its Python import path. The baseline environment and dependency pins were not upgraded. PEFT imported and executed successfully on Python 3.14.4; its published classifiers list Python through 3.13, so this runtime result should not be mistaken for an upstream Python 3.14 certification. See [PEFT package metadata](https://pypi.org/project/peft/0.21.3/).

The exact checkpoint loaded successfully: `Efficient-Large-Model/Sana_600M_512px_diffusers`, revision `83d7a190bfd1fd070570a793d2dab5c7a3231b9d`, variant `fp16`. Loading that variant into float32 uses the existing cached files; it does not recover precision beyond the stored weights.

## Selected implementation

The small probe [feasibility/one_step_mps.py](feasibility/one_step_mps.py) adapts the model loading, PEFT adapters, interpolation and loss from [NVlabs' official Sana training script](https://github.com/NVlabs/Sana/blob/main/train_scripts/train_dreambooth_lora_sana.py). It retains Apache attribution. It is a single-update diagnostic, not a new training framework.

The NVlabs source accepts SanaTransformer2DModel checkpoints, freezes transformer/VAE/text encoder, and targets `to_q`, `to_k`, `to_v`. It explicitly handles MPS, disables native AMP there, rejects bf16 mixed precision, but casts the text encoder to bf16 and hardcodes CUDA in latent caching. The probe instead uses fp32 and sequential MPS encoding, then releases frozen encoders. No CUDA-only optimizations are used.

The [NVlabs instructions](https://github.com/NVlabs/Sana/blob/main/docs/sana_lora_dreambooth.md) demonstrate a larger 1600M model, not an explicit 600M guarantee. The exact 600M architecture was verified by this local run. The [Diffusers Sana example](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sana.md) is the preferred upstream training reference for future work, but the downloaded main-branch script requires Diffusers `0.44.0.dev0`; the installed version is `0.43.0`. Therefore it cannot run unmodified here. The NVlabs copy requires `0.32.0.dev0` or newer.

References were saved under `feasibility/sources/`, with URLs, retrieval date and SHA-256 hashes in `sources/manifest.json`. These identify the inspected contents even when upstream main changes. No upstream assertion of a tested Sana/Python 3.14/MPS combination was found. [Accelerate documents MPS training generally](https://huggingface.co/docs/accelerate/usage_guides/mps), which is not a guarantee for this particular model.

## Dataset

Prepared eight image-caption pairs from OpenMoji release **17.0.0**: red apple, banana, car, rocket, soccer ball, guitar, house and pizza. All eight images were loaded and hash-validated; only the apple pair was used for the single batch. Captions are project-authored, separate from evaluation prompts. No baseline images were training inputs.

All emojis are designed by [OpenMoji](https://openmoji.org/), the open-source emoji and icon project by HfG Schwäbisch Gmünd and contributors, under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The [OpenMoji repository](https://github.com/hfg-gmuend/openmoji) supplies the assets and attribution guidance. Original PNGs, per-file URLs/hashes, attribution and license are in `feasibility/data/openmoji/`. Originals are unchanged. Runtime processing composites transparency onto white, resizes to 512×512 with bilinear interpolation and normalizes RGB to [-1, 1].

## Exact configuration

| Setting | Value |
| --- | --- |
| Model / revision | Sana 600M 512px / revision above |
| Compute device | MPS; no CPU operator fallback |
| Compute precision | Float32 transformer, LoRA, text encoder, VAE and loss; no AMP/scaler |
| Trainable weights | LoRA only, rank 4, alpha 4, Gaussian initialization |
| Adapter targets | Self- and cross-attention `to_q`, `to_k`, `to_v` in all 28 transformer blocks |
| Adapter parameter count | 1,548,288 of 593,296,160 transformer parameters including adapters |
| Frozen weights | Base transformer, text encoder, VAE; no text encoder adapters |
| Batch / accumulation | 1 / 1 |
| Resolution / sequence length | 512×512 / 300 tokens |
| Optimizer | AdamW, lr 1e-4, betas (0.9, 0.999), eps 1e-8, weight decay 1e-4, foreach=False |
| Gradient clipping | Global norm 1.0 |
| Gradient checkpointing | Enabled |
| Seed | 42; fresh CPU noise generator seeded 42, noise transferred to MPS |
| Augmentation / prior loss | None / none |
| Validation inference | None |

Frozen text/image encoding runs on MPS under `no_grad`; each encoder is released before transformer training. CPU use for checkpoint staging, image preprocessing, random-noise creation and adapter serialization is explicit; neural forward/backward computation does not silently fall back to CPU. CUDA, bitsandbytes/8-bit Adam, xFormers, compilation, distributed training and automatic Hub uploads are disabled/not used. The inference settings of 20 steps and CFG 4.5 are not training hyperparameters.

### Correct training objective

Following the official script, load **FlowMatchEulerDiscreteScheduler** from the checkpoint's scheduler subfolder. Sample an index uniformly over its 1,000 training timesteps and retrieve the matching sigma. For scaled VAE latent `x`, Gaussian noise `epsilon` and sigma `s`:

```text
x_s = (1 - s) * x + s * epsilon
target_velocity = epsilon - x
loss = mean((transformer(x_s, timestep, caption) - target_velocity)^2)
```

The selected `none` weighting scheme means uniform timestep sampling and unit loss weights. VAE latents use `vae.config.scaling_factor`. This predicts flow velocity, not an SD epsilon target or a clean image.

## Earlier diagnostic results and limitations (historical)

| Check | Initial optimizer-update attempt | Correct-schedule verification |
| --- | --- | --- |
| All 8 dataset pairs load | Passed | Passed |
| Exact checkpoint loads | Passed | Passed |
| Forward / finite loss | Passed; 4.149449 | Passed; 0.0666372 |
| Backward / finite gradients | Passed | Passed |
| LoRA gradient tensors present | 336 | 336 |
| Nonzero gradient tensors | 168 | 168 |
| Pre-clipping gradient norm | 0.422342 | 0.0200318 |
| Optimizer updates | **1** | **0**, intentionally |
| Changed LoRA tensors | 168 | 0, as expected |
| Synchronized computation duration | 2.49 s, includes optimizer update | 0.66 s, forward/backward only |
| Total process duration | 69.02 s | 55.17 s |
| Adapter save/reload | Saved; initial loader error | Reloaded saved adapter; exact equality |

Only LoRA parameters required gradients. At initialization, the LoRA B matrices are zero: gradients reach B first while A has zero gradients. Thus 168 nonzero gradients and 168 changed tensors are expected, not evidence of disconnected adapters.

**Scheduler mistake:** the first probe used `pipe.scheduler`, which is this checkpoint's DPMSolver inference scheduler. Its uninitialized training sigmas produced sigma **2.567981** at timestep 554, outside the intended interpolation interval. The optimizer update proves MPS computation and parameter updates, but is not a valid standard-schedule training result. This was corrected in the probe to instantiate the official FlowMatchEuler training scheduler explicitly, with an assertion that sigma is in [0, 1]. Verification used sigma **0.555**, timestep 555, and passed without updating weights. To respect the original task's one-update limit, that first corrected verification did **not** run an optimizer step. The subsequently authorized corrected update is reported below. Treat both saved adapters strictly as diagnostic artifacts; do not use it as an emoji-quality model.

**Reload error:** the initial generic loader searched for `pytorch_lora_weights.bin`, while saving produced `pytorch_lora_weights.safetensors`. Exact exception:

```text
OSError: Error no file named pytorch_lora_weights.bin found in directory
.../feasibility/results/one_step_fp32/adapter.
```

The fix specifies `weight_name="pytorch_lora_weights.safetensors"`, `use_safetensors=True`, and `local_files_only=True`. A separate load of the existing artifact succeeded with every tensor exactly matching the saved values. It did not perform another optimizer update. The original failed result and traceback are retained rather than overwritten.

No MPS operator error, non-finite loss/gradient or out-of-memory exception occurred. Loader warnings included `torch_dtype` deprecation, float32 casting advisories and an ignored/deprecated Hub symlink argument. They did not prevent computation. The initial process returned failure at adapter reload even though its optimizer update had already succeeded.

### Memory

Both processes sampled peaks of **9.94 GiB MPS live allocations** and **12.46 GiB MPS driver allocation**. These are sampled every 50 ms, across the entire process including encoding; they are lower bounds on peaks, not isolated training-step peaks. PyTorch reported recommended maximum memory of **17.76 GiB**. Process peak RSS was **4.26 GiB** on the first run and **14.54 GiB** on verification. RSS and MPS driver memory overlap under unified memory and must not be added.

Default allocator high/low watermarks remained intact. The probe refuses custom watermark environment variables or CPU operator fallback. No safety-limit bypass was attempted. See [PyTorch's MPS environment variable documentation](https://docs.pytorch.org/docs/2.15/mps_environment_variables.html).

## Corrected end-to-end optimizer smoke test

The subsequent test used `--corrected-smoke`, a fixed separate output folder with an exclusive one-update marker. It started from the same pretrained checkpoint with fresh rank-4 adapters, not the earlier diagnostic adapter. Configuration, dataset and corrected loss formulation above are unchanged. All eight OpenMoji pairs were loaded and hash-checked; the first apple pair supplied the single batch at batch size 1. Testing all eight with optimizer updates would exceed the one-update scope.

| Measurement | Corrected test result |
| --- | --- |
| Status | Success, including save/reload |
| Training scheduler | FlowMatchEulerDiscreteScheduler |
| Timestep / sigma | 555 / 0.5550000072 |
| Finite flow-matching loss | 0.0666371882 |
| Finite LoRA gradient tensors | 336 of 336 |
| Nonzero gradient tensors | 168; expected B-matrix gradients at initialization |
| Pre-clipping gradient norm | 0.0200317539 |
| Frozen base gradients | Absent, explicitly asserted |
| Optimizer updates in this test | **Exactly 1** |
| Changed LoRA parameter tensors | 168; all updated parameters finite |
| Synchronized forward/backward/update duration | **0.52182 s** |
| Synchronized optimizer.step() only | **0.01538 s** |
| Total process duration, including load/encoding/save/reload | 39.77615 s |
| Sampled peak MPS driver allocation | **12.45580 GiB** (13,374,308,352 bytes) |
| Sampled peak MPS live allocation | 9.92830 GiB |
| Process peak RSS | 10.17708 GiB |
| Adapter saved/reloaded | Exact equality for every adapter tensor |

Timing excludes model loading, frozen-encoder preprocessing and adapter serialization from the training-step duration. Memory samples cover the entire process every 50 ms and may miss brief peaks; driver memory is not an isolated training-step peak. RSS overlaps MPS unified memory. Default MPS safety limits and no-CPU-fallback checks remained enabled.

There were **no errors**, non-finite values, MPS operator failures or OOMs. Non-blocking warnings were `torch_dtype` deprecation, ignored `local_dir_use_symlinks`, float32 casting advisories for VAE/transformer, and the macOS multiprocessing-redirection note. No adapter image generation or quality evaluation was performed.

The corrected result is `feasibility/results/corrected_schedule_one_step/result.json`; its updated adapter is `feasibility/results/corrected_schedule_one_step/adapter/pytorch_lora_weights.safetensors`. Console output is `feasibility/corrected_smoke_run.log`. The exclusive update marker prevents a repeated invocation from silently making another update. The old results were not overwritten. There have now been two optimizer updates across separate authorized tasks: the earlier schedule diagnostic and this one corrected update.

All **87 protected files** in `feasibility/corrected_preservation_hashes.json` were verified unchanged, including baseline images/metadata, shared prompts/configs, inference scripts and the dataset.

## Artifacts and reproducibility

- `feasibility/summary.json`: concise outcome and total update count.
- `feasibility/results/one_step_fp32/result.json`: original update metrics and exact reload failure.
- `feasibility/results/verification_no_update/result.json`: corrected-schedule forward/backward and reload checks.
- `feasibility/results/one_step_fp32/adapter/pytorch_lora_weights.safetensors`: diagnostic adapter only.
- `feasibility/results_run.log`, `feasibility/verification_run.log`: complete console logs.
- `feasibility/preservation_hashes.json`: verified hashes of 68 baseline, inference and shared-input files; none changed.

Local dependencies/results/logs are ignored by Git. To recreate the isolated PEFT installation from the repository root:

```bash
sana-emoji/.venv/bin/python -m pip install --no-deps \
  --target sana-emoji/feasibility/vendor \
  -r sana-emoji/feasibility/requirements-feasibility.txt
```

The data preparation script downloads only the eight pinned images and license. The training probe has no epoch loop and refuses another optimizer attempt if its update marker exists, including after a post-update error. `--corrected-smoke` selects the separate corrected test folder; its marker is now present. **Do not remove that guard to rerun training as part of this feasibility task.** The separate `--verify-without-update` mode checks forward/backward and reload only. Running MPS may require a normal local terminal because the sandbox hides the GPU.

## Recommendation

Local MPS is now ready for a small, bounded pilot using rank 4, batch size 1, fp32, gradient checkpointing and sequential frozen-encoder preprocessing. The corrected schedule, finite loss/backward pass, optimizer update and exact adapter reload are verified end to end. A pilot still needs a separately agreed step budget, dataset/caption plan and sustained-memory monitoring; this one-batch result establishes neither image quality nor long-run stability. No pilot has been started.

A remote CUDA GPU remains the more mature path for longer training, larger datasets/batches and the upstream mixed-precision workflow. It is not required merely because MPS cannot compute gradients: gradients and an update worked here. Choose remote CUDA if local memory pressure, throughput or backend issues appear during later authorized tests. No remote resources were started.

# Text-to-Emoji Shared Evaluation Protocol (v1)

## Goal
Compare image generators on the **same text prompts** while distinguishing (a) pretrained model capability, (b) emoji adaptation, and (c) guidance/sampling interventions. Do **not** conclude that architectural differences alone caused observed changes.

## Primary benchmark
- `prompts.json`: 30 frozen prompts: 10 simple, 10 attributes, 10 relationships.
- Use the exact `prompt` field as written for the primary baseline; no rewriting or extra style phrases.
- 512 × 512 RGB images on white background.
- Fixed seed IDs: 42, 123, 456. **Same numerical seeds do not imply identical initial noise across architectures.**
- Run 3 images per prompt for a complete baseline = **90 images per model/condition**. For initial smoke tests, use 1 seed and 3 prompts, but label these preliminary.
- Keep held-out evaluation prompts out of training captions insofar as possible. Basic concepts may overlap; the **novel combinations** should not be intentionally placed in training data.

## Human scoring (blind to model/condition if practical)
1. **Object/attribute checks:** rate each `expected_elements` item as present = 1, absent/incorrect = 0, uncertain = NA. For relation/action elements, require the actual relationship, not merely two objects in the scene. Aggregate per prompt, then by category.
2. **Emoji style:** rate 1–5: 1 = not emoji-like, 3 = partial emoji characteristics, 5 = clean, recognizable emoji illustration with simple shapes and consistent outlines.
3. **Artifacts:** optional yes/no, with one-line explanation (duplicated limbs/objects, malformed faces, unreadable composition, etc.).
4. **Diversity:** inspect across seeds; diversity is not a replacement for prompt adherence.
5. Use at least two raters if feasible; keep scoring criteria stable and report disagreements.

## Efficiency
- Record synchronized wall-clock pipeline inference time (exclude checkpoint download, initial loading, and file saving). The baseline runner does no extra warmup generation: it marks the first generated image in each process `cold_generation: true`. Exclude cold generations from steady-state speed summaries; report them separately. A three-prompt smoke test is a functional test, not a speed benchmark. Keep device and hardware details in logs.
- Record model ID/revision, seed, precision, guidance, scheduler, schedule/steps, and peak memory where measurable.
- When comparing solvers, consider **neural network function evaluations (NFE)** as well as steps and elapsed time; one step may use multiple evaluations.
- For speed comparisons across Macs vs cloud GPUs, clearly separate hardware-specific results.

## Fair comparisons
- **Pretrained vs pretrained**: assesses out-of-box systems.
- **Adapted vs adapted** (if both trained): assesses practical adaptation pipelines; also disclose training budgets.
- **Before/after within one model**: best for causal claims about LoRA/fine-tuning.
- Person 3 custom guidance applies primarily to SD; do not imply it generalizes to Sana unless separately tested.
- Freeze this v1 benchmark before major experiments. Changes require a new version.

## Suggested exported metadata per generated image
`prompt_id, model_id, model_revision, condition, seed, steps, scheduler, guidance_scale, resolution, dtype, device, runtime_seconds, output_path`.

## Baseline execution and logging
- Sana uses the pinned checkpoint revision in `experiment_config.yaml`, 20 steps, CFG 4.5, fp16 transformer, float32 text encoder and VAE, and a fresh CPU generator for each prompt/seed pair. Start with seed 42; run all three shared seeds only when ready for the full benchmark.
- Stable Diffusion and custom guidance runs use the same full prompt text, resolution and seed IDs. Record their own checkpoint revisions, scheduler settings, precision, steps, guidance and negative prompt (empty for the primary baseline). Guidance scales and step counts are model-specific; equal numeric values do not establish equal compute budgets.
- Store every result, including failures of prompt adherence; no cherry-picking, retries for aesthetics, or image edits. Resizing in contact sheets is for inspection only. Score the original images; white background is a requested style property, not a postprocessing operation.
- Each PNG has a JSON sidecar written last after a successful save, with an image hash. Interrupted or corrupt pairs are regenerated. Matching completed pairs are skipped; conflicting settings require a new condition directory or explicit `--overwrite`.
- Contact sheets contain the current command's selected prompt/seed pairs, including resumed images. Per-image JSON records are authoritative across invocations; cold-generation flags refer to the original generating process.
- Pin and log software versions. Identical seeds support repeatability within an environment, but exact pixels are not guaranteed across hardware or library versions. Peak memory is null where reliable per-generation measurement is unavailable.

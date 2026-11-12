# Text-to-Emoji Generation — Team Roles and Implementation Plan

**Course:** Generative AI (USI)  
**Team:** 3 people  
**Project objective:** Generate novel emoji-style images from text prompts, then compare how different generative models and guidance strategies affect prompt adherence, style, diversity, and computational cost.

Example prompts: *“A sad robot drinking coffee,” “A nervous avocado astronaut,”* and *“A sleepy penguin holding an umbrella.”*

The project has three connected contributions: **diffusion-model adaptation (Person 1), flow-matching generation (Person 2), and compositional guidance (Person 3)**. Each person implements and evaluates a distinct method; everyone uses a shared prompt set and evaluation procedure.

---

## Person 1 — Diffusion-Based Text-to-Emoji Generation

### Goal

Adapt an existing text-to-image diffusion model to generate images in a recognizable, consistent emoji style. Determine whether emoji-specific fine-tuning improves style without sacrificing the ability to follow unusual prompts.

### Model and technical idea

- **Proposed model:** Stable Diffusion 1.5.
- **Adaptation method:** LoRA (Low-Rank Adaptation) fine-tuning.
- **Libraries:** PyTorch, Hugging Face Diffusers, PEFT/LoRA, and image-processing utilities.

Stable Diffusion turns text into embeddings, starts from noisy image latents, and repeatedly predicts the noise to remove. A decoder then converts the final latent into an image. Instead of retraining the entire model, LoRA learns small weight updates that help the pretrained generator acquire the target emoji style.

### What Person 1 implements

1. **Pretrained baseline:** Load Stable Diffusion 1.5 and generate images from the shared prompts without any emoji-specific training. Save images, seeds, settings, and inference times.
2. **Emoji training data:** Work with the team to collect OpenMoji assets, convert SVGs to raster images, and attach usable text captions. Check whether the descriptions really match the images.
3. **LoRA fine-tuning:** Train a lightweight adapter on the image–caption pairs, using a suitable resolution and batch size for the available GPU. Save the adapter separately from the base model.
4. **Adapted generation:** Load the original model with the LoRA adapter and regenerate the *same* prompts and seeds.
5. **Ablation experiments:** Compare the pretrained model versus the adapted model, and optionally test LoRA settings, sampling steps, and guidance scales.

### Experiments and evaluation

- Does the adapted model produce clearer and more consistent emoji-style outputs?
- Does it still generate combinations that are not directly represented in the emoji dataset?
- Does increasing guidance or changing sampling steps improve results, or create artifacts?
- What are the training and inference costs?

### Deliverable

A **working diffusion-based emoji generator**, a trained LoRA adapter, reproducible before/after image comparisons, and an analysis of improvements and failure cases.

**Dependency:** Person 3 can use this model for custom guidance experiments, but should initially work with the pretrained baseline so the two tasks can proceed in parallel.

---

## Person 2 — Flow-Matching-Based Text-to-Emoji Generation

### Goal

Investigate a second family of generative image models and compare its practical performance against Person 1's diffusion pipeline on the same text-to-emoji task.

### Model and technical idea

- **Proposed model:** Sana 0.6B (512-pixel variant), subject to a successful hardware test.
- **Method:** Flow matching, with iterative ODE-based sampling.
- **Libraries:** PyTorch and a compatible Diffusers pipeline, where supported.

Where a conventional diffusion implementation predicts noise to remove, a flow-matching model learns a velocity field that transports an initial noisy latent toward an image latent. During generation, a numerical solver follows the learned field across a sequence of sampling steps. The approach is related to diffusion in its use of latent iterative sampling, but its training target and sampling formulation differ.

### What Person 2 implements

1. **Feasibility test:** Load a pretrained text-conditioned flow-matching model and generate a small set of sample images. Measure memory, runtime, and compatibility with the available M4 Pro Macs or cloud GPU.
2. **Baseline generation:** Use the shared text prompts and document the model settings, seeds, output sizes, and generation time.
3. **Sampling experiments:** Vary the number of inference steps and guidance settings; where accessible, investigate solver choices or timestep schedules.
4. **Emoji adaptation (if feasible):** Attempt lightweight fine-tuning or another meaningful model adaptation using the shared emoji data. Verify training support and costs before committing.
5. **Comparison:** Evaluate outputs against Person 1's diffusion baseline and emoji-adapted model, recording differences in prompt adherence, emoji style, diversity, runtime, and memory usage.

### Experiments and evaluation

- How many sampling steps are required for usable results?
- Does the method preserve complex objects and relationships?
- How does its runtime and memory consumption compare with the diffusion pipeline?
- If adaptation is possible, does it improve emoji-style consistency?

### Deliverable

A **working flow-matching generation pipeline**, controlled sampling experiments, and a comparison with the diffusion approach. An emoji-adapted flow model is a stretch goal if compute and training support allow it.

**Feasibility note:** Unmodified inference alone is not a strong research contribution. If fine-tuning is impractical, Person 2 should deliver a substantive, reproducible sampling or conditioning investigation and confirm that this meets the course expectations.

**Comparison caveat:** Stable Diffusion and Sana differ in more than their generative objectives (e.g., architecture, size, and pretrained data). The comparison therefore measures *practical systems* rather than isolating diffusion versus flow matching as a single variable.

---

## Person 3 — Compositional Generation and Guidance

### Goal

Improve how well the generator handles prompts containing **multiple concepts, attributes, or relationships**, without training a new image model.

Example: *“A sad robot drinking coffee.”* The system should generate a robot that looks sad **and** is actually drinking coffee, rather than merely placing a robot beside a cup.

### Model and technical idea

- **Base model:** Person 1's Stable Diffusion model, initially without the LoRA adapter and later with it.
- **Methods:** Classifier-Free Guidance (CFG) and custom multi-condition guidance.
- **Implementation:** Modify the denoising/sampling loop in PyTorch and Diffusers so that predictions from different text conditions are combined at each step.

#### Part A — Establish a CFG baseline

Standard CFG compares two predictions: one using no prompt (`ε_u`) and one using the full prompt (`ε_c`). The predicted noise is adjusted according to guidance strength `s`:

\[
\hat{\epsilon}=\epsilon_u+s(\epsilon_c-\epsilon_u)
\]

Person 3 generates the same prompts using several guidance strengths (for example, **3, 5, 7.5, and 10**) to study the trade-off between prompt adherence, image quality, and variety.

#### Part B — Implement multi-condition guidance

Instead of treating a long prompt as one condition, split it into complementary subprompts:

- **Full prompt:** “A sad robot drinking coffee.”
- **Condition A:** “A sad robot.”
- **Condition B:** “A robot drinking coffee.”

At each denoising step, obtain the unconditional prediction (`ε_u`) and predictions for A and B (`ε_A`, `ε_B`). Combine them with a weight `w`:

\[
\hat{\epsilon}=\epsilon_u+s\left[w(\epsilon_A-\epsilon_u)+(1-w)(\epsilon_B-\epsilon_u)\right]
\]

The resulting prediction is passed to the scheduler to update the latent. This is repeated through the sampling process until the image is produced.

### What Person 3 implements

1. **Baseline:** Generate a fixed set of compositional prompts with standard single-prompt CFG.
2. **CFG sweep:** Test several guidance strengths using fixed prompts and seeds.
3. **Custom sampler:** Access the model's noise predictions for unconditional, A, and B conditions using the same latent and timestep; combine them with the equation above and run the scheduler step.
4. **Weight experiments:** Change the relative weights of the two conditions and evaluate which attributes or relationships are preserved.
5. **Direct comparison:** Compare full-prompt CFG, weighted multi-condition guidance, and alternative guidance settings on identical prompts.
6. **Failure analysis:** Identify missing attributes, wrong relationships, duplicated objects, composition artifacts, and effects on image variety and inference speed.

### Example of the core sampling modification

```python
eps_u = predict_noise(latents, t, uncond)
eps_a = predict_noise(latents, t, cond_a)
eps_b = predict_noise(latents, t, cond_b)

guided = eps_u + scale * (
    weight * (eps_a - eps_u)
    + (1 - weight) * (eps_b - eps_u)
)

latents = scheduler.step(guided, t, latents).prev_sample
```

This is **illustrative pseudocode**, not a standalone executable snippet: the implementation needs a real UNet prediction wrapper, text embeddings, correct scheduler scaling, and consistent timestep handling.

### Experiments and evaluation

- **Attribute accuracy:** Is the robot visibly sad?
- **Relationship accuracy:** Is it actually drinking coffee?
- **Style consistency:** Is the result recognizably emoji-like?
- **Diversity:** Are results varied across seeds or overly similar?
- **Efficiency:** What is the extra runtime from multiple model predictions per step?

Use approximately **20–30 prompts**, with multiple fixed seeds per prompt. The custom approach may be slower: standard CFG generally uses two predictions per step, while the proposed method uses three.

### Deliverable

A **custom compositional-guidance implementation**, controlled experiments against the standard CFG baseline, image comparison grids, and a quantitative/qualitative analysis of whether separating conditions improves prompt adherence.

**Important:** Combining conditional predictions is a hypothesis, not a guaranteed improvement. Finding that it fails for certain relationships is still a valid experimental result.

---

## Shared Work and Integration

The three contributions are coordinated through **one dataset, one evaluation protocol, and one shared set of prompts**.

| Shared activity | Responsibility |
|---|---|
| OpenMoji collection, licensing checks, image conversion, captions, and splits | All three; Person 1 drives training-data integration |
| Fixed prompt set, including simple objects and novel compositions | All three; Person 3 coordinates evaluation design |
| Diffusion baseline and emoji LoRA | Person 1 |
| Flow-matching baseline and sampling experiments | Person 2 |
| CFG and custom compositional guidance | Person 3 |
| Common output resolution, random seeds, metrics, and experiment logging | All three |
| Side-by-side comparisons, poster figures, and final findings | All three |

### Proposed evaluation protocol

1. Choose prompts spanning simple objects, multiple attributes, object interactions, and unusual combinations.
2. Keep output resolution and prompt text identical across methods; record model-specific settings and seeds.
3. Rate prompt adherence (objects, attributes, relationships) and emoji-style consistency, preferably with a clear rubric and more than one rater.
4. Measure generation time and peak memory where supported; generate multiple seeds to assess diversity.
5. Present side-by-side image grids and document failures, not just the best images.

**Dataset limitation:** Standard emoji datasets have few complex interactions. Fine-tuning may teach style more effectively than compositional understanding, which is why Person 3's experiments are complementary to Person 1's work.

## Final Project Outcome

The completed project should answer three connected questions:

1. **Person 1:** Does adapting a diffusion model with LoRA improve emoji style while preserving prompt adherence?
2. **Person 2:** What are the quality and efficiency trade-offs of a flow-matching generator for the same prompts?
3. **Person 3:** Can modified guidance improve generation of multiple concepts and relationships without retraining the model?

**Minimum viable result:** A working diffusion baseline, at least one meaningful emoji adaptation, a second investigated generative or sampling strategy, a reproducible compositional-guidance experiment, and a comparative evaluation suitable for the course poster.

**Timing:** Verify feasibility and submit the topic by **18 November**; implement baselines and experiments in late November/early December; prepare the poster for **15–16 December**. The course's **18 December report must be written by the team without AI assistance**, as specified in the original project outline.

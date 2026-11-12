# Person 1 — Stable Diffusion 1.5

Implement baseline generation, LoRA training orchestration, and adapted generation
here (or add a `genmoji.diffusion` module when implementation starts).

Use the shared 512px OpenMoji ImageFolder; the exact trainer arguments are in
[the dataset guide](../../docs/dataset.md#stable-diffusion-15--person-1).
Write adapters to `outputs/diffusion/lora/`, generated images and logs to
`outputs/diffusion/`. Log model revision, dataset manifest, prompt, seed, steps,
guidance, precision, resolution, elapsed time, and peak memory where available.
Coordinate model loading and scheduler choices with Person 3.

# Person 3 — Compositional guidance

Implement CFG sweeps and custom multi-condition sampling here (or add a
`genmoji.guidance` module when implementation starts). Start with SD 1.5's baseline,
then use Person 1's adapter. No additional training dataset is required.

Coordinate a fixed set of 20–30 evaluation prompts and multiple seeds with the team.
Include the full prompt plus conditions A/B for compositions, and use identical
full prompts for the SD and Sana comparisons. Keep these prompts separate from
the automatic OpenMoji captions: compositional evaluation needs novel interactions.
Write results under `outputs/guidance/`, recording guidance scale, weights,
prediction count, scheduler, seeds, runtime, model/adapter revisions and prompts.

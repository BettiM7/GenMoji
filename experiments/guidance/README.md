# Person 3 — Compositional guidance

Implement CFG sweeps and custom multi-condition sampling here (or add a
`genmoji.guidance` module when implementation starts). Start with SD 1.5's baseline,
then use Person 1's adapter. No additional training dataset is required.

Use the exact full prompts and seeds in
[Text-to-Emoji v1](../../benchmarks/text-to-emoji-v1/evaluation_protocol.md)
for the baseline and guidance runs. Record auxiliary conditions A/B with each
compositional-guidance run. Score the outputs against the benchmark's expected
elements and emoji-style rubric.
Write results under `outputs/guidance/`, recording guidance scale, weights,
prediction count, scheduler, seeds, runtime, model/adapter revisions and prompts.

# Person 2 — Sana 0.6B, 512px

Implement the hardware feasibility test and baseline/sampling experiments here
(or add a `genmoji.flow_matching` module when implementation starts).

Use `Efficient-Large-Model/Sana_600M_512px_diffusers`. Baseline inference uses the
shared evaluation prompts. Optional adaptation consumes the same RGB PNGs and
per-image captions as SD; see [the dataset guide](../../docs/dataset.md#sana-06b--person-2).
Tokenization and AutoencoderDC encoding belong in the Sana trainer, separately
from SD's CLIP tokenizer and VAE. Confirm adaptation support and memory costs
before a full run. Write adapters, images and logs under `outputs/flow_matching/`
using the same experiment fields as Person 1.

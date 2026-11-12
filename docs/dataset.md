# Shared OpenMoji dataset

## Source and reproducibility

The dataset named in `work_split.md` is [OpenMoji](https://github.com/hfg-gmuend/openmoji).
The root README downloads the fixed [17.0.0 release](https://github.com/hfg-gmuend/openmoji/releases/tag/17.0.0).
It supplies `color/svg/<hexcode>.svg`, `data/openmoji.json`, and `LICENSE.txt`.
Use assets and metadata from the same checkout. An existing local OpenMoji checkout
can also be passed with `--source /path/to/openmoji`.

With the Python 3.12 `.venv` activated, `genmoji prepare` records the Git commit, metadata SHA-256, each selected
SVG's SHA-256, original metadata including authors, processing settings and counts.
A modified source checkout is refused. Archive downloads without `.git` are supported,
but their revision is `null`; their source content is still hashed. Retain the
generated `provenance/` directory alongside experiment results. Record installed
package versions with `python -m pip freeze > outputs/environment.txt`.

## Processing choices

- Render the full color SVG canvas directly at **512 × 512**, without upscaling a
  72px PNG. Composite transparency onto **white** and save **RGB PNGs**.
- Remove undefined local SVG marker references in memory before rendering
  (one OpenMoji 17.0.0 flag references an absent `Dot` marker that crashes CairoSVG).
  Preserve defined markers and source files. Record removals in each example's
  `svg_adjustments` and the total `adjusted_svg_count` in the manifest.
- Caption: `OpenMoji style emoji of {annotation}, on a white background.`
  Annotations come from OpenMoji, without invented attributes or keyword expansion.
  Inspect image/caption pairs before training; annotations can be terse or ambiguous.
- Include all metadata entries (including flags, symbols, extras, and skin tones).
  There is no hidden category filtering. Record any future filtering consistently
  for both models, and regenerate both from that same selection.
- Assign roughly 80% train / 10% validation / 10% test using SHA-256 of seed 42
  and a skin-tone family key. Skin-tone variants and their base stay together.
  Fractions are approximate because families are indivisible. Other related or
  near-duplicate designs can still cross splits; this is not a semantic holdout.
- No random crop, flip, augmentation, tokenization, latent encoding, or model
  normalization is performed. Disable horizontal flips when training: direction,
  text, and symbols may change meaning. Let each trainer use its own tokenizer,
  text encoder, VAE, and normalization. SD latents are not compatible with Sana's.
- Existing output directories are refused. Use a new path when changing settings;
  processing failures do not publish a partial dataset.

CLI overrides: `--source`, `--output`, `--resolution`, `--seed`,
`--validation-fraction`, `--test-fraction`, and `--limit`. A smoke subset can have
empty held-out splits; use the full dataset for experiments.
If a split is empty, loading the entire ImageFolder root can fail in Datasets;
use the provided checker, which skips empty splits, or load a nonempty split's
directory directly for smoke tests.

## Shared ingestion contract

The layout follows [Hugging Face ImageFolder with metadata](https://huggingface.co/docs/diffusers/training/create_dataset).
Inside each split, `metadata.jsonl` has one line per local PNG:

```json
{"file_name":"1F600.png","text":"OpenMoji style emoji of grinning face, on a white background."}
```

`file_name` is resolved by ImageFolder into an `image` column. Both models consume
these same `image` and `text` columns:

```python
from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="data/processed/openmoji-512/imagefolder")
example = dataset["train"][0]
assert example["image"].mode == "RGB"
assert example["image"].size == (512, 512)
caption = example["text"]
```

Use the `imagefolder/` path, not its parent containing provenance. To make a training
example consume only the training split, pass `imagefolder/train` as the dataset
path. Neither model's baseline inference needs training data; use the team's fixed
evaluation prompts for inference. Emoji held-out images and novel-composition
evaluation prompts measure different things.

## Training-script setup (separate GPU environment)

These are ingestion examples for upstream trainers, not a hardware-validated
training recipe. Install PyTorch for the target GPU first, and create a separate
Python 3.12 environment. Pin Diffusers' code and package to the same release:

```bash
git clone --depth 1 --branch v0.35.1 https://github.com/huggingface/diffusers.git external/diffusers
python -m pip install -e external/diffusers
# SD 1.5 environment:
python -m pip install -r external/diffusers/examples/text_to_image/requirements.txt
# Sana environment (also explicitly install datasets for per-image captions):
python -m pip install -r external/diffusers/examples/dreambooth/requirements_sana.txt datasets
accelerate config
```

Run commands from the GenMoji repository root. Tune precision, memory options,
batch size, and training steps after the feasibility test. Pin base-model revisions
in experiment logs. Do not assume Mac MPS supports every upstream training option.

### Stable Diffusion 1.5 / Person 1

The [upstream text-to-image LoRA trainer](https://github.com/huggingface/diffusers/blob/v0.35.1/examples/text_to_image/train_text_to_image_lora.py)
accepts this ImageFolder directly:

```bash
accelerate launch external/diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --train_data_dir data/processed/openmoji-512/imagefolder/train \
  --image_column image --caption_column text \
  --resolution 512 --center_crop \
  --train_batch_size 1 --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 --lr_scheduler constant --lr_warmup_steps 0 \
  --max_train_steps 1000 --seed 42 \
  --output_dir outputs/diffusion/lora
```

### Sana 0.6B / Person 2

Use [Sana_600M_512px_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_600M_512px_diffusers).
The [upstream Sana LoRA trainer](https://github.com/huggingface/diffusers/blob/v0.35.1/examples/dreambooth/train_dreambooth_lora_sana.py)
loads local ImageFolder datasets via `--dataset_name` and supports per-image captions:

```bash
accelerate launch external/diffusers/examples/dreambooth/train_dreambooth_lora_sana.py \
  --pretrained_model_name_or_path Efficient-Large-Model/Sana_600M_512px_diffusers \
  --dataset_name data/processed/openmoji-512/imagefolder/train \
  --image_column image --caption_column text \
  --instance_prompt "OpenMoji style emoji" \
  --resolution 512 --center_crop \
  --train_batch_size 1 --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 --lr_scheduler constant --lr_warmup_steps 0 \
  --max_train_steps 1000 --seed 42 \
  --output_dir outputs/flow_matching/lora
```

Do not use `--instance_data_dir` for this captioned dataset: that branch of the
trainer opens files as images and uses one shared instance prompt. In the example
above, `text` supplies the per-image prompts; `--instance_prompt` is the required
fallback. Fine-tuning feasibility for the 600M model and available hardware still
needs verification. The upstream Sana trainer also caches processed training
images in memory, so begin with a small prepared subset.

## Verification and attribution

`scripts/check_dataset.py` loads every nonempty split through Hugging Face Datasets,
decodes every image, checks captions, counts and RGB dimensions, and checks family
isolation against provenance. It needs no model weights or GPU.

Artwork attribution: OpenMoji, https://openmoji.org, CC BY-SA 4.0. Retain the copied
license, attribution, author records, and notice of rasterization/caption changes
when sharing the prepared assets. These artifacts describe the source dataset;
base models have their own licenses. No claim about trained-weight licensing is
made by this preparation pipeline.

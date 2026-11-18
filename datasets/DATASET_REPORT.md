# Shared OpenMoji dataset report

Prepared 2026-11-18 from the official [OpenMoji release 17.0.0](https://github.com/hfg-gmuend/openmoji/tree/17.0.0). No baseline images or evaluation prompts were modified. No model training or generated-image evaluation was performed.

## Counts

| Measure | Count |
| --- | ---: |
| Official metadata entries inspected | 4495 |
| Original PNGs collected locally | 1190 |
| Final eligible and selected PNGs | 1131 |
| Metadata entries excluded | 3364 |
| Training images | 1018 |
| Validation images | 113 |
| Pilot training images (subset of train) | 192 |

Counts reconcile as 4,495 metadata entries = 1,131 selected + 3,364 excluded. The original PNG cache also retains 59 files fetched during initial review that were excluded by the finalized policy; a fresh build needs only the final 1,131 PNGs. Originals are never overwritten by preprocessing. Older generated previews removed from the final selection are archived under originals/unused_previews; images/ contains exactly the selected set. The complete local dataset directory is approximately 59 MiB including originals, metadata, manifests and previews, rather than a large repository/archive clone.

## Category coverage

| Category | All | Train | Validation | Pilot |
| --- | ---: | ---: | ---: | ---: |
| activities | 76 | 68 | 8 | 24 |
| animals-nature | 160 | 144 | 16 | 24 |
| extras-openmoji | 51 | 46 | 5 | 24 |
| food-drink | 131 | 118 | 13 | 24 |
| objects | 229 | 206 | 23 | 24 |
| people-body | 174 | 157 | 17 | 24 |
| smileys-emotion | 131 | 118 | 13 | 24 |
| travel-places | 179 | 161 | 18 | 24 |

Objects include clothing, tools, office/household items and instruments. Travel/places includes vehicles, buildings and recognizable weather/geographic illustrations. Selected extras are limited to animals, food, gardening and recognizable objects; brands/UI/flags are excluded. Pilot selection uses deterministic seed-42 hash ordering and category round-robin sampling: exactly 24 from each of eight categories. It deliberately oversamples smaller categories relative to the complete dataset, rather than representing raw frequency.

## Selection, duplicates and splits

Exclusion reasons (each excluded metadata entry receives one reason):

| Reason | Count |
| --- | ---: |
| abstract_emotion_symbol | 40 |
| skin_tone_variant | 2030 |
| complex_family_or_symbol_variants | 48 |
| flags_symbols_components | 640 |
| clock_variant | 40 |
| text_or_symbol_content | 27 |
| abstract_game_symbol | 5 |
| text_symbol_or_branded_object | 15 |
| extras_outside_illustration_scope | 353 |
| gender_variant | 166 |

The implementation distinguishes actual skin-tone variants from the metadata skintone_combination field, which describes applicability and is not itself an actual skin tone. Gender-equivalent role/action captions are normalized; one ungendered/preferred representative is retained before pixel comparison. This metadata filtering removes redundant families before image duplicate checks.

Every candidate is white-composited and checked for exact pixel duplication. Perceptual features use foreground-cropped grayscale DCT hashes (63 AC bits) plus mean absolute RGB difference of foreground thumbnails. Pairs with Hamming distance ≤8 and RGB MAE ≤12 are considered related; stricter ≤2/≤4 pairs would be removed as near duplicates. After final semantic variant pruning, **no additional exact/strict near duplicates remained**. There are **32 conservative similarity links**, whose retained endpoints are joined into split groups rather than discarded as identical. See near_duplicate_pairs.json. These thresholds are heuristics, not proof that all semantic near-duplicates have been found.

Split groups also join normalized annotation families (gender, color, size and facing-direction variants). Connected components are assigned deterministically with seed 42 and approximately 10% category-stratified validation targets. Whole groups stay in one split. Automated validation checks family and visual-link grouping, image/ID uniqueness, pilot membership and zero group overlap. General concepts can still overlap between train/validation; do not claim unseen-concept generalization. Preserve these manifests for both Sana and Stable Diffusion rather than letting each pipeline resplit.

## Formatting and captions

Source 618×618 PNGs are retained unchanged, including alpha. Processing composites on white, fits/pads aspect-preservingly to 512×512 using Lanczos and saves RGB PNG. It does not stretch non-square inputs. Each manifest records source metadata, author, release, original URL, original and processed SHA-256, pixel hash, perceptual hash, annotation, description, category, split and split-group ID.

Captions use the official annotation as the description plus the constant phrase `OpenMoji emoji illustration, clean outlines, white background`. Tags are preserved as metadata but not appended because they may be nonvisual associations. No inferred colors, expressions or object attributes are invented. One reviewed caption override maps the culturally specific E148 annotation kehrwoche to the depicted broom and dustpan. Its original annotation is preserved.

Examples:

- `1FABF`: goose, OpenMoji emoji illustration, clean outlines, white background
- `1FAD2`: olive, OpenMoji emoji illustration, clean outlines, white background
- `270F`: pencil, OpenMoji emoji illustration, clean outlines, white background
- `1F63B`: smiling cat with heart-eyes, OpenMoji emoji illustration, clean outlines, white background
- `E148`: broom and dustpan, OpenMoji emoji illustration, clean outlines, white background

Caption-to-image integrity is verified automatically through IDs, source metadata, exact caption construction and source/processed checksums. Representative 64-image train and 64-image validation contact sheets were visually inspected during iterative filtering; this is **not** an exhaustive manual audit of every source drawing. Remaining risks include upstream culturally specific labels, visually similar faces/poses, occasional lettering embedded in recognizable objects and imperfect semantic family matching. Review the pilot contact sheet before training, and version any further curation rather than silently changing a started run.

## Artifacts and reuse

All following paths are relative to datasets/openmoji-17.0.0/:

- `originals/openmoji.json`: full upstream metadata (4,495 entries), preserved.
- `originals/png/`: unmodified collected originals; `images/`: selected 512×512 RGB inputs.
- `manifests/all.jsonl`, `train.jsonl`, `validation.jsonl`, `pilot_train.jsonl`.
- `visualizations/pilot_train.png`, `validation.png`: representative contact sheets; companion JSON files contain every displayed caption/record.
- `report.json`: counts, categories, preparation script hash, software versions and metadata hash.
- `sample_pairs.json`: 16 deterministic representative image-caption records.
- `exclusions.json`, `near_duplicate_pairs.json`: auditable selection decisions.
- `LICENSE.txt`, `ATTRIBUTION.md`: original license and derived-image attribution.

Manifest paths are relative to the dataset root, independent of Sana. Person 1 can load each RGB image via dataset_root / record['image'] and use record['caption'] verbatim. Read train/validation manifests directly and respect split_group; no Sana-specific tensors are stored in this shared dataset. SHA-256 values can verify copying between machines. Raw/processed PNG directories are ignored by Git and can be regenerated; manifests, metadata and reports are retained.

## Reproduction and validation

From the repository root, using the existing environment (Pillow 13.0.0, NumPy 2.5.5, Python 3.14.4; no new packages):

```bash
sana-emoji/.venv/bin/python datasets/tools/prepare_openmoji.py --download
sana-emoji/.venv/bin/python datasets/tools/prepare_openmoji.py --validate-only
```

Without --download, preparation rebuilds using cached originals. Missing files fail rather than silently reducing the dataset. Downloads use the pinned tag, eight workers, timeout/retry and temporary files; no entire repository/model download is required. Validation checks every selected original/image checksum, RGB dimensions, white corner, caption construction, exact duplicate absence and split relationships. All four JSONL manifests were byte-identical across an offline rebuild; selected images also passed their recorded hashes. Changes to filtering, software or preprocessing should produce a new version, not overwrite a running experiment's inputs.

## License and preservation

Emojis designed by [OpenMoji](https://openmoji.org/), HfG Schwäbisch Gmünd and contributors. Source graphics and processed derivatives/contact sheets retain [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) attribution; per-item authors and modifications are recorded. Preserve LICENSE.txt and ATTRIBUTION.md when sharing. This records the asset license, not a separate legal conclusion about trained weights.

`datasets/preservation_hashes.json` records 110 protected baseline, feasibility, shared-config and inference files. Their contents were verified unchanged after this task. See `sana-emoji/pilot/PILOT_PLAN.md` for proposed training and its validation limits.

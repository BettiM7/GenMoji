"""Decode and validate prepared ImageFolder data without loading any models."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def check(root: Path) -> None:
    manifest = json.loads((root / "provenance/manifest.json").read_text())
    records = json.loads((root / "provenance/records.json").read_text())
    families = {}
    expected = {row["file_name"]: row for row in records}
    if len(expected) != len(records):
        raise ValueError("Duplicate files in provenance")
    for row in records:
        previous = families.setdefault(row["family"], row["split"])
        if previous != row["split"]:
            raise ValueError(f"Family crosses splits: {row['family']}")
    total = 0
    for split, count in manifest["counts"].items():
        folder = root / "imagefolder" / split
        metadata = [json.loads(line) for line in (folder / "metadata.jsonl").read_text().splitlines()]
        names = [row["file_name"] for row in metadata]
        if len(names) != len(set(names)) or len(names) != count:
            raise ValueError(f"Duplicate metadata or incorrect count: {split}")
        if set(names) != {path.name for path in folder.glob("*.png")}:
            raise ValueError(f"PNG/metadata mismatch: {split}")
        if count == 0:
            print(f"{split}: empty (acceptable only for a small smoke subset)")
            continue
        # Same local-folder ingestion used by the upstream training scripts.
        dataset = load_dataset("imagefolder", data_dir=str(folder), split="train")
        if len(dataset) != count:
            raise ValueError(f"ImageFolder count mismatch: {split}")
        for item in dataset:
            image = item["image"]
            if image.mode != "RGB" or image.size != (manifest["resolution"],) * 2:
                raise ValueError(f"Invalid image dimensions or mode: {image.filename}")
            # ImageFolder ordering need not match metadata ordering.
            key = f"{split}/{Path(image.filename).name}"
            if key not in expected or item["text"] != expected[key]["text"]:
                raise ValueError(f"Caption/provenance mismatch: {key}")
            if not item["text"].strip():
                raise ValueError(f"Empty caption: {key}")
        total += len(dataset)
        print(f"{split}: {count} decoded RGB images and captions OK")
    if total != len(records):
        raise ValueError("Provenance total mismatch")
    print(f"Verified {total} examples; no skin-tone family crosses splits.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    check(parser.parse_args().root)

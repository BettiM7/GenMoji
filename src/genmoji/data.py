"""OpenMoji SVGs -> shared RGB PNG + caption ImageFolder for SD 1.5 and Sana."""

import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from xml.etree.ElementTree import tostring


def prepare_svg(svg: bytes) -> tuple[bytes, list[dict]]:
    """Remove dangling local marker attributes that crash CairoSVG.

    OpenMoji 17.0.0 includes a flag with marker-end="url(#Dot)" but no
    corresponding marker definition. Undefined references cannot draw a marker;
    discard only those references and retain every defined marker and path.
    """
    from defusedxml.ElementTree import fromstring

    tree = fromstring(svg)
    marker_ids = {element.get("id") for element in tree.iter()
                  if element.tag == "{http://www.w3.org/2000/svg}marker"}
    adjustments = []
    for element in tree.iter():
        for attribute in ("marker", "marker-start", "marker-mid", "marker-end"):
            value = element.get(attribute, "")
            reference = re.fullmatch(r"url\(\s*['\"]?#([^\s'\")]+)['\"]?\s*\)", value)
            if reference and reference[1] not in marker_ids:
                adjustments.append({"attribute": attribute, "value": value})
                del element.attrib[attribute]
    return (tostring(tree) if adjustments else svg), adjustments


def family_key(row: dict) -> str:
    """Keep skin-tone variants and their base in the same split."""
    code = row.get("skintone_base_hexcode") or row["hexcode"]
    return "-".join(part for part in code.upper().split("-")
                    if part not in {"FE0F", "1F3FB", "1F3FC", "1F3FD", "1F3FE", "1F3FF"})


def split_for(key: str, seed: int, validation: float, test: float) -> str:
    # Stable across machines, input ordering, and subset selection.
    value = int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big") / 2**64
    if value < test:
        return "test"
    if value < test + validation:
        return "validation"
    return "train"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_dataset(
    source: Path, output: Path, resolution: int = 512, seed: int = 42,
    validation_fraction: float = 0.1, test_fraction: float = 0.1,
    limit: int | None = None,
) -> dict:
    """Build in a temporary directory; publish only when every SVG succeeds.

    Existing output directories are deliberately refused to avoid stale images
    and accidental mixing of training and held-out examples on reruns.
    """
    import cairosvg
    from PIL import Image

    source, output = Path(source).resolve(), Path(output).resolve()
    if resolution < 1 or (limit is not None and limit < 1):
        raise ValueError("Resolution and limit must be positive")
    if not (0 <= validation_fraction < 1 and 0 <= test_fraction < 1
            and validation_fraction + test_fraction < 1):
        raise ValueError("Split fractions must be nonnegative and sum to less than one")
    if output.exists():
        raise ValueError(f"Output already exists: {output}; choose a new output path")
    metadata_path = source / "data/openmoji.json"
    metadata_bytes = metadata_path.read_bytes()
    rows = json.loads(metadata_bytes)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty OpenMoji metadata list")
    # Sort before selecting a smoke-test subset, independently of upstream order.
    rows = sorted(rows, key=lambda row: row["hexcode"])
    seen = set()
    for row in rows:
        code = row["hexcode"]
        if not re.fullmatch(r"[0-9A-F]+(?:-[0-9A-F]+)*", code) or code in seen:
            raise ValueError(f"Invalid or duplicate hexcode: {code}")
        seen.add(code)
        if not isinstance(row.get("annotation"), str) or not row["annotation"].strip():
            raise ValueError(f"Missing caption: {code}")
    if limit is not None:
        rows = rows[:limit]
    for row in rows:
        if not (source / "color/svg" / f"{row['hexcode']}.svg").is_file():
            raise ValueError(f"Missing color SVG: {row['hexcode']}")
    license_path = source / "LICENSE.txt"
    if not license_path.is_file():
        raise ValueError("Source must include OpenMoji LICENSE.txt")
    revision = None
    if (source / ".git").exists():
        revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
            raise ValueError("OpenMoji checkout has local changes; use a clean source for provenance")
    counts = dict.fromkeys(("train", "validation", "test"), 0)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".openmoji-", dir=output.parent) as staging:
        root = Path(staging)
        records = []
        for split in counts:
            (root / split).mkdir()
            (root / split / "metadata.jsonl").touch()
        for row in rows:
            code = row["hexcode"]
            family = family_key(row)
            split = split_for(family, seed, validation_fraction, test_fraction)
            svg = (source / "color/svg" / f"{code}.svg").read_bytes()
            # Composite explicitly: converting transparent pixels directly to RGB
            # can create a black background. Preserve the full SVG canvas.
            try:
                render_svg, adjustments = prepare_svg(svg)
                png = cairosvg.svg2png(bytestring=render_svg, output_width=resolution, output_height=resolution)
            except Exception as error:
                raise ValueError(f"Could not render {code}.svg: {error}") from error
            with Image.open(io.BytesIO(png)) as image:
                rgba = image.convert("RGBA")
                rgb = Image.new("RGB", rgba.size, "white")
                rgb.paste(rgba, mask=rgba.getchannel("A"))
                rgb.save(root / split / f"{code}.png")
            caption = f"OpenMoji style emoji of {row['annotation'].strip()}, on a white background."
            with (root / split / "metadata.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"file_name": f"{code}.png", "text": caption}, ensure_ascii=False) + "\n")
            records.append({
                **row, "family": family, "split": split,
                "file_name": f"{split}/{code}.png", "text": caption,
                "svg_sha256": hashlib.sha256(svg).hexdigest(),
                "svg_adjustments": adjustments,
            })
            counts[split] += 1
        manifest = {
            "source": "https://github.com/hfg-gmuend/openmoji", "source_revision": revision,
            "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            "resolution": resolution, "mode": "RGB", "background": "white", "seed": seed,
            "validation_fraction": validation_fraction, "test_fraction": test_fraction,
            "limit": limit, "counts": counts, "license": "CC-BY-SA-4.0",
            "caption_template": "OpenMoji style emoji of {annotation}, on a white background.",
            "split_method": "sha256(seed:skin-tone-family), approximate fractions",
            "adjusted_svg_count": sum(bool(record["svg_adjustments"]) for record in records),
        }
        # Keep metadata outside the ImageFolder so its JSON isn't loaded as captions.
        provenance = root / "provenance"
        provenance.mkdir()
        write_json(provenance / "manifest.json", manifest)
        write_json(provenance / "records.json", records)
        shutil.copyfile(license_path, provenance / "LICENSE.txt")
        (provenance / "ATTRIBUTION.md").write_text(
            "Emoji artwork by OpenMoji (https://openmoji.org), CC BY-SA 4.0\n"
            "https://creativecommons.org/licenses/by-sa/4.0/\n\n"
            "Changes: SVG rasterization to RGB PNG, white background, templated captions, dataset splits.\n"
            "Undefined local SVG marker references removed during rendering; details in records.json.\n"
            "Individual authors and source hashes are in records.json.\n", encoding="utf-8")
        # Do not place provenance JSON inside the dataset tree: move images into
        # a dedicated ImageFolder and leave provenance alongside it.
        dataset = root / "imagefolder"
        dataset.mkdir()
        for split in counts:
            (root / split).rename(dataset / split)
        root.rename(output)
    return manifest

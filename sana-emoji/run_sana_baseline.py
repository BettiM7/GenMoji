#!/usr/bin/env python3
"""Reproducible Sana baseline; run with the existing sana-emoji environment."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import textwrap
import time

import yaml
from PIL import Image, ImageDraw

SANA_DIR = Path(__file__).resolve().parent
ROOT = SANA_DIR.parent


def read_inputs():
    config = yaml.safe_load((ROOT / "shared/experiment_config.yaml").read_text())
    benchmark = json.loads((ROOT / config["prompts_file"]).read_text())
    prompts = benchmark["prompts"]
    if Counter(p["category"] for p in prompts) != {
        "simple": 10, "attributes": 10, "relationships": 10
    }:
        raise ValueError("Expected 10 prompts in each of the three categories")
    ids = [p["id"] for p in prompts]
    if len(set(ids)) != 30 or any(not re.fullmatch(r"[a-z]+_\d+", x) for x in ids):
        raise ValueError("Prompt IDs must be unique safe identifiers")
    for p in prompts:
        if p["prompt"] != f'{p["content"]}, {benchmark["style_suffix"]}':
            raise ValueError(f'Inconsistent style suffix: {p["id"]}')
    return config, benchmark


def write_json(path, data):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def completed(image_path, metadata_path, expected):
    if not image_path.exists() or not metadata_path.exists():
        return False
    try:
        record = json.loads(metadata_path.read_text())
        if record.get("status") != "completed":
            return False
        if any(record.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Settings conflict at {metadata_path}; use a new output directory or --overwrite")
        with Image.open(image_path) as image:
            if image.size != (expected["width"], expected["height"]) or image.mode != "RGB":
                return False
            image.verify()
        return record.get("image_sha256") == hashlib.sha256(image_path.read_bytes()).hexdigest()
    except (OSError, json.JSONDecodeError):
        return False


def contact_sheet(jobs, output_dir):
    columns = min(5, len(jobs))
    cell_w, cell_h = 256, 336
    sheet = Image.new("RGB", (columns * cell_w, ((len(jobs) + columns - 1) // columns) * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (prompt, seed, path, _, _) in enumerate(jobs):
        x, y = (i % columns) * cell_w, (i // columns) * cell_h
        with Image.open(path) as image:
            sheet.paste(image.convert("RGB").resize((256, 256)), (x, y))
        label = f'{prompt["id"]} | seed {seed}\n' + "\n".join(textwrap.wrap(prompt["content"], 36))
        draw.multiline_text((x + 5, y + 260), label, fill="black", spacing=3)
    sheet.save(output_dir / "contact_sheet.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="First N prompts in benchmark order (not images)")
    parser.add_argument("--seeds", nargs="+", type=int, help="Space-separated seeds; default: 42")
    parser.add_argument("--output-dir", type=Path, default=SANA_DIR / "results/pretrained_baseline")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate selected outputs")
    parser.add_argument("--local-files-only", action="store_true", help="Use the cached checkpoint without network access")
    parser.add_argument("--validate-only", action="store_true", help="Validate shared files and print plan without loading torch/model")
    args = parser.parse_args()
    config, benchmark = read_inputs()
    if args.limit is not None and not 1 <= args.limit <= 30:
        parser.error("--limit must be between 1 and 30")
    seeds = args.seeds if args.seeds is not None else config["sana_candidate"]["default_seeds"]
    if len(set(seeds)) != len(seeds) or any(not 0 <= s < 2**32 for s in seeds):
        parser.error("Seeds must be unique integers in [0, 2**32)")
    prompts = benchmark["prompts"][:args.limit]
    settings = config["sana_candidate"]
    base = {
        "benchmark": config["benchmark"], "benchmark_version": benchmark["version"],
        "condition": config["logging"]["condition"],
        "model_id": settings["model_id"], "model_revision": settings["model_revision"],
        "variant": settings["variant"], "steps": settings["inference_steps"],
        "guidance_scale": settings["guidance_scale"], **{k: config["image"][k] for k in ("width", "height")},
        "dtype": "float16", "text_encoder_dtype": "float32", "vae_dtype": "float32",
        "generator_device": "cpu", "negative_prompt": "", "runner_version": 1,
    }
    jobs = []
    for prompt in prompts:
        for seed in seeds:
            path = args.output_dir / prompt["id"] / f"seed_{seed}.png"
            expected = {**base, "prompt_id": prompt["id"], "category": prompt["category"], "prompt": prompt["prompt"], "seed": seed}
            jobs.append((prompt, seed, path, path.with_suffix(".json"), expected))
    if args.validate_only:
        print(f"Valid benchmark: 30 prompts, 10 per category. Selected {len(prompts)} prompts x {len(seeds)} seeds = {len(jobs)} images.")
        return
    pending = [job for job in jobs if args.overwrite or not completed(job[2], job[3], job[4])]
    print(f"Selected {len(jobs)} images; generating {len(pending)}, skipping {len(jobs) - len(pending)}.", flush=True)
    if pending:
        import torch
        from diffusers import SanaPipeline

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        if device == "cpu":
            raise RuntimeError("MPS unavailable. Run from an Apple Silicon terminal with the working environment; CPU fp16 fallback is intentionally disabled.")
        packages = {name: version(name) for name in ("torch", "diffusers", "transformers", "accelerate", "huggingface_hub", "Pillow", "PyYAML")}
        load_start = time.perf_counter()
        pipe = SanaPipeline.from_pretrained(
            settings["model_id"], revision=settings["model_revision"],
            torch_dtype=torch.float16, variant=settings["variant"], local_files_only=args.local_files_only,
        )
        # Preserve the configuration verified by sana-emoji/test_sana.py.
        pipe.text_encoder.to(dtype=torch.float32)
        pipe.vae.to(dtype=torch.float32)
        pipe = pipe.to(device)
        torch.mps.synchronize()
        load_seconds = time.perf_counter() - load_start
        scheduler = dict(pipe.scheduler.config)
        for index, (prompt, seed, path, meta_path, expected) in enumerate(pending):
            path.parent.mkdir(parents=True, exist_ok=True)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            torch.mps.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                image = pipe(prompt=prompt["prompt"], height=base["height"], width=base["width"],
                             num_inference_steps=base["steps"], guidance_scale=base["guidance_scale"],
                             generator=generator).images[0].convert("RGB")
            torch.mps.synchronize()
            elapsed = time.perf_counter() - start
            temp = path.with_suffix(".png.tmp")
            image.save(temp, format="PNG")
            temp.replace(path)
            record = {
                **expected, "status": "completed", "runtime_seconds": elapsed,
                "cold_generation": index == 0, "warmup_generations": 0,
                "model_load_seconds": load_seconds, "device": device,
                "hardware": {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor()},
                "python": platform.python_version(), "packages": packages,
                "scheduler": type(pipe.scheduler).__name__, "scheduler_config": scheduler,
                "output_path": str(path.resolve()), "created_at": datetime.now(timezone.utc).isoformat(),
                "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "peak_memory_bytes": None,
            }
            write_json(meta_path, record)
            print(f'{prompt["id"]} seed={seed}: {elapsed:.2f}s -> {path}', flush=True)
    contact_sheet(jobs, args.output_dir)
    print(f"Contact sheet: {args.output_dir / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()

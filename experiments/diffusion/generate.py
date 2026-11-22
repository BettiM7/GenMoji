"""Generate paired SD 1.5 / SDXL baseline or LoRA images from the same prompts and seeds."""

import argparse
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import re
import time

from genmoji.models import PROFILES

BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "text-to-emoji-v1"


def comparison_key(metadata):
    fields = ("full_prompt", "seed", "model", "model_revision", "adapter_sha256",
              "training_steps", "scheduler", "steps", "guidance_scale", "architecture",
              "resolution", "device", "dtype")
    default_weight = 1.0 if metadata.get("adapter_sha256") else 0.0
    return (*[metadata.get(field) for field in fields], metadata.get("adapter_weight", default_weight))


def comparison_images(directories, export_size):
    images = {}
    for directory in directories:
        for path in sorted(directory.rglob("*.json")):
            png = path.with_suffix(".png")
            if not png.is_file():
                continue
            metadata = json.loads(path.read_text())
            if metadata.get("export_resolution", 0) >= export_size:
                images[comparison_key(metadata)] = (png, metadata)
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=PROFILES, default=None,
                        help="Model family; defaults to training record or sd15")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", action="append", help="Custom subject to wrap in the OpenMoji style template")
    prompt_group.add_argument("--raw-prompt", action="append", help="Use custom prompt text verbatim; repeat for multiple prompts")
    prompt_group.add_argument("--prompt-id", action="append", help="Select a benchmark prompt by ID; repeat for a subset")
    parser.add_argument("--seed", action="append", type=int, help="Repeat for multiple seeds; defaults to 42, 123, 456")
    parser.add_argument("--adapter", action="append", type=Path,
                        help="LoRA safetensors file or directory; repeat to compare adapters")
    parser.add_argument("--adapter-weight", action="append", type=float,
                        help="LoRA weight in (0, 1]; repeat to sweep weights for each adapter")
    parser.add_argument("--include-baseline", action="store_true", help="Also generate without an adapter")
    parser.add_argument("--reuse-from", action="append", type=Path, default=[],
                        help="Reuse matching PNG and JSON pairs from a previous output directory")
    parser.add_argument("--dry-run", action="store_true", help="Show planned image counts without loading a model")
    parser.add_argument("--training-record", type=Path, help="training.json to use the training base model and revision")
    parser.add_argument("--model", default=None, help="Base model ID; defaults to SD 1.5 or training record")
    parser.add_argument("--revision", default=None, help="Base revision; defaults to training record or main")
    parser.add_argument("--output", type=Path, help="Result root under outputs/diffusion by default")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--export-size", type=int, default=None,
                        help="Saved PNG size; defaults to 512 for the benchmark or native size for custom prompts")
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load((BENCHMARK_DIR / "experiment_config.yaml").read_text())
    benchmark = None
    if args.prompt or args.raw_prompt:
        prompts = [
            {"id": f"prompt-{i:02d}", "content": prompt,
             "prompt": prompt if args.raw_prompt else f"OpenMoji style emoji of {prompt}, on a white background."}
            for i, prompt in enumerate(args.raw_prompt or args.prompt)
        ]
    else:
        benchmark = json.loads((BENCHMARK_DIR / config["prompts_file"]).read_text())
        prompts = benchmark["prompts"]
        if args.prompt_id:
            known_ids = {prompt["id"] for prompt in prompts}
            unknown_ids = set(args.prompt_id) - known_ids
            if unknown_ids:
                parser.error(f"Unknown benchmark prompt IDs: {', '.join(sorted(unknown_ids))}")
            if len(set(args.prompt_id)) != len(args.prompt_id):
                parser.error("Duplicate benchmark prompt IDs")
            prompts = [prompt for prompt in prompts if prompt["id"] in args.prompt_id]
    if args.output is None:
        args.output = Path("outputs/diffusion") / (config["benchmark"] if benchmark else "images")

    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.adapter_weight and not args.adapter:
        parser.error("--adapter-weight requires --adapter")
    if args.adapter and len(args.adapter) > 1 and not args.training_record:
        parser.error("Multiple adapters require a shared --training-record")
    weights = args.adapter_weight or [None]
    if any(not math.isfinite(weight) or not 0 < weight <= 1 for weight in args.adapter_weight or []):
        parser.error("--adapter-weight must be greater than 0 and at most 1")
    if len(set(weights)) != len(weights):
        parser.error("Duplicate adapter weights")
    if any(not directory.is_dir() for directory in args.reuse_from):
        parser.error("--reuse-from requires an existing output directory")

    adapter_files = []
    for adapter in args.adapter or []:
        adapter_file = adapter / "pytorch_lora_weights.safetensors" if adapter.is_dir() else adapter
        if not adapter_file.is_file() or adapter_file.suffix != ".safetensors":
            parser.error(f"Expected a safetensors adapter: {adapter_file}")
        adapter_files.append(adapter_file)
    adapter_file = adapter_files[0] if adapter_files else None
    record_path = args.training_record
    if not record_path and adapter_file and (adapter_file.parent / "training.json").is_file():
        record_path = adapter_file.parent / "training.json"
    if not record_path and adapter_file and re.fullmatch(r"checkpoint-\d+", adapter_file.parent.name):
        candidate = adapter_file.parent.parent / "training.json"
        if candidate.is_file():
            record_path = candidate
    record = json.loads(record_path.read_text()) if record_path else None
    recorded_architecture = record["config"].get("architecture", "sd15") if record else None
    architecture = args.architecture or recorded_architecture or "sd15"
    if recorded_architecture and architecture != recorded_architecture:
        parser.error("Architecture differs from the supplied training record")
    profile = PROFILES[architecture]
    resolution = profile["resolution"]
    default_export_size = config["image"]["width"] if benchmark else resolution
    args.export_size = default_export_size if args.export_size is None else args.export_size
    if not 1 <= args.export_size <= resolution:
        parser.error(f"--export-size must be between 1 and {resolution}")
    model = args.model or (record["config"]["model"] if record else profile["model"])
    revision = args.revision or (record["model_revision"] if record else "main")
    if record and (model != record["config"]["model"] or revision != record["model_revision"]):
        parser.error("Model/revision differs from the supplied training record")

    conditions = []
    if args.include_baseline or not adapter_files:
        conditions.append((None, None, None, [None]))
    for adapter_file in adapter_files:
        adapter_hash = hashlib.sha256(adapter_file.read_bytes()).hexdigest()
        training_steps = None
        checkpoint = re.fullmatch(r"checkpoint-(\d+)", adapter_file.parent.name)
        if checkpoint:
            training_steps = int(checkpoint.group(1))
        elif record and record.get("status") == "completed" and record.get("adapter_sha256") == adapter_hash:
            training_steps = record["config"]["steps"]
        conditions.append((adapter_file, adapter_hash, training_steps, weights))

    import torch

    available = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(), "cpu": True}
    device = next(d for d in ("cuda", "mps", "cpu") if available[d]) if args.device == "auto" else args.device
    if not args.dry_run and not available[device]:
        parser.error(f"Device unavailable: {device}")
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    seeds = args.seed or config["seeds"]
    if len(set(seeds)) != len(seeds):
        parser.error("Duplicate seeds would overwrite images")

    resolved_revision = revision
    if not re.fullmatch(r"[0-9a-f]{40}", revision) and not args.dry_run:
        from huggingface_hub import HfApi
        resolved_revision = HfApi().model_info(model, revision=revision).sha
    reusable = comparison_images(args.reuse_from, args.export_size)
    planned = {}
    paths = set()
    for adapter_file, adapter_hash, training_steps, condition_weights in conditions:
        mode = "adapted" if adapter_file else "baseline"
        output = args.output / mode
        label = "base"
        if adapter_file:
            label = f"step-{training_steps}" if training_steps is not None else "step-unknown"
        for weight in condition_weights:
            suffix = label if weight is None else f"{label}_weight-{weight}"
            if args.export_size != default_export_size:
                suffix += f"_{args.export_size}px"
            jobs = []
            for prompt, seed in product(prompts, seeds):
                if benchmark:
                    relative = config["output_pattern"].format(
                        model=model.replace("/", "--"), condition=suffix, prompt_id=prompt["id"], seed=seed,
                    )
                    png = args.output / relative
                else:
                    png = output / f"{prompt['id']}_seed-{seed}_{suffix}.png"
                if png in paths or png.exists() or png.with_suffix(".json").exists():
                    parser.error(f"Output collision: {png}; use distinct checkpoint steps and a new --output directory")
                paths.add(png)
                metadata = {
                    "mode": mode, "prompt": prompt["prompt"] if benchmark else prompt["content"],
                    "seed": seed, "full_prompt": prompt["prompt"],
                    "model": model, "model_revision": resolved_revision, "requested_revision": revision,
                    "adapter_sha256": adapter_hash, "training_steps": training_steps,
                    "adapter_weight": (1.0 if weight is None else weight) if adapter_file else 0.0,
                    "scheduler": "DDIM", "steps": args.steps, "guidance_scale": args.guidance,
                    "architecture": architecture, "resolution": resolution, "export_resolution": args.export_size,
                    "device": device, "dtype": str(dtype),
                }
                if benchmark:
                    metadata.update(benchmark=config["benchmark"], benchmark_version=benchmark["version"],
                                    prompt_id=prompt["id"], category=prompt["category"],
                                    expected_elements=prompt["expected_elements"], condition=suffix)
                jobs.append((png, metadata, reusable.get(comparison_key(metadata))))
            planned.setdefault(adapter_file, []).append((weight, jobs))
            reused = sum(source is not None for _, _, source in jobs)
            print(f"{suffix}: {len(jobs) - reused} to generate, {reused} to reuse", flush=True)
    if args.dry_run:
        return

    from diffusers import DDIMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline
    from PIL import Image

    pipeline_class = StableDiffusionXLPipeline if architecture == "sdxl" else StableDiffusionPipeline
    pipe = pipeline_class.from_pretrained(model, revision=resolved_revision, torch_dtype=dtype, use_safetensors=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    synchronize = getattr(torch, device).synchronize if device != "cpu" else None
    for adapter_file, weight_groups in planned.items():
        if adapter_file:
            pipe.load_lora_weights(str(adapter_file.parent), weight_name=adapter_file.name, adapter_name="emoji")
        for weight, jobs in weight_groups:
            if weight is not None:
                pipe.set_adapters("emoji", adapter_weights=weight)
            for png, metadata, source in jobs:
                png.parent.mkdir(parents=True, exist_ok=True)
                if source:
                    source_png, source_metadata = source
                    with Image.open(source_png) as original:
                        image = original.resize((args.export_size, args.export_size), Image.Resampling.LANCZOS)
                    for field in ("elapsed_seconds", "nsfw_content_detected", "requested_revision"):
                        metadata[field] = source_metadata.get(field)
                    metadata["reused_from"] = str(source_png)
                else:
                    generator = torch.Generator(device="cpu").manual_seed(metadata["seed"])
                    if synchronize:
                        synchronize()
                    started = time.perf_counter()
                    result = pipe(metadata["full_prompt"], generator=generator, num_inference_steps=args.steps,
                                  guidance_scale=args.guidance, height=resolution, width=resolution)
                    if synchronize:
                        synchronize()
                    metadata["elapsed_seconds"] = time.perf_counter() - started
                    metadata["nsfw_content_detected"] = getattr(result, "nsfw_content_detected", None)
                    image = result.images[0]
                    if args.export_size != resolution:
                        image = image.resize((args.export_size, args.export_size), Image.Resampling.LANCZOS)
                image.save(png)
                png.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
                print(f"{'Reused' if source else 'Saved'} {png}", flush=True)
        if adapter_file:
            pipe.unload_lora_weights()


if __name__ == "__main__":
    main()

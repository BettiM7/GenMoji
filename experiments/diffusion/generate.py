"""Generate paired SD 1.5 / SDXL baseline or LoRA images from the same prompts and seeds."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from genmoji.models import PROFILES

DEFAULT_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_PROMPTS = [
    "a sad robot drinking coffee",
    "a nervous avocado astronaut",
    "a sleepy penguin holding an umbrella",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=PROFILES, default=None,
                        help="Model family; defaults to training record or sd15")
    parser.add_argument("--prompt", action="append", help="Repeat for multiple prompts; defaults to three examples")
    parser.add_argument("--seed", action="append", type=int, help="Repeat for multiple seeds; defaults to 42, 123, 2026")
    parser.add_argument("--adapter", type=Path, help="LoRA safetensors file or directory; omit for baseline")
    parser.add_argument("--training-record", type=Path, help="training.json to use the training base model and revision")
    parser.add_argument("--model", default=None, help="Base model ID; defaults to SD 1.5 or training record")
    parser.add_argument("--revision", default=None, help="Base revision; defaults to training record or main")
    parser.add_argument("--output", type=Path, default=Path("outputs/diffusion/images"))
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--export-size", type=int, default=None,
                        help="Saved PNG size; defaults to model resolution (512 or 1024)")
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.adapter and not args.adapter.exists():
        parser.error(f"Adapter not found: {args.adapter}")
    adapter_file = None
    if args.adapter:
        adapter_file = args.adapter / "pytorch_lora_weights.safetensors" if args.adapter.is_dir() else args.adapter
        if not adapter_file.is_file() or adapter_file.suffix != ".safetensors":
            parser.error(f"Expected a safetensors adapter: {adapter_file}")
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
    args.export_size = resolution if args.export_size is None else args.export_size
    if not 1 <= args.export_size <= resolution:
        parser.error(f"--export-size must be between 1 and {resolution}")
    model = args.model or (record["config"]["model"] if record else profile["model"])
    revision = args.revision or (record["model_revision"] if record else "main")
    if record and (model != record["config"]["model"] or revision != record["model_revision"]):
        parser.error("Model/revision differs from the supplied training record")

    adapter_hash = hashlib.sha256(adapter_file.read_bytes()).hexdigest() if adapter_file else None
    training_steps = None
    if adapter_file:
        checkpoint = re.fullmatch(r"checkpoint-(\d+)", adapter_file.parent.name)
        if checkpoint:
            training_steps = int(checkpoint.group(1))
        elif record and record.get("status") == "completed" and record.get("adapter_sha256") == adapter_hash:
            training_steps = record["config"]["steps"]
    suffix = (f"step-{training_steps}" if training_steps is not None else "step-unknown") if adapter_file else "base"
    if args.export_size != resolution:
        suffix += f"_{args.export_size}px"

    import torch
    from diffusers import DDIMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline
    from huggingface_hub import HfApi
    from PIL import Image

    available = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(), "cpu": True}
    device = next(d for d in ("cuda", "mps", "cpu") if available[d]) if args.device == "auto" else args.device
    if not available[device]:
        parser.error(f"Device unavailable: {device}")
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    prompts, seeds = args.prompt or DEFAULT_PROMPTS, args.seed or [42, 123, 2026]
    mode = "adapted" if adapter_file else "baseline"
    output = args.output / mode
    output.mkdir(parents=True, exist_ok=True)
    pairs = [(i, prompt, seed) for i, prompt in enumerate(prompts) for seed in seeds]
    if len({(i, seed) for i, _, seed in pairs}) != len(pairs):
        parser.error("Duplicate seeds would overwrite images")
    for i, _, seed in pairs:
        stem = f"prompt-{i:02d}_seed-{seed}_{suffix}"
        if (output / f"{stem}.png").exists() or (output / f"{stem}.json").exists():
            parser.error(f"Output already exists: {output / stem}; choose a new --output directory")

    resolved_revision = revision if re.fullmatch(r"[0-9a-f]{40}", revision) else HfApi().model_info(model, revision=revision).sha
    pipeline_class = StableDiffusionXLPipeline if architecture == "sdxl" else StableDiffusionPipeline
    pipe = pipeline_class.from_pretrained(model, revision=resolved_revision, torch_dtype=dtype, use_safetensors=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    if adapter_file:
        pipe.load_lora_weights(str(adapter_file.parent), weight_name=adapter_file.name)
    for i, prompt, seed in pairs:
        full_prompt = f"OpenMoji style emoji of {prompt}, on a white background."
        generator = torch.Generator(device="cpu").manual_seed(seed)
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        result = pipe(full_prompt, generator=generator, num_inference_steps=args.steps,
                      guidance_scale=args.guidance, height=resolution, width=resolution)
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - started
        stem = f"prompt-{i:02d}_seed-{seed}_{suffix}"
        image = result.images[0]
        if args.export_size != resolution:
            image = image.resize((args.export_size, args.export_size), Image.Resampling.LANCZOS)
        image.save(output / f"{stem}.png")
        metadata = {"mode": mode, "prompt": prompt, "full_prompt": full_prompt, "seed": seed,
                    "model": model, "model_revision": resolved_revision, "requested_revision": revision,
                    "adapter_sha256": adapter_hash, "training_steps": training_steps,
                    "scheduler": "DDIM", "steps": args.steps,
                    "guidance_scale": args.guidance, "architecture": architecture, "resolution": resolution,
                    "export_resolution": args.export_size, "device": device,
                    "dtype": str(dtype), "elapsed_seconds": elapsed,
                    "nsfw_content_detected": getattr(result, "nsfw_content_detected", None)}
        (output / f"{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Saved {output / (stem + '.png')} ({elapsed:.1f}s)", flush=True)


if __name__ == "__main__":
    main()

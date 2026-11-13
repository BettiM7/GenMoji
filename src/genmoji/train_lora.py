"""Single-device SD 1.5 LoRA training using the pinned Diffusers example."""

from argparse import BooleanOptionalAction
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid


MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
TRAINER = "examples/text_to_image/train_text_to_image_lora.py"
VERSION = "0.35.1"


def add_arguments(parser):
    parser.add_argument("--data", type=Path, default=Path("data/processed/openmoji-512"),
                        help="Prepared dataset root containing imagefolder/ and provenance/")
    parser.add_argument("--output", type=Path, default=Path("outputs/diffusion/lora"))
    parser.add_argument("--trainer-root", type=Path, default=Path("external/diffusers"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default="main", help="Hub revision; resolved to a commit before training")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--precision", choices=["auto", "no", "fp16", "bf16"], default="auto")
    parser.add_argument("--steps", type=int, default=1000, help="Total optimizer updates, including resumed updates")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--resume", help="latest or checkpoint-N inside the output directory")
    parser.add_argument("--gradient-checkpointing", action=BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Check data and print command without GPU imports or downloads")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_data(root):
    """Restrict training to the prepared train split and validate its metadata."""
    root = root.resolve()
    manifest = json.loads((root / "provenance/manifest.json").read_text())
    if manifest["resolution"] != 512 or manifest["mode"] != "RGB":
        raise ValueError("Training requires the shared 512px RGB dataset")
    folder = root / "imagefolder/train"
    rows = [json.loads(line) for line in (folder / "metadata.jsonl").read_text().splitlines()]
    names = set()
    for row in rows:
        name = row["file_name"]
        if (not isinstance(name, str) or Path(name).name != name
                or not name.endswith(".png") or name in names):
            raise ValueError(f"Invalid or duplicate training filename: {name}")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError(f"Missing training caption: {name}")
        if not (folder / name).is_file() or (folder / name).is_symlink():
            raise ValueError(f"Missing image or unsupported symlink: {name}")
        names.add(name)
    if not rows or len(rows) != manifest["counts"]["train"]:
        raise ValueError("Training metadata count does not match the dataset manifest")
    if names != {p.name for p in folder.glob("*.png")}:
        raise ValueError("Training PNG files and metadata disagree")
    return manifest


def resolve_resume(output, requested, steps):
    if not requested:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output is not empty; use a new --output or --resume latest")
        return None
    candidates = [p for p in output.glob("checkpoint-*")
                  if p.is_dir() and re.fullmatch(r"checkpoint-\d+", p.name)]
    if requested == "latest":
        checkpoint = max(candidates, key=lambda p: int(p.name.split("-")[1]), default=None)
    else:
        checkpoint = next((p for p in candidates if p.name == requested), None)
    if checkpoint is None:
        raise ValueError("Requested checkpoint does not exist in --output")
    if int(checkpoint.name.split("-")[1]) >= steps:
        raise ValueError("--steps must exceed the resumed checkpoint's update count")
    if not (checkpoint / "optimizer.bin").is_file():
        raise ValueError("Checkpoint has no optimizer state; an exported adapter cannot resume training")
    return checkpoint.name


def select_device(requested, precision):
    import torch

    available = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(), "cpu": True}
    device = next(d for d in ("cuda", "mps", "cpu") if available[d]) if requested == "auto" else requested
    if not available[device]:
        raise ValueError(f"Requested device {device} is unavailable")
    precision = ("fp16" if device == "cuda" else "no") if precision == "auto" else precision
    if device != "cuda" and precision != "no":
        raise ValueError("Use --precision no for CPU/MPS training")
    if device == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA GPU does not support BF16")
    return device, precision


def verify_trainer(root):
    script = root / TRAINER
    if not script.is_file():
        raise ValueError("Trainer missing. Run: git clone --depth 1 --branch v0.35.1 "
                         "https://github.com/huggingface/diffusers.git external/diffusers")
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    tag = subprocess.check_output(["git", "-C", str(root), "rev-parse", "v0.35.1^{commit}"], text=True).strip()
    original = subprocess.check_output(["git", "-C", str(root), "show", f"v0.35.1:{TRAINER}"])
    if commit != tag or script.read_bytes() != original:
        raise ValueError("Use the unmodified Diffusers v0.35.1 trainer")
    if metadata.version("diffusers") != VERSION:
        raise ValueError("Installed diffusers must be 0.35.1; install '.[diffusion]'")
    return commit


def build_command(args, model_path, precision, resume):
    options = {
        "pretrained_model_name_or_path": model_path,
        "train_data_dir": args.data.resolve() / "imagefolder/train",
        "image_column": "image", "caption_column": "text", "resolution": 512,
        "train_batch_size": args.batch_size, "gradient_accumulation_steps": args.accumulation,
        "rank": args.rank, "learning_rate": args.learning_rate,
        "lr_scheduler": "constant", "lr_warmup_steps": 0,
        "max_train_steps": args.steps, "seed": args.seed,
        "checkpointing_steps": args.checkpoint_every,
        "checkpoints_total_limit": args.keep_checkpoints,
        "mixed_precision": precision, "dataloader_num_workers": 0,
        "output_dir": args.output.resolve(), "report_to": "tensorboard",
    }
    if resume:
        options["resume_from_checkpoint"] = resume
    command = [sys.executable, str(args.trainer_root.resolve() / TRAINER)]
    for key, value in options.items():
        command.extend([f"--{key}", str(value)])
    command.append("--center_crop")
    if args.gradient_checkpointing:
        command.append("--gradient_checkpointing")
    return command


def dataset_identity(root):
    # Include actual image bytes as well as captions/settings so a changed local
    # PNG cannot silently reuse the identity of its source SVG.
    digest = hashlib.sha256()
    for path in sorted((root / "imagefolder/train").iterdir()):
        if path.is_file():
            digest.update(path.name.encode() + b"\0")
            digest.update(bytes.fromhex(sha256(path)))
    return {"manifest_sha256": sha256(root / "provenance/manifest.json"),
            "train_sha256": digest.hexdigest()}


def train(args):
    for name in ("steps", "batch_size", "accumulation", "rank", "checkpoint_every", "keep_checkpoints"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    manifest = validate_data(args.data)
    resume = resolve_resume(args.output, args.resume, args.steps)
    if args.dry_run:
        precision = args.precision if args.precision != "auto" else ("fp16" if args.device == "cuda" else "no")
        print(shlex.join(build_command(args, args.model, precision, resume)))
        print("Dry run only: auto device/precision and model revision resolve during execution.")
        return

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Launch this command directly, not through a distributed launcher")
    device, precision = select_device(args.device, args.precision)
    trainer_commit = verify_trainer(args.trainer_root.resolve())
    identity = dataset_identity(args.data.resolve())
    previous = None
    if resume:
        previous = json.loads((args.output / "training.json").read_text())
        # Preserve optimizer/scheduler semantics; increasing the update budget is allowed.
        for key in ("model", "rank", "learning_rate", "batch_size", "accumulation", "seed"):
            if previous["config"][key] != getattr(args, key):
                raise ValueError(f"Cannot resume with changed {key}")
        if previous["dataset_identity"] != identity:
            raise ValueError("Cannot resume with changed training data")
        if previous["precision"] != precision or previous["device"] != device:
            raise ValueError("Resume must use the original device type and precision")
        if previous["trainer_commit"] != trainer_commit:
            raise ValueError("Cannot resume with a different trainer revision")

    from huggingface_hub import HfApi, snapshot_download

    revision = previous["model_revision"] if previous else HfApi().model_info(args.model, revision=args.revision).sha
    # A local snapshot also pins the scheduler: the upstream script does not
    # forward --revision when loading that component.
    model_path = snapshot_download(args.model, revision=revision, allow_patterns=[
        "model_index.json", "scheduler/*.json", "tokenizer/*",
        "text_encoder/config.json", "text_encoder/model.safetensors",
        "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
        "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
    ])
    command = build_command(args, model_path, precision, resume)
    args.output.mkdir(parents=True, exist_ok=True)
    record = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "device": device, "precision": precision, "model_revision": revision,
        "trainer_commit": trainer_commit, "trainer_sha256": sha256(args.trainer_root / TRAINER),
        "dataset_identity": identity, "dataset_manifest": manifest,
        "packages": {dist.metadata["Name"]: dist.version for dist in metadata.distributions()},
        "command": command, "started_at": datetime.now(timezone.utc).isoformat(),
        "effective_batch_size": args.batch_size * args.accumulation,
        "status": "running",
    }
    # Keep a separate record for every invocation, including failed/resumed attempts.
    run_record = args.output / f"run-{uuid.uuid4().hex}.json"
    def save():
        value = json.dumps(record, indent=2) + "\n"
        run_record.write_text(value)
        (args.output / "training.json").write_text(value)
    save()
    (args.output / "dataset-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = os.environ.copy()
    env["ACCELERATE_USE_CPU"] = "true" if device == "cpu" else "false"
    env["ACCELERATE_MIXED_PRECISION"] = precision
    env["ACCELERATE_TORCH_DEVICE"] = device
    started = time.monotonic()
    print(shlex.join(command), flush=True)
    try:
        # Inherit stdout/stderr for live progress and TensorBoard logs for losses.
        subprocess.run(command, env=env, check=True)
        adapter = args.output / "pytorch_lora_weights.safetensors"
        if not adapter.is_file():
            raise RuntimeError("Trainer exited without exporting the LoRA adapter")
        record.update(status="completed", adapter_sha256=sha256(adapter))
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        save()
    print(f"Adapter saved to {adapter}")

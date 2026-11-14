"""Prepare shared emoji data and run generation experiments."""

import argparse
from pathlib import Path
import subprocess

from genmoji.models import PROFILES, dataset_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Rasterize OpenMoji into ImageFolder")
    prepare.add_argument("--architecture", choices=PROFILES, default="sd15",
                         help="Select PNG resolution and default dataset path")
    prepare.add_argument("--source", type=Path, default=Path("data/raw/openmoji"))
    prepare.add_argument("--output", type=Path, default=None)
    prepare.add_argument("--resolution", type=int, default=None)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--validation-fraction", type=float, default=0.1)
    prepare.add_argument("--test-fraction", type=float, default=0.1)
    prepare.add_argument("--limit", type=int, help="Small smoke-test subset; use a separate output")
    diffusion = commands.add_parser("diffusion", help="Stable Diffusion experiments")
    diffusion_commands = diffusion.add_subparsers(dest="diffusion_command", required=True)
    train = diffusion_commands.add_parser("train", help="Train an SD 1.5 or SDXL LoRA adapter")
    from genmoji.train_lora import add_arguments

    add_arguments(train)
    args = parser.parse_args()
    if args.command == "prepare":
        from genmoji.data import prepare_dataset

        args.output = args.output or Path(dataset_path(args.architecture))
        if args.resolution is None:
            args.resolution = PROFILES[args.architecture]["resolution"]
        try:
            result = prepare_dataset(**{k: v for k, v in vars(args).items()
                                        if k not in {"command", "architecture"}})
        except (ValueError, OSError) as error:
            parser.exit(1, f"Preparation failed: {error}\n")
        print(f"Prepared {sum(result['counts'].values())} images in {args.output}")
        print(f"Split counts: {result['counts']}")
    elif args.command == "diffusion" and args.diffusion_command == "train":
        from genmoji.train_lora import train

        try:
            train(args)
        except (ValueError, OSError, RuntimeError, ImportError, KeyError, subprocess.CalledProcessError) as error:
            parser.exit(1, f"Training failed: {error}\n")


if __name__ == "__main__":
    main()

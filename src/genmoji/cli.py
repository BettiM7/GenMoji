"""Prepare shared emoji data and run generation experiments."""

import argparse
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Rasterize OpenMoji into ImageFolder")
    prepare.add_argument("--source", type=Path, default=Path("data/raw/openmoji"))
    prepare.add_argument("--output", type=Path, default=Path("data/processed/openmoji-512"))
    prepare.add_argument("--resolution", type=int, default=512)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--validation-fraction", type=float, default=0.1)
    prepare.add_argument("--test-fraction", type=float, default=0.1)
    prepare.add_argument("--limit", type=int, help="Small smoke-test subset; use a separate output")
    diffusion = commands.add_parser("diffusion", help="Stable Diffusion experiments")
    diffusion_commands = diffusion.add_subparsers(dest="diffusion_command", required=True)
    train = diffusion_commands.add_parser("train", help="Train an SD 1.5 LoRA adapter")
    from genmoji.train_lora import add_arguments

    add_arguments(train)
    args = parser.parse_args()
    if args.command == "prepare":
        from genmoji.data import prepare_dataset

        try:
            result = prepare_dataset(**{k: v for k, v in vars(args).items() if k != "command"})
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

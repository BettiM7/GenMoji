"""Command-line entry point for the shared data pipeline."""

import argparse
from pathlib import Path


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
    args = parser.parse_args()
    if args.command == "prepare":
        from genmoji.data import prepare_dataset

        try:
            result = prepare_dataset(**{k: v for k, v in vars(args).items() if k != "command"})
        except (ValueError, OSError) as error:
            parser.exit(1, f"Preparation failed: {error}\n")
        print(f"Prepared {sum(result['counts'].values())} images in {args.output}")
        print(f"Split counts: {result['counts']}")


if __name__ == "__main__":
    main()

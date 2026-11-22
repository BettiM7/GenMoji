"""Preview or run the frozen text-to-emoji benchmark through MFLUX."""

import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = "text-to-emoji-v1"
REVISIONS = {
    "q4": "79c0034de040c460e9685bf0127f40af042a2abe",
    "q8": "7e46237dea3999a9792959f5e000aeb1b9f9e7df",
}
SMOKE_PROMPTS = ("simple_01", "attributes_01", "relationships_01")
BENCHMARK_SEEDS = (42, 123, 456)
CLI = "mflux-generate-flux3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Start inference. Default is preview.")
    parser.add_argument("--quantization", choices=REVISIONS, default="q4", help="Checkpoint to use (default: q4).")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--full", action="store_true", help="Run all 30 prompts with three seeds each.")
    selection.add_argument("--prompt-id", action="append", help="Select a benchmark prompt. Repeat to select more.")
    parser.add_argument("--seed", type=int, choices=BENCHMARK_SEEDS, help="Seed for a subset run (default: 42).")
    args = parser.parse_args()
    if args.full and args.seed is not None:
        parser.error("--full uses all benchmark seeds. Omit --seed.")

    prompt_file = ROOT / "benchmarks" / BENCHMARK / "prompts.json"
    rows = json.loads(prompt_file.read_text(encoding="utf-8"))["prompts"]
    prompts = {row["id"]: row["prompt"] for row in rows}
    prompt_ids = SMOKE_PROMPTS
    if args.full:
        prompt_ids = tuple(prompts)
    if args.prompt_id:
        prompt_ids = tuple(dict.fromkeys(args.prompt_id))
        for prompt_id in prompt_ids:
            if prompt_id not in prompts:
                parser.error(f"Unknown prompt ID {prompt_id!r}")
    seed = args.seed if args.seed is not None else BENCHMARK_SEEDS[0]
    seeds = BENCHMARK_SEEDS if args.full else (seed,)
    condition = "pretrained-standard" if args.full else "pretrained-standard-smoke"
    model = f"flux-3-image-klein-mflux-{args.quantization}"
    revision = REVISIONS[args.quantization]
    checkpoint = ROOT / "downloads" / f"{model}-{revision[:7]}"
    output_root = ROOT / "outputs" / BENCHMARK / model / condition
    if args.run:
        if shutil.which(CLI) is None:
            parser.error(f"{CLI} is missing from PATH. See the setup commands in experiments/flux3/README.md.")
        if not (checkpoint / "manifest.json").is_file():
            parser.error(f"Checkpoint missing at {checkpoint}. See the download command in experiments/flux3/README.md.")

    print(f"{len(prompt_ids) * len(seeds)} images planned in {output_root}", flush=True)
    for prompt_id in prompt_ids:
        for seed in seeds:
            output = output_root / prompt_id / f"seed_{seed}.png"
            command = [
                CLI, "--model", str(checkpoint), "--base-model", "flux3-image-klein",
                "--prompt", prompts[prompt_id], "--seed", str(seed), "--steps", "16",
                "--width", "512", "--height", "512", "--metadata",
            ]
            if not args.run:
                print(shlex.join([*command, "--output", str(output)]))
                continue

            metadata = output.with_suffix(".metadata.json")
            if output.is_file() and metadata.is_file():
                print(f"Skipping completed image {output}", flush=True)
                continue

            print(f"Generating {prompt_id} with seed {seed}", flush=True)
            output.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=output.parent) as staging:
                staged_image = Path(staging) / output.name
                staged_metadata = Path(staging) / metadata.name
                subprocess.run([*command, "--output", str(staged_image)], check=True)
                if not staged_image.is_file() or not staged_metadata.is_file():
                    raise SystemExit(f"MFLUX did not produce both output files for {prompt_id}, seed {seed}.")
                record = json.loads(staged_metadata.read_text(encoding="utf-8"))
                record.update(
                    prompt_id=prompt_id,
                    condition=condition,
                    model_id=f"mflux-community/{model}",
                    model_revision=revision,
                    scheduler="FLUX 3 flow-map schedule",
                    low_ram=False,
                )
                staged_metadata.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
                staged_metadata.replace(metadata)
                staged_image.replace(output)


if __name__ == "__main__":
    main()

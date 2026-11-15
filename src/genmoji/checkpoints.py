"""Run a pinned Diffusers trainer with milestone-aware checkpoint retention."""

import argparse
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import re
import runpy
import shutil
import sys


def prune_checkpoints(output, keep_recent, keep_every):
    """Keep the newest N checkpoint directories plus every milestone.

    Call only after a full save_state succeeds. This is filesystem retention,
    not confirmation that a mounted remote filesystem has uploaded the files.
    """
    if keep_recent < 1 or keep_every < 0:
        raise ValueError("keep_recent must be positive and keep_every nonnegative")
    checkpoints = sorted(
        (p for p in Path(output).glob("checkpoint-*")
         if p.is_dir() and not p.is_symlink() and re.fullmatch(r"checkpoint-\d+", p.name)),
        key=lambda p: int(p.name.split("-")[1]),
    )
    retained = set(checkpoints[-keep_recent:])
    if keep_every:
        retained.update(p for p in checkpoints
                        if int(p.name.split("-")[1]) > 0
                        and int(p.name.split("-")[1]) % keep_every == 0)
    for checkpoint in checkpoints:
        if checkpoint not in retained:
            shutil.rmtree(checkpoint)
    return [p.name for p in checkpoints if p in retained]


@contextmanager
def retention_after_save(accelerator_class, keep_recent, keep_every):
    """Apply retention after Accelerate saves models AND optimizer/RNG state.

    Upstream pre-save hooks run before optimizer serialization and are too early
    to prune recovery points. Keep the upstream trainer file unmodified and
    disable its own pre-save pruning in the launching command instead.
    """
    original = accelerator_class.save_state

    @wraps(original)
    def save_state(accelerator, output_dir=None, *args, **kwargs):
        result = original(accelerator, output_dir, *args, **kwargs)
        if output_dir is not None and accelerator.is_main_process:
            checkpoint = Path(output_dir)
            if re.fullmatch(r"checkpoint-\d+", checkpoint.name) and checkpoint.is_dir():
                retained = prune_checkpoints(checkpoint.parent, keep_recent, keep_every)
                print("Checkpoint retention: kept " + ", ".join(retained), flush=True)
        return result

    accelerator_class.save_state = save_state
    try:
        yield
    finally:
        accelerator_class.save_state = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-recent", type=int, required=True)
    parser.add_argument("--keep-every", type=int, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.keep_recent < 1 or args.keep_every < 0:
        parser.error("--keep-recent must be positive and --keep-every nonnegative")
    forwarded = args.trainer_args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    # The upstream deletion would remove milestones before our post-save policy.
    if any(option.split("=", 1)[0] == "--checkpoints_total_limit" for option in forwarded):
        parser.error("Use --keep-recent; upstream --checkpoints_total_limit must be disabled")
    from accelerate import Accelerator

    previous_argv = sys.argv
    sys.argv = [str(args.trainer), *forwarded]
    try:
        with retention_after_save(Accelerator, args.keep_recent, args.keep_every):
            runpy.run_path(str(args.trainer), run_name="__main__")
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()

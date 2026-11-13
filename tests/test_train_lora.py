"""Verify training orchestration without downloading weights or requiring torch."""

import argparse
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from genmoji.train_lora import add_arguments, build_command, resolve_resume, select_device, train, validate_data


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        (self.data / "provenance").mkdir(parents=True)
        self.folder = self.data / "imagefolder/train"
        self.folder.mkdir(parents=True)
        (self.data / "provenance/manifest.json").write_text(json.dumps({
            "resolution": 512, "mode": "RGB", "counts": {"train": 1}}))
        (self.folder / "emoji.png").write_bytes(b"fixture image bytes")
        (self.folder / "metadata.jsonl").write_text(json.dumps({
            "file_name": "emoji.png", "text": "OpenMoji style emoji of a robot"}) + "\n")
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        self.args = parser.parse_args(["--data", str(self.data), "--output", str(self.root / "output"),
                                      "--trainer-root", str(self.root / "trainer")])
        script = self.args.trainer_root / "examples/text_to_image/train_text_to_image_lora.py"
        script.parent.mkdir(parents=True)
        script.write_text('''import argparse
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--output_dir")
a, _ = p.parse_known_args()
root = Path(a.output_dir)
(root / "pytorch_lora_weights.safetensors").write_bytes(b"fixture adapter")
checkpoint = root / "checkpoint-250"
checkpoint.mkdir(exist_ok=True)
(checkpoint / "optimizer.bin").write_bytes(b"fixture optimizer")
''')

    def mocked_runtime(self):
        stack = ExitStack()
        stack.enter_context(patch("genmoji.train_lora.select_device", return_value=("cpu", "no")))
        stack.enter_context(patch("genmoji.train_lora.verify_trainer", return_value="trainer-commit"))
        api = stack.enter_context(patch("huggingface_hub.HfApi"))
        api.return_value.model_info.return_value.sha = "model-commit"
        stack.enter_context(patch("huggingface_hub.snapshot_download", return_value=str(self.root / "model")))
        stack.enter_context(redirect_stdout(io.StringIO()))
        self.addCleanup(stack.close)
        return stack

    def test_data_rejects_missing_images_duplicate_and_escaping_names(self):
        for rows in ([{"file_name": "missing.png", "text": "caption"}],
                     [{"file_name": "../emoji.png", "text": "caption"}],
                     [{"file_name": "emoji.png", "text": ""}],
                     [{"file_name": "emoji.png", "text": "caption"}] * 2):
            with self.subTest(rows=rows):
                (self.folder / "metadata.jsonl").write_text("\n".join(map(json.dumps, rows)))
                with self.assertRaises(ValueError):
                    validate_data(self.data)

    def test_dry_run_needs_no_torch_trainer_or_output(self):
        self.args.dry_run = True
        with patch("genmoji.train_lora.select_device", side_effect=AssertionError("must not import torch")):
            with redirect_stdout(io.StringIO()) as stream:
                train(self.args)
        self.assertIn("imagefolder/train", stream.getvalue())
        self.assertFalse(self.args.output.exists())

    def test_command_uses_only_train_and_no_flips(self):
        command = build_command(self.args, "/model path", "fp16", "checkpoint-250")
        self.assertEqual(command[command.index("--pretrained_model_name_or_path") + 1], "/model path")
        self.assertEqual(command[command.index("--train_data_dir") + 1], str(self.folder.resolve()))
        self.assertNotIn("--random_flip", command)
        self.assertIn("--gradient_checkpointing", command)
        self.assertEqual(command[command.index("--resume_from_checkpoint") + 1], "checkpoint-250")

    def test_subprocess_export_and_resume_preserve_revision_and_records(self):
        self.mocked_runtime()
        train(self.args)
        record = json.loads((self.args.output / "training.json").read_text())
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["model_revision"], "model-commit")
        self.assertEqual(record["effective_batch_size"], 4)
        with self.assertRaisesRegex(ValueError, "not empty"):
            train(self.args)
        self.args.resume = "latest"
        train(self.args)
        self.assertEqual(len(list(self.args.output.glob("run-*.json"))), 2)
        (self.folder / "emoji.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed training data"):
            train(self.args)

    def test_resume_requires_real_checkpoint_and_more_steps(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            resolve_resume(self.args.output, "latest", 1000)
        folder = self.args.output / "checkpoint-250"
        folder.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "optimizer"):
            resolve_resume(self.args.output, "latest", 1000)
        (folder / "optimizer.bin").touch()
        with self.assertRaisesRegex(ValueError, "must exceed"):
            resolve_resume(self.args.output, "latest", 250)

    def test_mps_defaults_to_fp32_and_rejects_mixed_precision(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)))
        with patch.dict("sys.modules", {"torch": torch}):
            self.assertEqual(select_device("auto", "auto"), ("mps", "no"))
            with self.assertRaisesRegex(ValueError, "precision no"):
                select_device("mps", "fp16")
            with self.assertRaisesRegex(ValueError, "unavailable"):
                select_device("cuda", "auto")

    def test_cuda_defaults_to_fp16_and_checks_bf16(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: False),
                                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))
        with patch.dict("sys.modules", {"torch": torch}):
            self.assertEqual(select_device("auto", "auto"), ("cuda", "fp16"))
            with self.assertRaisesRegex(ValueError, "BF16"):
                select_device("cuda", "bf16")

    def test_subprocess_failure_is_recorded(self):
        self.mocked_runtime()
        with patch("genmoji.train_lora.subprocess.run", side_effect=subprocess.CalledProcessError(1, "trainer")):
            with self.assertRaises(subprocess.CalledProcessError):
                train(self.args)
        record = json.loads((self.args.output / "training.json").read_text())
        self.assertEqual(record["status"], "failed")
        self.assertIn("elapsed_seconds", record)


if __name__ == "__main__":
    unittest.main()

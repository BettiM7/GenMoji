# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Adapted from NVlabs/Sana train_scripts/train_dreambooth_lora_sana.py.
# Licensed under the Apache License, Version 2.0:
# https://www.apache.org/licenses/LICENSE-2.0
"""Feasibility probe only: one batch, one optimizer update, no inference or epochs."""
import gc
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import sys
import threading
import time
import traceback

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "vendor"))  # isolated PEFT, never modifies baseline environment
import numpy as np
from PIL import Image
import torch
from diffusers import SanaPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

ARGS = argparse.ArgumentParser(description=__doc__)
MODE = ARGS.add_mutually_exclusive_group()
MODE.add_argument("--verify-without-update", action="store_true",
                  help="Correct-schedule forward/backward and saved adapter reload only; no optimizer update")
MODE.add_argument("--corrected-smoke", action="store_true",
                  help="One corrected-schedule update in a separate guarded output directory")
OPTIONS = ARGS.parse_args()
OUT = BASE / ("results/corrected_schedule_one_step" if OPTIONS.corrected_smoke else
              "results/verification_no_update" if OPTIONS.verify_without_update else "results/one_step_fp32")
MODEL = "Efficient-Large-Model/Sana_600M_512px_diffusers"
REVISION = "83d7a190bfd1fd070570a793d2dab5c7a3231b9d"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Prevent accidental additional updates on repeated execution, even after save/reload failure.
    if (OUT / "optimizer_update_started.json").exists():
        raise RuntimeError("An optimizer update has already been attempted; inspect results before any rerun.")
    result = {"status": "running", "stage": "preflight", "optimizer_updates": 0,
              "model": MODEL, "revision": REVISION, "device": "mps", "dtype": "float32",
              "rank": 4, "alpha": 4, "batch_size": 1, "resolution": [512, 512],
              "learning_rate": 1e-4, "seed": 42, "gradient_checkpointing": True,
              "target_modules": ["to_k", "to_q", "to_v"], "weighting_scheme": "none",
              "python": sys.version, "platform": platform.platform(),
              "packages": {p: version(p) for p in ("torch", "diffusers", "transformers", "accelerate", "peft")}}
    stop = threading.Event()
    peaks = {"sampled_peak_mps_allocated_bytes": 0, "sampled_peak_mps_driver_bytes": 0}

    def write():
        (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    def stage(name):
        result["stage"] = name
        write()
        print(name, flush=True)

    def sample():
        while not stop.wait(0.05):
            for key, fn in [("sampled_peak_mps_allocated_bytes", torch.mps.current_allocated_memory),
                            ("sampled_peak_mps_driver_bytes", torch.mps.driver_allocated_memory)]:
                peaks[key] = max(peaks[key], fn())

    monitor = None
    started = time.perf_counter()
    try:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; CPU fallback is forbidden for this probe")
        # Do not set allocator ratios or permit silent CPU fallback.
        if any(k in os.environ for k in ("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO")):
            raise RuntimeError("Remove custom MPS memory watermark variables; default safety limits required")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
            raise RuntimeError("Disable MPS CPU fallback before running the probe")
        result["recommended_max_memory_bytes"] = torch.mps.recommended_max_memory()
        monitor = threading.Thread(target=sample, daemon=True)
        monitor.start()
        torch.manual_seed(42)
        stage("dataset_load")
        manifest = json.loads((BASE / "data/openmoji/manifest.json").read_text())
        assert 8 <= len(manifest) <= 16
        images = []
        for record in manifest:
            path = BASE / "data/openmoji" / record["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
            with Image.open(path) as image:
                image = image.convert("RGBA")
                white = Image.new("RGBA", image.size, "white")
                white.alpha_composite(image)
                images.append(white.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR))
        result["dataset_pairs_loaded"] = len(images)
        result["training_example"] = manifest[0]
        stage("model_load")
        # Load existing cached fp16 files but run all computations in fp32; no AMP or scaler.
        pipe = SanaPipeline.from_pretrained(MODEL, revision=REVISION, variant="fp16",
                                            torch_dtype=torch.float32, local_files_only=True)
        # The inference checkpoint uses DPMSolver; use the official training scheduler explicitly.
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            MODEL, subfolder="scheduler", revision=REVISION, local_files_only=True)
        result["training_scheduler"] = type(noise_scheduler).__name__
        result["training_scheduler_config"] = dict(noise_scheduler.config)
        pipe.transformer.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        pipe.vae.requires_grad_(False)
        stage("encode_caption_mps")
        pipe.text_encoder.to("mps", dtype=torch.float32)
        with torch.no_grad():
            embeds, mask, _, _ = pipe.encode_prompt(
                prompt=manifest[0]["caption"], do_classifier_free_guidance=False,
                device="mps", max_sequence_length=300, complex_human_instruction=None)
        pipe.text_encoder.to("cpu")
        pipe.text_encoder = None
        gc.collect()
        torch.mps.empty_cache()
        stage("encode_image_mps")
        pixels = torch.from_numpy(np.array(images[0], dtype=np.float32) / 127.5 - 1).permute(2, 0, 1).unsqueeze(0).to("mps")
        pipe.vae.to("mps", dtype=torch.float32)
        with torch.no_grad():
            latents = pipe.vae.encode(pixels).latent * pipe.vae.config.scaling_factor
        pipe.vae.to("cpu")
        pipe.vae = None
        del pixels, images
        gc.collect()
        torch.mps.empty_cache()
        stage("add_lora")
        transformer = pipe.transformer.to("mps", dtype=torch.float32)
        transformer.add_adapter(LoraConfig(r=4, lora_alpha=4, init_lora_weights="gaussian",
                                          target_modules=["to_k", "to_q", "to_v"]))
        transformer.enable_gradient_checkpointing()
        transformer.train()
        trainable = {n: p for n, p in transformer.named_parameters() if p.requires_grad}
        assert trainable and all("lora_" in n for n in trainable)
        result["trainable_parameters"] = sum(p.numel() for p in trainable.values())
        result["total_transformer_parameters"] = sum(p.numel() for p in transformer.parameters())
        result["adapter_module_names"] = [n for n, m in transformer.named_modules() if hasattr(m, "lora_A")]
        before = {n: p.detach().cpu().clone() for n, p in trainable.items()}
        optimizer = torch.optim.AdamW(list(trainable.values()), lr=1e-4, betas=(0.9, 0.999),
                                     weight_decay=1e-4, eps=1e-8, foreach=False)
        torch.mps.synchronize()
        step_start = time.perf_counter()
        # Official flow-matching interpolation, uniform timestep sampling and velocity target.
        u = compute_density_for_timestep_sampling(weighting_scheme="none", batch_size=1,
                                                  logit_mean=0.0, logit_std=1.0, mode_scale=1.29)
        indices = (u * noise_scheduler.config.num_train_timesteps).long()
        timesteps = noise_scheduler.timesteps[indices].to("mps")
        sigmas = noise_scheduler.sigmas[indices].to("mps", dtype=torch.float32).reshape(1, 1, 1, 1)
        assert ((sigmas >= 0) & (sigmas <= 1)).all().item()
        noise = torch.randn(latents.shape, generator=torch.Generator(device="cpu").manual_seed(42)).to("mps")
        noisy_latents = (1 - sigmas) * latents + sigmas * noise
        stage("forward")
        pred = transformer(hidden_states=noisy_latents, encoder_hidden_states=embeds,
                           encoder_attention_mask=mask, timestep=timesteps, return_dict=False)[0]
        weight = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
        loss = (weight.float() * (pred.float() - (noise - latents).float()).square()).mean()
        assert torch.isfinite(loss).item(), "Non-finite flow matching loss"
        result["loss"] = loss.item()
        result["loss_finite"] = True
        result["timestep"] = timesteps.item()
        result["sigma"] = sigmas.item()
        stage("backward")
        loss.backward()
        torch.mps.synchronize()
        assert all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in trainable.values())
        result["all_lora_gradients_finite"] = True
        assert all(p.grad is None for p in transformer.parameters() if not p.requires_grad)
        result["frozen_base_gradients_absent"] = True
        result["gradient_tensors_present"] = len(trainable)
        result["gradient_tensors_nonzero"] = sum(bool(p.grad.count_nonzero().item()) for p in trainable.values())
        assert result["gradient_tensors_nonzero"] > 0
        result["gradient_norm"] = torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.0).item()
        assert np.isfinite(result["gradient_norm"])
        if not OPTIONS.verify_without_update:
            stage("optimizer_update")
            with (OUT / "optimizer_update_started.json").open("x") as marker:
                marker.write('{"attempted_updates": 1}\n')
            torch.mps.synchronize()
            optimizer_start = time.perf_counter()
            optimizer.step()  # Exactly one optimizer update; no loop.
            torch.mps.synchronize()
            result["optimizer_only_seconds"] = time.perf_counter() - optimizer_start
            result["optimizer_updates"] = 1
        torch.mps.synchronize()
        result["training_step_seconds"] = time.perf_counter() - step_start
        result["changed_parameter_tensors"] = sum(not torch.equal(before[n], p.detach().cpu()) for n, p in trainable.items())
        assert (result["changed_parameter_tensors"] == 0 if OPTIONS.verify_without_update
                else result["changed_parameter_tensors"] > 0)
        assert all(torch.isfinite(p).all().item() for p in trainable.values())
        stage("save_reload_adapter")
        if OPTIONS.verify_without_update:
            from safetensors.torch import load_file
            adapter_dir = BASE / "results/one_step_fp32/adapter"
            saved = load_file(adapter_dir / "pytorch_lora_weights.safetensors")
            state = {k.removeprefix("transformer."): v for k, v in saved.items()}
            result["reloaded_adapter_source"] = str(adapter_dir)
        else:
            adapter_dir = OUT / "adapter"
            state = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(transformer).items()}
            SanaPipeline.save_lora_weights(adapter_dir, transformer_lora_layers=state)
        transformer.delete_adapters("default")
        transformer.load_lora_adapter(adapter_dir, adapter_name="reloaded",
                                      weight_name="pytorch_lora_weights.safetensors",
                                      use_safetensors=True, local_files_only=True)
        reloaded = get_peft_model_state_dict(transformer, adapter_name="reloaded")
        assert set(reloaded) == set(state)
        assert all(torch.equal(state[k], v.detach().cpu()) for k, v in reloaded.items())
        result["adapter_reload_exact_match"] = True
        result["status"] = "success"
        stage("complete")
    except Exception:
        result["status"] = "failed"
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], flush=True)
    finally:
        stop.set()
        if monitor:
            monitor.join(timeout=1)
        result.update(peaks)
        result["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["total_elapsed_seconds"] = time.perf_counter() - started
        write()
    if result["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

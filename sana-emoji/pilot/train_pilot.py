# Adapted from the verified feasibility probe and NVlabs/Sana's official trainer.
# Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0
"""Bounded MPS pilot. Dry run by default; --train is required for optimizer updates."""
import argparse
from contextlib import contextmanager
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import sys
import threading
import time
import yaml

ROOT = Path(__file__).resolve().parents[2]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def sample_for_step(rows, step, seed):
    epoch, offset = divmod(step, len(rows))
    order = list(range(len(rows)))
    random.Random(seed + epoch).shuffle(order)
    return rows[order[offset]]

def plan(config):
    cfg = yaml.safe_load(Path(config).read_text())
    manifest = ROOT / cfg['train_manifest']
    rows = read_rows(manifest)
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Training manifest is empty or has duplicate IDs')
    if cfg['max_steps'] < 1 or cfg['max_steps'] > 200:
        raise ValueError('Pilot maximum must be within 1..200 updates')
    if cfg['rank'] != 4 or cfg['alpha'] != 4 or cfg['batch_size'] != 1 or cfg['dtype'] != 'float32':
        raise ValueError('This pilot requires the verified rank4/alpha4/batch1/fp32 settings')
    if any(not 1 <= s <= cfg['max_steps'] for s in cfg['checkpoint_steps']):
        raise ValueError('Checkpoint steps must be inside the training budget')
    val = read_rows(ROOT / cfg['validation_manifest'])
    if {r['split_group'] for r in rows} & {r['split_group'] for r in val}:
        raise ValueError('Train/validation group leakage')
    for r in rows + val:
        path = ROOT / cfg['dataset_root'] / r['image']
        if sha(path) != r['image_sha256']:
            raise ValueError(f'Changed image: {path}')
    identity = {'config': cfg, 'train_manifest_sha256':sha(manifest),
                'validation_manifest_sha256':sha(ROOT / cfg['validation_manifest']),
                'script_sha256':sha(__file__)}
    fingerprint = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    return cfg, rows, identity, fingerprint

@contextmanager
def lock_directory(output):
    output.mkdir(parents=True,exist_ok=True)
    lock = output / '.running'
    with lock.open('x') as f: f.write(str(os.getpid()))
    try: yield
    finally: lock.unlink()

class MemoryMonitor:
    def __init__(self, torch):
        self.torch=torch; self.peak_driver=0; self.peak_allocated=0; self.stop=threading.Event()
    def sample(self):
        self.peak_driver=max(self.peak_driver,self.torch.mps.driver_allocated_memory())
        self.peak_allocated=max(self.peak_allocated,self.torch.mps.current_allocated_memory())
    def run(self):
        while not self.stop.wait(.05): self.sample()
    def __enter__(self):
        self.thread=threading.Thread(target=self.run,daemon=True); self.thread.start(); return self
    def __exit__(self,*args):
        self.sample(); self.stop.set(); self.thread.join()

def train(cfg, rows, identity, fingerprint, output, resume):
    # Reuse the isolated, already verified PEFT; baseline environment remains untouched.
    sys.path.insert(0,str(ROOT/'sana-emoji/feasibility/vendor'))
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import SanaPipeline, FlowMatchEulerDiscreteScheduler
    from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file, save_file
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS required; no CPU fallback')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')!='0': raise RuntimeError('CPU fallback must be disabled')
    if any(k in os.environ for k in ('PYTORCH_MPS_HIGH_WATERMARK_RATIO','PYTORCH_MPS_LOW_WATERMARK_RATIO')):
        raise RuntimeError('Default MPS memory safety limits required')
    packages={p:version(p) for p in ('torch','diffusers','transformers','accelerate','peft','Pillow')}
    with lock_directory(output), MemoryMonitor(torch) as memory:
        checkpoints=output/'checkpoints'; checkpoints.mkdir(exist_ok=True)
        # An interrupted uncommitted optimizer update must never be silently repeated.
        if (output/'update_in_flight.json').exists():
            raise RuntimeError('Uncommitted update detected. Inspect run; automatic resume is unsafe.')
        complete=sorted(checkpoints.glob('step_[0-9]*'))
        if complete and not resume: raise RuntimeError('Checkpoints exist; use --resume')
        if (output/'run.json').exists():
            old=json.loads((output/'run.json').read_text())
            if old['fingerprint']!=fingerprint or old['packages']!=packages:
                raise RuntimeError('Config, manifest, code or package versions changed; use a new run directory')
        elif resume: raise RuntimeError('Cannot resume without run.json')
        else:
            (output/'run.json').write_text(json.dumps({'fingerprint':fingerprint,'identity':identity,'packages':packages},indent=2))
        step=0; resume_state=None
        if complete:
            latest=complete[-1]
            index=json.loads((latest/'checksums.json').read_text())
            if any(sha(latest/name)!=value for name,value in index.items()): raise RuntimeError('Corrupt checkpoint')
            # Only load this locally produced, trusted pickle; never untrusted downloads.
            resume_state=torch.load(latest/'state.pt',map_location='cpu',weights_only=False)
            step=resume_state['step']
        if step>=cfg['max_steps']:
            print(f'Already complete: {step} steps. No model load or update.'); return
        torch.manual_seed(cfg['seed'])
        pipe=SanaPipeline.from_pretrained(cfg['model_id'],revision=cfg['revision'],variant='fp16',torch_dtype=torch.float32,local_files_only=True)
        for component in (pipe.transformer,pipe.text_encoder,pipe.vae): component.requires_grad_(False)
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(cfg['model_id'],subfolder='scheduler',revision=cfg['revision'],local_files_only=True)
        # Cache frozen features one sample at a time. No model compute occurs on CPU.
        cache=output/'feature_cache'; cache.mkdir(exist_ok=True)
        missing=[r for r in rows if not (cache/f'{r["id"]}.safetensors').exists()]
        if missing:
            pipe.text_encoder.to('mps',dtype=torch.float32)
            for r in missing:
                with torch.no_grad():
                    emb,mask,_,_=pipe.encode_prompt(r['caption'],do_classifier_free_guidance=False,device='mps',max_sequence_length=300,complex_human_instruction=None)
                save_file({'embeds':emb.cpu().contiguous(),'mask':mask.cpu().contiguous()},cache/f'{r["id"]}.text.safetensors')
            pipe.text_encoder.to('cpu')
        pipe.text_encoder=None; gc.collect(); torch.mps.empty_cache()
        if missing:
            pipe.vae.to('mps',dtype=torch.float32)
            for r in missing:
                with Image.open(ROOT/cfg['dataset_root']/r['image']) as im:
                    pixels=torch.from_numpy(np.array(im,dtype=np.float32)/127.5-1).permute(2,0,1).unsqueeze(0).to('mps')
                with torch.no_grad(): latent=pipe.vae.encode(pixels).latent*pipe.vae.config.scaling_factor
                tensors=load_file(cache/f'{r["id"]}.text.safetensors'); tensors['latents']=latent.cpu().contiguous()
                tmp=cache/f'{r["id"]}.tmp'; save_file(tensors,tmp); tmp.replace(cache/f'{r["id"]}.safetensors')
                (cache/f'{r["id"]}.text.safetensors').unlink()
            del pixels,latent,tensors
            pipe.vae.to('cpu')
        pipe.vae=None; gc.collect(); torch.mps.empty_cache()
        model=pipe.transformer.to('mps',dtype=torch.float32)
        # Seed init independently of feature cache presence/encoding order.
        torch.manual_seed(cfg['seed'])
        model.add_adapter(LoraConfig(r=cfg['rank'],lora_alpha=cfg['alpha'],init_lora_weights='gaussian',target_modules=['to_q','to_k','to_v']))
        model.enable_gradient_checkpointing(); model.train()
        params=[p for n,p in model.named_parameters() if p.requires_grad]
        if not params or any('lora_' not in n for n,p in model.named_parameters() if p.requires_grad): raise RuntimeError('Trainable base weights')
        optimizer=torch.optim.AdamW(params,lr=cfg['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=1e-4,foreach=False)
        noise_rng=torch.Generator(device='cpu').manual_seed(cfg['seed'])
        torch.manual_seed(cfg['seed']+1)
        if resume_state:
            adapter=load_file(latest/'adapter/pytorch_lora_weights.safetensors')
            saved_adapter={k.removeprefix('transformer.'):v for k,v in adapter.items()}
            set_peft_model_state_dict(model,saved_adapter)
            restored=get_peft_model_state_dict(model)
            if set(restored)!=set(saved_adapter) or any(not torch.equal(restored[k].detach().cpu(),v) for k,v in saved_adapter.items()):
                raise RuntimeError('Adapter reload mismatch')
            optimizer.load_state_dict(resume_state['optimizer'])
            torch.set_rng_state(resume_state['cpu_rng']); torch.mps.set_rng_state(resume_state['mps_rng'])
            noise_rng.set_state(resume_state['noise_rng'])
        # Commit EVERY update for safe resume. Named 50/100/200 checkpoints remain available.
        for next_step in range(step+1,cfg['max_steps']+1):
            r=sample_for_step(rows,next_step-1,cfg['seed'])
            tensors=load_file(cache/f'{r["id"]}.safetensors')
            x=tensors['latents'].to('mps'); emb=tensors['embeds'].to('mps'); mask=tensors['mask'].to('mps')
            torch.mps.synchronize(); start=time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            u=compute_density_for_timestep_sampling(weighting_scheme='none',batch_size=1,logit_mean=0.,logit_std=1.,mode_scale=1.29)
            index=(u*scheduler.config.num_train_timesteps).long()
            timestep=scheduler.timesteps[index].to('mps'); sigma=scheduler.sigmas[index].to('mps').reshape(1,1,1,1)
            if not ((sigma>=0)&(sigma<=1)).all().item(): raise RuntimeError('Invalid training sigma')
            noise=torch.randn(x.shape,generator=noise_rng).to('mps')
            pred=model(hidden_states=(1-sigma)*x+sigma*noise,encoder_hidden_states=emb,encoder_attention_mask=mask,timestep=timestep,return_dict=False)[0]
            weight=compute_loss_weighting_for_sd3(weighting_scheme='none',sigmas=sigma)
            loss=(weight.float()*(pred.float()-(noise-x).float()).square()).mean()
            if not torch.isfinite(loss).item(): raise RuntimeError('Nonfinite loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in params): raise RuntimeError('Missing/nonfinite gradient')
            if not any(p.grad.count_nonzero().item() for p in params): raise RuntimeError('All gradients zero')
            norm=torch.nn.utils.clip_grad_norm_(params,1.)
            if not torch.isfinite(norm).item(): raise RuntimeError('Nonfinite gradient norm')
            in_flight=output/'update_in_flight.json'
            with in_flight.open('x') as f: json.dump({'step':next_step},f)
            optimizer.step(); torch.mps.synchronize(); seconds=time.perf_counter()-start
            if any(not torch.isfinite(p).all().item() for p in params): raise RuntimeError('Nonfinite updated weight')
            memory.sample()
            metrics={'step':next_step,'id':r['id'],'loss':loss.item(),'gradient_norm':norm.item(),'seconds':seconds,'sigma':sigma.item(),
                     'peak_driver_bytes':memory.peak_driver,'peak_allocated_bytes':memory.peak_allocated}
            tmp=checkpoints/f'.step_{next_step:06d}.tmp'; tmp.mkdir()
            state={k:v.detach().cpu().contiguous() for k,v in get_peft_model_state_dict(model).items()}
            SanaPipeline.save_lora_weights(tmp/'adapter',transformer_lora_layers=state)
            saved=load_file(tmp/'adapter/pytorch_lora_weights.safetensors')
            if any(not torch.equal(saved['transformer.'+k],v) for k,v in state.items()):
                raise RuntimeError('Adapter serialization mismatch')
            torch.save({'step':next_step,'optimizer':optimizer.state_dict(),'cpu_rng':torch.get_rng_state(),
                        'mps_rng':torch.mps.get_rng_state(),'noise_rng':noise_rng.get_state()},tmp/'state.pt')
            (tmp/'metrics.json').write_text(json.dumps(metrics,indent=2))
            names=['state.pt','metrics.json','adapter/pytorch_lora_weights.safetensors']
            (tmp/'checksums.json').write_text(json.dumps({n:sha(tmp/n) for n in names},indent=2))
            tmp.replace(checkpoints/f'step_{next_step:06d}')
            in_flight.unlink()
            # Per-checkpoint metrics are authoritative, including when the append log was interrupted.
            with (output/'loss.jsonl').open('a') as log: log.write(json.dumps(metrics)+'\n')
            print(json.dumps(metrics),flush=True)
            # Retain only latest plus planned checkpoints, bounding disk use.
            import shutil
            for old in checkpoints.glob('step_[0-9]*'):
                old_step=int(old.name.split('_')[1])
                if old_step!=next_step and old_step not in cfg['checkpoint_steps']:
                    shutil.rmtree(old)
        (output/'complete.json').write_text(json.dumps({'steps':cfg['max_steps'],'status':'complete'}))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('pilot_config.yaml'))
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--train',action='store_true'); p.add_argument('--resume',action='store_true')
    args=p.parse_args(); cfg,rows,identity,fingerprint=plan(args.config)
    output=(args.output_dir or ROOT/cfg['output_dir']).resolve()
    # Always isolate pilot artifacts, including caches, from baseline and feasibility paths.
    pilot_root=(ROOT/'sana-emoji/pilot/runs').resolve()
    if not output.is_relative_to(pilot_root) or output==pilot_root: p.error('Output must be a named directory inside sana-emoji/pilot/runs')
    if not args.train:
        print(json.dumps({'mode':'dry_run','samples':len(rows),'max_steps':cfg['max_steps'],'checkpoints':cfg['checkpoint_steps'],'output':str(output),'fingerprint':fingerprint},indent=2)); return
    train(cfg,rows,identity,fingerprint,output,args.resume)

if __name__=='__main__': main()

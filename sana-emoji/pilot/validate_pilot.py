"""Fixed held-out generation in a separate process; dry-run unless --generate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from train_pilot import ROOT, plan, read_rows, sha


def select_validation(cfg):
    rows=read_rows(ROOT/cfg['validation_manifest'])
    # Freeze these across checkpoints: one held-out sample per category, then fixed ordering.
    chosen=[]
    for category in cfg['validation']['categories']:
        members=[r for r in rows if r['category']==category]
        chosen.append(min(members,key=lambda r:hashlib.sha256(f'validation:{r["id"]}'.encode()).hexdigest()))
    return chosen[:cfg['validation']['samples']]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('pilot_config.yaml'))
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--generate',action='store_true')
    args=p.parse_args(); cfg,_,_,fingerprint=plan(args.config)
    checkpoint=args.checkpoint.resolve()
    run=checkpoint.parent.parent
    if not checkpoint.is_relative_to((ROOT/'sana-emoji/pilot/runs').resolve()): p.error('Pilot checkpoint path required')
    if (run/'.running').exists(): p.error('Stop training before separate-process validation to avoid overlapping MPS loads')
    record=json.loads((run/'run.json').read_text())
    if record['fingerprint']!=fingerprint: raise RuntimeError('Checkpoint/config fingerprint mismatch')
    checks=json.loads((checkpoint/'checksums.json').read_text())
    if any(sha(checkpoint/n)!=v for n,v in checks.items()): raise RuntimeError('Checkpoint checksum mismatch')
    chosen=select_validation(cfg)
    print(json.dumps({'checkpoint':str(checkpoint),'seed':cfg['validation']['seed'],'prompts':[r['caption'] for r in chosen],'mode':'generate' if args.generate else 'dry_run'},indent=2))
    if not args.generate: return
    sys.path.insert(0,str(ROOT/'sana-emoji/feasibility/vendor'))
    import os
    import torch
    from diffusers import SanaPipeline
    from PIL import Image,ImageDraw
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS required')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')!='0' or any(k in os.environ for k in ('PYTORCH_MPS_HIGH_WATERMARK_RATIO','PYTORCH_MPS_LOW_WATERMARK_RATIO')):
        raise RuntimeError('Default MPS safety limits and no fallback required')
    output=run/'validation'/checkpoint.name; output.mkdir(parents=True,exist_ok=True)
    adapter=checkpoint/'adapter/pytorch_lora_weights.safetensors'
    pending=[]
    for r in chosen:
        path=output/f'{r["id"]}.png'; meta=path.with_suffix('.json')
        expected={'prompt':r['caption'],'seed':cfg['validation']['seed'],'adapter_sha256':sha(adapter),'config_fingerprint':fingerprint}
        if path.exists() and meta.exists():
            old=json.loads(meta.read_text())
            if any(old.get(k)!=v for k,v in expected.items()) or old['image_sha256']!=sha(path):
                raise RuntimeError('Existing validation output conflicts; inspect instead of overwriting')
        else: pending.append((r,path,meta,expected))
    if pending:
        # Inference uses the checkpoint's original inference scheduler, not the training scheduler.
        pipe=SanaPipeline.from_pretrained(cfg['model_id'],revision=cfg['revision'],variant='fp16',torch_dtype=torch.float32,local_files_only=True)
        pipe.load_lora_weights(adapter.parent,weight_name=adapter.name,local_files_only=True)
        pipe.to('mps')
        for r,path,meta,expected in pending:
            torch.mps.synchronize(); started=time.perf_counter()
            with torch.inference_mode():
                image=pipe(r['caption'],height=512,width=512,num_inference_steps=cfg['validation']['steps'],guidance_scale=cfg['validation']['guidance_scale'],generator=torch.Generator(device='cpu').manual_seed(cfg['validation']['seed'])).images[0].convert('RGB')
            torch.mps.synchronize(); seconds=time.perf_counter()-started
            image.save(path)
            meta.write_text(json.dumps({**expected,'runtime_seconds':seconds,'steps':cfg['validation']['steps'],'guidance_scale':cfg['validation']['guidance_scale'],'image_sha256':sha(path)},indent=2))
    sheet=Image.new('RGB',(256*len(chosen),300),'white'); draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(chosen):
        with Image.open(output/f'{r["id"]}.png') as im: sheet.paste(im.resize((256,256)),(i*256,0))
        draw.text((i*256+4,260),r['annotation'],fill='black')
    sheet.save(output/'contact_sheet.png')

if __name__=='__main__': main()

"""Pinned OpenMoji preparation; offline validation by default, downloads only with --download."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import platform
from importlib.metadata import version
import re
import textwrap
import time
import urllib.request
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'datasets/openmoji-17.0.0'
URL = 'https://raw.githubusercontent.com/hfg-gmuend/openmoji/17.0.0/'
STYLE = 'OpenMoji emoji illustration, clean outlines, white background'
SEED = 42

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def fetch(relative, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            data = urllib.request.urlopen(URL + relative, timeout=40).read()
            tmp = path.with_suffix('.download')
            tmp.write_bytes(data)
            tmp.replace(path)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)

def reject(m):
    g, sub, a = m['group'], m['subgroups'], m['annotation'].lower()
    if m.get('skintone'):
        return 'skin_tone_variant'
    if g in ('flags', 'symbols', 'component', 'extras-unicode'):
        return 'flags_symbols_components'
    if g == 'extras-openmoji' and sub not in ('food-drink', 'animals-nature', 'objects', 'gardening'):
        return 'extras_outside_illustration_scope'
    if g == 'smileys-emotion' and sub in ('heart', 'emotion'):
        return 'abstract_emotion_symbol'
    if g == 'people-body' and sub in ('family', 'person-symbol'):
        return 'complex_family_or_symbol_variants'
    if g == 'travel-places' and (sub == 'time' or re.search(r'moon$', a)):
        return 'clock_variant'
    if a in ('spade suit','heart suit','diamond suit','club suit','mahjong red dragon'):
        return 'abstract_game_symbol'
    if sub in ('sound',) or a in ('musical note', 'musical notes', 'notebook', 'notebook with decorative cover', 'ledger', 'tardis'):
        return 'text_symbol_or_branded_object'
    if re.search(r'flag|button|input |alphabet|letter |banknote|currency|chart|calendar|page |pages |scroll|passport|id card|bookmark tabs|ticket|envelope|fax machine', a):
        return 'text_or_symbol_content'
    return None

def family(annotation):
    s = annotation.lower()
    s = re.sub(r'\b(man|woman|men|women|boy|girl)\b', 'person', s)
    s = re.sub(r'^person (?=.)', '', s)
    s = re.sub(r'\b(red|green|blue|yellow|orange|purple|black|white|brown|light|dark|medium|small|large)\b', '', s)
    s = re.sub(r'\b(facing left|facing right|left-facing|right-facing|leftwards|rightwards)\b', '', s)
    return re.sub(r'[^a-z0-9]+', ' ', s).strip()

def feature(image):
    rgb = np.array(image)
    ys, xs = np.where((rgb < 245).any(axis=2))
    if not len(xs):
        raise ValueError('Blank image')
    crop = image.crop((int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1))
    thumb = np.array(crop.resize((32, 32)), dtype=np.float32)
    gray = np.array(crop.convert('L').resize((32, 32)), dtype=float)
    n = np.arange(32)
    basis = np.cos(np.pi / 32 * (n[None, :] + .5) * n[:, None])
    coeff = (basis @ gray @ basis.T)[:8, :8].flatten()[1:]
    bits = coeff > np.median(coeff)
    return int(''.join('1' if b else '0' for b in bits), 2), thumb

def sheet(records, name, count=64):
    # category round-robin, deterministic and representative
    chosen = balanced(records, min(count, len(records)))
    cols, w, h = 8, 160, 220
    canvas = Image.new('RGB', (cols*w, ((len(chosen)+cols-1)//cols)*h), 'white')
    draw = ImageDraw.Draw(canvas)
    for i,r in enumerate(chosen):
        x,y = i%cols*w, i//cols*h
        with Image.open(DATA/r['image']) as im:
            canvas.paste(im.resize((160,160)), (x,y))
        label = (r['id'] if len(r['id'])<=20 else r['id'][:17]+'...') + '\n' + '\n'.join(textwrap.wrap(r.get('description',r['annotation']), 23)[:3])
        draw.multiline_text((x+3,y+162),label,fill='black')
    canvas.save(DATA/'visualizations'/name)
    dump(DATA/'visualizations'/name.replace('.png','.json'),chosen)

def balanced(records, count):
    buckets=defaultdict(list)
    for r in records: buckets[r['category']].append(r)
    for c in buckets:
        buckets[c].sort(key=lambda r: hashlib.sha256(f'{SEED}:{r["id"]}'.encode()).hexdigest())
    chosen=[]
    while len(chosen)<count:
        for c in sorted(buckets):
            if buckets[c] and len(chosen)<count: chosen.append(buckets[c].pop())
    return chosen

def prepare(download):
    for d in ('originals/png','images','manifests','visualizations'): (DATA/d).mkdir(parents=True,exist_ok=True)
    if download:
        fetch('data/openmoji.json', DATA/'originals/openmoji.json')
        fetch('LICENSE.txt', DATA/'LICENSE.txt')
    metadata=json.loads((DATA/'originals/openmoji.json').read_text())
    candidates=[]; excluded=[]
    for m in metadata:
        reason=reject(m)
        if reason: excluded.append({'id':m['hexcode'],'reason':reason,'annotation':m['annotation']})
        else: candidates.append(m)
    # Retain one gender-neutral/lexically canonical member per gender family.
    gender_families=defaultdict(list)
    other=[]
    for m in candidates:
        if m['group']=='people-body':
            key=re.sub(r'\b(man|woman|men|women|boy|girl)\b','person',m['annotation'].lower())
            key=re.sub(r'^person (?=.)','',key)
            gender_families[key].append(m)
        else: other.append(m)
    for members in gender_families.values():
        members.sort(key=lambda m:(bool(re.search(r'\b(man|woman|boy|girl)\b',m['annotation'])),m['hexcode']))
        other.append(members[0])
        excluded.extend({'id':m['hexcode'],'reason':'gender_variant','retained':members[0]['hexcode']} for m in members[1:])
    candidates=other
    print(f'{len(metadata)} metadata entries; {len(candidates)} candidate PNGs',flush=True)
    if download:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda m:fetch(f'color/618x618/{m["hexcode"]}.png',DATA/'originals/png'/f'{m["hexcode"]}.png'),candidates))
    records=[]; features=[]; exact={}; relations=[]
    for m in sorted(candidates,key=lambda m:m['hexcode']):
        code=m['hexcode']; src=DATA/'originals/png'/f'{code}.png'
        with Image.open(src) as image:
            image=image.convert('RGBA'); white=Image.new('RGBA',image.size,'white'); white.alpha_composite(image)
            rgb=ImageOps.pad(white.convert('RGB'),(512,512),method=Image.Resampling.LANCZOS,color='white')
        pixelhash=hashlib.sha256(rgb.tobytes()).hexdigest()
        if pixelhash in exact:
            excluded.append({'id':code,'reason':'exact_pixel_duplicate','retained':exact[pixelhash]}); continue
        ph, thumb=feature(rgb)
        near=[]; duplicate=None
        for prev,prevhash,prevthumb in features:
            distance=(ph^prevhash).bit_count()
            if distance<=8:
                mae=float(np.abs(thumb-prevthumb).mean())
                if mae<=12:
                    near.append(prev)
                    relations.append({'a':code,'b':prev,'phash_hamming':distance,'crop_rgb_mae':mae})
                    if distance<=2 and mae<=4: duplicate=prev
        if duplicate:
            excluded.append({'id':code,'reason':'near_duplicate','retained':duplicate}); continue
        dst=DATA/'images'/f'{code}.png'; rgb.save(dst)
        exact[pixelhash]=code; features.append((code,ph,thumb))
        description={'E148':'broom and dustpan'}.get(code,m['annotation'])
        records.append({'id':code,'annotation':m['annotation'],'description':description,'caption':f'{description}, {STYLE}',
                        'category':m['group'],'subcategory':m['subgroups'],'family':family(m['annotation']),
                        'image':f'images/{code}.png','original':f'originals/png/{code}.png',
                        'source_url':URL+f'color/618x618/{code}.png','release':'17.0.0',
                        'original_sha256':sha(src),'image_sha256':sha(dst),'pixel_sha256':pixelhash,
                        'phash':f'{ph:016x}','license':'CC-BY-SA-4.0','author':m['openmoji_author'],
                        'metadata':m})
    parent={r['id']:r['id'] for r in records}
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]]; a=parent[a]
        return a
    def union(a,b):
        a,b=find(a),find(b); parent[max(a,b)]=min(a,b)
    families={}
    for r in records:
        if r['family'] in families: union(r['id'],families[r['family']])
        families[r['family']]=r['id']
    for pair in relations:
        if pair['a'] in parent and pair['b'] in parent: union(pair['a'],pair['b'])
    grouped=defaultdict(list)
    for r in records:
        r['split_group']=find(r['id'])
        grouped[r['split_group']].append(r)
    targets={c:max(1,round(n*.1)) for c,n in Counter(r['category'] for r in records).items()}
    heldout=Counter()
    for gid in sorted(grouped,key=lambda x:hashlib.sha256(f'{SEED}:{x}'.encode()).hexdigest()):
        members=grouped[gid]
        primary=Counter(r['category'] for r in members).most_common(1)[0][0]
        split='validation' if heldout[primary]<targets[primary] else 'train'
        for r in members:
            r['split']=split
            if split=='validation': heldout[r['category']]+=1
    train=[r for r in records if r['split']=='train']; val=[r for r in records if r['split']=='validation']
    pilot=balanced(train,min(192,len(train)))
    for name, rows in [('all',records),('train',train),('validation',val),('pilot_train',pilot)]:
        (DATA/'manifests'/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    dump(DATA/'exclusions.json',excluded); dump(DATA/'near_duplicate_pairs.json',relations)
    report={'release':'17.0.0','seed':SEED,'metadata_entries':len(metadata),'eligible_pngs':len(candidates),'downloaded_pngs':len(list((DATA/'originals/png').glob('*.png'))),
            'selected':len(records),'excluded':len(excluded),'train':len(train),'validation':len(val),'pilot_train':len(pilot),
            'exclusions_by_reason':dict(Counter(r['reason'] for r in excluded)),
            'category_distribution':{name:dict(Counter(r['category'] for r in rows)) for name,rows in [('all',records),('train',train),('validation',val),('pilot_train',pilot)]},
            'near_duplicate_pairs':len(relations),'metadata_sha256':sha(DATA/'originals/openmoji.json'),'preparation_script_sha256':sha(Path(__file__)),
            'software':{'python':platform.python_version(),'Pillow':version('Pillow'),'numpy':version('numpy')},
            'caption_overrides':{'E148':'broom and dustpan'}}
    # Archive previews no longer selected after a policy revision; originals remain intact.
    selected_ids={r['id'] for r in records}
    archive=DATA/'originals/unused_previews'; archive.mkdir(exist_ok=True)
    for old in (DATA/'images').glob('*.png'):
        if old.stem not in selected_ids: old.replace(archive/old.name)
    dump(DATA/'report.json',report)
    sheet(pilot,'pilot_train.png'); sheet(val,'validation.png')
    dump(DATA/'sample_pairs.json',balanced(records,16))
    (DATA/'ATTRIBUTION.md').write_text('# OpenMoji 17.0.0\n\nEmojis designed by [OpenMoji](https://openmoji.org/), HfG Schwäbisch Gmünd and contributors. Graphics licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). See LICENSE.txt.\n\nOriginal metadata and selected PNGs are retained. Processed images are white-composited, aspect-preserving resized/padded to 512x512; derived images and contact sheets retain CC BY-SA 4.0 attribution. Authors and source URLs are in every manifest record. Captions use upstream annotations plus a constant style phrase, except reviewed E148 (broom and dustpan) replacing the culturally specific kehrwoche.\n')
    print(json.dumps(report,indent=2))

def validate():
    rows=[json.loads(s) for s in (DATA/'manifests/all.jsonl').read_text().splitlines()]
    ids=set(); hashes=set(); groups={}
    for r in rows:
        assert r['id'] not in ids; ids.add(r['id'])
        assert r['pixel_sha256'] not in hashes; hashes.add(r['pixel_sha256'])
        assert r['caption']==r['description']+', '+STYLE
        assert sha(DATA/r['image'])==r['image_sha256'] and sha(DATA/r['original'])==r['original_sha256']
        with Image.open(DATA/r['image']) as im:
            assert im.mode=='RGB' and im.size==(512,512)
            assert hashlib.sha256(im.tobytes()).hexdigest()==r['pixel_sha256']
            assert im.getpixel((0,0))==(255,255,255)
        assert groups.setdefault(r['split_group'],r['split'])==r['split']
    byid={r['id']:r for r in rows}
    families={}
    for r in rows:
        assert families.setdefault(r['family'],r['split_group'])==r['split_group']
    for pair in json.loads((DATA/'near_duplicate_pairs.json').read_text()):
        if pair['a'] in byid and pair['b'] in byid:
            assert byid[pair['a']]['split_group']==byid[pair['b']]['split_group']
    assert len(list((DATA/'images').glob('*.png')))==len(rows)
    for name in ('train','validation','pilot_train'):
        subset=[json.loads(s) for s in (DATA/'manifests'/f'{name}.jsonl').read_text().splitlines()]
        assert len({r['id'] for r in subset})==len(subset)
        assert all(r==byid[r['id']] and r['split']==('train' if name=='pilot_train' else name) for r in subset)
    print(f'Validated {len(rows)} unique images/captions, checksums, white corners, split groups and pilot membership.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--download',action='store_true'); p.add_argument('--validate-only',action='store_true'); args=p.parse_args()
    if not args.validate_only: prepare(args.download)
    validate()

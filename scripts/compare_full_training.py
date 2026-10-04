"""Paired validation of tiny-subset baseline and full-data training snapshot."""
import json
from pathlib import Path
import numpy as np
from render_small_scale import animate

ROOT=Path(__file__).resolve().parents[1]
old=ROOT/'artifacts/frozen_val96'
new=ROOT/'artifacts/overnight_13633_val96'
assert (old/'manifest.json').read_bytes()==(new/'manifest.json').read_bytes()
reports=[json.loads((p/'report.json').read_text()) for p in [old,new]]
with np.load(old/'embeddings.npz') as a,np.load(new/'embeddings.npz') as b:
    assert np.array_equal(a['real'],b['real'])
    assert np.array_equal(a['real_text'],b['real_text'])
    assert np.isfinite(b['generated']).all()
metrics=[r['metrics']['generated'] for r in reports]
summary={'same_manifest_and_real_embeddings':True,'old_updates':2000,'new_updates':13633,
    'old_microsteps':2000,'new_microsteps':218014,'old':metrics[0],'new':metrics[1],
    'fid_relative_decrease':1-metrics[1]['fid']/metrics[0]['fid'],
    'not_compute_matched':True,'note':'Changed dataset coverage, accumulation, length and training budget; no single-factor causal claim'}
(new/'comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2),flush=True)
manifest=json.loads((new/'manifest.json').read_text(encoding='utf-8'))
with np.load(old/'motions.npz') as a,np.load(new/'motions.npz') as b:
    assert np.array_equal(a['real'],b['real'])
    for i in [0,1]:
        length=manifest[i]['length']
        animate([b['real'][i,:length],a['generated'][i,:length],b['generated'][i,:length]],
            [f"Val ID {manifest[i]['id']}: real",'Old: 2000 updates / 32 clips','Full data: 13633 updates'],
            new/f'case_{i}_comparison.gif')

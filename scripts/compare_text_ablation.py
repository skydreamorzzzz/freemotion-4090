"""Compare completed frozen/unfrozen adapter trials without changing weights."""
import gc
import json
import numpy as np
import torch
from pathlib import Path
from render_small_scale import animate

ROOT=Path(__file__).resolve().parents[1]
a=ROOT/'artifacts/single_2000_8gb'
b=ROOT/'artifacts/single_2000_unfrozen'
for name in ['subset.json','subset.npz']:
    assert (a/name).read_bytes()==(b/name).read_bytes()
reports=[json.loads((p/'single_report.json').read_text()) for p in [a,b]]
logs=[[json.loads(x) for x in (p/'single_progress.jsonl').read_text().splitlines()] for p in [a,b]]
assert len(logs[0])==len(logs[1])==2000
assert all(x['example']==y['example'] and x['lr']==y['lr'] for x,y in zip(*logs))
for split in ['train','val']:
    for label in ['before','after']:
        with np.load(a/f'single_{split}_{label}.npz') as x,np.load(b/f'single_{split}_{label}.npz') as y:
            assert np.array_equal(x['ground_truth'],y['ground_truth'])
            if label=='before': assert np.array_equal(x['generated'],y['generated'])
            else:
                animate([y['ground_truth'],x['generated'],y['generated']],
                    [f'{split}: real reference','Frozen adapter: 2000','Trainable adapter: 2000'],
                    b/f'single_{split}_adapter_comparison.gif')
prefixes=('model.clipTransEncoder.','model.clip_ln.','model.clip_transformer.','model.token_embedding.','model.ln_final.','model.positional_embedding')
c=torch.load(ROOT/'artifacts/random_init/single_random_init.ckpt',map_location='cpu',weights_only=True)
initial={k:v.clone() for k,v in c['state_dict'].items() if k.startswith(prefixes)}
del c
gc.collect()
c=torch.load(b/'single_trial.ckpt',map_location='cpu',weights_only=True)
changed=[k for k,v in initial.items() if not torch.equal(v,c['state_dict'][k])]
assert changed and all(k.startswith(('model.clipTransEncoder.','model.clip_ln.')) for k in changed)
summary={'same_subset':True,'same_steps_order_lr':True,'same_before_samples':True,
    'original_clip_unchanged':True,'changed_adapter_state_keys':changed,
    'diagnostics':[{k:r[k] for k in ['after_denoising','after_samples','peak_training_memory']} for r in reports],
    'unfrozen_adapter_nonzero_gradient_steps':reports[1]['adapter_nonzero_gradient_steps']}
(b/'adapter_comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='diagnostics'}),flush=True)

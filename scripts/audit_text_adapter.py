"""CPU audit of frozen text adaptation weights and train/eval stochasticity."""
import gc
import json
import torch
from small_scale_trial import ROOT, get_config, InterGenSpatialControlNet, seed

torch.set_num_threads(4)
prefixes=('model.clipTransEncoder.','model.clip_ln.')
original=torch.load(ROOT/'artifacts/random_init/single_random_init.ckpt',map_location='cpu',weights_only=True)
initial={k:v.clone() for k,v in original['state_dict'].items() if k.startswith(prefixes)}
del original
gc.collect()
trained=torch.load(ROOT/'artifacts/single_2000_8gb/single_trial.ckpt',map_location='cpu',weights_only=True)
equal={k:torch.equal(v,trained['state_dict'][k]) for k,v in initial.items()}
seed(20261003)
model=InterGenSpatialControlNet(get_config('configs/model_single.yaml'))
model.load_state_dict({k.removeprefix('model.'):v for k,v in trained['state_dict'].items()},strict=True)
del trained
gc.collect()
prompts=['A person walks forward.','A person sits down.','A person jumps.']

@torch.no_grad()
def probe(mode,adapter_eval=False):
    model.train(mode=='train')
    if adapter_eval: model.clipTransEncoder.eval(); model.clip_ln.eval()
    seed(20261003)
    values=torch.stack([model.text_process({'text':prompts})['cond'].clone() for _ in range(6)])
    return {'repeat_max_abs_difference':float((values-values[0]).abs().max()),
        'repeat_mean_cosine':float(torch.nn.functional.cosine_similarity(values[1:],values[0].unsqueeze(0),dim=-1).mean()),
        'mean_condition_norm':float(values.norm(dim=-1).mean()),
        'adapter_training':model.clipTransEncoder.training,
        'dropout_training_flags':[m.training for m in model.clipTransEncoder.modules() if isinstance(m,torch.nn.Dropout)]}

report={'adapter_state_tensors':len(initial),'all_adapter_tensors_unchanged_after_2000_steps':all(equal.values()),
    'changed_keys':[k for k,v in equal.items() if not v],
    'adapter_parameter_count':sum(p.numel() for m in [model.clipTransEncoder,model.clip_ln] for p in m.parameters()),
    'adapter_trainable_parameter_count':sum(p.numel() for m in [model.clipTransEncoder,model.clip_ln] for p in m.parameters() if p.requires_grad),
    'layernorm_initial_weight_all_ones':bool(torch.all(initial['model.clip_ln.weight']==1)),
    'layernorm_initial_bias_all_zeros':bool(torch.all(initial['model.clip_ln.bias']==0)),
    'prompts':prompts,'train':probe('train'),'eval':probe('eval'),
    'train_with_adapter_eval_diagnostic_only':probe('train',True),
    'notes':['CPU diagnostic; no training or checkpoint mutation',
        'Frozen parameters do not automatically disable dropout',
        'Output variability does not establish contribution to FID']}
(ROOT/'artifacts/text_adapter_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2),flush=True)

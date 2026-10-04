"""Time-bounded full-data original FreeMotion single-stage training on 8GB."""
import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import traceback
import numpy as np
import torch
from small_scale_trial import ROOT, REPO, seed, get_config, SingleHumanDataset, InterGenSpatialControlNet, CosineWarmupScheduler, builtin_metadata, evaluate


def atomic_json(path,value):
    tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(tmp,path)


def rng_state():
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':[n[0],torch.tensor(n[1].astype(np.int64)),n[2],n[3],n[4]],
        'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}


def restore_rng(r):
    random.setstate(r['python'])
    n=r['numpy']; np.random.set_state((n[0],n[1].numpy().astype(np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(r['torch']); torch.cuda.set_rng_state_all(r['cuda'])


def main(args):
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    free,_=torch.cuda.mem_get_info()
    if free<5.5*1024**3: raise RuntimeError('Need 5.5 GiB free GPU memory')
    if not args.resume and (out/'latest.ckpt').exists(): raise RuntimeError('Existing run; use --resume')
    seed(args.seed)
    cfg=get_config(str(ROOT/'configs/datasets_single_local.yaml')).interhuman
    dataset=SingleHumanDataset(cfg)
    dataset.data_list.sort(key=lambda x:(int(x['name'].removesuffix('_swap')),bool(x['swap'])))
    n=len(dataset)
    ids=sorted({x['name'].removesuffix('_swap') for x in dataset.data_list},key=int)
    expected=json.loads((ROOT/'artifacts/local_data_policy.json').read_text(encoding='utf-8'))['eligible']['train']
    assert len(ids)==expected and n==2*expected
    assert dataset.max_gt_length==300
    names=[x['name'] for x in dataset.data_list]
    signature=hashlib.sha256(json.dumps(names).encode()).hexdigest()
    protocol={'stage':'single','original_frozen_text_adapter':True,'train_ids':len(ids),
        'augmented_epoch_samples':n,'sample_order_sha256':signature,'frames_cap':300,'microbatch':1,
        'accumulation':args.accumulation,'precision':'FP32','lr':1e-4,'weight_decay':2e-5,
        'gradient_clip':0.5,'warmup_epochs':10,'cosine_epochs':2500,'seed':args.seed,
        'source':str(ROOT/'artifacts/random_init/single_random_init.ckpt')}
    model=InterGenSpatialControlNet(get_config('configs/model_single.yaml'))
    saved=torch.load(out/'latest.ckpt' if args.resume else protocol['source'],map_location='cpu',weights_only=True)
    model.load_state_dict({k.removeprefix('model.'):v for k,v in saved['state_dict'].items()},strict=True)
    model.cuda().train()
    assert not any(p.requires_grad for p in model.clipTransEncoder.parameters())
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=2e-5)
    scheduler=CosineWarmupScheduler(optimizer,warmup=10,max_iters=2500)
    state={'updates':0,'microsteps':0,'epoch':0,'cursor':0,'order':[],'elapsed_seconds':0.0}
    if args.resume:
        assert saved['protocol']==protocol,'Resume protocol differs'
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        scheduler.load_state_dict(saved['scheduler_state_dict'])
        state=saved['training_state']; restore_rng(saved['rng_state'])
    else:
        seed(args.seed)
    del saved
    gc.collect(); torch.cuda.empty_cache()
    atomic_json(out/'protocol.json',protocol)
    atomic_json(out/'train_ids.json',ids)
    validation_data=np.load(ROOT/'artifacts/small_scale_8gb/subset.npz')
    start=time.monotonic(); previous=state['elapsed_seconds']; last_save=start
    torch.cuda.reset_peak_memory_stats()
    if os.name=='nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)

    def status(label):
        elapsed=previous+time.monotonic()-start
        value={'status':label,'pid':os.getpid(),'updates':state['updates'],'microsteps':state['microsteps'],
            'completed_epochs':state['epoch'],'epoch_fraction':state['cursor']/n,
            'elapsed_seconds':elapsed,'budget_hours':args.hours,'lr':optimizer.param_groups[0]['lr'],
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
            'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,
            'last_checkpoint':str(out/'latest.ckpt'),'stop_file':str(out/'STOP')}
        atomic_json(out/'status.json',builtin_metadata(value))
        return value

    def save(label):
        nonlocal last_save
        for p in model.parameters():
            if not torch.isfinite(p).all(): raise FloatingPointError('Nonfinite model parameter')
        state['elapsed_seconds']=previous+time.monotonic()-start
        payload={'state_dict':{'model.'+k:v.detach().cpu() for k,v in model.state_dict().items()},
            'optimizer_state_dict':builtin_metadata(optimizer.state_dict()),
            'scheduler_state_dict':builtin_metadata(scheduler.state_dict()),
            'training_state':dict(state),'rng_state':rng_state(),'protocol':protocol,
            'metadata':{'stage':'single','steps':state['updates'],'status':'full_data_time_bounded_training'}}
        tmp=out/'latest.ckpt.tmp'
        torch.save(payload,tmp)
        os.replace(tmp,out/'latest.ckpt')
        del payload
        last_save=time.monotonic()
        status(label)
        print(json.dumps({'checkpoint_saved':state['updates'],'label':label}),flush=True)

    def validate():
        saved_rng=rng_state()
        try:
            result=evaluate(model,validation_data,'single')
        finally:
            model.train(); restore_rng(saved_rng)
        value=result['val']['mean_normalized_mse']
        improved=value<state.get('best_validation_mse',float('inf'))
        if improved: state['best_validation_mse']=value
        record={'updates':state['updates'],'completed_epochs':state['epoch'],
            'fixed_four_val_denoising_mse':value,'best_so_far':state['best_validation_mse'],
            'note':'120-frame fixed diagnostic; not FID or full validation benchmark'}
        with (out/'validation.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        return improved

    status('running')
    try:
        with (out/'progress.jsonl').open('a',encoding='utf-8') as log:
            while True:
                if previous+time.monotonic()-start>=args.hours*3600:
                    reason='time_budget_complete'; break
                if args.max_updates and state['updates']>=args.max_updates:
                    reason='bounded_preflight_complete'; break
                if (out/'STOP').exists(): reason='stop_file_requested'; break
                if not state['order']:
                    state['order']=torch.randperm(n).tolist(); state['cursor']=0
                count=min(args.accumulation,n-state['cursor'])
                optimizer.zero_grad(set_to_none=True)
                total_loss=0.0
                for _ in range(count):
                    index=state['order'][state['cursor']]
                    name,text,_,motion,_,length,_=dataset[index]
                    batch={'text':[text],'text_multi_person':None,
                        'motions':torch.from_numpy(np.asarray(motion,dtype=np.float32)).unsqueeze(0).cuda(),
                        'motion_lens':torch.tensor([length],device='cuda'),'person_num':1}
                    loss,terms=model.compute_loss(batch)
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                    (loss/count).backward()
                    total_loss+=float(loss.detach())
                    state['microsteps']+=1; state['cursor']+=1
                norm=torch.nn.utils.clip_grad_norm_(params,0.5,error_if_nonfinite=True)
                optimizer.step(); state['updates']+=1
                lr_used=float(optimizer.param_groups[0]['lr'])
                finished_epoch=state['cursor']==n
                if finished_epoch:
                    state['epoch']+=1; state['cursor']=0; state['order']=[]; scheduler.step()
                row={'update':state['updates'],'microsteps':state['microsteps'],'completed_epochs':state['epoch'],
                    'epoch_fraction':state['cursor']/n,'effective_batch':count,'loss':total_loss/count,
                    'grad_norm':float(norm),'lr_used':lr_used,
                    'elapsed_seconds':previous+time.monotonic()-start,
                    'allocated_mib':torch.cuda.memory_allocated()/1024**2,'reserved_mib':torch.cuda.memory_reserved()/1024**2}
                log.write(json.dumps(row)+'\n'); log.flush()
                if state['updates']%10==0 or state['updates']<=3:
                    print(json.dumps(row),flush=True); status('running')
                del loss,terms,batch
                improved=validate() if finished_epoch else False
                if finished_epoch or time.monotonic()-last_save>=args.checkpoint_minutes*60:
                    save('running')
                    if improved:
                        shutil.copyfile(out/'latest.ckpt',out/'best_validation.ckpt.tmp')
                        os.replace(out/'best_validation.ckpt.tmp',out/'best_validation.ckpt')
        optimizer.zero_grad(set_to_none=True)
        if reason!='bounded_preflight_complete':
            improved=validate()
        else: improved=False
        save(reason)
        if improved:
            shutil.copyfile(out/'latest.ckpt',out/'best_validation.ckpt.tmp')
            os.replace(out/'best_validation.ckpt.tmp',out/'best_validation.ckpt')
    except Exception:
        status('failed_last_checkpoint_preserved')
        raise
    finally:
        if os.name=='nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/overnight_full_single')
    parser.add_argument('--hours',type=float,default=12)
    parser.add_argument('--accumulation',type=int,default=16)
    parser.add_argument('--checkpoint-minutes',type=float,default=15)
    parser.add_argument('--seed',type=int,default=20261003)
    parser.add_argument('--max-updates',type=int)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.hours<=0 or args.accumulation<1 or args.checkpoint_minutes<=0: parser.error('Positive budgets required')
    args.output_dir=args.output_dir.resolve()
    main(args)

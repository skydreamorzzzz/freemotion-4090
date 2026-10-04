"""96-ID pilot of the paper Table 2 independent-person protocol."""
import argparse
import gc
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from small_scale_trial import ROOT, REPO, seed, get_config, InterGenSpatialControlNet, MotionNormalizer, load_torch, strip_prefix
from datasets.interhuman import InterHumanPipelineInferDataset
from datasets.evaluator import EvaluatorModelWrapper
from utils.metrics import calculate_activation_statistics, calculate_frechet_distance, euclidean_distance_matrix, calculate_top_k

OUT=ROOT/'artifacts/single_2000_metrics96'
SEED=20261004


@torch.no_grad()
def main(checkpoint, split, requested_samples=96):
    OUT.mkdir(exist_ok=True)
    if (OUT/'report.json').exists():
        raise SystemExit('Completed report exists; preserve it')
    with checkpoint.open('rb') as stream:
        digest=hashlib.sha256()
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
        checkpoint_hash=digest.hexdigest()
    run_config={'checkpoint_sha256':checkpoint_hash,'split':split,'seed':SEED,'requested_samples':requested_samples}
    config_path=OUT/'run_config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text())==run_config, 'Cached run differs'
    elif (OUT/'motions.npz').exists():
        raise RuntimeError('Unverified cached motions; choose a fresh output directory')
    config_path.write_text(json.dumps(run_config,indent=2),encoding='utf-8')
    torch.set_num_threads(4)
    seed(SEED)
    cfg=get_config('configs/datasets_inter.yaml').interhuman_test.clone()
    cfg.defrost(); cfg.CACHE=False; cfg.MODE=split; cfg.freeze()
    dataset=InterHumanPipelineInferDataset(cfg)
    audit=json.loads((ROOT/'artifacts/data_audit.json').read_text(encoding='utf-8'))
    excluded=set(audit['upstream_short_ids']) | set(audit['splits'][split]['missing_ids'])
    eligible=[i for i,x in enumerate(dataset.data_list) if x['name'] not in excluded and not x['swap']]
    sample_count=len(eligible) if requested_samples==0 else requested_samples
    if sample_count>len(eligible) or sample_count<96: raise ValueError('Need 96 to all eligible samples')
    indices=random.Random(SEED).sample(eligible,sample_count)
    items=[dataset[i] for i in indices]
    training=json.loads((ROOT/'artifacts/small_scale_8gb/subset.json').read_text(encoding='utf-8'))
    assert not {x[0] for x in items} & set(training['train_ids']+(training['val_ids'] if split=='test' else []))
    manifest=[{'id':x[0],'text':x[1],'person_texts':[x[2],x[3]],'length':int(x[6])} for x in items]
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    if not (OUT/'motions.npz').exists():
        free,_=torch.cuda.mem_get_info()
        if free<5.5*1024**3: raise RuntimeError('Need 5.5 GiB free GPU memory')
        model=InterGenSpatialControlNet(get_config('configs/model_single.yaml'))
        ckpt=load_torch(checkpoint,map_location='cpu')
        model.load_state_dict({strip_prefix(k,'model.'):v for k,v in ckpt['state_dict'].items()},strict=True)
        del ckpt
        model.cuda().eval()
        normalizer=MotionNormalizer()
        generated=[]
        start=time.time()
        for i,item in enumerate(items):
            name,text,text1,text2,a,b,length,_=item
            people=[]
            for person,prompt in enumerate([text1,text2]):
                seed(SEED+1000+2*i+person)
                output=model.forward_test({'text':[prompt],'text_multi_person':None,
                    'motion_lens':torch.tensor([length],device='cuda'),'person_num':1,
                    'motion_guidance':None})['output']
                value=normalizer.backward(output.cpu().numpy()[0])
                assert np.isfinite(value).all()
                people.append(np.pad(value,((0,dataset.max_gt_length-length),(0,0))))
            generated.append(np.stack(people,axis=1))
            if (i+1)%8==0: print(json.dumps({'generated':i+1,'seconds':time.time()-start}),flush=True)
        np.savez(OUT/'motions.npz',generated=np.stack(generated),
            real=np.stack([np.stack([x[4],x[5]],axis=1) for x in items]))
        del model,output
        gc.collect(); torch.cuda.empty_cache()
    arrays=np.load(OUT/'motions.npz')
    evaluator=EvaluatorModelWrapper(get_config('configs/eval_model.yaml'),'cuda')
    embeddings={}
    for kind in ['real','generated']:
        texts,motions=[],[]
        # Batch 1 encoding preserves input order. Retrieval below still has 96 candidates.
        for i,item in enumerate(items):
            m=torch.from_numpy(arrays[kind][i:i+1]).float()
            batch=([item[0]],[item[1]],[item[2]],[item[3]],m[:,:,0],m[:,:,1],
                torch.tensor([item[6]]),torch.zeros(1,0))
            t,e=evaluator.get_co_embeddings(batch)
            texts.append(t.cpu().numpy()); motions.append(e.cpu().numpy())
        embeddings[kind]=np.concatenate(motions)
        embeddings[kind+'_text']=np.concatenate(texts)
    assert np.allclose(embeddings['real_text'],embeddings['generated_text'])
    np.savez(OUT/'embeddings.npz',**embeddings)
    # Official drop_last=True: metrics use complete retrieval groups of 96.
    metric_count=(sample_count//96)*96
    mu,cov=calculate_activation_statistics(embeddings['real'][:metric_count])
    metrics={}
    for kind in ['real','generated']:
        motion=embeddings[kind][:metric_count]; text=embeddings[kind+'_text'][:metric_count]
        assert np.isfinite(motion).all() and np.isfinite(text).all()
        hits=np.zeros(3); distance_sum=0.0
        for begin in range(0,metric_count,96):
            dist=euclidean_distance_matrix(text[begin:begin+96],motion[begin:begin+96])
            hits+=calculate_top_k(np.argsort(dist,axis=1),3).sum(axis=0)
            distance_sum+=float(np.trace(dist))
        top=hits/metric_count
        m,c=calculate_activation_statistics(motion)
        metrics[kind]={'r_precision':top.tolist(),'mm_distance':distance_sum/metric_count,
            'fid':float(calculate_frechet_distance(mu,cov,m,c))}
    report={'status':'pilot_not_full_paper_benchmark','checkpoint':str(checkpoint),
        'checkpoint_sha256':checkpoint_hash,
        'seed':SEED,'split':split,'samples':sample_count,'metric_samples':metric_count,
        'drop_last_samples':sample_count-metric_count,'retrieval_candidates':96,'replications':1,
        'max_gt_length':dataset.max_gt_length,'length_range':[min(x[6] for x in items),max(x[6] for x in items)],
        'protocol':'Table 2: independently generate both people using personal text only; score concatenated motions with scene text',
        'metrics':metrics,'paper_table2':{'fid':12.975,'r_precision':[.264,.394,.473],'mm_distance':3.885},
        'paper_source':'https://arxiv.org/html/2405.15763v1#S5.T2',
        'limitations':[f'{sample_count} {split} samples generated, {metric_count} evaluated; one replication, no confidence interval',
            'Finite-sample FID; covariance rank deficient when metric_samples <= 512',
            'Training protocol depends on checkpoint; inspect its metadata/protocol',
            'Diversity and Multimodality not evaluated',
            'Real FID uses same reference set and is zero by construction, unlike paper independent draws']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'artifacts/single_2000_8gb/single_trial.ckpt')
    parser.add_argument('--split',choices=['test','val'],default='test')
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--samples',type=int,default=96,help='0 uses all eligible IDs; metrics keep complete groups of 96')
    parser.add_argument('--seed',type=int,default=SEED)
    args=parser.parse_args()
    OUT=args.output_dir.resolve()
    SEED=args.seed
    main(args.checkpoint.resolve(),args.split,args.samples)

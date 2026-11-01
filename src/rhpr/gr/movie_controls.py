"""Names-in-text multimodal TIGER and capacity-matched item-ID SASRec controls."""
import gc,json,shutil
from pathlib import Path
import numpy as np
from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _write


def validate_controls(c):
    from rhpr.gr.movie_multimodal import MODALITIES
    pairs={(t['predictor'],t['seed']) for t in c['trials']}
    expected={(p,s) for p in ['tiger_names','sasrec_id'] for s in [42,101,202]}
    q=c['quantizer'];t=c['training']
    if (len(c['trials'])!=6 or pairs!=expected or c['replicate_seeds']!=[42,101,202]
            or c['data']['history_items']!=20 or c['data']['split']!='timestamp_group_leave_two'
            or t['max_epochs']!=20 or t['min_epochs']!=20 or t['validation_interval']!=5 or t['validate_first_epoch']
            or c['compare_initialization_by_shape'] is not True
            or c['fields']!=['title_with_year','genres','explicit_year','runtime','tmdb_keywords','cast','directors']
            or q['max_steps']!=3000 or q['num_hierarchies']!=3 or q['codebook_size']!=256
            or q['ae_warmup']!={'max_steps':5000,'probe_interval':500,'stopping':{'kind':'water_filling_plateau','min_steps':2000,'patience_probes':4}}
            or c['capacity']['target_active_nonembedding_parameters']!=4590464
            or c['capacity']['maximum_relative_difference']!=.01
            or c['sasrec']!={'d_model':256,'n_heads':4,'n_layers':7,'ffn_multiplier':3,'dropout':.1}
            or c['benchmark']!={'batch_sizes':[1,32],'warmup':10,'iterations':50}):
        raise ValueError('movie controls differ from declared matched protocol')
    for trial in c['trials']:
        expected_modalities=list(MODALITIES[:3]) if trial['predictor']=='tiger_names' else []
        if trial['modalities']!=expected_modalities or trial['stage']!=(3 if trial['predictor']=='tiger_names' else 0):
            raise ValueError('graph people must be disabled in both controls')
    for pins in [q.get('expected_initial_state_sha256_by_seed',{}),c['expected_initial_state_sha256_by_seed'],c.get('expected_sasrec_initial_state_sha256_by_seed',{})]:
        if set(pins)!={'42','101','202'} or any(not isinstance(h,str) or len(h)!=64 or any(x not in '0123456789abcdef' for x in h) for h in pins.values()):
            raise ValueError('each control initialization must be pinned for all three seeds')


def prepare_names_modalities(config,base,source_ids,output):
    """Reuse exact previously encoded cast/director text, preserving other vectors."""
    from rhpr.gr.movie_multimodal import MODALITIES
    spec=config['names_input'];parent=None
    for candidate in [spec['path'],spec['legacy_path']]:
        if Path(candidate).is_dir():parent=Path(candidate);break
    if parent is None:raise FileNotFoundError('completed full-text notebook output is not mounted')
    if file_sha256(parent/'run-manifest.json')!=spec['parent_manifest_sha256'] or json.loads((parent/'run-manifest.json').read_text())['git_sha']!=spec['parent_git_sha']:
        raise ValueError('full-text parent notebook changed')
    trial=parent/spec['trial_directory'];embedding=trial/'item-embeddings.npy'
    if file_sha256(embedding)!=spec['embedding_sha256'] or file_sha256(trial/'representation.json')!=spec['representation_sha256']:
        raise ValueError('names-in-text vectors or encoder record changed')
    if json.loads((trial/'sid-catalogue.json').read_text())['original_item_ids']!=source_ids:
        raise ValueError('names-in-text catalogue order differs')
    representation=json.loads((trial/'representation.json').read_text())
    if representation['fields']!=config['fields'] or representation['truncated_items']!=0 or representation['model']!=config['representation']:
        raise ValueError('names-in-text fields/encoder/truncation differ')
    base=Path(base);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with np.load(base/'features.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    text=np.load(embedding,allow_pickle=False)
    if text.shape!=arrays['text'].shape or len(text)!=len(source_ids) or not np.isfinite(text).all():raise ValueError('invalid names-in-text vectors')
    arrays['text']=text;arrays['text_observed']=np.ones(len(text),dtype=bool)
    np.savez_compressed(output/'features.npz',**arrays);shutil.copy2(base/'source-ids.json',output/'source-ids.json')
    provenance=json.loads((base/'provenance.json').read_text())
    provenance['text_replacement']={'source':spec,'representation':representation,'active_modalities':list(MODALITIES[:3]),
        'graph_people_used':False,'base_features_sha256':file_sha256(base/'features.npz'),'output_features_sha256':file_sha256(output/'features.npz')}
    _write(output/'provenance.json',provenance)
    return output


def capacity_preflight(config,source,users,output):
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.experiment import _index_model_batch,_loss
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.movie_sasrec import ItemIDIndex,build_movie_sasrec,parameter_counts
    from rhpr.sasrec.benchmark import dense_encoder_flops
    n=config['data']['items'];width=config['data']['history_items']
    index=SidIndex(np.column_stack([np.arange(n)//256,np.arange(n)%256,np.zeros(n,dtype=int)]),width=256)
    contexts=(np.arange(width)%n)[None,:];uid=np.array([1]);target=np.array([0]);reports={}
    for name,model,idx in [('tiger',build_grid_generator(index.sids,config['generator'],42,'cpu',source),index),('sasrec',build_movie_sasrec(config,42,'cpu'),ItemIDIndex(n))]:
        model.train();batch=_index_model_batch(contexts,uid,target,idx,'cpu');_loss(model,batch).backward()
        report=parameter_counts(model);model.zero_grad(set_to_none=True)
        # Math SDPA exposes attention matmuls to the CPU FLOP counter.
        with sdpa_kernel(SDPBackend.MATH),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as profiler:
            _loss(model,batch)
        report['supported_forward_flops']=sum(e.flops for e in profiler.key_averages())
        report['flop_scope']='B1,20valid history items; training forward including token/item output heads; profiler-supported ops, FMA=2; math SDPA, no backward/beam/AE/encoding'
        if name=='sasrec':report['analytic_forward_matmul_flops']=dense_encoder_flops(batch_size=1,sequence_length=width,**{k:config['sasrec'][k] for k in ['d_model','n_layers','ffn_multiplier']})+2*config['sasrec']['d_model']*n
        reports[name]=report;del model;gc.collect()
    target=reports['tiger']['active_nonembedding_parameters'];actual=reports['sasrec']['active_nonembedding_parameters'];delta=abs(actual/target-1)
    if target!=config['capacity']['target_active_nonembedding_parameters'] or delta>config['capacity']['maximum_relative_difference']:
        raise ValueError('active nonembedding predictor capacity exceeds declared matching tolerance')
    reports.update(relative_parameter_difference=delta,definition='Unique trainable parameters reached by loss backward; excludes every nn.Embedding on both sides, including position/user/relative-bias tables; tied weights deduplicated',
        fairness='Parameter capacity and examples/epochs are matched; tokenized encoder-decoder and item-only encoder have different FLOPs and serving algorithms. No equal wall-clock/FLOP claim.')
    _write(Path(output)/'capacity-preflight.json',reports)
    return reports


def run_control_trial(trial,config,plan,users,source,device):
    import torch
    if trial['predictor']=='sasrec_id':
        from rhpr.gr.movie_sasrec import run_sasrec_trial
        return run_sasrec_trial(trial,config,plan,users,device)
    from rhpr.gr.movie_multimodal import run_multimodal_trial
    from rhpr.gr.movie_sasrec import benchmark_calls,benchmark_examples
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.experiment import _index_model_batch
    from rhpr.gr.sid import SidIndex
    result=run_multimodal_trial(trial,config,plan,users,source,device)
    result.update(predictor='tiger_names',initialization_group=f"tiger_seed{trial['seed']}")
    root=Path(plan['output'])/'trials'/trial['id'];_write(root/'trial-result.json',result)
    index=SidIndex(np.load(root/'semantic-codes.npy'),width=config['quantizer']['codebook_size'])
    model=build_grid_generator(index.sids,config['generator'],trial['seed'],device,source)
    state=torch.load(root/'generator-best.pt',map_location=device,weights_only=False);model.load_state_dict(state['model']);model.eval();del state
    examples=benchmark_examples(users,config['data']['history_items'])
    def factory(size):
        ix=np.arange(size)%len(examples[2]);batch=_index_model_batch(*(v[ix] for v in examples),index,device)
        return lambda:model.generate(input_ids=batch['input_ids'],attention_mask=batch['attention_mask_encoder'],user_id=batch['user_id'])
    _write(root/'inference-benchmark.json',benchmark_calls(factory,examples,device,config['benchmark']))
    del model;gc.collect()
    return result

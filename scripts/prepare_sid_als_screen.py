"""CPU-only ALS/priority candidates, exported as a byte-pinned private input."""
import argparse,json,shutil,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from rhpr.gr.data import load_beauty,file_sha256
from rhpr.gr.index_proxies import IndexProxies
from rhpr.gr.sid import SidIndex
from rhpr.gr.sid_graph_swap import NeighborGraph,cosine_neighbors,heldout_gain
from rhpr.gr.sid_graph_search import fit_als_signal,budgeted_search,owner_after_moves,randomize_changed_paths


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def event(**kw):print(json.dumps(kw),flush=True)
def pin(p,root):return dict(path=p.relative_to(root).as_posix(),sha256=file_sha256(p),size_bytes=p.stat().st_size)


def run(config):
    begin=time.monotonic();out=Path(config['output']);out.mkdir(parents=True,exist_ok=False)
    assets=out/'input';assets.mkdir();write(out/'config.json',config)
    for record in config['sources'].values():
        if file_sha256(Path(record['path']))!=record['sha256']:raise ValueError('source input hash mismatch')
    _,users,source_ids,counts=load_beauty(Path(config['data']))
    n=len(source_ids);embedding=np.load(config['sources']['embeddings']['path'])
    with np.load(config['sources']['user_split']['path']) as f:fit_rows=f['fit_rows'];held_rows=f['heldout_rows']
    fit_users=[users[i] for i in fit_rows];held_users=[users[i] for i in held_rows]
    factors,user_factors,frequency,als_report=fit_als_signal(fit_users,n,config['als'])
    np.save(out/'als-user-factors.npy',user_factors);np.save(assets/'als-item-factors.npy',factors)
    np.save(assets/'cf-fit-user-frequency.npy',frequency);write(assets/'als.json',als_report)
    masked=factors.copy();masked[frequency<config['minimum_cf_users']]=0
    cf=cosine_neighbors(masked,config['neighbors'])
    np.savez_compressed(assets/'als-graph.npz',source=cf.source,target=cf.target,weight=cf.weight)
    def graph(key):
        with np.load(config['sources'][key]['path']) as f:return NeighborGraph(f['source'],f['target'],f['weight'],n).top(config['neighbors'])
    semantic=graph('semantic_graph');held=graph('heldout_graph')
    np.savez_compressed(assets/'semantic-graph.npz',source=semantic.source,target=semantic.target,weight=semantic.weight)
    shutil.copyfile(config['sources']['user_split']['path'],assets/'user-split.npz')
    with np.load(config['sources']['fields']['path']) as f:price,observed=f['price'][:,0],f['price_observed']
    all_proxy=IndexProxies(users,price,observed);fit_proxy=IndexProxies(fit_users,price,observed);held_proxy=IndexProxies(held_users,price,observed)
    write(assets/'catalogue.json',dict(original_item_ids=source_ids))
    event(phase='als_graph_ready',als_seconds=als_report['seconds'],edges=len(cf.source))
    indices={};curves=[];families={}
    def probe(family,variant,initial,codes,accepted,proposals):
        row=dict(family=family,variant=variant,accepted=accepted,proposals=proposals,
            changed_items=int(np.any(codes!=initial,axis=1).sum()),
            cf_score=cf.score(codes),semantic_score=semantic.score(codes),heldout_cf_score=held.score(codes),
            **all_proxy.first(codes[:,0]),**heldout_gain(codes[:,0],fit_proxy,held_proxy))
        curves.append(row);return row
    for family in ['t5_rqvae','t5_rqkmeans']:
        original_path=Path(config['sources'][family]['path']);initial=np.load(original_path);SidIndex.from_full_sids(initial)
        original_name=family+'-original.npy';shutil.copyfile(original_path,assets/original_name)
        results={}
        for variant in ['uniform','priority']:
            codes,owner,moves,snapshots,report=budgeted_search(initial,cf,semantic,embedding,frequency,config['search'],variant)
            results[variant]=dict(moves=moves,report=report)
            directory=out/family/variant;directory.mkdir(parents=True)
            np.save(directory/'accepted-swaps.npy',moves);write(directory/'search.json',report)
            for position,accepted,paths,who in snapshots:
                np.savez_compressed(directory/f'accepted-{accepted:05d}.npz',sids=paths,owner=who)
                probe(family,variant,initial,paths,accepted,position)
            event(phase='search_complete',family=family,variant=variant,**report)
        budget=min(config['search']['accepted_cap'],*(int(x['report']['accepted']) for x in results.values()))
        if budget<2:raise RuntimeError('insufficient common edit budget;do not launch GPU screen')
        families[family]=dict(common_accepted_swaps=budget,search_reports={k:v['report'] for k,v in results.items()})
        owners={variant:owner_after_moves(n,r['moves'],budget) for variant,r in results.items()}
        owners['random']=randomize_changed_paths(owners['priority'],config['random_seed'])
        np.testing.assert_array_equal(np.flatnonzero(owners['random']!=np.arange(n)),np.flatnonzero(owners['priority']!=np.arange(n)))
        for variant,owner in owners.items():
            paths=initial[owner];index=SidIndex.from_full_sids(paths)
            for depth in range(1,5):
                a,c=np.unique(initial[:,:depth],axis=0,return_counts=True);b,d=np.unique(paths[:,:depth],axis=0,return_counts=True)
                np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(c,d)
            name=family+'_'+variant;np.save(assets/(name+'-sids.npy'),paths);np.save(assets/(name+'-owner.npy'),owner)
            row=probe(family,variant+'_export',initial,paths,budget,
                int(results[variant]['moves'][budget-1,0]) if variant!='random' else 0)
            indices[name]=dict(family=family,role='random' if variant=='random' else 'swapped',
                variant='als_'+variant if variant!='random' else 'random_changed_set',
                semantic_lambda=config['search']['semantic_lambda'] if variant!='random' else 0.,
                sids=name+'-sids.npy',owner=name+'-owner.npy',original_sids=original_name,
                selected_snapshot=row,common_accepted_budget=budget,
                random_control='same changed item set and available SID paths as priority;not matched by transposition count or semantic distortion'
                    if variant=='random' else None)
            event(phase='index_exported',family=family,variant=variant,budget=budget,changed=row['changed_items'],heldout_gain=row['heldout_gain_bits'])
    write(out/'curves.json',curves);write(assets/'index-curves.json',curves)
    write(assets/'preparation.json',dict(config=config,counts=counts,families=families,seconds=time.monotonic()-begin,
        validation_and_test_used=False,train_user_split='same75/25as original graph probe'))
    files=[pin(p,assets) for p in sorted(assets.iterdir())]
    write(assets/'resources-manifest.json',dict(kind='frozen_sid_graph_placement',catalogue_items=n,
        data_sha256={name:file_sha256(Path(config['data'])/name) for name in ['catalog.json','sequences.json']},
        files=files,indices=indices,quantizer_retrained=False,full_paths_preserved=True))
    write(out/'summary.json',dict(families=families,seconds=time.monotonic()-begin,
        manifest_sha256=file_sha256(assets/'resources-manifest.json'),input_bytes=sum(p.stat().st_size for p in assets.iterdir())))
    event(phase='preparation_complete',seconds=time.monotonic()-begin,manifest_sha256=file_sha256(assets/'resources-manifest.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    with threadpool_limits(limits=2):run(json.loads(Path(a.config).read_text()))

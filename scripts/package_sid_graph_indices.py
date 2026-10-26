"""Create a small PRIVATE Kaggle input from exact completed CPU swap outputs."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from rhpr.gr.data import file_sha256, load_beauty
from rhpr.gr.sid import SidIndex


def package(source, destination):
    source,destination=Path(source),Path(destination)
    c=json.loads((source/'config.json').read_text());rows=json.loads((source/'curves.json').read_text())
    destination.mkdir(parents=True,exist_ok=False)
    _,_,ids,_=load_beauty(Path(c['data']))
    (destination/'catalogue.json').write_text(json.dumps({'original_item_ids':ids})+'\n')
    for name in ['manifest.json','config.json','user-split.npz']:
        shutil.copyfile(source/name,destination/('graph-'+name))
    indices={}
    for index in c['indices']:
        family=index['id'];original=np.load(index['codes'])
        SidIndex.from_full_sids(original)
        orig_name=f'{family}-original-sids.npy';shutil.copyfile(index['codes'],destination/orig_name)
        for role in ['original','swapped']:
            asset=f'{family}_{role}';prefix=destination/asset
            if role=='original':
                sids=original;owner=np.arange(len(original))
                shutil.copyfile(index['codes'],destination/f'{asset}-sids.npy')
                np.save(destination/f'{asset}-owner.npy',owner)
                record={'accepted':0,'proposals':0}
            else:
                d=source/family/'lambda_0p3';sids=np.load(d/'sids.npy');owner=np.load(d/'owner.npy')
                for old,new in [('sids.npy','sids.npy'),('owner.npy','owner.npy'),('accepted-swaps.npy','accepted-swaps.npy')]:
                    shutil.copyfile(d/old,destination/f'{asset}-{new}')
                record=next(r for r in rows if r['index']==family and r['trial']=='lambda_0p3' and r['proposals']==c['search']['proposals'])
            np.testing.assert_array_equal(sids,original[owner]);SidIndex.from_full_sids(sids)
            indices[asset]=dict(family=family,role=role,semantic_lambda=.3 if role=='swapped' else 0,
                sids=f'{asset}-sids.npy',owner=f'{asset}-owner.npy',original_sids=orig_name,
                selected_snapshot=record,original_codes_sha256=file_sha256(Path(index['codes'])))
    files=[dict(path=p.name,sha256=file_sha256(p),size_bytes=p.stat().st_size) for p in sorted(destination.iterdir())]
    data_hashes={name:file_sha256(Path(c['data'])/name) for name in ['catalog.json','sequences.json']}
    manifest=dict(kind='frozen_sid_graph_placement',catalogue_items=len(ids),data_sha256=data_hashes,
        files=files,indices=indices,graph_optimization='fixed CPU candidates,75% TRAIN users;lambda0.3;k20;20kproposals',
        full_paths_preserved=True,quantizer_retrained=False,source_manifest_sha256=file_sha256(source/'manifest.json'))
    (destination/'resources-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(path=str(destination),manifest_sha256=file_sha256(destination/'resources-manifest.json'),
        files=len(files),size_bytes=sum(p.stat().st_size for p in destination.iterdir()))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    package(a.source,a.output)

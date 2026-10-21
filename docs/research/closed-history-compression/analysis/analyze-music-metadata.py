# BEGIN ARCHIVE PATH ADAPTER: no research logic below this block is changed.
import os as _archive_os
from pathlib import Path as _ArchivePath

_archive_repo_root = _ArchivePath(__file__).resolve().parents[4]
_archive_workspace_root = _ArchivePath(
    _archive_os.environ.get("RHPR_RESEARCH_WORKSPACE", "/workspace")
).expanduser().resolve()
if (_archive_workspace_root == _archive_repo_root
        or _archive_repo_root in _archive_workspace_root.parents):
    raise ValueError("RHPR_RESEARCH_WORKSPACE must be outside the repository checkout")


def _archive_workspace_path(relative):
    return str(_archive_workspace_root / relative)


def _archive_repo_src_path():
    return str(_archive_repo_root / "src")
# END ARCHIVE PATH ADAPTER

from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import json,time,sys,gzip
from collections import Counter,defaultdict
sys.path.insert(0,_archive_repo_src_path())
from rhpr.sasrec.long_history_data import load_prepared_cohort
from rhpr.sasrec.daily_evaluation_data import load_prepared_daily_evaluation,iter_daily_evaluation_examples,daily_context_sha256

started=time.monotonic();root=Path(_archive_workspace_path('music-metadata'))
cohort=load_prepared_cohort(_archive_workspace_path('plateau-cohort'))
evaluation=load_prepared_daily_evaluation(_archive_workspace_path('baseline-evaluation'),cohort)
catalog=cohort.catalog_item_ids
print(json.dumps({'phase':'cohort_loaded','seconds':time.monotonic()-started}),flush=True)

def entity_csr(name):
    t=pq.read_table(root/f'{name}_item_mapping.parquet')
    item=t['item_id'].combine_chunks().to_numpy();entity=t[f'{name}_id'].combine_chunks().to_numpy()
    mapped=np.searchsorted(catalog,item)
    known=(mapped<len(catalog));known[known] &= catalog[mapped[known]]==item[known]
    mapped=mapped[known].astype(np.int32)+1;entity=entity[known].astype(np.uint32)
    order=np.lexsort((entity,mapped));mapped=mapped[order];entity=entity[order]
    unique=np.r_[True,(mapped[1:]!=mapped[:-1]) | (entity[1:]!=entity[:-1])]
    mapped=mapped[unique];entity=entity[unique]
    lengths=np.bincount(mapped,minlength=cohort.num_items+1)
    offsets=np.r_[0,np.cumsum(lengths)]
    result={'source_rows':len(t),'unique_catalogue_pairs':len(mapped),'covered_catalogue_items':int(np.count_nonzero(lengths)),
            'catalogue_items':cohort.num_items,'catalogue_coverage':float(np.count_nonzero(lengths)/cohort.num_items),
            'catalogue_items_with_multiple_entities':int(np.count_nonzero(lengths>1)),
            'max_entities_per_catalogue_item':int(lengths.max()),'distinct_entities_in_catalogue':int(len(np.unique(entity))),
            'dropped_duplicate_pairs':int(np.count_nonzero(known)-len(mapped))}
    print(json.dumps({'phase':'entity_indexed','entity':name,**result,'seconds':time.monotonic()-started}),flush=True)
    return entity,offsets,result

indices={name:entity_csr(name) for name in ['artist','album']}
with gzip.open(_archive_workspace_path('ssm2-state128-final-output/query_ssm2_m32_s128-fixed_lr-p60-validation-input-contexts.json.gz'),'rt') as stream:
    expected={r['anchor_id']:r['actual_context_sha256'] for r in json.load(stream)}

def expand(flat,offsets,ids):
    starts=offsets[ids];lengths=offsets[ids+1]-starts
    if bool((lengths==1).all()):return flat[starts]
    total=int(lengths.sum())
    locations=np.repeat(starts,lengths)+np.arange(total)-np.repeat(np.cumsum(lengths)-lengths,lengths)
    return flat[locations]

counts=Counter();user_counts=defaultdict(Counter);rows=[]
for ex in iter_daily_evaluation_examples(cohort,evaluation,split_name='validation',max_seq_len=1152):
    assert len(ex.context)==1152 and daily_context_sha256(ex.user_id,ex.context,ex.timestamps)==expected[ex.anchor_id]
    assert int(ex.timestamps[-1])<ex.target_timestamp
    r=ex.context[-128:];h=ex.context[:-128];target=int(ex.target)
    flags={'anchors':1,'target_known':int(target>0),'target_oov':int(target==0)}
    track_r=bool(target>0 and np.any(r==target));track_h=bool(target>0 and np.any(h==target))
    flags.update(track_seen_R128=int(track_r),track_seen_only_H1024=int(not track_r and track_h),track_unseen_visible1152=int(target>0 and not track_r and not track_h))
    for name,(flat,offsets,metadata) in indices.items():
        targets=flat[offsets[target]:offsets[target+1]] if target>0 else flat[:0]
        covered=bool(len(targets));flags[f'{name}_target_covered']=int(covered)
        flags[f'{name}_target_single_entity']=int(len(targets)==1)
        flags[f'{name}_target_multiple_entities']=int(len(targets)>1)
        r_entities=expand(flat,offsets,r);h_entities=expand(flat,offsets,h)
        seen_r=bool(covered and np.isin(targets,r_entities).any())
        seen_h=bool(covered and np.isin(targets,h_entities).any())
        flags.update({f'{name}_seen_R128':int(seen_r),f'{name}_seen_only_H1024':int(not seen_r and seen_h),
                      f'single_entity_target_{name}_seen_R128':int(len(targets)==1 and seen_r),
                      f'single_entity_target_track_unseen_R128_but_{name}_seen_R128':int(len(targets)==1 and target>0 and not track_r and seen_r),
                      f'{name}_unseen_visible1152':int(covered and not seen_r and not seen_h),
                      f'track_unseen_R128_but_{name}_seen_R128':int(target>0 and not track_r and seen_r),
                      f'track_unseen_visible1152_but_{name}_seen_visible1152':int(target>0 and not track_r and not track_h and (seen_r or seen_h)),
                      f'{name}_known_context_slots':int(np.count_nonzero((ex.context>0) & (offsets[ex.context+1]>offsets[ex.context]))),
                      f'known_context_slots':int(np.count_nonzero(ex.context>0))})
    counts.update(flags);user_counts[ex.user_id].update(flags)
    rows.append(flags)
assert counts['anchors']==26335 and len(user_counts)==4384
report={'complete':True,'no_gpu':True,'no_repo_edits':True,'split':'validation','source_manifest':json.loads((root/'fetch-manifest.json').read_text()),
        'support':{'anchors':counts['anchors'],'users':len(user_counts),'known_targets':counts['target_known'],'oov_targets':counts['target_oov'],
                   'actual_context_hashes_match_saved_neural_inputs':True,'all_contexts_strictly_before_target':True},
        'mapping_coverage':{name:data[2] for name,data in indices.items()},'counts':dict(counts),
        'anchor_fractions_all_anchors':{key:value/counts['anchors'] for key,value in counts.items() if not key.endswith('_slots')},
        'macro_user_fractions_all_user_anchors':{key:float(np.mean([v[key]/v['anchors'] for v in user_counts.values()])) for key in counts if not key.endswith('_slots')},
        'conditional_fractions_among_covered_targets':{},'conditional_fractions_among_single_entity_targets':{},'seconds':time.monotonic()-started,
        'limitations':['Artist and album IDs are observed metadata identities; they do not label playlist, genre, intent or mood.',
                       'Artist/album overlap uses any associated entity for many-to-many mappings, never an arbitrary first entity. Multiple album links may include compilations/editions; any-shared-album is a relation diagnostic, not proof of intentional album listening.',
                       'OOV observed IDs were already mapped to zero in prepared data; their original entity identities cannot be reconstructed here. Unknown target entities remain unassessed.',
                       'An entity unseen in the visible1152 may have been present earlier in the complete user history; this analysis does not call it globally new.',
                       'User cohort was selected for exceptionally long histories and is not representative of all music listeners.',
                       'All-target fractions retain missing entities and OOV as no observed overlap; covered-target fractions give complementary coverage-aware descriptions.',
                       'Only original validation anchors analysed, no test metrics or neural logits were computed.']}
for name in indices:
    denominator=counts[f'{name}_target_covered']
    report['conditional_fractions_among_covered_targets'][name]={key:counts[key]/denominator for key in counts if name in key and key!=f'{name}_target_covered' and not key.endswith('_slots')}
    report['mapping_coverage'][name]['target_coverage_among_known']=denominator/counts['target_known']
    report['mapping_coverage'][name]['context_slot_coverage_among_known']=counts[f'{name}_known_context_slots']/counts['known_context_slots']
    report['mapping_coverage'][name]['target_fraction_multiple_entities_among_covered']=counts[f'{name}_target_multiple_entities']/denominator
    report['conditional_fractions_among_single_entity_targets'][name]={
        'targets':counts[f'{name}_target_single_entity'],
        'seen_R128':counts[f'single_entity_target_{name}_seen_R128']/counts[f'{name}_target_single_entity'],
        'track_unseen_R128_but_entity_seen_R128':counts[f'single_entity_target_track_unseen_R128_but_{name}_seen_R128']/counts[f'{name}_target_single_entity']}
Path(_archive_workspace_path('music-metadata-audit.json')).write_text(json.dumps(report,indent=2)+'\n')

lines=['# Artist/album return patterns on the existing validation anchors','',
       'Read-only CPU analysis; no training or neural inference. Mappings were downloaded from the exact pinned Yambda revision.',
       '',f"Support: {counts['anchors']:,} anchors, {len(user_counts):,} users, {counts['target_known']:,} known targets and {counts['target_oov']:,} OOV targets.",'',
       '| Pattern | Count | Fraction of all anchors |','|---|---:|---:|']
keys=['track_seen_R128','artist_seen_R128','album_seen_R128','track_unseen_R128_but_artist_seen_R128','track_unseen_R128_but_album_seen_R128',
      'track_unseen_visible1152','track_unseen_visible1152_but_artist_seen_visible1152','track_unseen_visible1152_but_album_seen_visible1152']
for key in keys:lines.append(f"| {key} | {counts[key]:,} | {100*counts[key]/counts['anchors']:.2f}% |")
lines+=['','Metadata coverage:']
for name in indices:
    c=report['mapping_coverage'][name]
    lines.append(f"- {name}: {c['catalogue_coverage']:.4%} catalogue items; {c['target_coverage_among_known']:.4%} known targets; {c['catalogue_items_with_multiple_entities']:,} catalogue items have multiple associated entities.")
    s=report['conditional_fractions_among_single_entity_targets'][name]
    lines.append(f"  Target ambiguity: {100*c['target_fraction_multiple_entities_among_covered']:.2f}% of covered targets have multiple entities. Among {s['targets']:,} single-entity targets, entity overlap with R128 is {100*s['seen_R128']:.2f}%; unseen R128 track but familiar entity is {100*s['track_unseen_R128_but_entity_seen_R128']:.2f}%.")
lines+=['','The actual pinned sequential500m listens Parquet schema, read by just2125 footer bytes, includes `is_organic`, `played_ratio_pct` and `track_length_seconds` in addition to user/item/timestamp. Earlier partial-projection manifests do not describe the full source schema. These fields can therefore be restored from the same source revision. No full listening file was downloaded.','',
        'Interpretation: track-level novelty and artist/album-level familiarity can differ. This motivates separate multi-resolution diagnostics; it does not identify moods or prove a particular recommendation objective.','',
        'Limitations:']+[f'- {s}' for s in report['limitations']]
lines+=['','[Full JSON](/workspace/music-metadata-audit.json), [fetch manifest](/workspace/music-metadata/fetch-manifest.json), [analysis script](/workspace/analyze-music-metadata.py).']
Path(_archive_workspace_path('music-metadata-audit.md')).write_text('\n'.join(lines)+'\n')
print(json.dumps({'complete':True,'seconds':report['seconds'],'counts':dict(counts),'mapping_coverage':report['mapping_coverage']},indent=2))

"""Deterministic earlier singleton holdouts, keeping intervening test context."""
import collections,json,shutil
from pathlib import Path
from rhpr.gr.data import file_sha256
from rhpr.gr.movie_data import load_temporal_users,temporal_examples,case_digest
from rhpr.gr.experiment import _write


def singleton_rows(sequences, embargo):
    events={r['user_id']:[*map(tuple,r['train']),tuple(r['validation']),tuple(r['test'])] for r in sequences}
    for e in embargo:events[e['user_id']].append((e['movie_id'],e['timestamp']))
    rows=[];excluded=[];audit=[]
    for uid,g in sorted(events.items()):
        if len({i for i,t in g})!=len(g):raise ValueError('duplicate user-item in source')
        groups=collections.defaultdict(list)
        for i,t in g:groups[t].append(i)
        singles=sorted(t for t,items in groups.items() if len(items)==1)
        if len(singles)<2:
            excluded.append(dict(user_id=uid,reason='fewer_than_two_singleton_groups',events=len(g)));continue
        vt,tt=singles[-2:];train=sorted(((i,t) for i,t in g if t<vt),key=lambda x:(x[1],x[0]))
        if len({t for i,t in train})<2:
            excluded.append(dict(user_id=uid,reason='fewer_than_two_training_groups',events=len(g)));continue
        between=sorted(((i,t)for i,t in g if vt<t<tt),key=lambda x:(x[1],x[0]))
        rows.append(dict(user_id=uid,train=[list(v)for v in train],validation=[groups[vt][0],vt],test=[groups[tt][0],tt],test_between=[list(v)for v in between]))
        audit.append(dict(user_id=uid,validation_time=vt,test_time=tt,validation_group_size=1,test_group_size=1,train_events=len(train),test_between_events=len(between),discarded_after_test=sum(t>tt for i,t in g)))
    return rows,excluded,audit


def prepare_singleton_input(original,output,expected):
    original=Path(original);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    load=lambda p:json.loads(p.read_text())
    seq,excluded,audit=singleton_rows(load(original/'sequences.json'),load(original/'embargoed-events.json'))
    _write(output/'sequences.json',seq);_write(output/'singleton-audit.json',dict(users=audit,excluded=excluded,source_files={p:file_sha256(original/p)for p in ['sequences.json','embargoed-events.json','split-report.json','catalog-stage-01.json']}))
    shutil.copyfile(original/'catalog-stage-01.json',output/'catalog-stage-01.json')
    ids=[r['item_id']for r in load(output/'catalog-stage-01.json')];users=load_temporal_users(output/'sequences.json',ids)
    report=dict(split='latest_two_singleton_timestamp_groups',items=len(ids),users=len(users),train_events=sum(len(u.train)for u in users),train_examples=len(temporal_examples(users,20,'train')[2]),evaluation_examples=len(users),excluded_users=excluded,
        test_between_events=sum(len(u.test_between)for u in users),discarded_after_test=sum(a['discarded_after_test']for a in audit),case_sha256={s:case_digest(temporal_examples(users,20,s))for s in ['train','validation','test']},
        target_selection='last two groups containing exactly one retained catalogue item; at least two earlier training timestamp groups',test_context='all retained events strictly before test, including full intervening timestamp groups; train strictly before validation',timestamp_semantics='rating submission, not viewing',catalogue_policy='same1273films; metadata available independently of retained interactions')
    for k in ['items','users','train_events','train_examples']:
        if report[k]!=expected[k]:raise ValueError(f'singleton input count differs: {k}')
    if 'case_sha256' in expected and report['case_sha256']!=expected['case_sha256']:
        raise ValueError('singleton train/validation/test cases differ from pinned preflight')
    _write(output/'split-report.json',report)
    return output

#!/usr/bin/env python3
"""Use archived full-catalogue scores to bound local decision repairs, CPU only."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,duckdb
from rhpr.gr.study_ceiling import repair_true_decisions,target_rank


def analyze(root,output,epoch):
 root=Path(root).resolve();output=Path(output);output.mkdir(parents=True,exist_ok=True)
 db=duckdb.connect();db.execute("SET threads=2");db.execute("SET memory_limit='1GB'")
 def view(name):
  path=str(root/'tables'/name/'**/*.parquet').replace("'","''")
  db.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path}',hive_partitioning=false)")
 for name in ['cases','runs','item_codes','target_path']:view(name)
 cases=db.execute("SELECT case_id,target_item,panel_probability FROM cases WHERE split='validation' AND exact_panel ORDER BY case_id").fetchdf()
 runs=db.execute('SELECT run_id FROM runs ORDER BY run_id').fetchdf().run_id.tolist()
 paths=db.execute("""SELECT t.run_id,t.case_id,t.level,t.raw_logprob,t.local_logprob FROM target_path t JOIN cases c USING(case_id)
  WHERE c.split='validation' AND c.exact_panel AND t.epoch=?""",[epoch]).fetchdf()
 choices={'baseline':[],'first':[0],'second':[1],'third':[2],'suffix':[3],'first_two':[0,1],'last_two':[2,3],'all':[0,1,2,3]}
 rows=[];max_parity=0.
 for run in runs:
  sids=np.stack(db.execute('SELECT sid FROM item_codes WHERE run_id=? ORDER BY item_id',[run]).fetchnumpy()['sid'])
  pp=paths[paths.run_id==run].sort_values(['case_id','level']).set_index('case_id')
  for case in cases.itertuples():
   file=root/run/'catalogue-scores'/f"{epoch}-{case.case_id.replace(':','-')}.npz"
   with np.load(file,allow_pickle=False) as archive:
    for scoring in ['raw','local']:
     original=archive[scoring];lp=pp.loc[case.case_id,f'{scoring}_logprob'].to_numpy()
     error=abs(original[case.target_item]-lp.sum());max_parity=max(max_parity,error)
     if error>3e-5:raise ValueError('Full scores disagree with source target path')
     oldrank=target_rank(original,case.target_item)
     for condition,levels in choices.items():
      repaired=repair_true_decisions(original,sids,int(case.target_item),lp,levels)
      rank=target_rank(repaired,case.target_item)
      assert rank<=oldrank
      if condition=='all':assert rank==1
      if scoring=='local' and not np.isclose(np.exp(repaired).sum(),1,rtol=3e-4,atol=2e-6):raise ValueError('Local repaired mass is not one')
      rows.append(dict(run_id=run,case_id=case.case_id,item_id=case.target_item,scoring=scoring,condition=condition,
       panel_probability=case.panel_probability,rank=rank,recall10=float(rank<=10),ndcg10=1/np.log2(rank+1) if rank<=10 else 0.,
       target_log_score=float(repaired[case.target_item])))
  print(json.dumps(dict(run_id=run,processed_cases=len(cases))),flush=True)
 df=pd.DataFrame(rows);df.to_parquet(output/'decision_ceiling_cases.parquet',index=False)
 summaries=[]
 for (scoring,condition),g in df.groupby(['scoring','condition']):
  w=1/g.panel_probability.to_numpy();base=df[(df.scoring==scoring)&(df.condition=='baseline')].set_index(['run_id','case_id'])
  b=base.loc[pd.MultiIndex.from_frame(g[['run_id','case_id']])].reset_index()
  summaries.append(dict(scoring=scoring,condition=condition,ndcg10=float(np.average(g.ndcg10,weights=w)),
     recall10=float(np.average(g.recall10,weights=w)),ndcg10_gain=float(np.average(g.ndcg10.to_numpy()-b.ndcg10.to_numpy(),weights=w))))
 summary=pd.DataFrame(summaries);summary.to_csv(output/'decision_ceiling_summary.csv',index=False)
 (output/'verification.json').write_text(json.dumps(dict(validation_cases=len(cases),runs=len(runs),epoch=epoch,
   max_target_score_parity_error=max_parity,rank_never_worsens=True,all_repaired_target_rank_one=True,
   local_probability_mass_preserved=True,interpretation='target-aware local repair ceilings, not a trained method'),indent=2)+'\n')
 print(summary.to_string(index=False),flush=True)
 return summary


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epoch',type=int,default=20)
 a=p.parse_args();analyze(a.root,a.output,a.epoch)

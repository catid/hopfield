"""Validate saved objective evidence and regenerate published/local comparisons."""
import csv,hashlib,json,math,statistics,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'artifacts/objective_studies';started=time.monotonic()
def read(p):return json.loads((R/p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def csvout(name,rows):
 with (O/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
s=read('artifacts/objectives/summary.json');f=read('artifacts/figure4/summary.json');cur=read('artifacts/figure4/curves.json');d=read('artifacts/d2/summary.json');plot=read('artifacts/figure4/figures/figure4.json');aggregate=read('artifacts/objective_studies/summary.json')
assert aggregate['status']=='complete' and aggregate['all_required_inputs_validated'] and not aggregate['incomplete_schedules']
assert s['scope_complete'] and len(s['jobs'])==81 and len(s['external_jobs'])==3
assert len(s['operation_counts'])==24 and all(c['matmul_class_equals_flopcountermode'] and c['classes']['matmul']==c['flopcountermode_next_update'] for c in s['operation_counts'].values())
assert len(s['rows'])==28
hashed=0
for j in s['jobs']:
 assert j['status']=='complete'
 e=read(j['execution_path']);run=R/'artifacts/objectives'/j['lane'].split(':')[0]/j['run_relative']
 for name,h in e['outputs'].items():assert sha(run/name)==h,(j['id'],name);hashed+=1
 assert j['final']['stop']=='schedule' and max(p['update'] for p in j['final']['curve'])==j['final']['updates']
 assert j['result']['test']['n']==(50000 if j['task']=='sudoku' else 1000)
for j in s['external_jobs']:
 assert j['status']=='complete' and j['timing_log_updates']==3840000
 for name,h in j['files_sha256'].items():assert sha(R/j['collection_directory']/name)==h,(j['id'],name);hashed+=1
 assert j['final']['stop']=='schedule' and max(p['update'] for p in j['final']['curve'])==3840000
alljobs={j['id']:j for j in s['jobs']+s['external_jobs']}
for key,row in s['rows'].items():
 assert row['status']=='complete' and row['completed_seeds']==[0,1,2]
 jobs=sorted((alljobs[i] for i in row['job_ids']),key=lambda x:x['seed'])
 rates=[100*j['result']['test']['solved']/j['result']['test']['n'] for j in jobs]
 mean=sum(rates)/3;sd=math.sqrt(sum((x-mean)**2 for x in rates)/2)
 assert math.isclose(mean,row['solved_percent_mean'],abs_tol=1e-12) and math.isclose(sd,row['solved_percent_sample_sd'],abs_tol=1e-12)
 probes=[{p['update']:p for p in j['final']['curve']} for j in jobs];shared=sorted(set.intersection(*(set(x) for x in probes)))
 crossing=next((u for u in shared if sum(x[u]['solved']/x[u]['n'] for x in probes)/3>=.99),None)
 assert crossing==row['T99_update_common_mean']
 if crossing is not None:
  adjusted=statistics.median(p[crossing]['fit_s']-j['startup_s']+j['result']['ready_s'] for p,j in zip(probes,jobs))
  assert math.isclose(adjusted,row['paper_adjusted_T99_s_median'],abs_tol=1e-12)
  assert row['T99_fullcount_flop']==crossing*row['fullcount_flop_per_update']
 else:assert row['paper_adjusted_T99_s_median'] is None and row.get('T99_fullcount_flop') is None
assert len(f['jobs'])==len(cur['rows'])==100 and all(j['status']=='complete' for j in f['jobs'])
assert sum(j['validation']['n_probes'] for j in f['jobs'])==9740
dense=[]
for j in f['jobs']:
 assert j['output_hashes_match'] and j['validation']['schedule_completed']
 host=j['lane'].split(':')[0];e=read(f'artifacts/figure4/{host}/jobs/{j["id"]}/execution.json');run=R/'artifacts/figure4'/host/j['run_relative']
 for name,h in e['outputs'].items():assert sha(run/name)==h,(j['id'],name);hashed+=1
 if j['dense']:
  proof=e['validation']['clock_match'];assert proof['matched'];dense.append({'job':j['id'],'checkpoints':len(proof['compared_checkpoints'])})
  curve=next(r for r in cur['rows'] if r['id']==j['id']);assert sha(R/curve['clock_from'])==proof['final_log_sha256']
assert len(dense)==12 and sum(x['checkpoints'] for x in dense)==228
figure_rows=[];pubfig={'pl':.048,'mpf':.21,'cd':33.,'bptt1':4.4,'bptt':None}
for m in ['pl','mpf','bptt1','cd','bptt']:
 rows=[r for r in cur['rows'] if r['method']==m];assert sorted(r['seed'] for r in rows)==list(range(20))
 assert all(r['status']=='complete' and r['schedule_completed'] and all(p['n']==512 for p in r['curve']) for r in rows)
 xs=[np.asarray([p['paper_adjusted_fit_s'] for p in r['curve']]) for r in rows];ys=[np.asarray([p['solved_percent'] for p in r['curve']]) for r in rows]
 knots=np.unique(np.concatenate(xs));vals=np.vstack([np.interp(knots,x,y,left=0.,right=y[-1]) for x,y in zip(xs,ys)]);avg=vals.sum(axis=0)/20
 ix=np.where(avg>=99)[0];cross=None
 if len(ix):
  k=ix[0];cross=float(knots[k] if k==0 else knots[k-1]+(99-avg[k-1])*(knots[k]-knots[k-1])/(avg[k]-avg[k-1]))
 expected=plot['methods'][m]['mean_curve_T99_s'];assert (cross==expected if cross is None else math.isclose(cross,expected,abs_tol=1e-12))
 end=float(np.mean([y[-1] for y in ys]));assert math.isclose(end,plot['methods'][m]['final_mean_solved_percent'],abs_tol=1e-12)
 times=[float(x[-1]) for x in xs]
 figure_rows.append({'method':m,'local_n_seeds':20,'development_puzzles':512,'local_T99_s':cross,'published_H200_T99_s':pubfig[m],'local_final_mean_solved_percent':end,'local_final_time_median_s':statistics.median(times),'local_final_time_min_s':min(times),'local_final_time_max_s':max(times),'published_final_mean_solved_percent':27. if m=='bptt' else None,'published_final_time_s':245. if m=='bptt' else None})
assert len(d['jobs'])==9
d2rows=[]
for j in d['jobs']:
 assert j['status']=='complete' and j['schedule_completed'] and j['completed_updates']==60000 and j['probe_n']==512 and j['test_evaluated'] is False
 assert all(p['n']==512 for p in j['probe_curve'])
 best=max(j['probe_curve'],key=lambda p:p['solved']);assert best['solved']*100/512==j['max_probe_solved_percent']
 run=R/'artifacts/objectives'/j['job']['lane'].split(':')[0]/j['job']['run_relative']
 for name,h in j['outputs'].items():assert sha(run/name)==h,(j['job']['id'],name);hashed+=1
 d2rows.append({'lr':j['job']['lr'],'seed':j['job']['seed'],'max_development_solved':best['solved'],'probe_n':512,'max_development_percent':j['max_probe_solved_percent'],'first_max_update':best['update'],'fit_s':j['fit_s'],'updates':60000,'test_evaluated':False})
assert max(x['max_development_solved'] for x in d2rows)==3
# Published values transcribed from reviewed arXiv-v1 Table 7 and its footnote.
# None means no numerical value reported at this exact schedule, or no99% crossing.
pubs={
 'pl':(.050,3.42e10,100.,0.,.043,6.15e7),
 'mpl':(.198,1.95e11,99.99,0.,.022,3.11e7),
 'mpf':(.225,1.95e11,99.99,0.,.030,4.22e7),
 'dsm':(1.6,2.20e12,99.98,.01,.023,3.44e7),
 'cd':(33.7,1.26e14,99.99,0.,.258,3.37e9),
 'pcd':(30.2,1.26e14,99.96,.06,.262,3.37e9),
 'chl':(.348,1.92e12,100.,0.,.111,2.15e9),
 'eqp':(None,None,1.10,.72,3.7,6.66e10),
 'imp':(None,None,.45,.50,2.3,4.23e10),
 'bptt1':(4.6,2.35e13,99.57,.29,1.1,7.41e9),
 'bptt5':(None,None,98.51,2.15,4.,1.24e10),
 'bptt':(None,None,.72,.92,7.7,1.86e10)}
extended={'sudoku/bptt5/60000':(174.,1.37e15,99.89,.07),'sudoku/eqp/240000':(None,None,16.05,8.75),'sudoku/bptt/480000':(None,None,46.24,2.86),'sudoku/bptt/3840000':(None,None,48.82,3.73)}
rows=[]
order=['pl','mpl','mpf','dsm','cd','pcd','chl','eqp','imp','bptt1','bptt5','bptt']
keys=[f'{task}/{m}/standard' for task in ['sudoku','maze'] for m in order]+list(extended)
for key in keys:
 r=s['rows'][key];task,m,schedule=key.split('/')
 if key in extended:p=extended[key]
 elif task=='sudoku':p=pubs[m][:4]
 else:p=(*pubs[m][4:],100.,0.)
 jobs=[alljobs[i] for i in r['job_ids']]
 rows.append({'recipe':key,'local_status':'complete','seeds':'0;1;2','n_test_per_seed':r['n_test_per_seed'][0],'updates':jobs[0]['final']['updates'],'lr':r['frozen_choice']['lr'],'knob':r['frozen_choice'].get('knob'),'l1':r['frozen_choice']['l1'],'local_solved_percent_mean':r['solved_percent_mean'],'local_solved_percent_sample_sd':r['solved_percent_sample_sd'],'published_solved_percent_mean':p[2],'published_solved_percent_sd':p[3],'local_common_T99_update':r['T99_update_common_mean'],'local_T99_s':r['paper_adjusted_T99_s_median'],'published_H200_T99_s':p[0],'local_FLOPs99':r.get('T99_fullcount_flop'),'published_FLOPs99':p[1],'local_full_fit_s_min':min(r['full_schedule_fit_s_per_seed']),'local_full_fit_s_max':max(r['full_schedule_fit_s_per_seed']),'published_reference':'arXiv2611.17309v1 Table7 or extended-schedule footnote','published_reference_kind':'published_not_locally_reproduced_measurement'})
csvout('table7-comparison.csv',rows);csvout('figure4-comparison.csv',figure_rows);csvout('d2-implicit-results.csv',d2rows)
save(O/'published-comparisons.json',{'published_source':'references/review/combined_draft.md:867-884 and references/review/01_main.md:280','reference_kind':'Published H200 reference numbers; not local baseline training','table7':rows,'figure4':figure_rows})
names={'pl':'Pseudolikelihood','mpl':'Masked PL','mpf':'Minimum probability flow','dsm':'Denoising score matching','cd':'Contrastive divergence','pcd':'Persistent CD','chl':'Contrastive Hebbian','eqp':'Equilibrium propagation','imp':'Implicit differentiation','bptt1':'One-step BPTT','bptt5':'Five-step BPTT','bptt':'Full BPTT'}
def secs(v):return '—' if v is None else f'{v:.6g}'
def flops(v):return '—' if v is None else f'{v:.3e}'
def solved(r,which):
 places=5 if which=="local" else 2
 return f'{r[which+"_solved_percent_mean"]:.{places}f} ± {r[which+"_solved_percent_"+("sample_sd" if which=="local" else "sd")]:.{places}f}'
lines=['# Objective-family reproduction: final measured results','',
'**Complete for this objective family:** all 84 Table 7 fits/tests (81 managed plus three 3.84-million-update controls), nine Appendix D.2 development-only fits, and all 100 Figure 4 curves. Every schedule finished. This does not declare the broader paper reproduction complete.','',
'Pinned source: `62bbe830c7d6f87f2ba18f30a28039258b986454`. Local runtime: Python 3.13.13, PyTorch 2.11.0+cu128, CUDA 12.8, Triton 3.6.0. GPUs are RTX PRO 6000 Blackwell Workstation and Max-Q devices on four recorded hosts; published timings are H200 measurements. All published columns below are quoted reference results, not locally retrained looped baselines.','',
'Each Table 7 row uses seeds 0–2, the frozen published recipe and development checkpoint selection, then 50,000 Sudoku-Extreme or 1,000 Maze-Unique test puzzles per seed. Means and sample standard deviations are recomputed from integer solved counts. No test inference was repeated for this consolidation. These experiments use the paper\'s SGD-family training; BP AdamW is outside this objective protocol and was not run.','',
'`T99` is the first common probe update at which the three-run development mean reaches 99%; its reported time is the median adjusted fit clock at that update. Adjustment removes startup and includes readiness-statistic fitting. A dash means no crossing, unless explicitly noted as unreported at that schedule. FLOPs99 is locally counted training arithmetic to that same update, not inference cost. The shared objective harness differs from the optimized main-result harness; do not compare these times directly with Table 2.','',
'## Sudoku-Extreme: standard schedules','',
'Every method uses 2,000 updates except PL (32). Published five-step BPTT, full BPTT and EqProp accuracies here come from the standard-schedule footnote; their main Table 7 cells instead use the longer schedules below. Their standard T99/FLOPs99 values are not separately published.','',
'| Method | Local solved (%) | Published solved (%) | Local T99 (s) | Published H200 T99 (s) | Local FLOPs99 | Published FLOPs99 |','|---|---:|---:|---:|---:|---:|---:|',
'| BP AdamW | Outside protocol; not run | Not in this study | — | — | — | — |']
for r in rows[:12]:
 m=r['recipe'].split('/')[1];lines.append(f'| {names[m]} | {solved(r,"local")} | {solved(r,"published")} | {secs(r["local_T99_s"])} | {secs(r["published_H200_T99_s"])} | {flops(r["local_FLOPs99"])} | {flops(r["published_FLOPs99"])} |')
lines+=['','## Sudoku-Extreme: extended schedules','',
'| Method and updates | Local solved (%) | Published solved (%) | Local T99 (s) | Published H200 T99 (s) | Local FLOPs99 | Published FLOPs99 |','|---|---:|---:|---:|---:|---:|---:|',
'| BP AdamW | Outside protocol; not run | Not in this study | — | — | — | — |']
for r in rows[24:]:
 m=r['recipe'].split('/')[1];lines.append(f'| {names[m]}, {r["updates"]:,} | {solved(r,"local")} | {solved(r,"published")} | {secs(r["local_T99_s"])} | {secs(r["published_H200_T99_s"])} | {flops(r["local_FLOPs99"])} | {flops(r["published_FLOPs99"])} |')
lines+=['',
'Full 3.84-million-update BPTT scores were 52.740%, 44.502% and 48.460%, with complete raw fit clocks of 2.463–2.508 hours per seed; the published raw fits took 4.8–5.0 hours. These are different GPU platforms, not a controlled hardware speedup measurement. Five-step BPTT used LR 6 and 60,000 updates, EqProp LR 6 and 240,000 updates, and full BPTT LR 2 at 480,000 and 3,840,000 updates.','',
'## Maze-Unique','',
'All 36 standard runs solved 1,000/1,000 test mazes, with zero wrong stops and zero budget failures. Published solved accuracy is also 100.00 ± 0.00% for every method. PL uses 20 updates, sampling objectives 1,500, and other objectives 2,000.','',
'| Method | Local solved (%) | Published solved (%) | Local T99 (s) | Published H200 T99 (s) | Local FLOPs99 | Published FLOPs99 |','|---|---:|---:|---:|---:|---:|---:|',
'| BP AdamW | Outside protocol; not run | Not in this study | — | — | — | — |']
for r in rows[12:24]:
 m=r['recipe'].split('/')[1];lines.append(f'| {names[m]} | {solved(r,"local")} | {solved(r,"published")} | {secs(r["local_T99_s"])} | {secs(r["published_H200_T99_s"])} | {flops(r["local_FLOPs99"])} | {flops(r["published_FLOPs99"])} |')
lines+=['','## Figure 4: 20-seed development curves','',
'All five objectives have seeds 0–19, each with its declared 1,000-puzzle training split and the fixed 512-puzzle development probe. All 9,740 scheduled probes completed (4,986,880 puzzle evaluations). The 12 dense trajectories matched their corresponding Table 7 clocks at all 228 shared checkpoints.','',
'![Locally measured Figure 4](../figure4/figures/figure4.png)','',
'[PDF](../figure4/figures/figure4.pdf) · [SVG](../figure4/figures/figure4.svg) · [All individual curves](../figure4/curves.json)','',
'| Method | Local mean-curve T99 (s) | Published H200 T99 (s) | Local final development mean (%) | Local final adjusted time, median [min, max] (s) |','|---|---:|---:|---:|---:|',
'| BP AdamW | Outside protocol; not run | Not in this study | — | — |']
for r in figure_rows:lines.append(f'| {names[r["method"]]} | {secs(r["local_T99_s"])} | {secs(r["published_H200_T99_s"])} | {r["local_final_mean_solved_percent"]:.5f} | {r["local_final_time_median_s"]:.3f} [{r["local_final_time_min_s"]:.3f}, {r["local_final_time_max_s"]:.3f}] |')
lines+=['',
'Full BPTT uses LR 60 and 60,000 updates here; its final 26.74805% development mean is close to the published 27% after 245 s. It is a distinct protocol from the longer LR-2 Table 7 controls. Local PL reaches 99% first, followed by MPF, one-step BPTT and CD, reproducing their reported ordering.','',
'Each curve uses its recorded fit clock minus startup plus the median of the three standard-run readiness measurements for that method. Curves use linear interpolation in elapsed seconds, zero before the first observation, and hold the final value afterward; bands are the 25th–75th percentiles over 20 seeds. Readiness pooling and interpolation are reconstructed conventions because the final upstream plotting script is missing. The displayed clocks combine documented Blackwell hosts and are not H200 timing reproduction.','',
'## Appendix D.2 implicit differentiation','',
'All nine 60,000-update fits completed with knob 10 and L1 multiplier 0.01. They evaluate development data only. The largest observed solve rate was 3/512 = **0.5859375%**, agreeing with the published “at most 0.6%” bound for these released-script configurations.','',
'| Learning rate | Seed 0 max solved / 512 | Seed 1 max solved / 512 | Seed 2 max solved / 512 |','|---:|---:|---:|---:|']
for lr in [6,20,60]:
 rr=sorted([r for r in d2rows if r['lr']==lr],key=lambda r:r['seed']);lines.append(f'| {lr} | '+ ' | '.join(str(r['max_development_solved']) for r in rr)+' |')
lines+=['',
'The separate declared gradient diagnostic gave cosine ≈0.99999967 between beta=0.01 EqProp and full BPTT, using the recorded 32-example initialization test. The original diagnostic recipe and beta=0.5 divergence trajectory are unreleased; this check does not reproduce that unspecified trajectory.','',
'## Differences and limits','',
'- Nine of twelve objectives reach at least 99% mean test accuracy on both tasks when the designated extended schedules are used, matching the paper\'s qualitative conclusion.','- Standard one-step BPTT reaches 99.576 ± 0.343% on test, but its three-seed development mean peaks at 98.828125%; hence its local Table 7 T99 is absent, unlike the published 4.6 s. The distinct 20-seed Figure 4 mean does cross 99%.','- DSM crosses the three-seed development target at update 1,448 locally rather than the update 1,024 implied by the published arithmetic. Five-step BPTT crosses at 46,341 rather than 32,768; their higher local FLOPs99 reflect different convergence updates, not a different per-update counter. All 12 Maze FLOPs99 values match the paper at its printed precision.','- PL selected updates 23, 32 and 32 locally, versus published 16, 23 and 16. The cause of these trajectory/selection differences has not been isolated. Published choices were reused without local tuning.','- Private two-client MPS improved the matched two-checkpoint PL probe test from 56.962 s to 31.661 s (1.799×). This measured throughput improvement concerns deferred development evaluation; it does not alter fit clocks or establish the same speedup on every method/host. Source, configuration and checkpoint checks plus per-GPU ownership guards were retained.','- Optional legacy ALT-free/Adam Maze curves are excluded because they are not a reported arXiv-v1 figure or numerical claim. The required Maze objective result uses Table 7\'s SGD/free-kernel protocol.','',
'## Evidence','',
'- [Final validation receipt](final-validation.json): hashes, independently recomputed statistics, complete schedules, all dense-clock proofs and visual plot review.','- [Machine-readable Table 7 comparison](table7-comparison.csv), [Figure 4 comparison](figure4-comparison.csv), [D.2 results](d2-implicit-results.csv), [published-value provenance](published-comparisons.json).','- Raw objective and external-control evidence: [objective summary](../objectives/summary.json) and [long-BPTT summary](../long_bptt/collected/summary.json).','- Baseline figures are transcribed from the reviewed arXiv-v1 Table 7, its schedule footnote, and Figure 4 caption; published and locally measured values remain separate.']
report_text='\n'.join(lines)+'\n'
(O/'FINAL_REPORT.md').write_text(report_text)
# Keep the substantive report with the source, alongside ignored raw artifacts.
import os,re
def document_link(match):
 target=match.group(1)
 if target.startswith(('https:', 'http:', '#')):return match.group(0)
 return ']('+os.path.relpath((O/target).resolve(),R/'docs')+')'
(R/'docs/OBJECTIVE_REPRODUCTION.md').write_text(re.sub(r'\]\(([^)]+)\)',document_link,report_text))
inputs=['artifacts/objectives/summary.json','artifacts/d2/summary.json','artifacts/figure4/summary.json','artifacts/figure4/curves.json','artifacts/figure4/figures/figure4.json','artifacts/figure4/figures/figure4.png','artifacts/figure4/figures/figure4.pdf','artifacts/figure4/figures/figure4.svg','references/review/combined_draft.md','references/review/01_main.md']
receipt={'status':'passed','utc':datetime.now(timezone.utc).isoformat(),'managed_jobs':81,'external_jobs':3,'independently_recomputed_table_rows':28,'arithmetic_count_rows':24,'d2_complete_schedules':9,'figure4_complete_curves':100,'figure4_complete_probes':9740,'dense_clock_validations':dense,'checkpoint_comparisons':228,'hashed_outputs':hashed,'input_sha256':{p:sha(R/p) for p in inputs},'plot_visual_review':{'file':'artifacts/figure4/figures/figure4.png','status':'passed','notes':'Five labeled curves, log time axis, visible interquartile bands and four99percent crossing markers; no clipping or overlapping labels.'},'new_test_inference':False,'wall_seconds':time.monotonic()-started,'report':'artifacts/objective_studies/FINAL_REPORT.md'}
receipt['finalization_script']=str(Path(__file__).resolve().relative_to(R))
receipt['finalization_script_sha256']=sha(Path(__file__))
receipt['reproduction_command']='artifacts/plotting-venv/bin/python '+receipt['finalization_script']
save(O/'final-validation.json',receipt)
print(json.dumps({'status':'passed','wall_seconds':receipt['wall_seconds'],'hashed_outputs':hashed,'report':str(O/'FINAL_REPORT.md')}))

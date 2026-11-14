"""Export compact public evidence without checkpoints or raw puzzle datasets."""
import csv
import json
from pathlib import Path
from run import ROOT, HERE, save, sha, now

BASE=ROOT/'artifacts/research/20261114/ripper'
DEST=ROOT/'results/research/ripper'


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    specs=[('discovery','',512,'protocol.json','REPORT.md'),
           ('confirmation','confirmation',2048,'protocol-confirmation.json','REPORT-confirmation.md'),
           ('lowdata','lowdata',512,'protocol-lowdata.json','REPORT-lowdata.md'),
           ('mechanism','mechanism',2048,'protocol-mechanism.json','REPORT-mechanism.md')]
    studies=[];points=[];grouped=[];provenance=[]
    for name,sub,n,proto,report in specs:
        ap=BASE/sub/'audit.json';a=json.loads(ap.read_text());p=json.loads((HERE/proto).read_text())
        ci=a.get('primary_ci95_pp',a.get('two_way_bootstrap_95pct_interval_pp',a.get('two_way_paired_bootstrap_95pct_interval_pp')))
        studies.append({'study':name,'hypothesis':p.get('hypothesis'),
                        'effect_pp':a['primary_effect_pp'],'ci95_pp':ci,
                        'decision':a.get('primary_decision',a.get('decision')),
                        'n_per_run':n,'jobs':a['jobs'],
                        'saved_boards_verified':a.get('saved_boards_verified',a.get('saved_boards_independently_verified')),
                        'rows_half_open':p.get('rows_half_open',p.get('development_rows_half_open',[0,512])),
                        'protocol':str((HERE/proto).relative_to(ROOT)),
                        'report':str((HERE/report).relative_to(ROOT)),
                        'audit_sha256':sha(ap),'protocol_sha256':sha(HERE/proto)})
        for r in a['rows']:
            points.append({'study':name,'k':r['k'],'arm':r['arm'],'seed':r['seed'],
                           'n':n,'solved':r['solved'],'accuracy_pct':100*r['solved']/n,
                           'wrong_stops':r['wrong_stops'],'mean_moves':r['mean_moves'],
                           'move_cap_per_chain':262144,'chains':4})
        for g in a['grouped']:
            grouped.append({'study':name,**g})
        save(DEST/f'audit-{name}.json',a)
        provenance.append({'path':str(ap.relative_to(ROOT)),'sha256':sha(ap)})
    cert=json.loads((BASE/'mechanism/repulsive-certificate-audit.json').read_text())
    save(DEST/'certificate.json',cert)
    summary={'generated_utc':now(),'upstream_commit':'62bbe830c7d6f87f2ba18f30a28039258b986454',
             'comparison_control':{'method':'BP AdamW','status':'Outside post-training intervention protocol; not run'},
             'scope':'New post-fit intervention research on fixed 262144-move/chain, four-chain budget; these are not paper-budget main accuracies.',
             'studies':studies,'grouped':grouped,
             'mechanism_secondary':{'gauge_effect_pp':a['secondary_gauge_effect_pp'],'gauge_ci95_pp':a['secondary_gauge_ci95_pp']},
             'certificate':{'certified_models':cert['certified_models'],'total_models':cert['total_models'],'proof':cert['proof'],'numerical_limit':cert['numerical_limit']},
             'uncertainty':'Approximate paired two-way bootstrap with 5000 draws, seed20261114; resample model-seed clusters and shared puzzle indices, preserving paired arms and k. Only3 discovery or6 confirmation/model-mechanism seeds.',
             'limitations':['Official development rows used for discovery/confirmation/mechanism; no external distribution claim.',
                            'k1000 checkpoint tensors are identical across seed IDs; variation there is search-only.',
                            'Projection fails at k1/3 and can worsen wrong stopping.',
                            'Small-seed bootstrap intervals do not imply independent evidence for every repeated puzzle evaluation.',
                            'Logical FLOP proxies are not measured hardware speedups.'],
             'provenance':provenance}
    save(DEST/'summary.json',summary)
    with (DEST/'per_seed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(points[0]));w.writeheader();w.writerows(points)
    with (DEST/'certificate.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(cert['rows'][0]));w.writeheader();w.writerows(cert['rows'])
    (DEST/'README.md').write_text('''# Ripper research evidence

The strongest confirmed result is digit-label symmetry projection: **+48.31 percentage points** (approximate paired 95% interval **43.11–53.65**) over unchanged fitted CRAM parameters, on new development rows 512–2559, k17/21 and six model/search seeds. The fixed search budget is 262,144 moves per chain, four chains; this is not the paper main-result budget.

| Method/control | Scope |
| --- | --- |
| BP AdamW | Outside this post-training intervention protocol; not run |
| Original CRAM | Measured matched controls, same models, puzzle rows and move budgets |
| Digit-symmetric CRAM | Mean diagonal/off-diagonal pair tables and digit-mean unary terms; no supplied Sudoku peer graph |
| Canonical repulsive CRAM | Drop the constant energy gauge, then clip negative equal-digit couplings |

The preregistered repulsion hypothesis of at least +10 points was falsified: observed **+1.87 points**, interval **0.86–2.82**. Repulsion removed all observed wrong stops in its study. All 18 repulsive models also satisfy a post-hoc sufficient exact-energy validity condition; this is not a worst-case floating-point drift guarantee. Low-data k1/3 models still solve zero puzzles and projection worsens their stopping behavior.

`summary.json` contains grouped plot inputs, hypotheses, decisions, confidence intervals, scope and hashes. `per_seed.csv` is the common source for per-seed plots. The four `audit-*.json` files preserve compact audited outcomes. `certificate.csv` and `certificate.json` retain the post-hoc energy-gap evidence and numerical caveat.

Frozen protocols and implementation are under [research/20261114/ripper](../../../research/20261114/ripper/). Detailed reports there describe matched controls, saved-board validation and protocol errata. Large saved boards, fitted parameters, training examples and source dataset files remain under ignored `artifacts/` and `upstream/` paths; a source-only checkout does not include these inputs. Reconstruct them using the repository reproduction/data setup and the frozen per-input SHA256 manifests before running the research workers. The research entrypoints intentionally require matching archived inputs and Ripper host ownership.

Subsequent sparse-backend and frozen held-out-test studies will be added after their own audits; no such result is included in this initial export.
''')
    print(f'Exported {len(points)} measured per-seed rows to {DEST}')


if __name__=='__main__':main()

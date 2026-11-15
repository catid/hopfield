"""Independent reconstruction from each accepted observed mask and label row."""
import csv
import json
import time
import numpy as np
from positive_witness_study import ROOT,HERE,PARENT,OUT,PUBLIC,PROTOCOL,sha,save,verify


def audit():
    p=verify();started=time.monotonic();mapping=json.loads((OUT/'mapping.json').read_text())
    assert mapping['protocol_sha256']==sha(PROTOCOL) and len(mapping['models'])==32
    parent=json.loads((PARENT/'audit.json').read_text());assert parent['status']=='validated'
    manifest=json.loads((PARENT/'input_manifest.json').read_text());i,j=np.triu_indices(81,1)
    true=(i//9==j//9)|(i%9==j%9)|((i//27==j//27)&(i%9//3==j%9//3));assert true.sum()==810
    rows=[];hashes={};witnesses=[];baselines=[]
    for spec in mapping['models']:
        b=spec['bank'];d=PARENT/f's{b}'/'filtered';receipt=json.loads((d/'receipt.json').read_text())
        prior_audit=json.loads((d/'audit.json').read_text());assert prior_audit['status']=='validated' and prior_audit['missing_true_edges']==0
        assert prior_audit['receipt_sha256']==sha(d/'receipt.json')
        for name in ['fit.npz','observations.npz']:assert receipt['artifact_sha256'][name]==sha(d/name)
        entry=next(e for e in manifest['inputs']if e['seed']==b)
        with np.load(ROOT/entry['training_path'],allow_pickle=False)as z:clean=z['solutions']
        with np.load(d/'observations.npz',allow_pickle=False)as z:obs={k:z[k]for k in z.files}
        count=np.zeros(3240,np.int64);equal=np.zeros(3240,np.int64);accepted=0
        for row in range(1110):
            mask=obs['given'][row];values=obs['values'][row];assert mask.sum()==25 and (values[~mask]==-1).all()
            if not obs['accepted'][row]:continue
            accepted+=1;assert np.array_equal(values[mask],clean[row,mask])
            assert max(np.count_nonzero(values[mask]==digit)for digit in range(9))<=9
            queried=mask[i]&mask[j];count+=queried;equal+=queried&(values[i]==values[j])
        with np.load(d/'fit.npz',allow_pickle=False)as z:
            assert np.array_equal(count,z['joint_count']) and np.array_equal(equal,z['equal_count'])
            baseline=z['support'].copy();assert np.array_equal(baseline,z['first_crossing']<0)
        assert accepted==prior_audit['accepted_boards']
        independently_retained=~(equal>0)
        assert independently_retained[count==0].all() and independently_retained[true].all()
        assert np.all(~independently_retained|baseline)
        path=ROOT/spec['path'];assert sha(path)==spec['sha256']
        with np.load(path,allow_pickle=False)as z:
            assert np.array_equal(z['baseline_support'],baseline) and np.array_equal(z['witness_support'],independently_retained)
            assert np.array_equal(z['joint_count'],count) and np.array_equal(z['equal_count'],equal)
        missing=int((true&~independently_retained).sum());extra=int((~true&independently_retained).sum());baseextra=int((~true&baseline).sum())
        rows.append(dict(bank=b,accepted_boards=accepted,baseline_missing=int((true&~baseline).sum()),baseline_extras=baseextra,
            baseline_exact=bool(np.array_equal(baseline,true)),witness_missing=missing,witness_extras=extra,witness_exact=missing==extra==0,
            removed_baseline_extras=int((baseline&~independently_retained).sum()),unexposed_pairs=int((count==0).sum()),positive_witness_pairs=int((equal>0).sum())))
        witnesses.append(independently_retained);baselines.append(baseline);hashes[spec['path']]=sha(path)
        save(OUT/'progress.json',dict(stage='independent_audit',completed_banks=b+1,elapsed_s=time.monotonic()-started))
    assert [r['bank']for r in rows]==list(range(32))
    grouped=[]
    for arm,prefix in [('archived_lr','baseline'),('positive_witness','witness')]:
        grouped.append(dict(arm=arm,exact_graph_banks=sum(r[prefix+'_exact']for r in rows),missing_true_edges=sum(r[prefix+'_missing']for r in rows),extra_edges=sum(r[prefix+'_extras']for r in rows)))
    primary=dict(target='At least29/32 positive-witness graphs are exact.',exact_banks=grouped[1]['exact_graph_banks'],
        decision='supported_finite_bank_target'if grouped[1]['exact_graph_banks']>=29 else'falsified_finite_bank_target')
    PUBLIC.mkdir(parents=True,exist_ok=True);np.savez_compressed(PUBLIC/'positive_witness_models.npz',banks=np.arange(32),baseline_support=np.stack(baselines),witness_support=np.stack(witnesses))
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),scope=p['scope'],primary=primary,grouped=grouped,rows=rows,
        theorem_checks=dict(zero_missing_true_edges=grouped[1]['missing_true_edges']==0,accepted_labels_clean=True,unexposed_pairs_retained=True),
        conditional_scope=p['supplied_assumptions'],independent_observed_rows_replayed=35520,pair_counts_checked=103680,
        new_queries=0,new_source_rows=0,new_solver_inputs=0,gpu_seconds=0,artifact_sha256=hashes,
        parent_audit_sha256=sha(PARENT/'audit.json'),model_export_sha256=sha(PUBLIC/'positive_witness_models.npz'))
    save(OUT/'audit.json',result);save(PUBLIC/'positive_witness.json',result)
    with (PUBLIC/'positive_witness_per_bank.csv').open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# H1ao: positive equality witnesses on unchanged filtered evidence','',
        f'The target of at least 29 exact graphs is **{primary["decision"]}**: {primary["exact_banks"]}/32 positive-witness graphs are exact.','',
        '| Retention rule | Exact graphs | Missing true edges | Retained extras |','|---|---:|---:|---:|',
        '| BP AdamW: outside this source-only study; not run | — | — | — |']
    for g in grouped:lines.append(f'| {g["arm"]} | {g["exact_graph_banks"]}/32 | {g["missing_true_edges"]} | {g["extra_edges"]} |')
    lines+=['','This post-hoc mechanism study reuses all 32 H1an filtered streams unchanged: the same acquired masks, observed labels and accepted rows. It changes only final pair retention. A pair is removed after any observed equal-digit coquery among accepted rows; an unexposed pair remains retained. No adaptive trajectory is changed or rerun.','',
        'The rule is conditional on the explicitly assumed shared-collapse law and supplied digit-multiplicity prior. Those assumptions establish that all accepted labels are clean. True peer cells therefore cannot have equal labels, proving true-edge retention separately from the empirical exact-recovery target. The rule is unsafe under arbitrary undetected label errors.','',
        f'The independent auditor reconstructed every joint and equal count from all 35,520 observed rows, updating only accepted rows. All 103,680 pair-bank counts matched the archived fits; the final rules, unexposed-pair handling and true-edge retention were independently checked. The new rule removes {sum(r["removed_baseline_extras"]for r in rows)} of the archived 54 extra edge-bank pairs.','',
        'This isolates the effect of evidence conservatism at fixed acquired data. It is not a fresh-generalization or revised adaptive-acquisition experiment. No new queries, source rows, solver inputs or GPU work were used.','',
        '[Protocol](protocol-positive-witness.json) · [Independent reconstruction](audit_positive_witness.py) · [Results](../../../results/research/ripper/positive_witness.json)','']
    (HERE/'REPORT-positive-witness.md').write_text('\n'.join(lines));print(json.dumps(dict(primary=primary,grouped=grouped)))

if __name__=='__main__':audit()

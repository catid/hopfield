"""Post hoc certificate compatibility and returned-board decomposition."""
import csv
import json
import numpy as np
from run import ROOT,HERE,sha,save,now


def main():
    out=ROOT/'artifacts/research/20261114/ripper/mixture_law'
    audit=json.loads((out/'audit.json').read_text());assert audit['status']=='validated'
    p=json.loads((HERE/'protocol-mixture-law.json').read_text())
    queue=json.loads((out/'queue.json').read_text());mapping=json.loads((out/'search_mapping.json').read_text())
    assert all(j['status']=='complete' for j in queue)
    i,j=np.triu_indices(81,1);canonical={};hashes={}
    for job in queue:
        d=out/'jobs'/job['job_id'];receipt=json.loads((d/'receipt.json').read_text())
        path=ROOT/job['model']['path'];assert sha(path)==job['model']['sha256'];support=np.load(path)['support']
        assert sha(d/'boards.npz')==receipt['output_sha256']['boards.npz']
        z=np.load(d/'boards.npz');target=z['solutions'];free=z['free'];exact=z['exact'];stopped=z['stopped']
        ii,jj=i[support],j[support]
        conflict=target[:,ii]==target[:,jj]
        target_conflicts=conflict.sum(1)
        given_conflicts=(conflict&~free[:,ii]&~free[:,jj]).sum(1)
        compatible=target_conflicts==0;given_impossible=given_conflicts>0
        assert np.all(given_conflicts<=target_conflicts)
        assert not (stopped&given_impossible).any()
        assert not (stopped&exact&~compatible).any()
        assert not (stopped&~exact).any()
        saved=next(r for r in audit['oracle_diagnostic'] if r['seed']==job['seed'] and r['arm']==job['arm'])
        assert int(compatible.sum())==saved['zero_energy_targets']
        assert int((stopped&exact).sum())==saved['correct_certified_stops']
        assert int((~stopped&exact).sum())==saved['correct_uncertified_returns']
        row={'seed':job['seed'],'arm':job['arm'],'puzzles':len(target),
            'canonical_compatible':int(compatible.sum()),
            'canonical_incompatible':int((~compatible).sum()),
            'given_given_infeasible':int(given_impossible.sum()),
            'no_given_conflict_but_canonical_incompatible':int((~given_impossible&~compatible).sum()),
            'correct_certified_stops':int((stopped&exact).sum()),
            'compatible_correct_capped_returns':int((compatible&~stopped&exact).sum()),
            'incompatible_correct_capped_returns':int((~compatible&~stopped&exact).sum()),
            'compatible_target_misses':int((compatible&~exact).sum()),
            'incompatible_target_misses':int((~compatible&~exact).sum()),
            'mean_canonical_conflicts':float(target_conflicts.mean()),
            'mean_given_given_conflicts':float(given_conflicts.mean())}
        assert row['canonical_compatible']==row['correct_certified_stops']+row['compatible_correct_capped_returns']+row['compatible_target_misses']
        assert row['canonical_incompatible']==row['incompatible_correct_capped_returns']+row['incompatible_target_misses']
        canonical[job['job_id']]=row
        hashes[str(path.relative_to(ROOT))]=sha(path);hashes[str((d/'boards.npz').relative_to(ROOT))]=sha(d/'boards.npz')
    rows=[]
    for entry in mapping['rows']:
        r=dict(canonical[entry['canonical_job']]);assert r['seed']==entry['seed'];r['arm']=entry['arm'];rows.append(r)
    aggregate=[]
    for arm in p['arms']:
        part=[r for r in rows if r['arm']==arm];total=sum(r['puzzles'] for r in part)
        a={'arm':arm,'bank_puzzle_instances':total,'banks':len(part),'distinct_test_inputs':10000}
        for key in rows[0]:
            if key in ('seed','arm','puzzles'):continue
            if key.startswith('mean_'):a[key]=float(np.mean([r[key] for r in part]))
            else:a[key]=sum(r[key] for r in part);a[key+'_pct']=100*a[key]/total
        aggregate.append(a)
    report={'status':'validated_descriptive','utc':now(),'scope':'Post hoc diagnosis using saved H1ag models and boards; no new labels, queries, inference trajectories, tuning or hypothesis tests.',
        'uniqueness':'No independent uniqueness proof is established by this diagnostic. Report canonical-solution compatibility, not exact capacity for all valid solutions.',
        'bound_scope':'Canonical compatibility bounds correct canonical zero-energy certificates. Positive given-given conflicts preclude every zero-energy completion respecting the clues. Neither bounds returned-best-board accuracy at positive energy.',
        'weight_scope':'Strictly positive reweighting preserves the zero set and all given-given obstructions, but can change positive-energy best-board rankings and capped-return accuracy.',
        'rows':rows,'aggregate':aggregate,'audit_sha256':sha(out/'audit.json'),'script_sha256':sha(__file__),'artifact_sha256':hashes}
    save(out/'compatibility_diagnostic.json',report);save(ROOT/'results/research/ripper/mixture_law_compatibility.json',report)
    with (ROOT/'results/research/ripper/mixture_law_compatibility.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# H1ag: canonical compatibility and certified versus capped returns','',report['scope'],'',report['uniqueness'],'',
        '| Evidence law / bound | Canonical zero-energy compatibility (%) | Given–given infeasibility (%) | Correct certified (%) | Correct capped, compatible target (%) | Correct capped, incompatible target (%) | Compatible target misses (%) |',
        '|---|---:|---:|---:|---:|---:|---:|','| BP AdamW: outside this study; not run | — | — | — | — | — | — |']
    for a in aggregate:
        lines.append(f'| {a["arm"]} | {a["canonical_compatible_pct"]:.4f} | {a["given_given_infeasible_pct"]:.4f} | {a["correct_certified_stops_pct"]:.4f} | {a["compatible_correct_capped_returns_pct"]:.4f} | {a["incompatible_correct_capped_returns_pct"]:.4f} | {a["compatible_target_misses_pct"]:.4f} |')
    lines+=['',report['bound_scope'],'',report['weight_scope'],'',
        'Each arm has 32 bank/search seeds on the same 10,000 puzzles. Percentages average over 320,000 bank–puzzle instances, not 320,000 distinct inputs. A compatible target miss means the search did not return the stored target within the frozen budget; no uniqueness or universal search claim is implied.',
        '', '[Study report](REPORT-mixture-law.md) · [Diagnostic script](diagnose_mixture_compatibility.py) · [Diagnostic data](../../../results/research/ripper/mixture_law_compatibility.json)','']
    (HERE/'REPORT-mixture-compatibility.md').write_text('\n'.join(lines));print(json.dumps(aggregate))


if __name__=='__main__':main()

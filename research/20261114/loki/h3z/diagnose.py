"""Post-seal known-target comparison and existing census identity mapping."""
import argparse
import datetime as dt
import shutil
import socket
import numpy as np
from source import ROOT,SRC,OUT,OLD,sha,read,save
PUBLIC=ROOT/'results/research/loki/h3z'

def main(expected):
    assert socket.gethostname()=='loki';p=read(SRC/'protocol.json');seal=read(OUT/'source-seals.json');assert sha(OUT/'source-seals.json')==expected
    assert seal['status']=='all6source_results_sealed_before_reused_target_diagnosis'and len(seal['source_result_sha256'])==6
    results={}
    for f,h in seal['source_result_sha256'].items():
        folder=OUT/f'cases/{f}';assert sha(folder/'result.json')==h;r=read(folder/'result.json');assert r['protocol_sha256']==sha(SRC/'protocol.json')
        assert dt.datetime.fromisoformat(r['completed_utc'])<=dt.datetime.fromisoformat(seal['utc'])
        if r['status']=='source_certified':
            for name,key in [('certificate.npz','certificate_sha256'),('checked-subsets.json','checked_subsets_sha256'),('verification.json','verification_sha256')]:assert sha(folder/name)==r[key]
            assert read(folder/'verification.json')['status']=='passed'
        results[f]=r
    # Boundary: all six source choices have been checked and sealed.
    assert sha(OLD/'summary.json')==p['reference_summary_sha256']and sha(OLD/'q11-census.json')==p['reference_census_sha256'];previous=read(OLD/'summary.json');census=read(OLD/'q11-census.json')
    i,j=np.triu_indices(121,1);dr=j//11-i//11;dc=j%11-i%11
    def graph(lines,generators):
        return np.array([any(a*int(r)+b*int(c)==0 for a,b in lines)or sorted((int(r),abs(int(c))))in generators for r,c in zip(dr,dc)])
    details={};rows=[];mapped=None
    for f in p['family_ids']:
        r=results[f];old=previous['case_diagnostics'][f'{f}-n10'];truth=graph([[0,1],[1,-1],[1,0],[1,1]],previous['families'][f]);minimums=[]
        if r['status']=='source_certified':
            for selected in r['selected_generator_sets']:
                pred=graph(r['inferred_lines'],selected);fp=int((pred&~truth).sum());fn=int((~pred&truth).sum())
                minimums.append(dict(generators=selected,false_pairs=fp,missed_pairs=fn,exact_graph=fp==fn==0,target_pairs=int(pred.sum()),upper_graph_bytes_sha256=__import__('hashlib').sha256(pred.tobytes()).hexdigest()))
            if f=='f4'and len(minimums)==1:
                before=graph(old['source_result']['inferred_lines'],old['all_minima'][0]['generators']);after=graph(r['inferred_lines'],r['selected_generator_sets'][0])
                assert before.tobytes()==after.tobytes()and census['n']==11
                mapped=dict(status='identity_derived_prior_census',reference='results/research/loki/h3y/q11-census.json',reference_sha256=sha(OLD/'q11-census.json'),
                    reference_boards='results/research/loki/h3y/q11-census-boards.npz',reference_boards_sha256=sha(OLD/'q11-census-boards.npz'),q=11,
                    upper_graph_bytes_sha256=minimums[0]['upper_graph_bytes_sha256'],selected_complete_boards=census['selected_rule_complete_boards'],true_complete_boards=census['true_rule_complete_boards'],
                    invalid_zero_conflict_boards=census['selected_but_invalid_complete_boards'],new_census_enumerations=0,new_searches=0)
        if r['status'].startswith('incomplete'):outcome='incomplete'
        elif r['status']!='source_certified':outcome=r['status']
        elif len(minimums)!=1:outcome='ambiguous_minimum'
        elif not minimums[0]['exact_graph']:outcome='unique_minimum_wrong_graph'
        else:outcome='unique_exact_transfer'
        oldrow=next(x for x in previous['rows']if x['id']==f'{f}-n10')
        rows.append(dict(family_id=f,family=previous['families'][f],source_patterns=r['source_patterns_by_size'],
            n10_only_minimum_count=oldrow['minimum_count'],pooled_minimum_count=len(r.get('minima',[])),n10_only_outcome=oldrow['outcome'],pooled_outcome=outcome,
            minimum_cardinality=r.get('minimum_cardinality'),candidate_orbits=len(r.get('candidate_generators',[])),checked_subsets=r.get('checked_subsets',0),
            false_pairs_best=min((x['false_pairs']for x in minimums),default=None),false_pairs_worst=max((x['false_pairs']for x in minimums),default=None),
            missed_pairs_best=min((x['missed_pairs']for x in minimums),default=None),missed_pairs_worst=max((x['missed_pairs']for x in minimums),default=None)))
        details[f]=dict(source_result=r,target_minima=minimums)
    # Explain elimination with existing N8 source incidences, without changing selection.
    r=results['f5'];z=np.load(OUT/'cases/f5/certificate.npz');assert not z['n8_is_positive'].any();mechanism=[]
    for old in previous['case_diagnostics']['f5-n10']['all_minima']:
        ids=[r['candidate_generators'].index(g)for g in old['generators']];unrejected=int((~z['n8_incidence'][:,ids].any(1)).sum())
        mechanism.append(dict(generators=old['generators'],unrejected_complete_N8_negatives=unrejected,survives_pooled_minimum=old['generators']in r['selected_generator_sets']))
        assert (unrejected==0)==mechanism[-1]['survives_pooled_minimum']
    counts={key:sum(row['pooled_outcome']==key for row in rows)for key in sorted({row['pooled_outcome']for row in rows})}
    failed=sum(row['pooled_outcome']not in['unique_exact_transfer','incomplete']for row in rows);decision='falsified'if failed else'inconclusive'if counts.get('incomplete',0)else'supported'
    PUBLIC.mkdir(parents=True,exist_ok=True);shutil.copytree(OUT/'cases',PUBLIC/'cases',dirs_exist_ok=True);shutil.copy2(OUT/'source-seals.json',PUBLIC/'source-seals.json')
    summary=dict(status='validated',utc=dt.datetime.now(dt.timezone.utc).isoformat(),hypothesis='H3z',decision=decision,
        interpretation='Adaptive known-target mechanistic follow-up; all source decisions were sealed before reusing N11diagnostics, but N11truth was already known fromH3y.',
        rows=rows,outcome_counts=counts,details=details,empty_source_constraint_mechanism=mechanism,inherited_census=mapped,
        source_cpu_seconds=sum(r['cpu_seconds']for r in results.values()),source_worker_wall_seconds=sum(r['wall_seconds']for r in results.values()),
        source_campaign_wall_seconds=seal['campaign_wall_seconds'],total_checked_subsets=sum(r.get('checked_subsets',0)for r in results.values()),
        source_seal_sha256=expected,protocol_sha256=sha(SRC/'protocol.json'),reference_summary_sha256=p['reference_summary_sha256'],
        new_source_generation=0,new_gpu_search_jobs=0,new_optimizer_fits=0,new_census_enumerations=0,limits=p['limits'],diagnostic_source_sha256=sha(__file__),
        export_sha256={str(path.relative_to(PUBLIC)):sha(path)for path in sorted(PUBLIC.rglob('*'))if path.is_file()and path.name not in['summary.json','README.md','plot-provenance.json']and path.suffix not in['.png','.svg']})
    save(PUBLIC/'summary.json',summary);print('DECISION',decision,'COUNTS',counts,'CPU',summary['source_cpu_seconds'],'SUBSETS',summary['total_checked_subsets'],flush=True)
    print('MECHANISM',mechanism,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--source-seal-sha256',required=True);args=a.parse_args();main(args.source_seal_sha256)

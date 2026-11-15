"""Public reconstruction of complete H2bk charged costs and bounded archives."""
import argparse
import json
from pathlib import Path
import numpy as np


def verify(directory):
    a=json.loads((directory/'audit.json').read_text());p=json.loads((directory/'protocol.json').read_text());t=json.loads((directory/'timing-accounting.json').read_text());assert len(t['rows'])==48
    baseline=main=strict=loads=0.;actual=0;seen=set()
    for row in t['rows']:
        job=row['job'];assert job in p['jobs'] and job['id'] not in seen;seen.add(job['id']);assert len(row['prefix_batch_timings'])==len(row['hybrid_batch_timings'])==8
        assert row['archive_writer_metrics']['maximum_pending']<=2 and row['archive_writer_metrics']['maximum_rss_kib']<=8*1024**2
        assert all(np.isfinite(row[k]) and row[k]>=0 for k in ['input_transfer_seconds','hidden_seconds','diagnostic_seconds','archive_seconds','job_seconds','bp_load_charge_here'])
        for prefix,hybrid in zip(row['prefix_batch_timings'],row['hybrid_batch_timings']):
            unique=np.array(prefix['unique_operation_seconds']);returns=np.array(prefix['cpu_return_seconds']);charged=np.array(prefix['charged_answer_seconds']);assert unique.shape==returns.shape==charged.shape==(8,)
            wanted=unique.copy();wanted[1:]=np.cumsum(unique[1:]);wanted+=returns;assert np.allclose(wanted,charged,rtol=0,atol=1e-12)
            operations=hybrid['operations'];assert all(np.isfinite(v) and v>=0 for v in operations.values())
            added=sum(v for k,v in operations.items() if k not in ['lp_cross_certificate','strict_certificate_flags']);all_added=sum(operations.values())
            assert abs(added-hybrid['main_added_seconds'])<1e-12 and abs(all_added-hybrid['strict_added_seconds'])<1e-12
            assert abs(charged[7]+added-hybrid['main_charged_seconds'])<1e-12 and abs(charged[7]+all_added-hybrid['strict_charged_seconds'])<1e-12
            baseline+=charged[7];main+=charged[7]+added;strict+=charged[7]+all_added;actual+=hybrid['actual_subset_inputs']
        loads+=row['bp_load_charge_here']
    expected_load=sum(r['additional_bp_load_seconds_charged_once'] for r in t['worker_model_loads']);assert abs(loads-expected_load)<1e-12
    for row in t['worker_model_loads']:assert abs(sum(r['pl'] for r in row['model_load_seconds_by_seed'].values())-row['additional_bp_load_seconds_charged_once'])<1e-12
    main+=loads;strict+=loads;c=a['cost_endpoint'];assert actual==a['actual_fallback_trajectories']
    assert abs(baseline-c['lp96_charged_seconds'])<1e-9 and abs(main-c['main_hybrid_charged_seconds_including_bp_load'])<1e-9 and abs(strict-c['strict_hybrid_charged_seconds_including_bp_load'])<1e-9
    overhead=100*(main-baseline)/baseline;assert abs(overhead-c['main_overhead_percent'])<1e-9 and c['decision']==('supported' if overhead<=10 else 'falsified')
    return {'status':'passed','jobs':48,'batches':384,'actual_subset_inputs':actual,'bp_model_load_seconds_charged':loads,'lp96_seconds':baseline,'main_hybrid_seconds':main,'strict_hybrid_seconds':strict,'main_overhead_percent':overhead,'decision':c['decision'],'scope':'Published complete-prefix/incremental arithmetic, explicit BP model-load charge and queue/RSS bounds. Full files and states were independently audited on Work.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);print(json.dumps(verify(parser.parse_args().directory),indent=2))

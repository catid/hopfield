"""H2br fixed descriptive strata and operation costs after the full audit."""
import csv
import json
import numpy as np
import study as S
DEST=S.OUT/'bp_first_endpoint_dispatch'
ARMS=['BP_AdamW','LP96','cascade','dispatcher']


def main():
    audit=json.loads((DEST/'audit.json').read_text());assert audit['status']=='passed'
    with (DEST/'per-model-input.csv').open() as f:rows=list(csv.DictReader(f))
    with (DEST/'per-input.csv').open() as f:joint=list(csv.DictReader(f))
    assert len(rows)==49152 and len(joint)==4096
    distances=np.asarray(json.loads((DEST/'independent-input-audit.json').read_text())['distances']).reshape(-1);assert distances.shape==(4096,)
    primary={int(r['input_index']):r for r in rows if r['arm']=='dispatcher' and r['model_seed']=='0'};assert set(primary)==set(range(4096))
    groups=[('mixture_component',[('uniform',lambda i:primary[i]['mixture_component']=='0'),('opposite_regions',lambda i:primary[i]['mixture_component']=='1')]),
            ('token_dispatch',[('D<182',lambda i:primary[i]['dispatch_high']=='0'),('D>=182',lambda i:primary[i]['dispatch_high']=='1')]),
            ('shortest_edges',[('1..64',lambda i:distances[i]<=64),('65..128',lambda i:64<distances[i]<=128),('129..256',lambda i:128<distances[i]<=256),('>=257',lambda i:distances[i]>=257)])]
    partitions={}
    for partition,bins in groups:
        records=[]
        for name,predicate in bins:
            indices={i for i in range(4096) if predicate(i)};arms={}
            for arm in ARMS:
                selected=[r for r in rows if r['arm']==arm and int(r['input_index']) in indices];assert len(selected)==3*len(indices)
                arms[arm]={'maze_denominator':len(indices),'joint_successes':sum(int(joint[i][arm]) for i in indices),'model_input_denominator':len(selected),'model_input_successes':sum(int(r['success']) for r in selected),
                    'numerical_rejections':sum(r['numerically_accepted']=='0' for r in selected),'BP_inputs':sum(int(r['BP_executed']) for r in selected),'direct_LP_inputs':sum(int(r['direct_LP']) for r in selected),'BP_to_LP_rescues':sum(int(r['routed']) for r in selected),
                    'underlying_BP_valid_longer':sum(int(r['BP_geometry_valid_longer']) for r in selected),'retained_BP_valid_longer':sum(r['chosen_model']=='0' and r['BP_geometry_valid_longer']=='1' for r in selected)}
            records.append({'bin':name,'empty':not indices,'arms':arms})
        assert sum(r['arms']['dispatcher']['maze_denominator'] for r in records)==4096;partitions[partition]=records
    operations=[]
    for block in range(6):
        for mode in ['LP96','BP_geometry_LP96','endpoint_dispatch']:
            path=DEST/'runs'/f'block{block}-{mode}'/'result.json';cohort=json.loads(path.read_text());sums={}
            for row in cohort['rows']:
                for name,value in row['timing']['components'].items():sums[name]=sums.get(name,0.)+value
                sums['unattributed_in_segment']=sums.get('unattributed_in_segment',0.)+row['timing']['unattributed_in_segment_seconds']
            sums['model_readiness']=cohort['model_readiness_seconds'];assert abs(sum(sums.values())-cohort['complete_answer_seconds'])<1e-8
            operations.append({'block':block,'mode':mode,'receipt_sha256':S.sha(path),'component_seconds':sums,'complete_answer_seconds':cohort['complete_answer_seconds']})
    result={'status':'complete','protocol_sha256':S.sha(DEST/'protocol.json'),'audit_sha256':S.sha(DEST/'audit.json'),'source_sha256':S.sha(__file__),'partitions':partitions,'operation_component_costs':operations,'scope':'Prespecified descriptive partitions and operation-level full-cohort cost components only. All inputs and empty bins retained. Shared mixed-batch costs are not attributed to individual inputs or mixture strata. No extra confidence or causal claims.'}
    S.save(DEST/'descriptive-strata-and-costs.json',result)


if __name__=='__main__':main()

"""H2bp frozen descriptive distance strata, only after the full audit passes."""
import csv
import json
from pathlib import Path
import numpy as np
import study as S
DEST=S.OUT/'bp_first_endpoint_shift'


def distance_bin(distance):
    if not isinstance(distance,(int,np.integer)) or distance<1:raise ValueError('Positive integer graph distance required')
    return '1..64' if distance<=64 else '65..128' if distance<=128 else '129..256' if distance<=256 else '>=257'


def main():
    audit=json.loads((DEST/'audit.json').read_text());assert audit['status']=='passed'
    protocol=json.loads((DEST/'protocol.json').read_text());assert protocol['diagnostic_strata']['shortest_edge_bins']==['1..64','65..128','129..256','>=257']
    distances={}
    for ds in json.loads((DEST/'datasets.json').read_text())['datasets']:
        assert S.sha(S.ROOT/ds['path'])==ds['sha256']
        with np.load(S.ROOT/ds['path'],allow_pickle=False) as z:
            for index,distance in zip(z['indices'],z['shortest_edges']):distances[int(index)]=int(distance)
    assert set(distances)==set(range(4096))
    with (DEST/'per-model-input.csv').open() as stream:rows=list(csv.DictReader(stream))
    with (DEST/'per-input.csv').open() as stream:joint=list(csv.DictReader(stream))
    assert len(rows)==36864 and len(joint)==4096
    result=[]
    for name in ['1..64','65..128','129..256','>=257']:
        indices={i for i,d in distances.items() if distance_bin(d)==name}
        by_arm={}
        for arm in ['BP_AdamW','LP96','cascade']:
            selected=[r for r in rows if r['arm']==arm and int(r['input_index']) in indices]
            assert len(selected)==len(indices)*3
            by_arm[arm]={'maze_denominator':len(indices),'joint_successes':sum(int(r[arm]) for r in joint if int(r['input_index']) in indices),
                'model_input_denominator':len(selected),'model_input_successes':sum(int(r['success']) for r in selected),
                'numerical_rejections':sum(r['numerically_accepted']=='0' for r in selected),
                'routed':sum(int(r['routed']) for r in selected),'BP_geometry_valid_longer':sum(int(r['BP_geometry_valid_longer']) for r in selected)}
        result.append({'shortest_edge_bin':name,'empty':not indices,'arms':by_arm})
    assert sum(r['arms']['cascade']['maze_denominator'] for r in result)==4096
    out={'status':'complete','protocol_sha256':S.sha(DEST/'protocol.json'),'audit_sha256':S.sha(DEST/'audit.json'),
        'source_sha256':S.sha(Path(__file__)),'bins':result,
        'scope':'Prespecified descriptive strata only. All mazes retained, empty bins explicit, no additional hypothesis decisions or confidence claims.'}
    S.save(DEST/'distance-strata.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':main()

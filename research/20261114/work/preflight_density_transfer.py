"""H2ay generator/schema gate with tiny synthetic labels and inherited exact source replay."""
import ast
import json
import time
import numpy as np
import study as S
import cap_transfer_batch as Q
from audit import independent


def main():
    started=time.perf_counter();target=S.OUT/'density_transfer_preflight';target.mkdir(exist_ok=True);source=S.OUT/'cap_transfer_preflight/numerical-gate.json';r=json.loads(source.read_text());assert r['status']=='passed' and r['batch_helper_sha256']==S.sha(S.HERE/'cap_transfer_batch.py') and r['script_sha256']==S.sha(S.HERE/'preflight_cap_transfer.py') and r['all_frozen_nonsimple_starts']==84
    for path,h in r['source_gate_hashes'].items():assert S.sha(S.ROOT/path)==h
    cases=[]
    for rate in [0.,.2,.5,1.]:
        for index in range(8):
            t,y,d,opened,eligible=S.maze(6,rate,np.random.SeedSequence([2026111599,6,int(rate*100),index]));ok,distance=independent(t,y);assert ok and distance==d and opened==round(rate*eligible) and np.array_equal(y[t!=1],(t>=2)[t!=1]);cases.append({'synthetic_length':6,'loop_rate':rate,'index':index,'opened':opened,'eligible':eligible,'shortest_edges':d})
    tree=ast.parse((S.HERE/'density_transfer.py').read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='make_one');expected=ast.parse('def make_one(index):return S.maze(180,.5,np.random.SeedSequence([BASE,180,50,index]))').body[0];assert ast.dump(fn)==ast.dump(expected);base=next(x for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BASE' for t in x.targets));assert ast.literal_eval(base.value)==2026111523
    assert Q.ARMS==('pl','old_binary','connectivity_veto','lexical256','lexical512_from_lexical256','energy512_from_lexical256');assert Q.MODEL_INDEX==(0,1,1,1,1,1)
    density=[]
    for path in S.HERE.glob('*protocol*.json'):
        text=json.dumps(json.loads(path.read_text()))
        if 'loop50' in text or '"loop_rate": 0.5' in text or '"loop_percent": 50' in text:density.append(str(path.relative_to(S.ROOT)))
    assert not density
    out={'status':'passed','kind':'pre-science synthetic-generator/schema and inherited complete source parity','new_study_inputs':0,'new_learner_inference':0,'existing_loop50_protocols':density,'source180_preflight':{'path':str(source.relative_to(S.ROOT)),'sha256':S.sha(source),'source_cases':180,'all_nonsimple_source_starts':84},'driver_sha256':S.sha(S.HERE/'density_transfer.py'),'batch_helper_sha256':S.sha(S.HERE/'cap_transfer_batch.py'),'script_sha256':S.sha(__file__),'synthetic_label_checks':cases,'seconds':time.perf_counter()-started,'scope':'No existing loop50 source cohort. Reuse unchanged six-arm/model/180 numerical parity and all synthetic size/selection/restoration gates by hash. New tiny6x6 generator fixtures check exact loop count, clamps, independent shortest-label correctness at0/20/50/100percent. AST checks exact180loop50 seed namespace; no fresh study draws or scientific outcomes.'};S.save(target/'numerical-gate.json',out);S.emit('density_preflight_passed',synthetic_cases=len(cases),seconds=out['seconds'])


if __name__=='__main__':main()

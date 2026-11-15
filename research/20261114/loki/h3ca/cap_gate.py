"""Fixed weight-sensitive work cap and no-routing-on-incomplete-plan checks."""
from pathlib import Path
import numpy as np
from common import SRC,save,sha
from compiler import Budget,ResourceCap,compile_tree
from verify_tree import weighted_certificate
from weights import integer_weights
from arm import ArmRunner
from ledger import Ledger
from oracle import PaidOracle


class LocalCap:
    def __init__(self,parent,states=100000,work=100000):
        self.parent=parent;self.max_states=states;self.max_splits=work;self.states=0;self.splits=0
    def state(self):
        if self.states>=self.max_states:raise ResourceCap('Synthetic local state cap')
        self.parent.state();self.states+=1
    def split(self):
        if self.splits>=self.max_splits:raise ResourceCap('Synthetic local work cap')
        self.parent.split();self.splits+=1


def run(folder,budget):
    folder.mkdir(parents=True,exist_ok=False);start_states,start_work=budget.states,budget.splits
    catalog=[dict(signature=s,lexfirst_board_index=s-1,board_count=1) for s in range(1,8)]
    checks=0
    def check(name,value):
        nonlocal checks
        assert value,name;checks+=1
    full=[]
    for label,weights in [('uniform',[1,1,1,1]),('skewed',[1,2,4,8])]:
        local=LocalCap(budget);proof=compile_tree(list(range(4)),weights,catalog,4,local)
        ns,nt,_=weighted_certificate(proof,catalog,check,label+'_')
        check('full_counter_identity',ns==local.states and nt==local.splits)
        save(folder/(label+'-full-proof.json'),proof)
        full.append(dict(weights=weights,states=ns,work=nt,optimal_cost=proof['optimal_cost']))
    check('fixed_weight_sensitive_work',full[0]['states']==full[1]['states']==7 and
          full[0]['work']==23 and full[1]['work']==28)
    local=LocalCap(budget,work=23)
    proof=compile_tree(list(range(4)),[1]*4,catalog,4,local)
    check('uniform_finishes_at_same_cap',proof['feasible'] and local.splits==23)
    save(folder/'uniform-capped-proof.json',proof)
    failures=[]
    for mode,local in [('work',LocalCap(budget,work=23)),('state',LocalCap(budget,states=6))]:
        try:compile_tree(list(range(4)),[1,2,4,8],catalog,4,local)
        except ResourceCap as error:
            traceback=error.__traceback__;incumbent=False
            while traceback:
                frame=traceback.tb_frame
                if frame.f_code.co_name=='solve' and frame.f_locals.get('best') is not None:incumbent=True
                traceback=traceback.tb_next
            failures.append(dict(mode=mode,error=str(error),states=local.states,work=local.splits,
                                 feasible_incumbent_existed=incumbent,complete_proof_returned=False))
        else:raise AssertionError('Fixed cap did not interrupt exact proof')
    check('work_cap_after_incumbent_still_incomplete',failures[0]['feasible_incumbent_existed'])
    try:integer_weights([0,1],['3','5'],max_bits=2)
    except ResourceCap:integer_rejected=True
    else:raise AssertionError('Integer cap did not interrupt')
    predictions=np.array([[bool(s&(1<<c)) for s in range(1,8)] for c in range(4)])
    bits=np.packbits(predictions,axis=1,bitorder='little');boards=np.arange(7,dtype=np.uint8)[:,None]
    armfolder=folder/'T';armfolder.mkdir();np.save(armfolder/'predictions-bits.npy',bits)
    selector=['1','1','1','1'];tree=['8','4','2','1']
    save(armfolder/'selector-scores.json',selector);save(armfolder/'tree-scores.json',tree)
    ledger=Ledger(folder/'query-ledger.jsonl');oracle=PaidOracle(ledger,boards,bits)
    runner=ArmRunner('T',armfolder,predictions,selector,tree,LocalCap(budget,work=23),100000,7)
    statuses=[dict(case=0,T='pending')]
    try:runner.run([dict(case=0,candidate=0,initial_candidates=list(range(4)),source_record_sha256='a'*64)],ledger,oracle.ask,statuses)
    except ResourceCap as error:arm_error=str(error)
    else:raise AssertionError('Arm routed after a capped tree')
    finally:ledger.close()
    check('cap_propagates_without_fallback_or_paid_query',oracle.calls==0 and runner.results==[] and
          statuses==[dict(case=0,T='pending')] and runner.worker is None and
          not list(armfolder.glob('group*/proof.json')) and not list(armfolder.glob('cost*')))
    result=dict(status='passed_weight_sensitive_cap_and_no_fallback_gate',checks=checks,
        fixed_full_runs=full,fixed_cap_failures=failures,integer_cap_rejected=integer_rejected,
        arm_cap_error=arm_error,whole_study_required_status='failed_or_incomplete',partial_primary_permitted=False,
        no_incumbent_route_no_MAP_fallback_no_actual_query=True,
        synthetic_memo_states=budget.states-start_states,synthetic_work_steps=budget.splits-start_work,
        actual_oracle_calls=0,fresh_truth_draws=0,fresh_source_draws=0,
        source_sha256={name:sha(SRC/name) for name in ['compiler.py','weights.py','arm.py','cap_gate.py']})
    save(folder/'summary.json',result);return result

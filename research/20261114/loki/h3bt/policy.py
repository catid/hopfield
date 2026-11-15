"""Online compilation receives only the current candidates and immutable models."""
import hashlib
import time
import numpy as np
from common import SRC,OUT,save,sha,now
from compiler import compile_tree
from weights import integer_weights
from ledger import canonical


class OnlineCompiler:
    def __init__(self,predictions,scores,predictions_sha256,budget,max_bits):
        self.predictions=predictions;self.predictions.flags.writeable=False;self.scores=tuple(scores);self.predictions_sha256=predictions_sha256;self.budget=budget;self.max_bits=max_bits;self.cache={};self.rows=[]

    def get(self,candidates):
        candidates=list(candidates);assert candidates==sorted(set(candidates)) and len(candidates)>1
        # All other compiler inputs are fixed immutable members for this run.
        if tuple(candidates) in self.cache:return self.cache[tuple(candidates)]
        group=len(self.rows);folder=OUT/f'group{group}';folder.mkdir();before_states=self.budget.states;before_splits=self.budget.splits;start=time.process_time();wall=time.perf_counter();catalog_sha=None;receipt=None;weights=None
        if len(candidates)>16:
            self.budget.state();full=(1<<len(candidates))-1;root=f'{full}:4';proof=dict(candidates=candidates,depth_limit=4,root_state=root,feasible=False,states={root:dict(mask=full,depth_remaining=4,models=len(candidates),feasible=False,reason='binary_information_bound',attempts=[])},leaf_depths={},worst_leaf_depth=None);method='binary_information_bound';compiler_sha=sha(SRC/'policy.py')
        else:
            codes=np.zeros(self.predictions.shape[1],np.uint32)
            for local,cid in enumerate(candidates):codes|=self.predictions[cid].astype(np.uint32)<<local
            signatures,indices,counts=np.unique(codes,return_index=True,return_counts=True);catalog=[dict(signature=int(s),lexfirst_board_index=int(i),board_count=int(n)) for s,i,n in zip(signatures,indices,counts)]
            save(folder/'catalog.json',dict(candidates=candidates,universe_boards=self.predictions.shape[1],signatures=catalog));catalog_sha=sha(folder/'catalog.json');weights,receipt=integer_weights(candidates,self.scores,self.max_bits);save(folder/'weights.json',receipt);proof=compile_tree(candidates,weights,catalog,4,self.budget);method='posterior_expected_depth_optimum';compiler_sha=sha(SRC/'compiler.py')
        inputs=dict(policy='early_handoff_posterior',method=method,candidates=candidates,weights=list(map(str,weights)) if weights is not None else None,depth=4,catalog_sha256=catalog_sha,predictions_sha256=self.predictions_sha256,compiler_sha256=compiler_sha,allowed_inputs_only=True);save(folder/'compiler-input.json',inputs);save(folder/'proof.json',proof)
        row=dict(group=group,candidates=candidates,feasible=proof['feasible'],method=method,utc=now(),catalog_sha256=catalog_sha,weights_sha256=sha(folder/'weights.json') if receipt else None,compiler_input_sha256=sha(folder/'compiler-input.json'),cache_key=hashlib.sha256(canonical(inputs)).hexdigest(),proof_sha256=sha(folder/'proof.json'),memo_states=self.budget.states-before_states,canonical_split_attempts=self.budget.splits-before_splits,CPU_seconds=time.process_time()-start,wall_seconds=time.perf_counter()-wall)
        save(folder/'result.json',row);self.rows.append(row);self.cache[tuple(candidates)]=(row,proof);return row,proof

"""Exact depth-indexed online trees, restricted to at most sixteen candidates."""
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

    def get(self,candidates,depth=4):
        candidates=list(candidates);assert candidates==sorted(set(candidates)) and 2<=len(candidates)<=16 and type(depth) is int and 4<=depth<=8
        key=(tuple(candidates),depth)
        if key in self.cache:return self.cache[key]
        group=len(self.rows);folder=OUT/f'group{group}';folder.mkdir();before_states=self.budget.states;before_splits=self.budget.splits;start=time.process_time();wall=time.perf_counter();codes=np.zeros(self.predictions.shape[1],np.uint32)
        for local,cid in enumerate(candidates):codes|=self.predictions[cid].astype(np.uint32)<<local
        signatures,indices,counts=np.unique(codes,return_index=True,return_counts=True);catalog=[dict(signature=int(s),lexfirst_board_index=int(i),board_count=int(n)) for s,i,n in zip(signatures,indices,counts)];save(folder/'catalog.json',dict(candidates=candidates,universe_boards=self.predictions.shape[1],signatures=catalog));catalog_sha=sha(folder/'catalog.json');weights,receipt=integer_weights(candidates,self.scores,self.max_bits);save(folder/'weights.json',receipt);proof=compile_tree(candidates,weights,catalog,depth,self.budget);method='posterior_expected_depth_optimum'
        inputs=dict(policy='budget_matched_posterior',method=method,candidates=candidates,weights=list(map(str,weights)),depth=depth,catalog_sha256=catalog_sha,predictions_sha256=self.predictions_sha256,compiler_sha256=sha(SRC/'compiler.py'),allowed_inputs_only=True);save(folder/'compiler-input.json',inputs);save(folder/'proof.json',proof);row=dict(group=group,candidates=candidates,depth=depth,feasible=proof['feasible'],method=method,utc=now(),catalog_sha256=catalog_sha,weights_sha256=sha(folder/'weights.json'),compiler_input_sha256=sha(folder/'compiler-input.json'),cache_key=hashlib.sha256(canonical(inputs)).hexdigest(),proof_sha256=sha(folder/'proof.json'),memo_states=self.budget.states-before_states,canonical_split_attempts=self.budget.splits-before_splits,CPU_seconds=time.process_time()-start,wall_seconds=time.perf_counter()-wall);save(folder/'result.json',row);self.rows.append(row);self.cache[key]=(row,proof);return row,proof

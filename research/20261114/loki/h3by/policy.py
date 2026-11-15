"""Requested-depth receipts with canonical reuse and compact in-memory routing."""
import hashlib
import time
import numpy as np
from common import SRC,OUT,save,read,sha,now
from compiler import compile_tree
from weights import integer_weights
from ledger import canonical


def compact(proof):
    result={key:value for key,value in proof.items() if key not in ['states','huffman_bounds']};chosen={}
    def retain(key):
        node=proof['states'][key];chosen[key]={name:value for name,value in node.items() if name!='attempts'}
        if node['reason']!='singleton_leaf':retain(node['negative_state']);retain(node['positive_state'])
    if proof['feasible']:retain(proof['root_state'])
    result['states']=chosen;result['certificate_payload_in_memory']=False;return result


class OnlineCompiler:
    def __init__(self,predictions,scores,predictions_sha256,budget,max_bits,compact_cache=True,reuse_effective=True):
        self.predictions=predictions;self.predictions.flags.writeable=False;self.scores=tuple(scores);self.predictions_sha256=predictions_sha256;self.budget=budget;self.max_bits=max_bits;self.cache={};self.effective_cache={};self.rows=[];self.compact_cache=compact_cache;self.reuse_effective=reuse_effective

    def get(self,candidates,depth=4):
        candidates=list(candidates);assert candidates==sorted(set(candidates)) and 2<=len(candidates)<=16 and type(depth) is int and 4<=depth<=8;key=(tuple(candidates),depth);effective=(tuple(candidates),min(depth,len(candidates)-1))
        if key in self.cache:return self.cache[key]
        group=len(self.rows);folder=OUT/f'group{group}';folder.mkdir();before_states=self.budget.states;before_splits=self.budget.splits;start=time.process_time();wall=time.perf_counter();codes=np.zeros(self.predictions.shape[1],np.uint32)
        for local,cid in enumerate(candidates):codes|=self.predictions[cid].astype(np.uint32)<<local
        signatures,indices,counts=np.unique(codes,return_index=True,return_counts=True);catalog=[dict(signature=int(s),lexfirst_board_index=int(i),board_count=int(n)) for s,i,n in zip(signatures,indices,counts)];save(folder/'catalog.json',dict(candidates=candidates,universe_boards=self.predictions.shape[1],signatures=catalog));catalog_sha=sha(folder/'catalog.json');weights,receipt=integer_weights(candidates,self.scores,self.max_bits);save(folder/'weights.json',receipt);reuse=None
        if self.reuse_effective and effective in self.effective_cache:
            original=self.effective_cache[effective];original_path=OUT/f'group{original["group"]}/proof.json';assert sha(original_path)==original['proof_sha256'];proof=read(original_path);assert proof['candidates']==candidates and proof['weights']==list(map(str,weights)) and min(proof['depth_limit'],len(candidates)-1)==effective[1];proof['depth_limit']=depth;reuse=dict(group=original['group'],source_requested_depth=original['depth'],effective_root_depth=effective[1],proof_sha256=original['proof_sha256'],group_result_sha256=sha(OUT/f'group{original["group"]}/result.json'))
        else:proof=compile_tree(candidates,weights,catalog,depth,self.budget)
        method='posterior_expected_depth_optimum';inputs=dict(policy='budget_matched_posterior',method=method,candidates=candidates,weights=list(map(str,weights)),depth=depth,catalog_sha256=catalog_sha,predictions_sha256=self.predictions_sha256,compiler_sha256=sha(SRC/'compiler.py'),allowed_inputs_only=True);save(folder/'compiler-input.json',inputs);save(folder/'proof.json',proof);tree_states=len(proof['states']);bound_states=len(proof['huffman_bounds']);partitions=sum(len(node['attempts']) for node in proof['states'].values());persistent_merges=sum(len(item['merges']) for item in proof['huffman_bounds'].values());inline_merges=sum(len(attempt['ordering_bound'][side]['merges']) for node in proof['states'].values() for attempt in node['attempts'] if 'ordering_bound' in attempt for side in ['negative_certificate','positive_certificate'] if attempt['ordering_bound'][side]['kind']=='inline_recomputed');merges=persistent_merges+inline_merges;charged_states=self.budget.states-before_states;charged_steps=self.budget.splits-before_splits;assert charged_states==(0 if reuse else tree_states+bound_states) and charged_steps==(0 if reuse else partitions+merges)
        row=dict(group=group,candidates=candidates,depth=depth,effective_root_depth=effective[1],feasible=proof['feasible'],method=method,utc=now(),catalog_sha256=catalog_sha,weights_sha256=sha(folder/'weights.json'),compiler_input_sha256=sha(folder/'compiler-input.json'),cache_key=hashlib.sha256(canonical(inputs)).hexdigest(),proof_sha256=sha(folder/'proof.json'),canonical_computation_reuse=reuse,memo_states=charged_states,canonical_split_attempts=0 if reuse else partitions,huffman_merge_operations=0 if reuse else merges,work_steps=charged_steps,logical_tree_states=tree_states,logical_bound_states=bound_states,logical_canonical_partitions=partitions,logical_bound_merges=merges,logical_inline_merges=inline_merges,logical_memoized_merges=persistent_merges,CPU_seconds=time.process_time()-start,wall_seconds=time.perf_counter()-wall);save(folder/'result.json',row);cached=compact(proof) if self.compact_cache else proof;self.rows.append(row);self.cache[key]=(row,cached)
        if effective not in self.effective_cache:self.effective_cache[effective]=row
        return row,cached

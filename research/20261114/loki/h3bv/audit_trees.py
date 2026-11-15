"""Independent bound-ordered certificates, reuse sources and charged/logical work."""
import hashlib
import math
import numpy as np
from verify_tree import weighted_certificate


class TreeAudit:
    def __init__(self,folder,source,rows,predictions,scores,predictions_sha,check,read,sha,canonical):
        self.folder=folder;self.source=source;self.rows=rows;self.pred=predictions;self.scores=scores;self.pred_sha=predictions_sha;self.check=check;self.read=read;self.sha=sha;self.canonical=canonical;self.lookup={(tuple(row['candidates']),row['depth']):row['group'] for row in rows};check('tree_depth_cache_unique',len(self.lookup)==len(rows) and [r['group'] for r in rows]==list(range(len(rows))));self.proofs={};self.fingerprints={};self.used=set();self.states=0;self.splits=0;self.paths=0;self.logical_states=0;self.logical_steps=0;self.reuse_hits=0

    def get(self,candidates,bound,depth):
        read=self.read;sha=self.sha;check=self.check;group=self.lookup[(tuple(candidates),depth)];row=self.rows[group];check('tree_completed_before_reference',row['utc']<=bound['utc']);self.used.add(group)
        if group in self.proofs:return row,self.proofs[group]
        folder=self.folder/f'group{group}';proof=read(folder/'proof.json');codes=np.sum(self.pred[candidates].astype(np.uint32)*(1<<np.arange(len(candidates),dtype=np.uint32))[:,None],axis=0,dtype=np.uint32);signatures,counts=np.unique(codes,return_counts=True);catalog=[dict(signature=int(s),lexfirst_board_index=int(np.flatnonzero(codes==s)[0]),board_count=int(n)) for s,n in zip(signatures,counts)];denominators=[int(self.scores[c]) for c in candidates];L=1;lengths=[]
        for d in denominators:L=math.lcm(L,d);lengths.append(L.bit_length())
        weights=[L//d for d in denominators];receipt=dict(score_denominators=list(map(str,denominators)),common_denominator=str(L),LCM_intermediate_bit_lengths=lengths,integer_weights=list(map(str,weights)),weight_bit_lengths=[w.bit_length() for w in weights],max_bits=100000);inputs=dict(policy='budget_matched_posterior',method='posterior_expected_depth_optimum',candidates=candidates,weights=list(map(str,weights)),depth=depth,catalog_sha256=sha(folder/'catalog.json'),predictions_sha256=self.pred_sha,compiler_sha256=sha(self.source/'compiler.py'),allowed_inputs_only=True)
        check('tree_full_input_identity',row==read(folder/'result.json') and row['group']==group and row['candidates']==proof['candidates']==candidates and row['depth']==proof['depth_limit']==depth and 4<=depth<=8 and 2<=len(candidates)<=16 and row['feasible']==proof['feasible'] and row['method']=='posterior_expected_depth_optimum' and row['catalog_sha256']==sha(folder/'catalog.json') and read(folder/'catalog.json')==dict(candidates=candidates,universe_boards=self.pred.shape[1],signatures=catalog) and row['weights_sha256']==sha(folder/'weights.json') and read(folder/'weights.json')==receipt and proof['weights']==list(map(str,weights)) and row['compiler_input_sha256']==sha(folder/'compiler-input.json') and read(folder/'compiler-input.json')==inputs and row['cache_key']==hashlib.sha256(self.canonical(inputs)).hexdigest() and row['proof_sha256']==sha(folder/'proof.json') and max(lengths)<=100000 and max(w.bit_length() for w in weights)<=100000)
        ns,nt,_=weighted_certificate(proof,catalog,check,'bound_ordered_tree_');tree_states=len(proof['states']);bound_states=len(proof['huffman_bounds']);partitions=sum(len(node['attempts']) for node in proof['states'].values());merges=sum(len(item['merges']) for item in proof['huffman_bounds'].values());fingerprint=hashlib.sha256(self.canonical({k:v for k,v in proof.items() if k!='depth_limit'})).hexdigest();reuse=row['canonical_computation_reuse'];check('logical_work_counts',ns==tree_states+bound_states and nt==partitions+merges and row['logical_tree_states']==tree_states and row['logical_bound_states']==bound_states and row['logical_canonical_partitions']==partitions and row['logical_bound_merges']==merges and row['effective_root_depth']==min(depth,len(candidates)-1))
        if reuse is None:check('original_computation_charges',row['memo_states']==ns and row['canonical_split_attempts']==partitions and row['huffman_merge_operations']==merges and row['work_steps']==nt)
        else:
            check('reuse_source_precedes_requested_receipt',reuse['group']<group);source,_=self.get(candidates,row,reuse['source_requested_depth']);check('independent_complete_body_reuse',source['group']==reuse['group'] and source['canonical_computation_reuse'] is None and source['effective_root_depth']==row['effective_root_depth'] and reuse==dict(group=source['group'],source_requested_depth=source['depth'],effective_root_depth=row['effective_root_depth'],proof_sha256=source['proof_sha256'],group_result_sha256=sha(self.folder/f'group{source["group"]}/result.json')) and self.fingerprints[source['group']]==fingerprint and row['memo_states']==row['canonical_split_attempts']==row['huffman_merge_operations']==row['work_steps']==0);self.reuse_hits+=1
        self.states+=row['memo_states'];self.splits+=row['work_steps'];self.logical_states+=ns;self.logical_steps+=nt;self.fingerprints[group]=fingerprint
        chosen={}
        def retain(key):
            node=proof['states'][key];chosen[key]={k:v for k,v in node.items() if k!='attempts'}
            if node['reason']!='singleton_leaf':retain(node['negative_state']);retain(node['positive_state'])
        if proof['feasible']:
            for i,cid in enumerate(candidates):
                key=proof['root_state'];queries=[]
                while proof['states'][key]['reason']!='singleton_leaf':
                    node=proof['states'][key];index=node['chosen_split']['board_index'];check('tree_prediction_branch',index not in queries and bool(self.pred[cid,index])==bool(node['chosen_split']['positive_mask']&(1<<i)));queries.append(index);key=node['positive_state'] if self.pred[cid,index] else node['negative_state']
                check('tree_correct_candidate_depth',proof['states'][key]['mask']==1<<i and len(queries)==proof['leaf_depths'][str(cid)]<=depth);self.paths+=1
            retain(proof['root_state'])
        proof['states']=chosen;proof.pop('huffman_bounds');self.proofs[group]=proof;return row,proof

    def finish(self):self.check('all_completed_tree_inputs_reachable',self.used==set(range(len(self.rows))));return self.states,self.splits

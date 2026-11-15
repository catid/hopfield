"""Rebuild complete enumerations and every clue count using only published files.

Run from any publication snapshot with NumPy and g++ installed. This is a
post-publication correctness check, not a rerun of the experimental timing.
"""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import tempfile
import time
import numpy as np

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]);args=parser.parse_args();R=args.root;P=R/'results/research/loki/h3aj';S=R/'research/20261114/loki/h3aj';checks=[];start=time.perf_counter();sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    def check(name,value):assert value,name;checks.append(name)
    def histogram(columns):
        h=np.zeros(226**3,np.int64)
        for k in [1,2,3]:
            ix=np.array(list(itertools.combinations(range(15),k)));powers=np.array([226**(2-j)for j in range(k)],np.int64)
            for lo in range(0,len(columns),2048):
                cells=15*np.arange(15)[None,:]+columns[lo:lo+2048];keys=((cells[:,ix]+1)*powers).sum(-1);h+=np.bincount(keys.reshape(-1),minlength=226**3)
        check('all575subsets',h.sum()==len(columns)*575);return h
    s=json.loads((P/'summary.json').read_text());p=json.loads((S/'protocol.json').read_text());check('falsified',s['decision']=='falsified');check('protocol',s['protocol_sha256']==sha(S/'protocol.json'));check('budget',s['aggregate_CPU_seconds']<=120 and s['wall_seconds']<=180 and s['GPU_seconds']==s['target_cohorts_created']==0)
    for rel,h in p['source_sha256'].items():check('source '+rel,sha(R/rel)==h)
    for rel,h in s['public_sha256'].items():check('export '+rel,sha(P/rel)==h)
    Ftrue=np.load(P/'truth/F.npy');truecache=np.load(P/'truth/cache.npz');truecols=truecache['columns'];check('true count',len(truecols)==26960==s['universe']['true_states'])
    for i,j in itertools.combinations(range(225),2):
        ar,ac=divmod(i,15);br,bc=divmod(j,15);valid=ar==br or ac==bc or abs(ar-br)==abs(ac-bc)or(abs(ar-br),abs(ac-bc))in[(1,3),(3,1)];assert Ftrue[i,j]==Ftrue[j,i]==valid
    check('literal true geometry',True)
    with tempfile.TemporaryDirectory(prefix='h3aj-public-census-')as temp:
        temp=Path(temp)
        for name in ['reverse_literal','truth_camel']:subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-Wall','-Wextra','-Werror',str(S/f'{name}.cpp'),'-o',str(temp/name)],check=True)
        def reverse(F,tag):
            raw=temp/f'{tag}.F.bin';out=temp/f'{tag}.columns.bin';F.astype(np.uint8).tofile(raw);r=subprocess.run([str(temp/'reverse_literal'),'15',str(raw),str(out)],text=True,capture_output=True,check=True);a=np.fromfile(out,np.uint8).reshape(-1,15);check(tag+'enumeration count',len(a)==int(r.stdout));return a
        check('complete truth reverse parity',np.array_equal(reverse(Ftrue,'truth'),truecols));out=temp/'direct.columns.bin';r=subprocess.run([str(temp/'truth_camel'),'15',str(out)],text=True,capture_output=True,check=True);check('complete independent direct truth parity',int(r.stdout)==len(truecols)and np.array_equal(np.fromfile(out,np.uint8).reshape(-1,15),truecols));truehist=histogram(truecols);allkeys=np.flatnonzero(truehist);old=np.load(R/'results/research/loki/h3ag/per-input-metrics.npz');excluded=np.union1d(old['b0_keys'],old['b1_keys']);eligible=np.setdiff1d(allkeys,excluded);clues=sum((eligible//226**power%226)>0 for power in [2,1,0]);u=np.load(P/'input-universe.npz');check('full true input universe',np.array_equal(allkeys,u['all_keys'])and np.array_equal(truehist[allkeys],u['all_true_counts']));check('prior exclusions',np.array_equal(excluded,u['excluded_keys'])and len(excluded)==8192);check('remaining eligible universe',np.array_equal(eligible,u['eligible_keys'])and len(eligible)==831605 and np.array_equal(clues,u['eligible_clues'])and np.array_equal(truehist[eligible],u['eligible_true_counts']));truekeys={row.tobytes()for row in truecols};learned_F=[]
        for info in s['support_results']:
            group=info['group'];bank=info['banks'][0];m=np.load(P/f'b{bank}/model.npz');F=m['F'];learned_F.append(F);cache=np.load(P/f'support{group}/cache.npz');columns=cache['columns'];check(f'support{group}complete compiled census',np.array_equal(reverse(F,f'support{group}'),columns));valid=np.array([row.tobytes()in truekeys for row in columns]);check(f'support{group}valid states',int(valid.sum())==info['valid_zero_F_states']and int((~valid).sum())==info['invalid_zero_F_states']);h=histogram(columns[valid]);cap=np.load(P/f'support{group}/capacity.npz');kh=np.load(P/f'support{group}/key-histogram.npz');check(f'support{group}complete key histogram',np.array_equal(kh['keys'],np.flatnonzero(h))and np.array_equal(kh['counts'],h[h>0]));check(f'support{group}every capacity count',np.array_equal(cap['eligible_keys'],eligible)and np.array_equal(cap['compatible_true_zero_F'],h[eligible])and np.array_equal(cap['valid_state_mask'],valid));check(f'support{group}finite ratio',info['capacity_pct']==100*float((h[eligible]>0).mean())and info['inputs_with_valid_zero_F_completion']==int((h[eligible]>0).sum()));check(f'support{group}state retention',info['valid_state_retention_pct']==100*float(valid.sum()/len(truecols)))
            for k in [1,2,3]:v=info['per_clue'][str(k)];part=clues==k;check(f'support{group}clue{k}',v['eligible_inputs']==int(part.sum())and v['inputs_with_valid_zero_F_completion']==int((h[eligible][part]>0).sum())and v['capacity_pct']==100*float((h[eligible][part]>0).mean()))
            for b in info['banks']:
                other=np.load(P/f'b{b}/model.npz');row=s['rows'][b];check(f'bank{b}full support equality',np.array_equal(other['F'],F));check(f'bank{b}retained failed outcome',row['bank']==b and row['capacity_pct']==info['capacity_pct']and row['decision']=='falsified'and row['capacity_pct']<95)
        check('distinct full supports',not np.array_equal(*learned_F))
    check('audit source',sha(S/'analyze.py')==s['audit_source_sha256']);check('figure inputs',all(sha(R/rel)==h for rel,h in json.loads((P/'plot-provenance.json').read_text())['input_sha256'].items()));print(json.dumps(dict(status='passed',checks=len(checks),all_complete_sets_reenumerated_from_public_source=True,all_true_and_learned_key_histograms_rebuilt=True,elapsed_seconds=time.perf_counter()-start)))
if __name__=='__main__':main()

"""Public-only exact arithmetic check of all zero-origin KKT certificates."""
from pathlib import Path
from fractions import Fraction
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
PUB=ROOT/'results/research/foureyes/uniform_pl_origin'


def unpack(value):return Fraction(value['numerator'],value['denominator'])


def main():
    result=json.loads((PUB/'summary.json').read_text());receipt=json.loads((PUB/'count-export.json').read_text())
    file=PUB/'sufficient-counts.npz';assert hashlib.sha256(file.read_bytes()).hexdigest()==receipt['sha256']
    data=np.load(file,allow_pickle=False);A=data['pair_exposure'].astype(np.int64);F=data['free_totals'].astype(np.int64)
    assert A.shape==(64,3240) and F.shape==(64,) and (A>0).all() and (A<=1200).all()
    checked=0;decisions={}
    for proof in result['certificates']:
        bank=proof['bank'];key='clean_equal' if proof['source']=='clean' else 'noisy_equal'
        C=data[key][bank].astype(np.int64);assert (C>=0).all() and (C<=A[bank]).all()
        numerators=9*C-A[bank];index=int(np.abs(numerators).argmax());maximum=Fraction(abs(int(numerators[index])),81*int(F[bank]))
        penalty=Fraction(1,3000) if proof['coefficient']=='nominal_proximal' else Fraction(1,6000)
        margin=penalty-maximum
        assert maximum==unpack(proof['max_gradient_exact']) and penalty==unpack(proof['penalty_exact']) and margin==unpack(proof['margin_exact'])
        origin='unique_global_minimum' if margin>0 else 'not_a_minimum' if margin<0 else 'boundary'
        assert origin==proof['origin']
        if margin<0:
            L=Fraction(int(A[bank,index]),4*int(F[bank]));step=unpack(proof['descent_step'])
            gain=unpack(proof['guaranteed_objective_decrease'])
            assert step>0 and gain>0 and margin*step+L*step*step/2==-gain
            assert int(proof['coordinate_sign'])*int(numerators[index])<0
        key=(proof['source'],proof['coefficient'],origin);decisions[key]=decisions.get(key,0)+1;checked+=1
    assert checked==256
    for row in result['grouped']:
        prefix=(row['source'],row['coefficient'])
        assert decisions.get(prefix+('unique_global_minimum',),0)==row['unique_zero']
        assert decisions.get(prefix+('not_a_minimum',),0)==row['nonminimum_zero']
        assert decisions.get(prefix+('boundary',),0)==row['boundary']
    print(json.dumps(dict(status='passed',exact_certificates=checked,all_pair_gradient_bounds_checked=64*3240*2)))


if __name__=='__main__':main()

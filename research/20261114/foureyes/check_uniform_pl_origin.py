"""Exact source-only KKT certificates for uniformly digit-averaged PL at zero.

This is an objective diagnostic, not a claim about finite stochastic iterates.
The nominal proximal coefficient and the constant-step heavy-ball fixed-point
coefficient are deliberately tested and reported separately.
"""
from pathlib import Path
from fractions import Fraction
import csv
import hashlib
import itertools
import json
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
RAW=ROOT/'artifacts/research/20261114/foureyes/noisy_pl_long'
PUBLIC=ROOT/'results/research/foureyes/uniform_pl_origin'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def packed(f):return dict(numerator=f.numerator,denominator=f.denominator,decimal=float(f))


def literal_fixture():
    """Enumerate all S3 permutations and every coordinate of a four-cell PL."""
    v=3;n=4
    sol=np.array([[0,1,2,0],[1,1,0,2],[2,0,1,1],[0,2,0,1]])
    free=np.array([[1,0,1,1],[0,1,1,1],[1,1,0,1],[1,1,1,0]],dtype=bool)
    total=int(free.sum());checks=0
    for i,j in itertools.combinations(range(n),2):
        A=int(free[:,i].sum()+free[:,j].sum())
        C=int(((free[:,i].astype(int)+free[:,j])*(sol[:,i]==sol[:,j])).sum())
        diag=Fraction(v*C-A,v*v*total)
        for a,b in itertools.product(range(v),repeat=2):
            literal=Fraction(0)
            for permutation in itertools.permutations(range(v)):
                relabeled=np.asarray(permutation)[sol]
                for row,f in zip(relabeled,free):
                    if f[i]:literal+=Fraction(int(row[i]==a and row[j]==b),1)-Fraction(int(row[j]==b),v)
                    if f[j]:literal+=Fraction(int(row[i]==a and row[j]==b),1)-Fraction(int(row[i]==a),v)
            literal/=6*total
            expected=diag if a==b else -diag/(v-1)
            assert literal==expected;checks+=1
    return dict(permutations=6,all_pair_digit_coordinates_checked=checks,exact_rational_match=True)


def main():
    start=time.monotonic();fixture=literal_fixture()
    manifest=json.loads((RAW/'inputs.json').read_text());audit=json.loads((RAW/'source-audit.json').read_text())
    assert audit['status']=='validated' and manifest['protocol_sha256']==audit['protocol_sha256']
    ii,jj=np.triu_indices(81,1);rows=[];certificates=[];evidence={}
    for entry in manifest['banks']:
        path=ROOT/entry['path'];digest=sha(path)
        assert digest==entry['sha256']==audit['input_sha256'][entry['path']];evidence[entry['path']]=digest
        z=np.load(path);free=z['free'];total=int(free.sum());A=free[:,ii].sum(0)+free[:,jj].sum(0)
        for kind,key in [('clean','solutions'),('nine_corruptions','noisy')]:
            sol=z[key];C=((free[:,ii].astype(np.int64)+free[:,jj])*(sol[:,ii]==sol[:,jj])).sum(0)
            numerator=9*C-A;index=int(np.abs(numerator).argmax());maximum=Fraction(int(abs(numerator[index])),81*total)
            # Every diagonal coordinate has this gradient; off-diagonals are -1/8
            # times it, and every unary coordinate is zero by permutation symmetry.
            for label,penalty in [('nominal_proximal',Fraction(1,3000)),('heavy_ball_fixed_point',Fraction(1,6000))]:
                margin=penalty-maximum
                row=dict(bank=entry['bank'],source=kind,coefficient=label,max_abs_gradient=float(maximum),penalty=float(penalty),
                         origin='unique_global_minimum' if margin>0 else 'not_a_minimum' if margin<0 else 'boundary',
                         kkt_margin=float(margin),max_pair_i=int(ii[index]),max_pair_j=int(jj[index]),free_cell_count=total)
                proof=dict(**row,max_gradient_exact=packed(maximum),penalty_exact=packed(penalty),margin_exact=packed(margin),
                           maximum_numerator=int(abs(numerator[index])),denominator=81*total)
                if margin<0:
                    # One-coordinate smoothness bound L=A/(4F) is valid uniformly
                    # over relabelings. A small explicit signed step lowers F.
                    smooth=Fraction(int(A[index]),4*total);step=(-margin)/smooth
                    proof.update(descent_step=packed(step),coordinate_sign=-int(np.sign(numerator[index])),
                                 guaranteed_objective_decrease=packed(margin*margin/(2*smooth)))
                rows.append(row);certificates.append(proof)
            np.savez_compressed(RAW/f'origin-{entry["bank"]:02d}-{kind}.npz',pair_exposure=A,free_weighted_equal=C,gradient_numerator=numerator)
    grouped=[]
    for kind in ['clean','nine_corruptions']:
        for penalty in ['nominal_proximal','heavy_ball_fixed_point']:
            rs=[r for r in rows if r['source']==kind and r['coefficient']==penalty]
            grouped.append(dict(source=kind,coefficient=penalty,banks=len(rs),unique_zero=sum(r['origin']=='unique_global_minimum' for r in rs),
                                nonminimum_zero=sum(r['origin']=='not_a_minimum' for r in rs),boundary=sum(r['origin']=='boundary' for r in rs),
                                gradient_max_range=[min(r['max_abs_gradient'] for r in rs),max(r['max_abs_gradient'] for r in rs)]))
    PUBLIC.mkdir(exist_ok=True,parents=True)
    result=dict(status='exact_source_objective_diagnostic',scope='Post hoc on64frozen H4s trainingbanks and pairedcleancontrols. Uniformaverageoverall9!digitpermutations; not a finiteaugmentationobjective or optimizerconvergence claim.',
                formula='g_diag=(9*C-A)/(81*F),g_off=-g_diag/8,g_bias=0; C=sum(f_i+f_j)*1[y_i=y_j], A=sum(f_i+f_j), F=sumallfreecells.',
                proof='Convex PL plus lambdaL1 is >=log9+(lambda-max|g|)*||parameters||1. Positive margin proves uniquezero; negative margin has explicit one-coordinate descent with HessianboundA/(4F).',
                momentum='Implemented v=m*v+grad,theta=prox_(alpha*lambda)(theta-alpha*v) gives constant-step stationary threshold (1-m)*lambda at zero. This does not establish convergence of the finite cosine stochastic schedule or equivalence of its whole trajectory to an optimizer of that objective.',
                fixture=fixture,grouped=grouped,certificates=certificates,input_sha256=evidence,producer_sha256=sha(__file__),wall_s=time.monotonic()-start)
    (PUBLIC/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (PUBLIC/'per_bank.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(grouped=grouped,wall_s=result['wall_s'])),flush=True)


if __name__=='__main__':main()

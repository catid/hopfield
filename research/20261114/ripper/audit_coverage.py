"""Frozen 160-test validation family for the ideal augmentation theorem."""
import csv
import json
import numpy as np
import scipy
from scipy.stats import beta,binomtest
from run import ROOT,HERE,save,sha,now

OUT=ROOT/'artifacts/research/20261114/ripper/coverage'
PUBLIC=ROOT/'results/research/ripper'


def main():
    p=json.loads((HERE/'protocol-coverage.json').read_text());alpha=.05/160;tests=[];allcounts=[];basecurves=[];hashes={}
    for name,h in p['inputs'].items():assert sha(OUT/name)==h
    boards=np.load(OUT/'bases.npz')['solutions'].reshape(-1,9,9)
    for digit in range(9):
        assert np.all((boards==digit).sum(1)==1) and np.all((boards==digit).sum(2)==1)
        assert np.all((boards.reshape(-1,3,3,3,3).transpose(0,1,3,2,4).reshape(-1,9,9)==digit).sum(2)==1)
    for b in range(32):
        d=OUT/f'samples/base{b:02d}';r=json.loads((d/'receipt.json').read_text())
        assert r['status']=='complete' and r['missing_true_peer_events']==0 and r['draws']==32512
        assert r['numpy']=='2.3.4' and r['protocol_sha256']==sha(HERE/'protocol-coverage.json') and r['wrapper_sha256']==sha(HERE/'coverage_sample.py')
        assert sha(d/'counts.npz')==r['counts_sha256'];z=np.load(d/'counts.npz');extra=z['extra_edges'];hits=z['representative_pair_equalities']
        assert extra.shape==(256,9) and np.all(extra[:,0]==2106) and np.all(np.diff(extra,axis=1)<=0) and np.all(extra>=0)
        allcounts.append(extra);hashes[str((d/'receipt.json').relative_to(ROOT))]=sha(d/'receipt.json')
        for i,(orbit,q) in enumerate([('A',1/6),('B',1/9)]):
            k,n=int(hits[i]),32512;pv=float(binomtest(k,n,q).pvalue)
            lo=0. if k==0 else float(beta.ppf(alpha/2,k,n-k+1));hi=1. if k==n else float(beta.ppf(1-alpha/2,k+1,n-k))
            tests.append({'base':b,'kind':'orbit_equality','target':orbit,'successes':k,'trials':n,'observed_fraction':k/n,'null_probability':q,'raw_p':pv,'bonferroni_p':min(1.,160*pv),'lower':lo,'upper':hi,'interval':'two-sided simultaneous Clopper-Pearson'})
        for i,v in enumerate(p['variants_grid']):
            expectation=810*(5/6)**v+1296*(8/9)**v
            basecurves.append({'base':b,'variants':v,'mean_extra_edges':float(extra[:,i].mean()),'exact_graphs':int((extra[:,i]==0).sum()),'streams':256,'predicted_mean_extra_edges':expectation})
            if v not in (63,99,127):continue
            k,n=int((extra[:,i]>0).sum()),256;q=min(1.,expectation);pv=float(binomtest(k,n,q,alternative='greater').pvalue)
            lo=0. if k==0 else float(beta.ppf(alpha,k,n-k+1))
            tests.append({'base':b,'kind':'coverage_failure_upper_bound','target':str(v),'successes':k,'trials':n,'observed_fraction':k/n,'null_probability':q,'raw_p':pv,'bonferroni_p':min(1.,160*pv),'lower':lo,'upper':1.,'interval':'one-sided simultaneous Clopper-Pearson lower bound'})
    assert len(tests)==160;counts=np.stack(allcounts);curve=[]
    for i,v in enumerate(p['variants_grid']):
        values=counts[:,:,i];se=float(np.sqrt(values.var(axis=1,ddof=1).sum()/256)/32)
        curve.append({'variants':v,'predicted_mean_extra_edges':810*(5/6)**v+1296*(8/9)**v,'measured_mean_extra_edges':float(values.mean()),'descriptive_mean_se':se,'exact_graphs':int((values==0).sum()),'streams':8192,'measured_exact_fraction':float((values==0).mean()),'union_failure_upper_bound':min(1.,810*(5/6)**v+1296*(8/9)**v)})
    contrad=[t for t in tests if t['bonferroni_p']<.05]
    rec={'status':'validated','utc':now(),'ideal_theorem':'E(extra)=810*(5/6)^V+1296*(8/9)^V; failure_probability<=min(1,E(extra)) under independent uniform spatialgroupdraws.','bases':32,'streams':8192,'fresh_variants':1040384,'family_tests':160,'family_alpha':.05,'contradictions':contrad,'decision':'no_statistical_contradiction_in_frozen_family' if not contrad else 'statistical_contradiction_requires_investigation','minimum_raw_p':min(t['raw_p'] for t in tests),'minimum_bonferroni_p':min(t['bonferroni_p'] for t in tests),'curve':curve,'tests':tests,'base_curves':basecurves,'protocol_sha256':sha(HERE/'protocol-coverage.json'),'input_sha256':hashes,'software':{'numpy':np.__version__,'scipy':scipy.__version__},'scope':'Source-specific mathematical theorem with ideal sampling assumptions; deterministic pseudorandom experiment is separate empirical validation. No GPU searches.','limitations':['Absence of a statistical contradiction does not prove PRNG independence.','Bonferroni family is this160-test validation study, not all adaptive research hypotheses.','Curve standarderrors are descriptive; no additional unregistered expectation tests are claimed.','Different paircoverage events are dependent; theorem uses onlylinearity andunionbound.','Spatialaugmentation supplies Sudoku taskstructure.']}
    save(OUT/'audit.json',rec);save(PUBLIC/'coverage.json',rec)
    for name,rs in [('coverage_curve.csv',curve),('coverage_tests.csv',tests),('coverage_per_base.csv',basecurves)]:
        with (PUBLIC/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    summary=json.loads((PUBLIC/'summary.json').read_text());summary['augmentation_coverage']=rec;save(PUBLIC/'summary.json',summary)
    lines=['# H1m: fresh-stream validation of the augmentation coverage theorem','',
           f'The frozen validation ran **1,040,384 fresh spatial variants** across 32 fixed valid base boards and 8,192 PRNG streams. There were **{len(contrad)} statistical contradictions** in the prespecified 160-test Bonferroni family at level 0.05. The minimum adjusted p-value was **{rec["minimum_bonferroni_p"]:.6g}**. This empirical check is distinct from the exact ideal-sampling theorem.','',
           '| Spatial variants | Predicted extra edges | Measured mean extra edges | Exact graphs / 8,192 |','|---:|---:|---:|---:|']
    for c in curve:lines.append(f'| {c["variants"]} | {c["predicted_mean_extra_edges"]:.6g} | {c["measured_mean_extra_edges"]:.6g} | {c["exact_graphs"]} |')
    lines+=['','For each base, two representative-pair equality rates were tested against the predicted probabilities 1/6 and 1/9. Three per-base graph-failure rates were tested against the union bounds at 63, 99 and 127 variants. All 64+96 tests share a Bonferroni family; the CSV reports raw and adjusted p-values and simultaneous exact-binomial intervals.','',
            'The theorem uses independent uniform group draws. The experiment uses deterministic NumPy PRNG streams and the unchanged pinned source function. No statistical contradiction is evidence of implementation consistency, not proof of ideal independence. The mean-curve standard errors are descriptive; no extra unregistered expectation tests are asserted.','',
            '[Theorem and source assumptions](AUGMENTATION_COVERAGE_THEOREM.md) · [Frozen validation protocol](protocol-coverage.json) · [Sampling code](coverage_sample.py) · [Audit](audit_coverage.py) · [Tracked results](../../../results/research/ripper/coverage.json)','']
    (HERE/'REPORT-coverage.md').write_text('\n'.join(lines));print(json.dumps({'decision':rec['decision'],'contradictions':len(contrad),'min_adjusted_p':rec['minimum_bonferroni_p'],'curve':curve}))


if __name__=='__main__':main()

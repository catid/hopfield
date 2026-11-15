"""Post hoc descriptive exposure diagnostics; no decisions or tuning inputs."""
import csv
import hashlib
import json
import numpy as np
from run import ROOT,HERE,sha,save,now
from lr_selector_common import load_protocol


def main():
    p=load_protocol();out=ROOT/'artifacts/research/20261114/ripper/lr_selector'
    i,j=np.triu_indices(81,1)
    peer=(i//9==j//9)|(i%9==j%9)|((i//27==j//27)&(i%9//3==j%9//3))
    rows=[];hashes={};all_counts={}
    for seed in p['seeds']:
        receipt=json.loads((out/f's{seed}/receipt.json').read_text())
        for model in receipt['models']:
            path=ROOT/model['path'];assert sha(path)==model['sha256'];hashes[model['path']]=sha(path)
            z=np.load(path);m=z['joint_count'];c=z['equal_count'];s=z['support']
            assert m.shape==c.shape==s.shape==(3240,) and np.all(c<=m) and int(m.sum())==2220*300
            groups={'all_pairs':np.ones(3240,bool),'true_edges':peer,'nonedges':~peer,
                'retained_nonedges':~peer&s,'excluded_nonedges':~peer&~s}
            for name,mask in groups.items():
                vals=m[mask];all_counts.setdefault((model['arm'],name),[]).append(vals)
                row={'seed':seed,'arm':model['arm'],'group':name,'pairs':int(mask.sum()),
                    'total_joint_exposures':int(vals.sum()),'never_queried':int((vals==0).sum()),
                    'mean_joint_exposures':float(vals.mean()) if len(vals) else None}
                for q,v in zip(['min','q25','median','q75','max'],np.quantile(vals,[0,.25,.5,.75,1]) if len(vals) else [None]*5):row[q]=None if v is None else float(v)
                rows.append(row)
    pooled=[]
    for (arm,name),parts in all_counts.items():
        vals=np.concatenate(parts)
        row={'arm':arm,'group':name,'pair_bank_instances':len(vals),'never_queried':int((vals==0).sum()),
            'mean_joint_exposures':float(vals.mean()) if len(vals) else None,'total_joint_exposures':int(vals.sum())}
        for q,v in zip(['min','q25','median','q75','max'],np.quantile(vals,[0,.25,.5,.75,1]) if len(vals) else [None]*5):row[q]=None if v is None else float(v)
        pooled.append(row)
    report={'status':'validated_descriptive','utc':now(),'scope':'Post hoc descriptive mechanism analysis only; no confidence intervals, causal attribution, or changes to frozen H1ae endpoints.',
        'rows':rows,'pooled':pooled,'protocol_sha256':sha(HERE/'protocol-lr-selector.json'),'script_sha256':sha(__file__),'model_sha256':hashes,
        'interpretation_limit':'Final counts include exposures before and after pair exclusion. True/nonedge labels appear only in this postfit analysis and were unavailable to the selector.'}
    save(out/'exposure_diagnostic.json',report);save(ROOT/'results/research/ripper/lr_selector_exposure_diagnostic.json',report)
    with (ROOT/'results/research/ripper/lr_selector_exposure_diagnostic.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# H1ae: descriptive exposure diagnostic','',
        'This analysis was added after acquisition and while inference was finishing. It is descriptive and was not used to tune or change H1ae. It does not establish a causal mechanism.',
        '', '| Selector / bound | Retained nonedge-bank pairs | Never queried nonedge-bank pairs | Median exposures among retained nonedges | Mean exposures among retained nonedges |',
        '|---|---:|---:|---:|---:|','| BP AdamW: outside this study; not run | — | — | — | — |']
    for arm in p['arms']:
        retained=next(r for r in pooled if r['arm']==arm and r['group']=='retained_nonedges')
        nonedges=next(r for r in pooled if r['arm']==arm and r['group']=='nonedges')
        lines.append(f'| {arm} | {retained["pair_bank_instances"]:,} | {nonedges["never_queried"]:,} | {retained["median"]:g} | {retained["mean_joint_exposures"]:.2f} |')
    lines+=['','Counts pool 32 banks, each with 2,430 possible nonedges. They are pair-bank instances, not distinct pairs or independent test puzzles. Every arm has exactly 666,000 joint pair exposures per bank. Final counts include exposures before and after exclusion.',
        '', 'The LR selector leaves many nonedges with few or no observations. This is consistent with allocation concentrating on pairs that already accumulated positive evidence. The exposure control has no unqueried pairs and its residual nonedges have substantially more observations. True-edge and nonedge labels were used only in this postfit analysis, never by the acquisition policy.',
        '', '[Frozen study](LR_EVIDENCE_SELECTOR.md) · [Final inference report](REPORT-lr-selector.md) · [Diagnostic script](diagnose_lr_selector.py) · [Diagnostic data](../../../results/research/ripper/lr_selector_exposure_diagnostic.json)','']
    (HERE/'REPORT-lr-selector-diagnostic.md').write_text('\n'.join(lines))
    print(json.dumps(pooled))


if __name__=='__main__':main()

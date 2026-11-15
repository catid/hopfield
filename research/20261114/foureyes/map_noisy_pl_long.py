"""Freeze unique support/search jobs after all fixed H4s fits finish."""
import hashlib
import json
from noisy_pl_long import ROOT, OUT, PROTOCOL, sha, write, dependencies


def main():
    np,torch,CF=dependencies();p=json.loads(PROTOCOL.read_text())
    q=json.loads((OUT/'fit_queue.json').read_text());assert all(j['status']=='complete' for j in q)
    jobs=[];mappings=[];seen={};ineligible=[]
    for bank in range(p['banks']):
        dest=OUT/'fits'/f'b{bank:02d}';rec=json.loads((dest/'receipt.json').read_text())
        assert rec['protocol_sha256']==sha(PROTOCOL)
        for name,digest in rec['output_sha256'].items():assert sha(dest/name)==digest
        z=np.load(dest/'models.npz')
        for arm in p['arms']:
            if not rec['arms'][arm]['eligible']:
                ineligible.append(dict(bank=bank,arm=arm));continue
            support=z[f'{arm}_support'];digest=hashlib.sha256(support.tobytes()).hexdigest()
            if digest not in seen:
                job=f'm{len(seen):03d}';seen[digest]=job
                path=OUT/'models'/f'{job}.npz';path.parent.mkdir(exist_ok=True)
                np.savez_compressed(path,support=support)
                jobs.append(dict(job_id=job,bank=bank,arm=arm,n=p['n'],seed=p['search_seed'],threshold='frozen-support',
                            model_path=str(path.relative_to(ROOT)),model_sha256=sha(path),support_sha256=digest,status='pending'))
            mappings.append(dict(bank=bank,arm=arm,actual=seen[digest],support_sha256=digest,seed=p['search_seed']))
    write(OUT/'prepared.json',dict(protocol_sha256=sha(PROTOCOL),producer_sha256=sha(__file__),jobs=jobs,mappings=mappings,
          ineligible=ineligible,unique_searches=len(jobs),logical_cells=len(mappings),reuse='Same support, unit1/B0/tau0/ebar0, input cohort and seed0.'))
    print(json.dumps(dict(unique_searches=len(jobs),logical_cells=len(mappings),ineligible=ineligible)),flush=True)


if __name__=='__main__':main()

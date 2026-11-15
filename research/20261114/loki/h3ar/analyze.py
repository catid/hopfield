"""Export selected raw evidence and independently audit the full likelihood study."""
import csv
import shutil
from common import ROOT, SRC, OUT, frozen, read, save, sha, now
from verify_public import verify
PUBLIC = ROOT/'results/research/loki/h3ar'


def main():
    p = frozen(True); source = read(OUT/'source-selection.json')
    files = ['capacity-gate.json', 'study-result.json', 'source-selection.json', 'campaign.log']
    files += [f'candidate{i}/{name}' for i in range(56) for name in ('source-columns.npy', 'source-F.npy')]
    files += [f'b{i}/source-evidence.npz' for i in range(8)]
    files += [f'family{i}/{name}' for i in range(2) for name in ('true-columns.npy', 'true-F.npy', 'input-universe.npz')]
    files += [f"support{r['group']}/{name}" for r in source['supports'] for name in ('model.npz', 'capacity.npz')]
    for rel in files:
        path = PUBLIC/rel; path.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(OUT/rel, path)
    audit = verify(ROOT, False); g = read(PUBLIC/'capacity-gate.json')
    s = dict(status='validated', utc=now(), hypothesis='H3ar', **{k: g[k] for k in ('decision', 'rows', 'bank_decisions', 'support_results')}, aggregate_CPU_seconds=g['aggregate_cpu_seconds'], wall_seconds=g['wall_seconds'], GPU_seconds=0, distinct_full_supports=len(source['supports']), audit=audit, prior=p['candidate_prior'], likelihood=p['likelihood'], control=p['control'], protocol_sha256=sha(SRC/'protocol.json'), audit_source_sha256=sha(__file__), verifier_source_sha256=sha(SRC/'verify_public.py'), public_sha256={rel: sha(PUBLIC/rel) for rel in files})
    save(PUBLIC/'summary.json', s)
    with (PUBLIC/'capacity.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(s['rows'][0])); writer.writeheader(); writer.writerows(s['rows'])
    print(dict(decision=s['decision'], banks=s['bank_decisions'], audit=audit))


if __name__ == '__main__': main()

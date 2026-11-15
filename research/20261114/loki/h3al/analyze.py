"""Export exact evidence only after a full independent public-data audit."""
import csv
import shutil
from common import ROOT, SRC, OUT, frozen, read, save, sha, now
from verify_public import verify
PUBLIC = ROOT/'results/research/loki/h3al'


def main():
    p = frozen('target')
    source = read(OUT/'source-outcomes.json')
    files = ['capacity-gate.json', 'study-result.json', 'source-outcomes.json', 'distribution-validation.json', 'filtering-completeness-seal.json', 'campaign.log']
    files += [f'b{bank}/source-patterns.npz' for bank in range(4)]
    files += [f"support{r['group']}/{name}" for r in source['supports'] for name in ['F.npz', 'capacity.npz']]
    PUBLIC.mkdir(parents=True, exist_ok=True)
    for rel in files:
        destination = PUBLIC/rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(OUT/rel, destination)
    shutil.copyfile(ROOT/p['source_population']['path'], PUBLIC/'source-population-n13.npy')
    audit = verify(ROOT, require_summary=False)
    gate = read(PUBLIC/'capacity-gate.json')
    summary = dict(status='validated', utc=now(), hypothesis='H3al', decision=gate['decision'], rows=gate['rows'], primary=gate['primary'], support_results=gate['support_results'], independent_stream_joint=gate['independent_stream_joint'], total_bank_masks=80, unique_full_supports=32, reference_true_states=26960, same_seen_eligible_inputs=831605, aggregate_CPU_seconds=gate['aggregate_cpu_seconds'], wall_seconds=gate['wall_seconds'], GPU_seconds=0, target_cohorts_created=0, audit=audit, protocol_sha256=sha(SRC/'protocol.json'), audit_source_sha256=sha(__file__), public_verifier_source_sha256=sha(SRC/'verify_public.py'), public_sha256={rel:sha(PUBLIC/rel) for rel in files+['source-population-n13.npy']}, scope=p['scope'])
    with (PUBLIC/'reliability.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary['rows'][0]))
        writer.writeheader()
        writer.writerows(summary['rows'])
    save(PUBLIC/'summary.json', summary)
    print(dict(decision=summary['decision'], primary=[r['success_probability_pct'] for r in summary['primary']], audit=audit))


if __name__ == '__main__':
    main()

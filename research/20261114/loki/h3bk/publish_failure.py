"""Preserve the unsuccessful frozen run without converting it to an endpoint result."""
import shutil
from common import ROOT,SRC,OUT,frozen,read,save,sha,now
PUBLIC=ROOT/'results/research/loki/h3bk'


def main():
    frozen();PUBLIC.mkdir(parents=True,exist_ok=True)
    for path in sorted(OUT.rglob('*')):
        if path.is_file():target=PUBLIC/path.relative_to(OUT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    g=read(PUBLIC/'capacity-gate.json');assert g['status']=='failed_or_incomplete' and '4300 digits' in g['error'];save(PUBLIC/'summary.json',dict(status='incomplete',utc=now(),hypothesis='H3bk',primary_decision=None,error=g['error'],completed_group_receipts=len(g['completed_groups']),global_memo_states=g['global_memo_states'],global_canonical_split_attempts=g['global_canonical_split_attempts'],aggregate_CPU_seconds=g['aggregate_cpu_seconds'],wall_seconds=g['wall_seconds'],GPU_seconds=0,new_source_draws=0,new_oracle_labels=0,partial_results_usable=False,integrity_only_no_optimality_audit=True,protocol_sha256=sha(SRC/'protocol.json'),public_sha256={str(path.relative_to(PUBLIC)):sha(path) for path in sorted(PUBLIC.rglob('*')) if path.is_file() and path.name not in ['summary.json','README.md']}))
    (SRC/'REPORT.md').write_text('# H3bk: incomplete after exact-rational serialization error\n\nThe frozen run is incomplete. Its final exact aggregate numerator or denominator exceeded Python\'s default 4,300-decimal-digit integer-to-string conversion limit. All 125 group receipts were written, but the whole-study result could not be serialized; no primary decision is reported. This is an implementation/output error, not an infeasibility or a failed 10% scientific hypothesis.\n\nThe original source and protocol remain unchanged. All partial catalogs, proofs, receipts, the traceback and failed whole-study record are retained. Their integrity can be checked, but they are not presented as an independently validated scientific result. The run used 1.864954 aggregate CPU seconds / 1.873281 wall seconds, 8,558 memo states and 50,439 split attempts. Zero new draws, labels or GPU work.\n\nA corrected serialization environment must receive a new frozen study directory and preserve the same threshold and scope. Run `upstream/.venv/bin/python research/20261114/loki/h3bk/verify_incomplete.py --root SNAPSHOT` for integrity-only verification.\n');(PUBLIC/'README.md').write_text('# H3bk: incomplete\n\n[Failure report](../../../../research/20261114/loki/h3bk/REPORT.md). Exact-rational serialization exceeded Python\'s decimal conversion guard. No scientific endpoint decision; frozen code and all partial evidence retained.\n')


if __name__=='__main__':main()

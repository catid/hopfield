"""Freeze H3x before source set-cover computation."""
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / 'research/20261114/loki/h3x'
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert not (SRC / 'protocol.json').exists()
    assert not (ROOT / 'artifacts/research/20261114/loki/h3x/source-selection.json').exists()
    source = ROOT / 'artifacts/research/20261114/loki/h3t/solutions-n12.npy'
    protocol = dict(id='H3x-complete-source-minimum-D4-rule', frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        hypothesis='Under supplied closed-world source completeness and D4/minimum-cardinality priors, the unique minimum finite-orbit generator set that rejects all absent inferred-line-compatible N12 boards preserves all176source positives and transfers to the exact N13 queen-plus-camel attack graph.',
        antecedent='H3w permutation search approaches the filtered support ceiling, but308false target attack pairs persist. This adaptive follow-up uses known complete source coverage to distinguish redundant global absences from the smallest local rule set.',
        source=dict(path=str(source.relative_to(ROOT)), n=12, positives=176, completeness='Explicitly supplied: this bank contains every valid source board. Only under this closed-world assumption may absent line-compatible boards be treated as negative.'),
        inference=dict(offsets='Canonical half-plane dr=0,dc>0 or dr>0,dc in[-11,11]. Count occupied cell pairs in all176positive source boards.',
            lines='Unchanged primitive-normal selector: a,b in[-2,2],gcd(abs(a),abs(b))=1, canonical positive first nonzero entry; retain if at least3member offsets have zero source count and no member offset has positive count. Require inferred row andcolumn lines to justify complete row-wise q12 enumeration.',
            universe='Enumerate every q12placement satisfying every inferred line. With inferred row exclusion, q12implies one object perrow. Positive iff exactly present in complete source bank; every absent board isnegative.',
            candidates='All D4orbits fully contained in the off-line source-zero offsets, under sign flips and coordinate exchange. Canonical generator is sorted absolute displacement pair. No geometry or target labels enter selection.',
            objective='Exhaustively enumerate every subset of candidate orbits. A selected orbit rejects a board iff any occupied pair has a displacement in that orbit. Minimize number of selected orbitgenerators while rejecting every source negative and preserving all176positives. Preserve ALLminima; no tie selection.',
            certificate='Save full universe, positive flags, all orbit incidences, every subset cardinality/coverage and first uncovered negative witness for each non-cover. Independently re-enumerate via attack-bitmask propagation, recompute pairwise incidence literally, and check every subset/minimum/witness.'),
        primary=dict(metric='Unique cardinality-minimum source cover and exact N13 graph extension',
            support='Exactly one source minimum; its unchanged finite offsets plus inferred infinite lines give zero false and zero missed N13 attack pairs.',
            falsify='No finite cover, more than one minimum, or unique minimum with any false/missed target pair. Multiple minima are all diagnosed with best/worst errors; never select a tie using target geometry.',
            inconclusive='Timeout, incomplete exhaustive certificate, or failed source validation. Retain partial evidence; no hidden subset truncation, budget extension or scientific retuning.'),
        target_diagnostic='Only after source-result hash is sealed: extend the selected finite offsets unchanged and inferred lines toN13. Compare all target unordered cell pairs with privileged queen plus camel(1,3)/(3,1) attacks. All target comparisons are diagnostics after source selection.',
        model='For every minimum use J=float32(F),h=float32zeros169,F=int32symmetric169square,unit=float64(1),q=int64(13). No fit or coefficient tuning.',
        identity_mapping=dict(reference='H3w oracle-permutation', reference_protocol_sha256='93b34dfb26e26ef9a1c7b9008fe23e0f6c751d542b8252810ed3e26b4c385706',
            reference_model_sha256='ac29a077b07629a74079c0b42dc1756b01af17c3c1e01680bba8414c1875ef7e',
            backend_sha256='1bb03dc45e735347226c177fab2c237c42fe69229a1a8bd6d93b32bb90f60861',
            data_sha256='4fc37451c20a8af8e004c17faa7cc96c88939100dd134b07f66445136c4ea163',
            canonical_arrays='Compare dtype,shape and C-order bytes for J,h,F,unit,q. Also record whole NPZ hash, but ZIP-container metadata does not alter scientific identity.',
            settings=dict(q=13, unique_inputs=4096, chains=20, cap=20000, seeds=[0,1,2], end_temperature_ratio=.01, block=65536, unit=1., kernel='permutation',
                initialization='uniform unused columns in nongiven rows', proposal='symmetric column transposition in two nongiven rows', certificate='exact integer occupied F-edge count zero including given-given edges'),
            reuse='If arrays,backend,calibration,input and seedsettings are identical, map all three audited H3w oracle outcomes. Label as inherited identity-derived evidence, with zero new search runs and no independent confirmation. If nonidentical, report structural results before any new GPUsearch budget.'),
        budget=dict(source_cpu_seconds=60, source_wall_seconds=60, selection_and_independent_certificate_share_budget=True, gpu_search_jobs=0, optimizer_fits=0,
            policy='SIGALRM60seconds bounds source selection plus independent certificate. No subset-count truncation. Targetdiagnostics occur afterward under a separate stated runtime, without altering source choices.'),
        limits=['Closed-world completeness is supplied and substantially stronger than positive examples alone.',
                'D4 symmetry, translation-invariant finite offsets, inferred-line extrapolation and minimum-cardinality are explicit inductive priors.',
                'One complete source universe; source uniqueness is only within this declared candidate class and size.',
                'Success would be a symbolic derivative with added priors, not paper reproduction or general positive-only identifiability.',
                'H3w outcome mapping reuses previously observed target results and is not an independent search experiment.'],
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [SRC/'solve_source.py', SRC/'verify_source.py', Path(__file__)]},
        source_data_sha256={str(source.relative_to(ROOT)):sha(source)})
    (SRC / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print(json.dumps(dict(status='frozen_before_cover_computation', protocol_sha256=sha(SRC/'protocol.json'))))

if __name__ == '__main__':
    main()

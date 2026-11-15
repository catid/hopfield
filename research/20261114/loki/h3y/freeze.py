"""Preregister the full eighteen-case structural stress grid."""
import datetime as dt
from pathlib import Path
from common import ROOT,SRC,OUT,sha,save

def main():
    assert not(SRC/'protocol.json').exists()and not(OUT/'generation.json').exists()
    families={'f0':[[1,4]],'f1':[[2,3]],'f2':[[2,5]],'f3':[[3,4]],'f4':[[1,3],[2,3]],'f5':[[1,4],[2,3]]}
    sizes=[8,10,12];cases=[dict(id=f'{f}-n{n}',family_id=f,n=n,target_n=n+1)for f in families for n in sizes]
    p=dict(id='H3y-prespecified-multifamily-minimum-D4-transfer',frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        hypothesis='The complete-source minimum-D4-rule recipe identifies one unique minimum with exact N+1attack support in ALL18prespecified family/size cases.',
        antecedent='H3x uniquely recovered one camel orbit atN12given completeness and a minimum-description prior, while16different full source-equivalent rule sets remained without that prior. H3ytests this recipe across a fixed grid rather than choosing a successful family.',
        families=families,source_sizes=sizes,cases=cases,
        family_definition='Every family is ordinary row/column/diagonal queens plus the listed D4leaperorbits under sign changes and coordinate exchange. All six combinations and all three source sizes are fixed before any source enumeration. No family replacement, source-size screening or omitted empties.',
        source_generation='Separate privileged source-generation process. Enumerate every ordinary queen board with independent line-coordinate-set and attack-bitmask recursions. Filter each complete universe by its fixed family and compare against a separate family-constrained attack-bitmask enumeration. Save all source columns and completeness receipts. No target enumeration.',
        learner_inputs='Only source columns, source size, and explicit completeness are used by source selection. Family geometry is used exclusively by the separate source generator and later target diagnostic, never in candidate selection or objective.',
        inference='Same H3x primitive-line selector: canonical coprime normals in[-2,2],atleast3zero member offsets and no observed member. Require inferred row/column axes. Enumerate every q=Nboard satisfying inferred lines. Absent boards are negatives solely under supplied completeness. Candidate finite rules are all D4orbits wholly contained in off-line zero-count offsets.',
        objective='Enumerate all orbit subsets at cardinality0,then1,etc until the first cardinality admitting a cover of all negatives while preserving all positives. Finish EVERYsubset at this first successful cardinality and preserve EVERYminimum. This is an exhaustive minimality/uniqueness certificate without enumerating larger irrelevant supersets.',
        certificate='Save complete source/universe coordinates, positive flags, orbit incidence, every checked lower/equal-cardinality subset with coverage and first uncovered negative witness. Independent verifier reconstructs source counts,lines,candidates and literal pair incidence, validates the independent universe and verifies every lower/equal-size subset,witness and all minima. No candidate-count cap or hidden subset truncation.',
        cache='Reuse source line-compatible universe only under SHA256of canonical(N,inferred-line-tuple), with separate independently generated universe, exact array hashes,constructor hash and equality receipt. Cache does not depend on target geometry or outcomes. Construction on first use counts within that case budget; cached uses retain provenance.',
        primary=dict(unit='All18fixed cases, not just eligible successful ones',
            support='All18have nonempty complete sources,eligible row/column inference,exhaustive source certificates,oneunique minimum,andzero false/missed N+1attack pairs.',
            falsify='Any empty source,ineligible row/column inference,no source cover,ambiguous minimum,or unique minimum with any false/missed target pair falsifies the universal claim. Still finish/report everycase.',
            incomplete='A timeout or failed validation remains incomplete and is never dropped. If no completed case falsifies but at least one is incomplete,the overall decision is inconclusive; an observed falsification remains a falsification even if other cases are incomplete.'),
        source_sealing='Finish all18source outcomes and seal every source-result hash in one manifest before any N+1target truth comparison. Empty/failed/incomplete cases remain in that manifest.',
        target_diagnosis='After the all-case source seal,extend each minimum finite orbit unchanged and inferred lines toN+1. Compare every unordered target cell pair to the supplied family oracle. Diagnose ALLminima and report best/worst false/missed counts; never choose a target-favored tie. No target-solution enumeration is needed for this structural endpoint.',
        budget=dict(generation_cpu_seconds=120,generation_wall_seconds=120,source_cpu_seconds_per_case=60,source_wall_seconds_per_case=60,
            source_cases=18,maximum_aggregate_source_cpu_seconds=1080,concurrent_source_processes=2,new_gpu_search_jobs=0,new_optimizer_fits=0,
            enforcement='Generation and each inference+certificate subprocess useSIGALRM. TwoCPUprocesses overlap independent cases. On timeout retain incomplete evidence,do not extend budget or replace a family.'),
        limits=['Supplied closed-world completeness and D4/minimum-description priors are stronger information than positive examples alone.',
                'Uniqueness concerns only this finite candidate class; completeness need not identify one rule set without the minimum-description prior.',
                'Eighteen fixed synthetic cases are a structural stress test,not independent random draws from a rule-family population.',
                'No GPUsearch or optimizer fitting is performed. Structural exactness is separate from search performance and paper reproduction.',
                'All target comparisons follow the global source seal; ambiguous source minima remain ambiguous even if only one transfers.'],
        source_sha256={str(path.relative_to(ROOT)):sha(path)for path in [SRC/'common.py',SRC/'generate.py',SRC/'run_source.py',SRC/'verify_source.py',Path(__file__)]})
    save(SRC/'protocol.json',p);print('FROZEN',sha(SRC/'protocol.json'))

if __name__=='__main__':main()

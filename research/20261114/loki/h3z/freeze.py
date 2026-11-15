"""Freeze pooled-source mechanistic follow-up before solving any pooled case."""
import datetime as dt
from source import ROOT,SRC,OUT,OLD,sha,read,save

def main():
    assert not(SRC/'protocol.json').exists()and not(OUT/'source-seals.json').exists()
    files=[OLD/f'sources/f{k}-n{n}.npy'for k in range(6)for n in[8,10]]
    files.extend(path for path in(OLD/'cache').rglob('*')if path.is_file())
    p=dict(id='H3z-pooled-N8-N10-source-completeness',frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        hypothesis='Pooling complete N8andN10source banks makes ALLsix fixed H3yfamilies identify unique minimumD4rules with exact N11attack support.',
        scope='Adaptive mechanistic follow-up to known H3yN11targetdiagnostics,not fresh target validation. Prior H3y structural/census results remain unchanged.',
        family_ids=[f'f{k}'for k in range(6)],source_sizes=[8,10],target_n=11,comparison='Six already published H3yN10-only source outcomes; no baseline rerun.',
        families='Exactly all six H3yfamilies: {(1,4)},{(2,3)},{(2,5)},{(3,4)},{(1,3),(2,3)},{(1,4),(2,3)}. No selection or replacement.',
        source_information='Source banks are separately COMPLETE atN8andN10. An emptyN8positive bank still labels its entire inferred-line-compatible universe negative. This is supplied closed-world information,not inferred from finite positive samples.',
        offsets='Pooled occurrence counts over canonical offsets representable in at least one source size,up toN10extent. A displacement not representable atN8 supplies no N8occurrence/absence evidence and has no N8incidence. Its status can still be learned fromN10where representable.',
        lines='Unchanged primitive-normal rule fromH3x/H3y on pooled counts: canonical coprime normals in[-2,2],atleast3zero member offsets and no positive member. Require row/column lines. The complete compatible universe is separately enumerated for each source size.',
        objective='Candidate finiteD4orbits are entirely off-line pooled-zero offsets. Minimize orbit-generator count subject to preserving ALLpositives and rejecting ALLnegatives from BOTHcomplete source universes. Enumerate every lower-cardinality subset and the entire first covering layer; preserve every minimum and all noncover witnesses.',
        validation='Independent pooled count/line/candidate reconstruction and literal pair incidences for both sizes; exact all lower/equal-layer coverage/witness/minimum checks. Cached universe use requires exact(N,lines),array hashes,independent array equality and frozen constructorhash fromH3y.',
        sealing='Seal all six source outcomes before reusing any N11truth diagnostic. No target-favored tie selection. Diagnose every minimum and record best/worst graph errors.',
        primary=dict(support='All6cases have one unique minimum and zero false/missed N11attackpairs.',
            falsify='Any source ineligibility,ambiguous minimum or unique wrong graph falsifies the universal hypothesis; retain all6cases.',
            incomplete='Timeout/validation failure remains incomplete. Support requires all6complete; observed falsification takes precedence over incomplete othercases.'),
        reuse='If a unique transferred graph equals an already audited H3ygraph at identicalq11,its existing complete-state census may be explicitly identity-mapped. No new census/search implied; no independent confirmation.',
        budget=dict(source_cpu_seconds_per_case=60,source_wall_seconds_per_case=60,source_cases=6,aggregate_cpu_envelope=360,concurrent_source_processes=2,new_source_generation=0,new_gpu_search_jobs=0,new_optimizer_fits=0),
        limits=['Known-target adaptive mechanistic follow-up; no fresh empirical validation.',
                'Supplied source completeness,D4,translation invariance,line extrapolation and minimum-description remain strong priors.',
                'Six deterministic fixed families are not random independent population draws.',
                'Structural inference and any inherited certificate counterexample are distinct from measured search performance.'],
        source_sha256={str(path.relative_to(ROOT)):sha(path)for path in[SRC/'source.py',SRC/'freeze.py',SRC.parent/'h3y/common.py']},
        source_data_sha256={str(path.relative_to(ROOT)):sha(path)for path in files},
        reference_summary_sha256=sha(OLD/'summary.json'),reference_census_sha256=sha(OLD/'q11-census.json'))
    save(SRC/'protocol.json',p);print('FROZEN',sha(SRC/'protocol.json'))
if __name__=='__main__':main()

"""Freeze H3aa before source gates, fresh inputs or initialization draws."""
import datetime as dt
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3aa';OUT=ROOT/'artifacts/research/20261114/loki/h3aa';BASE=ROOT/'artifacts/research/20261114/loki'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not(SRC/'protocol.json').exists()and not(OUT/'data.npz').exists();OUT.mkdir(parents=True,exist_ok=True)
    source=[SRC/f'{name}.py'for name in['build_adapters','arbitrary_backend','permutation_backend','initialization','prepare','gate','driver','preflight','freeze']]
    source.extend([SRC.parent/'h3v/backend.py',SRC.parent/'h3w/backend.py',SRC.parent/'h3w/protocol.json',SRC.parent/'analyze.py'])
    data=[BASE/'h3w/models/bp-adamw-permutation/model.npz',BASE/'h3w/data.npz',BASE/'h3t/input-universe.npz',BASE/'h3t/solutions-n13.npy',BASE/'h3u/source-design.npz',BASE/'h3u/models/bp-adamw-grouped/model.pt',BASE/'h3w/kernel-validation-cpu.json',BASE/'h3w/kernel-validation-cuda.json']
    p=dict(id='H3aa-initialization-proposal-triangle',frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        hypothesis='Conditional on byte-identical permutation initial states and unchanged frozen positive BPAdamWenergy/temperature,permutation proposals improve certified valid success by atleast50percentage points over arbitrary-cell proposals,withzero wrongstops.',
        antecedent='H3w jointly changed initialization and proposals and gained67.22points. H3aa separates these interventions using a third arm and explicit shared initialstates. No H3wtrajectory is reused.',
        arms=[dict(arm='arbitrary-init-arbitrary-swaps',initialization='arbitrary',kernel='arbitrary'),
              dict(arm='permutation-init-arbitrary-swaps',initialization='permutation',kernel='arbitrary'),
              dict(arm='permutation-init-permutation-swaps',initialization='permutation',kernel='permutation')],
        queue_order=['permutation-init-permutation-swaps','permutation-init-arbitrary-swaps','arbitrary-init-arbitrary-swaps'],
        model='Same frozen H3wpositive BPAdamWJ,h,F,q,unitbytes in allarms. No refit,supportedit,weightchange or recalibration. Fretains308false targetattackpairs and thus limits certifiedprimary.',
        model_sha256=sha(BASE/'h3w/models/bp-adamw-permutation/model.npz'),unit=3.0367188453674316,
        initialization='Generate all input×chain initialstates onceperkind/searchseed using a dedicated CPUtorch.Generator withseed2026112600+searchseed. Arbitrary:uniform nongiven-cell subset reachingq13. Permutation:uniform unused-column assignment to nongiven rows. Save packed arrays and canonical unpacked-byte hashes. Bothprimaryarms reference the exactsame saved permutationfile. Initialization neverdepends on priorblock searchdraws.',
        rng='All three arms reset a separate proposal torch.Generator on the GPU to searchseed0/1/2. This split-stream convention applies to everyarm. No historical trajectory equivalence is claimed. Different proposalshapes/compactions consume different randomstream portions; shared initialstates are exact.',
        transitions='Copy frozen H3varbitrary andH3wpermutation backends; onlyadd optional explicit initialstates. Preserve the complete per-step transition loop AST exactly. Arbitrary removesone occupied nongiven cell and fillsone emptycell; permutation swaps columns of twonongiven rows. Bothuse frozen Metropolis acceptance/schedule and exactFconflict updates.',
        search=dict(q=13,unique_inputs=4096,chains=20,cap=20000,seeds=[0,1,2],block=65536,end_temperature_ratio=.01),
        temperature='T(k)=3.0367188453674316*0.01**(k/19999),k0..19999. Unit remains sourcecalibrated,including row/column contributions even when permutationstatesmake themzero.',
        certificate='Exact integer occupied F-edge countzero,including given-given pairs. AllFweightspositive,zeroelsewhere; allstatesq13. Abstain if none of20chains yields independentlyvalid stoppedcompletion.',
        cohort='After source-only gatespass,reserve4096uniform unique validN13one/two/three-clue inputs usingseed2026112514. Exclude unionofH3wprior_target_test_keys andH3wtest_keys. Allthreearms share freshcohort; no targetoutcomes inspected beforefreeze.',
        primary=dict(arm='permutation-init-permutation-swaps',control='permutation-init-arbitrary-swaps',required_gain_pp=50.,wrong_stops=0,
            support='Paired-input bootstrap one-sided95%lowergain>=50pp,zero wrongstops in permutation-proposalarm,and numericalgatepasses.',
            falsify='One-sided95%uppergain<50pp ORany wrongstop in permutation-proposalarm.',
            otherwise='Inconclusive; a numericalgatefailure prevents support but does not erase an observed falsification.'),
        statistics='2000paired bootstrapdraws,RNG20261114,allthreesearchseeds retained inside each uniqueinputcluster. One inherited fitted model/sourceuniverse; searchseeds are not independentlearningreplicates.',
        secondary='Permutation-init/arbitrary-swaps minus arbitrary-init/arbitrary-swaps is the descriptive initializationeffect. Also report original/certificate-first fallback accuracy,coverage,exact supportcapacity,proposal/accept counts,capped states,and measured searchseconds/proposal. No post-hocgain threshold forsecondaryeffects.',
        gates='Before freshdata: exact legacy outputs ANDalltracebytes when explicitinitialization omitted,withmultipleblocks; AST-identical transitionloops; explicitshared permutationstarts inbothkernels; CPU/GPUfull-energy,fields,integercounts,q/givens audits; source-onlyN12pilot128inputs×20chains×2000steps for allthreearms. Existing exhaustivepermutation/uniforminitialization/detailed-balance gates remain hashpinned because the transitionloopsareunchanged.',
        preflight='Independently reconstruct freshinputdraw/exclusion,verifyallmodelbytes,and regenerateall six complete saved initializationarrays from dedicatedCPUstreams. Workers hash-check preflight artifacts. Every job storesinitialfile/unpacked hashes; primarypair hashes mustmatchfor eachseed.',
        audit='Audit all737280savedstates against exhaustivetargetmembership,givens,q,integerFconflicts,selectors,coverage andcapcounters. Permutation-proposaloutputs additionallypreserve allrows/columns. Compare running versus independentlyrecomputedfloat64energies; maxerror>.1blocks support. Preserveallcapreturns.',
        budget=dict(search_gpu_minutes_envelope=20,shared_wall_seconds=600,concurrent_gpu_workers=2,logical_search_jobs=9,optimizer_refits=0,identity_mappings=0,
            enforcement='Shared absolute10minutedeadline plusSIGALRM boundsaggregateworkerwallat20GPUminutes. Retaincompleted evidence on timeout; no budget/cap/recipe retuning. Source-only pilots precede targetsearch and their runtimesare reportedseparately.'),
        limits=['This isolates the proposal intervention conditional on shared permutationinitialization; the descriptive middle-minus-first comparison isolates initialization conditional on arbitraryproposals.',
                'Proposal types change workpermove and accessible states. Equalcaps do not imply equalFLOPs or equalwalltime; no automatic speedupclaim.',
                'Fsupport,energy andtemperature are unchanged; source-globalfalseexclusions and their capacityceilingremain.',
                'Split randomstreams mean no historicalH3woutcomeidentity. All9targetsearchesarefresh.',
                'One inherited sourcefit and sourceuniverse; repeated searchseeds are not independent optimizerreplicas.'],
        source_sha256={str(x.relative_to(ROOT)):sha(x)for x in source},data_sha256={str(x.relative_to(ROOT)):sha(x)for x in data})
    (SRC/'protocol.json').write_text(json.dumps(p,indent=2)+'\n');print('FROZEN',sha(SRC/'protocol.json'))
if __name__=='__main__':main()

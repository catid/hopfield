"""Freeze source-only compute calibration and prospective matched-time test."""
from pathlib import Path
from common import ROOT,SRC,OUT,OLD,BASE,sha,save,now
def main():
    assert not(SRC/'protocol.json').exists()and not(OUT/'calibration-data.npz').exists();OUT.mkdir(parents=True,exist_ok=True)
    sources=[SRC/f'{name}.py'for name in['common','calibration','prepare','driver','freeze']]+[SRC.parent/f'h3aa/{name}.py'for name in['initialization','arbitrary_backend','permutation_backend','gate']]+[SRC.parent/'h3aa/protocol.json',SRC.parent/'analyze.py']
    data=[OLD/'model.npz',OLD/'data.npz',OLD/'gate-cpu.json',OLD/'gate-cuda.json',BASE/'h3t/solutions-n12.npy',BASE/'h3t/solutions-n13.npy',BASE/'h3t/input-universe.npz']
    p=dict(id='H3ab-source-calibrated-matched-compute-proposals',frozen_utc=now(),
        hypothesis='With identical positive BPAdamWenergy,source temperature andpermutation starts,permutation proposals gain atleast50percentage points certifiedsuccess at source-calibrated matched GPUsolve time,withzero wrongstops.',
        antecedent='H3aa isolated a66.36point proposal gain under equal20000proposal caps but permutation consumedmoretime. H3abseparates practical quality from this compute mismatch.',
        arms=['arbitrary','permutation'],model_sha256=sha(OLD/'model.npz'),unit=3.0367188453674316,
        frozen_model='Unchanged H3aapositive BPAdamWJ,h,F,qandunit. Same308falsepairs; nofit,supportchange,reweighting orsourcecalibrationchange.',
        calibration_inputs='4096uniqueN13-shaped inputs derived ONLYfrom176completeN12sourceboards. Embed each sourcecell(r,c)at13*r+c,enumerate unique1/2/3clue subsets,then draw2/205/3889unique subsets bycluecount usingRNG2026112602. No N13targetbanks,labels,heldoutkeys or targetaccuracy accessed during calibration. Exclude everycalibrationkey fromlaterfreshcohort.',
        initialization='Reuse frozen H3aaCPUinitializer: samecomplete saved permutationarray for bothkernels at everyinput/searchseed; initialization seed2026112600+searchseed. Calibration streams101/102/103; targetstreams0/1/2. ProposalGPUgenerators reset separately to correspondingsearchseed.',
        calibration=dict(reference_kernel='arbitrary',reference_cap=20000,permutation_grid=[4000,8000,12000,16000],streams=[101,102,103],physical_gpus=[0,1],timings_per_configuration=6,
            order='EachphysicalGPUrunsallthree streams and everyconfiguration. Configurationorder reverses when(gpu+stream_index)isodd,balancing ascending/descendingorder acrossdevices/streams.',
            timing='SynchronizeGPUbefore/after complete solve-call; elapsedincludes initialtransfer and returnedCPUarrays. No targetvalidityscoring. Record proposals/accepted/self-stopped counts for forwardprogress,never targetaccuracy.',
            warmup='Each calibrationworkerphase first runs BOTHkernels for32proposals on thefullsource-derived4096×20shape. Checkq/givens,sampledexactconflicts,and permutationinvariants. Warmup time counts within totalGPUworkerbudget but not timed solve-trial medians.',
            selection='Choose permutationcap minimizing abs(medianpermtime/medianarbitrarytime-1); ties choose smallercap. Accept if within5%. Ifnonequalifies,one deterministic linear interpolation in thefirst adjacentascending-cap pair bracketingreference time,rounded to nearest128proposals with ties-to-even; measureits same6trials. No extrapolation. If stilloutside5%,no bracket,or roundedcap repeatsanexistingcap,declare sourcecalibrationinconclusive anddonotlaunch targetsearch. No additionalretuning.',
            uncertainty='Report all6timings,range,device medians and pairedsource-stream bootstrap5/95ratio endpoints using2000draws,RNG2026112604,retainingbothphysicalGPUsinsideeachstream. Sixmeasurements are3searchstreams×2devices,not6independentlearningreplicates.',
            freeze='Seal selectedfixedcaps andcomplete timingreceipt BEFOREreserving freshtargetcohort.'),
        target=dict(unique_inputs=4096,input_seed=2026112515,chains=20,seeds=[0,1,2],block=65536,q=13,end_temperature_ratio=.01,
            exclusion='Union ofH3aaprior_target_test_keys,H3aatest_keys,andallcalibrationinputkeys; uniformdrawfromremainingvalid1/2/3-clueN13universe.',
            device_assignment='GPU0:arbitraryseed0,permutationseed1,arbitraryseed2.GPU1:permutationseed0,arbitraryseed1,permutationseed2.',
            warmup='Beforeeach of6timed targetjobs,run anunscored fullshape32-proposal warmup withproposal seed100and thejobinitialarray; no targetaccuracy iscomputedfromwarmups. Sixmeasuredsearches plus6unscoredwarmups; alltimecounts towardstudybudget.',
            timing='Same synchronized complete solve-call scope ascalibration. Exclude valid-output scoring/serialization fromsolve timingsbutinclude them in aggregateworkerbudget.'),
        temperature='T(k)=u*0.01**(k/(cap-1)),u=3.0367188453674316. Endpointtemperaturesunchanged; cap-specific annealingduration is explicitly partof each matched-computerecipe.',
        certificate='Exactinteger zero occupiedFedges,including given-given edges; q13always. Primary abstains if no independentlyvalid stoppedcompletion among20chains.',
        primary=dict(required_gain_pp=50.,source_time_match_tolerance=.05,target_time_match_tolerance=.10,wrong_stops=0,
            quality='Paired-input bootstrap gain permutation-minus-arbitrary. Supportquality iff one-sided95%lower>=50pp,zero wrongstops,and numericalgatepasses; falsifyquality iff upper<50pp ORanywrongstop; otherwiseinconclusive.',
            matched_compute='Supportmatchedcompute onlyif source mediansmatchwithin5%,fresh target median solve-time ratio matcheswithin10%,andqualityissupported. If targettimegatefails,matchedcomputeisinconclusive regardlessquality; reportqualityandfreshcostsseparately withoutretuning. Anywrongstopfalsifiesthezero-wrongstopclaim.',
            statistics='2000pairedinputbootstrapdraws,RNG20261114,all3searchseedsretainedinsideeachinputcluster. Oneinheritedfit; timingandsearchseedsarenotlearningreplicates.'),
        diagnostics='Original/certificate-first fallbackaccuracy,coverage,exactFcapacity,allcapreturns,proposals/accepts,per-proposal cost,source timing uncertainty,freshmediantimesandphysicalGPUassignment. Auditall491520timedtargetstates;maxrunningenergyerror>.1preventsqualitysupport.',
        gates='Frozen H3aabackends/initializer andpassedCPU/CUDAgates are hashpinned. Sourcefullshape warmupsverifyarray/counterinvariants beforetiming. Targetpreflight checksfrozenselection,sourceexclusion,allvaliditycounts andregeneratessharedinitials.',
        budget=dict(whole_study_gpu_minutes=20,whole_study_wall_seconds=600,concurrent_gpu_workers=2,grid_timing_trials=30,maximum_refinement_trials=6,measured_target_searches=6,
            includes='ALLcalibrationgrid/refinement workers,warmups,andfresh targetworkers. One shareddeadline startswithfirstcalibrationGPUworker; no freshdeadlinefortargetphase. Actualaggregateworkerwallmuststay<=1200s. CPUsourceinputpreparation precedesdeadline; noGPUworkthere.',
            on_expiry='Retaincompleted trials/results andreportincomplete; do notextendtime,dropfailedtrial,changegrid,retuneontargetorclaimmatchedtimewithoutgate.'),
        limits=['Samefixed support/weights but cap-specific schedule duration; this tests declaredmatched-computerecipes,notpureproposalcountscaling.',
                'Source-derived target-shaped timing inputs can differ in stoppingdifficulty fromfresh targets; the realized10%gate mayfail.',
                'Alltargetcostsreportedwithoutretuning; sourcecalibrationmatchingdoesnotguaranteetargetmatching.',
                'OneinheritedBPAdamWfit andsourcebank; search/timingrepeatsarenotindependenttrainingdraws.',
                'No targetaccuracyin sourcecountselection; knownhistoricalH3aaaccuracydoesnotenterselectioncode.'],
        source_sha256={str(path.relative_to(ROOT)):sha(path)for path in sources},data_sha256={str(path.relative_to(ROOT)):sha(path)for path in data})
    save(SRC/'protocol.json',p);print('FROZEN',sha(SRC/'protocol.json'))
if __name__=='__main__':main()

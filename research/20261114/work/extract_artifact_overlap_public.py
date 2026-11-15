"""Portable complete-board/flag replay evidence plus immutable full-file hashes."""
import json
import numpy as np
import study as S
import fast_npz as Z
from artifact_overlap_study import DEST

KEYS=['predictions','iterations','stopped','finite','minimum_flip_margin64','block_native_cap','block_cap','hidden_good']


def snapshot(directory,expected):
    arrays={key:[] for key in KEYS}
    for job in expected['jobs']:
        path=directory/job['id']/'outputs.npz';assert S.sha(path)==job['files_sha256']['outputs.npz']
        with np.load(path) as z:
            for key in KEYS:arrays[key].append(z[key])
    return {key:np.stack(values) for key,values in arrays.items()}


def main():
    a=json.loads((DEST/'audit.json').read_text());assert a['status']=='passed';p=json.loads((DEST/'protocol.json').read_text());expected=p['source_cohort'];target=DEST/'public-fixtures';target.mkdir(exist_ok=True);source=S.OUT/'loop80_confirmation';reference=snapshot(source/'runs',expected);ds=expected['dataset'];assert S.sha(S.ROOT/ds['path'])==ds['sha256']
    with np.load(S.ROOT/ds['path']) as z:tokens=z['tokens'];labels=z['labels']
    Z.save(target/'source-cohort.npz',tokens=tokens,labels=labels,**reference);rows=[]
    for gpu in [0,1]:
        worker=json.loads((DEST/f'benchmark-gpu{gpu}.json').read_text())
        for pair in worker['rows']:
            for mode in ['serialized','async']:
                r=pair['runs'][mode];arrays=snapshot(S.ROOT/r['directory'],expected);assert all(np.array_equal(arrays[k],reference[k]) for k in KEYS);name=f'gpu{gpu}-rep{pair["repetition"]}-{mode}.npz';Z.save(target/name,**arrays);rows.append({'file':name,'sha256':S.sha(target/name),'gpu':gpu,'repetition':pair['repetition'],'mode':mode,'all8prefix_boards_counters_flags_and_margins_exact':True,'full_output_sha256_by_job':{j['id']:j['files_sha256']['outputs.npz'] for j in expected['jobs']}})
    S.save(target/'manifest.json',{'source_cohort_sha256':S.sha(target/'source-cohort.npz'),'source_protocol_sha256':expected['source_protocol_sha256'],'source_audit_sha256':expected['source_audit_sha256'],'rows':rows,'files':{path.name:S.sha(path) for path in sorted(target.glob('*.npz'))},'extractor_sha256':S.sha(__file__),'scope':'All768source modelinputs/eightprefixes and every8timed replay:49,152board/counter/flag/margin copies. Full1368scientificNPZfiles rehashed independently onWork and bound inprotocol/audit; rawcouplings/tracearchives remain onWork.'})


if __name__=='__main__':main()

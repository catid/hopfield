"""Trusted reconstruction of previously acquired H1bw noisy query records."""
import json
import numpy as np
from posterior_query_labels import ROOT, HERE, OUT, PARENT, PROTOCOL, sha, save, npz, verify
from repeat_query_math import corrupted, decode
RAW_PARENT = ROOT/'artifacts/research/20261114/ripper/repeat_query_tradeoff'


def reconstruct(mask, clean, noise):
    result = dict(given=mask)
    for name, n in [('first',270), ('repeat1',90), ('repeat2',90)]:
        labels = corrupted(clean[:n], {key:noise[name+'_'+key] for key in ['iid_uniforms','iid_offsets']})
        observed = np.full(labels.shape,-1,np.int8)
        observed[mask[:n]] = labels[mask[:n]]
        result[name] = observed
    return result


def independent_reconstruction(seed, data, protocol):
    from audit_higher_noise import reference_corrupt
    from audit_repeat_query_tradeoff import reference_decode
    parent = RAW_PARENT/f's{seed}'
    source = RAW_PARENT/f'training-s{seed}.npz'
    old_audit = json.loads((PARENT/'summary.json').read_text())['raw_audits'][seed]
    assert old_audit['seed'] == seed
    for name in ['masks.npz','potential_noise.npz','single270-observations.npz','triple90-observations.npz']:
        assert sha(parent/name) == old_audit['artifact_sha256'][name]
    clean = npz(source)['solutions']
    mask = npz(parent/'masks.npz')['balanced']
    assert np.array_equal(data['given'], mask)
    noise = npz(parent/'potential_noise.npz')
    expected = {}
    for name,n in [('first',270),('repeat1',90),('repeat2',90)]:
        labels = reference_corrupt(clean[:n], {key:noise[name+'_'+key] for key in ['iid_uniforms','iid_offsets']})
        values = np.full((n,81),-1,np.int8)
        values[mask[:n]] = labels[mask[:n]]
        assert np.array_equal(data[name], values)
        expected[name] = labels
    first = npz(parent/'single270-observations.npz')
    triple = npz(parent/'triple90-observations.npz')
    assert np.array_equal(first['given'], mask) and np.array_equal(first['values'], data['first'])
    decoded = reference_decode(expected['first'][:90], expected['repeat1'], expected['repeat2'])
    values = np.full((90,81),-1,np.int8);values[mask[:90]] = decoded[mask[:90]]
    assert np.array_equal(triple['given'], mask[:90]) and np.array_equal(triple['values'], values)
    manifest = json.loads((PARENT/'input_manifest.json').read_text())
    assert manifest['inputs'][seed]['completion_sha256'] == sha(source)
    for path in [source, parent/'masks.npz', parent/'potential_noise.npz', parent/'single270-observations.npz', parent/'triple90-observations.npz']:
        assert sha(path) == protocol['raw_input_sha256'][str(path.relative_to(ROOT))]
    return dict(status='validated', seed=seed, protocol_sha256=sha(PROTOCOL),
                parent_completion_sha256=sha(source), parent_first_observation_sha256=sha(parent/'single270-observations.npz'),
                parent_decoded_observation_sha256=sha(parent/'triple90-observations.npz'),
                original_queried_labels_reconstructed=True, inherited_CSV_provenance=True,
                new_queries=0, scorer_receives_clean_values=False)


def prepare():
    p = verify(True)
    assert not (OUT/'query-record-seal.json').exists()
    receipts, audits = {}, {}
    for seed in range(64):
        directory = OUT/f's{seed}';directory.mkdir()
        inherited = json.loads((PARENT/'summary.json').read_text())['raw_audits'][seed]
        for name in ['masks.npz','potential_noise.npz','single270-observations.npz','triple90-observations.npz']:
            assert sha(RAW_PARENT/f's{seed}/{name}') == inherited['artifact_sha256'][name]
        clean = npz(RAW_PARENT/f'training-s{seed}.npz')['solutions']
        mask = npz(RAW_PARENT/f's{seed}/masks.npz')['balanced']
        noise = npz(RAW_PARENT/f's{seed}/potential_noise.npz')
        records = reconstruct(mask,clean,noise)
        np.savez_compressed(directory/'query-records.npz',**records)
        audit = independent_reconstruction(seed,records,p)
        audit['query_records_sha256'] = sha(directory/'query-records.npz')
        save(directory/'query-record-audit.json',audit)
        receipts[str(seed)] = sha(directory/'query-records.npz')
        audits[str(seed)] = sha(directory/'query-record-audit.json')
    save(OUT/'query-record-seal.json',dict(status='sealed',protocol_sha256=sha(PROTOCOL),
         all64_query_records_validated_before_scoring=True,record_sha256=receipts,audit_sha256=audits,
         scope='Reconstructed already-acquired noisy labels. Independent charged raw correspondence passed; original CSV provenance inherited.'))
    print(json.dumps(dict(status='sealed',banks=64,new_queries=0)),flush=True)

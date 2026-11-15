"""H4be immutable bindings, independent-repeat assembly and paired endpoints."""
from pathlib import Path
import json
import time
import numpy as np
from fresh_pooled_donor_common import ROOT, HERE, RIPPER, CPU_PYTHON, FAMILIES, N, sha, save, bounds

PROTOCOL = HERE/"repeat-versus-diversity.json"
PROPOSAL = HERE/"repeat-versus-diversity-proposal.json"
PARENT_PROTOCOL = HERE/"fresh-pooled-donor.json"
PARENT = ROOT/"results/research/foureyes/fresh_pooled_donor_corrected"
PARENT_RAW = ROOT/"artifacts/research/20261114/foureyes/fresh_pooled_donor"
OUT = ROOT/"artifacts/research/20261114/foureyes/repeat_diversity"
PUBLIC = ROOT/"results/research/foureyes/repeat_diversity"


def read(path):return json.loads(Path(path).read_text())


def npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def protocol(inputs=True):
    p=read(PROTOCOL)
    assert p["status"]=="frozen_before_new_repeat_noise_and_fits"
    for rel,digest in (p["source_sha256"] | (p["input_sha256"] if inputs else {})).items():
        assert sha(ROOT/rel)==digest,rel
    return p


def assemble(first,repeat1,repeat2):
    for rows in [first,repeat1,repeat2]:
        assert rows.shape==(600,81) and rows.dtype==np.uint8 and (rows<9).all()
    return np.concatenate([first,repeat1,repeat2])


def repeats(clean,hot,seeds):
    from regularity_intensity_study import corrupt
    assert clean.shape==(600,81) and clean.dtype==np.uint8 and (clean<9).all()
    assert hot.shape==(18,) and hot.dtype==np.int64 and np.array_equal(hot,np.unique(hot))
    arrays={}
    for number,seed in enumerate(seeds,1):
        rng=np.random.default_rng(seed)
        uniforms=1-rng.random((600,81))
        offsets=rng.integers(1,9,(600,81),dtype=np.int64)
        noisy,cells=corrupt(np,clean,np.ones((600,81),bool),uniforms,offsets,hot,"hot18_weight32")
        arrays.update({f"r{number}_{key}":value for key,value in dict(uniforms=uniforms,offsets=offsets,noisy=noisy,cells=cells).items()})
    return arrays


def independent_repeats(clean,hot,seeds,arrays):
    from audit_regularity_intensity import literal_corruption
    assert set(arrays)=={f"r{n}_{k}" for n in [1,2] for k in ["uniforms","offsets","noisy","cells"]}
    for number,seed in enumerate(seeds,1):
        rng=np.random.default_rng(seed)
        uniform=1-rng.random((600,81))
        offsets=rng.integers(1,9,(600,81),dtype=np.int64)
        noisy,cells=literal_corruption(clean,np.ones((600,81),bool),uniform,offsets,hot,"hot18_weight32")
        for key,expected in dict(uniforms=uniform,offsets=offsets,noisy=noisy,cells=cells).items():
            actual=arrays[f"r{number}_{key}"]
            assert actual.dtype==expected.dtype and np.array_equal(actual,expected),key
        assert ((noisy!=clean).sum(1)==9).all()


def phase_files(phase):
    if phase=="source":return {f"{name}/b{i:03d}.{ext}" for i in range(N) for name,ext in
        [("observed","npz"),("noise","npz"),("source_audit","json")]}
    assert phase=="fit"
    return {f"fits/b{i:03d}.{ext}" for i in range(N) for ext in ["json","npz","log"]}


def check_seal(phase,base=OUT):
    rec=read(base/f"{phase}-seal.json")
    assert rec["phase"]==phase and rec["protocol_sha256"]==sha(PROTOCOL)
    assert set(rec["files"])==phase_files(phase)
    for rel,digest in rec["files"].items():assert sha(base/rel)==digest,rel
    return rec


def seal(phase):
    path=OUT/f"{phase}-seal.json"
    assert not path.exists() and not list((OUT/"audits").glob("*.json"))
    names=phase_files(phase)
    assert all((OUT/name).is_file() for name in names)
    rec=dict(phase=phase,protocol_sha256=sha(PROTOCOL),sealed_monotonic_ns=time.monotonic_ns(),
             files={name:sha(OUT/name) for name in sorted(names)})
    save(path,rec)
    return rec


def aggregate(rows):
    assert [r["index"] for r in rows]==list(range(N))
    statuses=["success","resolved_failure_nonexact","resolved_failure_fractional","unresolved"]
    new=[r["repeat"]["classification"] for r in rows]
    old=[r["baseline"]["classification"] for r in rows]
    assert set(new+old)<=set(statuses)
    return dict(primary=bounds(new,old),all256_banks_retained=True,
        grouped=[dict(arm=arm,**{k:values.count(k) for k in statuses},
            families=[dict(family=f,exact=sum(r["family"]==f and r[arm]["classification"]=="success" for r in rows)) for f in FAMILIES])
            for arm,values in [("baseline",old),("repeat",new)]],
        paired_gains=sum(a=="success" and b.startswith("resolved_failure") for a,b in zip(new,old)),
        paired_losses=sum(b=="success" and a.startswith("resolved_failure") for a,b in zip(new,old)),
        new_certificates_checked=sum(r["repeat"]["certified"] for r in rows),
        inherited_selected_certificates_checked=sum(r["baseline"]["certified"] for r in rows),
        mutations_rejected=3*sum(r["repeat"]["certified"]+r["baseline"]["certified"] for r in rows))

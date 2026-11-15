"""H2bh pre-draw seed reservation and prior-data registry, no generation."""
import json
import study as S
BASE=2026111527


def main():
    target=S.OUT/'loop80_confirmation';target.mkdir(exist_ok=True);records=[];hits=[]
    for path in sorted(S.HERE.glob('*protocol*.json')):
        data=json.loads(path.read_text());hit=str(BASE) in json.dumps(data);records.append({'path':str(path.relative_to(S.ROOT)),'sha256':S.sha(path),'contains_proposed_seed_base':hit})
        if hit:hits.append(str(path))
    assert not hits,hits;prior=[]
    for path in sorted([S.OUT/'datasets.json',*S.OUT.glob('*/datasets.json')]):
        if not path.exists() or path.parent==target:continue
        data=json.loads(path.read_text())
        if 'datasets' in data:prior.append({'path':str(path.relative_to(S.ROOT)),'sha256':S.sha(path),'dataset_records':len(data['datasets'])})
    out={'status':'passed','proposed_seed_base':BASE,'existing_protocol_references':0,'protocol_registry':records,'prior_data_manifests':prior,'fresh_inputs_drawn':0,'source_sha256':S.sha(__file__),'scope':'Beforefreshdraws: proposed seed base absent from every Work protocol; all extant Work dataset manifests registered for exact post-generation token-hash exclusion.'};S.save(target/'seed-registry.json',out);S.emit('cycle_seed_registry_passed',protocols=len(records),prior_manifests=len(prior),base=BASE)


if __name__=='__main__':main()

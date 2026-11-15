"""Bounded parser benchmark on archived H1bk IDs only, before H1bl freeze."""
import csv,hashlib,json,platform,time
import numpy as np
from coverage_tradeoff_study import ROOT,HERE,sha,save,selected_answers


def old_answers(source,wanted,row_limit):
    with source.open(newline='') as f:
        for index,row in enumerate(csv.DictReader(f)):
            if index>=row_limit:break
            if index in wanted:yield index,np.frombuffer(row['answer'].encode('ascii'),np.uint8).astype(np.int8)-ord('1')


def main():
    assert platform.node()=='ripper';output=HERE/'coverage-tradeoff-csv-benchmark.json';assert not output.exists();protocol=json.loads((HERE/'protocol-uniform32.json').read_text());source=ROOT/protocol['source'];assert sha(source)==protocol['source_csv_sha256']
    parent=ROOT/'results/research/ripper/uniform32/selected_sources.npz'
    with np.load(parent,allow_pickle=False) as z:wanted=set(map(int,z['all_source_row'][z['all_source_row']<200000]))
    records=[];checks=[]
    for repeat in range(3):
        for name,fn in ([('dict',old_answers),('positional',selected_answers)] if repeat%2==0 else [('positional',selected_answers),('dict',old_answers)]):
            start=time.monotonic();cpu=time.process_time();observed=list(fn(source,wanted,200000));wall=time.monotonic()-start;used=time.process_time()-cpu
            assert set(index for index,_ in observed)==wanted and all(board.shape==(81,) for _,board in observed)
            digest=hashlib.sha256(b''.join(int(index).to_bytes(8,'little')+board.tobytes() for index,board in observed)).hexdigest();checks.append(digest)
            records.append(dict(repeat=repeat,parser=name,wall_seconds=wall,cpu_seconds=used,selected_ids=len(observed),output_sha256=digest))
    assert len(set(checks))==1
    medians={name:float(np.median([r['wall_seconds'] for r in records if r['parser']==name])) for name in ['dict','positional']};assert medians['positional']<medians['dict']
    save(output,dict(status='passed',host=platform.node(),scope='First200000 archived official rows, restricted to already examined H1bk IDs; no H1bl draws. Alternating order,3trials each, warm filesystem cache; no general speed guarantee.',
                     records=records,median_wall_seconds=medians,archived_subset_speedup=medians['dict']/medians['positional'],exact_selected_answer_parity=True,source_csv_sha256=sha(source),archived_id_file_sha256=sha(parent),
                     producer_sha256=sha(HERE/'coverage_tradeoff_study.py'),benchmark_sha256=sha(__file__)))
    print(json.dumps(dict(status='passed',selected_archived_ids=len(wanted),median_wall_seconds=medians,archived_subset_speedup=medians['dict']/medians['positional'])))


if __name__=='__main__':main()

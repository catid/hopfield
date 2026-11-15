"""H2bs explicit charged host segments and paired exact cost endpoint."""
from fractions import Fraction
import math
from endpoint_dispatch_accounting import batch_cost


def segment(start,finish):
    if not all(math.isfinite(x) for x in [start,finish]) or finish<start:raise ValueError('Invalid charged host segment')
    return {'started_monotonic':start,'finished_monotonic':finish,'seconds':finish-start}


def complete(readiness,models,rows):
    if not math.isfinite(readiness) or readiness<=0 or len(models)!=3:raise ValueError('All fresh models and preparation required')
    total=readiness
    for seed,record in enumerate(models):
        if record['seed']!=seed or record['actual_inputs']<=0:raise ValueError('Complete ascending model records required')
        for name in ['preparation','finalization']:
            s=record[name]
            if not isinstance(s,dict):raise ValueError('Missing charged model host segment')
            checked=segment(s['started_monotonic'],s['finished_monotonic'])
            if checked!=s:raise ValueError('Host stage accounting mismatch')
            total+=s['seconds']
    seen=set()
    for row in rows:
        key=(row['seed'],row['slot'])
        if key in seen:raise ValueError('Duplicate host slot')
        seen.add(key)
        if row['zero_work']:
            if row['indices'] or row['answer_timing'] is not None or row['materialization'] is not None or row['scatter'] is not None:raise ValueError('Zero-work slot executed or charged input work')
        else:
            if not 1<=len(row['indices'])<=32 or row['answer_timing']['n']!=len(row['indices']):raise ValueError('No padded inputs allowed')
            total+=batch_cost(row['answer_timing'])
            for name in ['materialization','scatter']:
                s=row[name]
                if not isinstance(s,dict):raise ValueError('Missing charged batch host segment')
                checked=segment(s['started_monotonic'],s['finished_monotonic'])
                if checked!=s:raise ValueError('Batch host stage accounting mismatch')
                total+=s['seconds']
    for model in models:
        selected=[r for r in rows if r['seed']==model['seed']]
        if [r['slot'] for r in selected]!=list(range(model['slots_per_model'])):raise ValueError('Every predetermined host slot required')
        ids=[i for row in selected for i in row['indices']]
        if len(ids)!=len(set(ids)) or sorted(ids)!=model['indices'] or len(ids)!=model['actual_inputs']:raise ValueError('Dropped, duplicate or replaced input')
    return total


def endpoint(rows):
    if len(rows)!=6 or sorted(r['pair'] for r in rows)!=list(range(6)):raise ValueError('Exactly six prespecified pairs required')
    values=[];out=[]
    for row in sorted(rows,key=lambda r:r['pair']):
        a,b=row['original_order'],row['packed']
        if not all(math.isfinite(x) and x>0 for x in [a,b]):raise ValueError('Positive finite complete costs required')
        saving=Fraction(1)-Fraction(b)/Fraction(a);values.append(saving)
        out.append({'pair':row['pair'],'original_order_seconds':a,'packed_seconds':b,'saving':float(saving)})
    ordered=sorted(values);median=(ordered[2]+ordered[3])/2
    return {'pairs':out,'median_saving':float(median),'exact_median_fraction':str(median),'minimum_saving':.10,'decision':'supported' if median>=Fraction(1,10) else 'falsified'}

"""Independent SciPy CP paired-net endpoints from all H2ax joint rows."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from scipy.stats import beta


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(directory):
    p=json.loads((directory/'protocol.json').read_text());a=json.loads((directory/'audit.json').read_text());rows=list(csv.DictReader((directory/'per-input.csv').open()));assert len(rows)==4096 and [int(r['input_index']) for r in rows]==list(range(4096));results=[]
    def bounds(k,alpha=.025):return (float(beta.ppf(alpha,k,4096-k+1)) if k else 0.,float(beta.ppf(1-alpha,k+1,4096-k)) if k<4096 else 1.)
    for name,d in p['primary_endpoints'].items():
        g=sum(int(r[d['candidate']]) and not int(r[d['control']]) for r in rows);l=sum(not int(r[d['candidate']]) and int(r[d['control']]) for r in rows);gl,gu=bounds(g);ll,lu=bounds(l);lo=100*(gl-lu);hi=100*(gu-ll);expected=a['primary_endpoints'][name];assert g==expected['gains'] and l==expected['losses'];error=max(abs(lo-expected['separately_one_sided95_net_lower_pp']),abs(hi-expected['separately_one_sided95_net_upper_pp']));assert error<1e-9
        result={'name':name,'gains':g,'losses':l,'net_lower_pp':lo,'net_upper_pp':hi}
        if d['kind']=='paired_net':decision='supported' if lo>=d['minimum_net_gain_pp'] else 'falsified' if hi<d['minimum_net_gain_pp'] else 'inconclusive'
        else:
            low,high=bounds(l,.05);low*=100;high*=100;error=max(error,abs(low-expected['one_sided95_regression_lower_pp']),abs(high-expected['one_sided95_regression_upper_pp']));assert error<1e-9;result.update(regression_lower_pp=low,regression_upper_pp=high);decision='supported' if high<=d['maximum_regression_probability_pp'] else 'falsified' if low>d['maximum_regression_probability_pp'] else 'inconclusive'
        assert decision==expected['decision'];result.update(decision=decision,maximum_error_pp=error);results.append(result)
    r={'status':'passed','protocol_sha256':sha(directory/'protocol.json'),'paired_rows_sha256':sha(directory/'per-input.csv'),'audit_sha256':sha(directory/'audit.json'),'source_sha256':sha(Path(__file__)),'n_independent_inputs':4096,'endpoints':results,'scope':'SciPy beta quantiles independently reconstruct separately one-sided95 pairednetbounds usingCPmarginalalpha.025. Three endpoints decided separately: paired-net gain/BP NI use CP marginal alpha.025; standalone regression uses alpha.05. No simultaneous claim.'};(directory/'statistics-verification.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);main(p.parse_args().directory)

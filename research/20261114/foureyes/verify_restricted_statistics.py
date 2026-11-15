"""Independent beta-quantile verification of the frozen paired net bounds."""
from pathlib import Path
import hashlib
import json
import scipy
from scipy.stats import beta
ROOT=Path(__file__).resolve().parents[3]
path=ROOT/'results/research/foureyes/restricted_gap/summary.json';s=json.loads(path.read_text());p=s['primary'];n=p['banks'];a=.025

def cp(k):return (float(beta.ppf(a,k,n-k+1)) if k else 0.,float(beta.ppf(1-a,k+1,n-k)) if k<n else 1.)
gl,gu=cp(p['gains']);ll,lu=cp(p['losses'])
values=dict(one_sided95_lower_pp=100*(gl-lu),one_sided95_upper_pp=100*(gu-ll));error=max(abs(p[k]-v) for k,v in values.items())
assert error<1e-10
out=dict(status='passed',scipy=scipy.__version__,max_absolute_error_pp=error,independent_endpoints=values,summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),coverage='For each one-sided net bound two binomial tails each alpha=.025; union bound gives at least95% coverage. Upper and lower are separate families.')
(path.parent/'statistics-verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

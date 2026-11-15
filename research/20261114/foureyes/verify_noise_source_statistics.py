"""SciPy check for independent source-and-corruption recovery bounds."""
from pathlib import Path
import hashlib
import json
import scipy
from scipy.stats import beta
ROOT=Path(__file__).resolve().parents[3]
p=ROOT/'results/research/foureyes/noise_source_replication/summary.json';summary=json.loads(p.read_text());errors=[]
for r in summary['grouped']:
    k,n=r['exact_graphs'],r['banks'];lo=beta.ppf(.05,k,n-k+1) if k else 0;hi=beta.ppf(.95,k+1,n-k) if k<n else 1
    errors.extend([abs(lo-r['lower_one_sided95']),abs(hi-r['upper_one_sided95'])])
assert max(errors)<1e-12
result=dict(status='passed',scipy=scipy.__version__,bounds_checked=len(errors),max_absolute_error=float(max(errors)),summary_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(p.parent/'statistics-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

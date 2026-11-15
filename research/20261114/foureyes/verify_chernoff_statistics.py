"""Independent exact-binomial endpoint check for H4p."""
from pathlib import Path
import hashlib
import json
import scipy
from scipy.stats import beta
ROOT=Path(__file__).resolve().parents[3];p=ROOT/'results/research/foureyes/chernoff_replication/summary.json'
s=json.loads(p.read_text());errors=[]
for row in s['grouped']:
    k,n=row['exact_graphs'],row['banks'];lo=float(beta.ppf(.05,k,n-k+1)) if k else 0.;hi=float(beta.ppf(.95,k+1,n-k)) if k<n else 1.
    errors.extend([abs(lo-row['lower_one_sided95']),abs(hi-row['upper_one_sided95'])])
assert max(errors)<1e-12
result=dict(status='passed',scipy=scipy.__version__,bounds_checked=len(errors),max_absolute_error=max(errors),summary_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(p.parent/'statistics-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

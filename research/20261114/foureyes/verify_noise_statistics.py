"""Independent SciPy beta-quantile verification of exact binomial bounds.

Run: uv run --no-project --with scipy python <this script>
"""
from pathlib import Path
import hashlib
import json
import scipy
from scipy.stats import beta
ROOT=Path(__file__).resolve().parents[3]
p=ROOT/'results/research/foureyes/noise_replication/summary.json'
d=json.loads(p.read_text());errors=[]
for row in d['grouped']:
    k,n,a=row['exact_graphs'],row['draws'],row['per_bank_alpha']
    lo=beta.ppf(a,k,n-k+1) if k else 0
    hi=beta.ppf(1-a,k+1,n-k) if k<n else 1
    errors.extend([abs(lo-row['lower_one_sided']),abs(hi-row['upper_one_sided'])])
assert max(errors)<1e-12
out=dict(status='passed',scipy=scipy.__version__,bounds_checked=len(errors),
    max_absolute_error=float(max(errors)),summary_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    method='SciPy beta quantiles vs direct binomial-tail inversion using Python math.lgamma')
(p.parent/'statistics-verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))

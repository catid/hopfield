"""H4ba larger-donor comparison, including the donor-only diagnostic control."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'assets/figures'
matplotlib.rcParams['svg.hashsalt']='h4ba-larger-donor'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=ROOT/'results/research/foureyes/larger_donor/summary.json';s=json.loads(p.read_text());g=s['grouped'][0]
 donor=sum(r['donors'][0]['classification']=='success' for r in s['rows']);assert donor==g['success']
 assert all((r['donors'][0]['classification']=='success')==(r['classification']=='success') for r in s['rows'])
 fig,ax=plt.subplots(figsize=(9.5,4.7));vals=[sum(r['parent_detector_exact'] for r in s['rows']),donor,g['success']]
 ax.bar(range(3),vals,color=['#85929b','#278a9b','#19798b'],width=.6)
 ax.set_xticks(range(3),['Target-trained detector\n600 rows total','Donor graph alone\n1,200 donor rows','Donor + masked target\n1,800 rows total'])
 for i,v in enumerate(vals):ax.text(i,v+4,f'{v}/256',ha='center',fontsize=12)
 ax.axhline(162,color='#855988',linestyle='--');ax.text(-.35,166,'Fixed target: at least 162',fontsize=10)
 ax.set_ylim(0,270);ax.set_ylabel('Exact graphs / 256 examined banks');ax.spines[['top','right']].set_visible(False)
 fig.suptitle('More donor data reaches 229/256; masking adds no exact recoveries',x=.02,ha='left',fontsize=13)
 fig.text(.02,.085,'Both fixed targets pass: 229 exact graphs; 8 losses versus the original 130 successes (limit 8).',fontsize=10)
 fig.text(.02,.03,'All donor/final exact flags agree. Examined banks; larger observation budget; inherited degree and 9 flags/row. BP AdamW not run.',fontsize=9)
 fig.tight_layout(rect=[0,.14,1,.92]);OUT.mkdir(exist_ok=True)
 for ext in ['png','svg']:fig.savefig(OUT/f'larger-donor.{ext}',dpi=160,metadata={'Date':None} if ext=='svg' else {'Software':'Matplotlib'})
 plt.close(fig);(OUT/'larger-donor-provenance.json').write_text(json.dumps(dict(renderer_sha256=sha(Path(__file__)),source_sha256={str(p.relative_to(ROOT)):sha(p)},scope='All examined banks and donor-only control; no benefit from masking established by exact recovery.'),indent=2)+'\n')
if __name__=='__main__':main()

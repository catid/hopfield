"""Fresh H2av paired failures and fully prefix-charged answer times."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
PUBLIC=ROOT/'results/research/work'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    source=PUBLIC/'cycle-transfer-180-summary.json';a=json.loads(source.read_text());assert a['status']=='passed' and a['all_final_states']==61440;names=['pl','old_binary','connectivity_veto','energy_cycle_cut','lexical_cycle_cut'];labels=['BP AdamW, native','LP, binary','LP + connectivity veto','LP + energy-ranked cycle cuts','LP + lexicographic cycle cuts'];fail=[4096-a['joint_success'][k] for k in names];seconds=[a['answer_seconds_by_arm'][k] for k in names];colors=['#667085','#9AA4B2','#547AA5','#2E8B77','#13755E'];plt.rcParams.update({'font.size':11,'axes.titlesize':13,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'h2av-fresh-cycle-transfer'})
    fig,(left,right)=plt.subplots(1,2,figsize=(13,5.2),gridspec_kw={'width_ratios':[1.15,1]},sharey=True);y=list(range(5))
    left.barh(y,fail,color=colors,height=.60);right.barh(y,seconds,color=colors,height=.60);left.set_yticks(y,labels);left.invert_yaxis();left.set_title('Joint failures on the same 4,096 mazes',loc='left',pad=13);right.set_title('Answer time with full prefix charged',loc='left',pad=13);left.set_xlim(0,max(20,max(fail)*1.2));right.set_xlim(0,max(seconds)*1.18);left.set_xlabel('Failed inputs (all three models must solve)');right.set_xlabel('Aggregate answer seconds');right.tick_params(axis='y',left=False);left.tick_params(axis='y',length=0)
    for ax in [left,right]:ax.spines['left'].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='x',color='#D0D5DD',alpha=.55)
    for i,v in enumerate(fail):left.text(v+3,i,str(v),va='center',fontweight='bold',color=colors[i]);left.plot([0],[i],marker='o',color=colors[i],markersize=4 if v==0 else 0)
    for i,v in enumerate(seconds):right.text(v+.8,i,f'{v:.2f}',va='center',fontweight='bold',color=colors[i])
    fig.suptitle('Fresh 180×180 test: fixed cycle-cut policies at a larger grid size',x=.025,ha='left',fontsize=16,fontweight='bold',y=.99)
    fig.text(.025,.91,'4,096 new loop20 mazes · 3 fixed models · explicit graph prior with energy-increasing cycle cuts',fontsize=11,color='#344054')
    bp=a['primary_endpoints']['lexical_noninferiority_vs_bp'];veto=a['primary_endpoints']['lexical_gain_vs_veto'];fig.text(.025,.115,f"Lexical paired gains / losses: {bp['gains']} / {bp['losses']} versus BP; {veto['gains']} / {veto['losses']} versus veto.",fontsize=11,fontweight='bold')
    hidden=sum(r['hidden_seconds'] for r in a['rows']);fig.text(.025,.065,f'Every cycle arm is charged its full binary + veto prefix. Shared hidden inference: {hidden:.3f} s separately; diagnostics and archiving excluded.',fontsize=9,color='#475467')
    fig.text(.025,.025,f"Prespecified gain versus veto: {veto['decision']}. Net noninferiority versus BP: {bp['decision']}. Finite fresh-cohort result.",fontsize=9,color='#475467')
    fig.subplots_adjust(left=.255,right=.975,top=.81,bottom=.23,wspace=.20);out=PUBLIC/'cycle-transfer-180';png=out/'fresh-cycle-transfer-180.png';svg=out/'fresh-cycle-transfer-180.svg';fig.savefig(png,dpi=180,facecolor='white');fig.savefig(svg,facecolor='white',metadata={'Date':None});plt.close(fig);receipt={'status':'rendered','source_audit_sha256':sha(source),'script_sha256':sha(Path(__file__)),'figures':{png.name:sha(png),svg.name:sha(svg)},'data':{'joint_failures':dict(zip(names,fail)),'fully_prefix_charged_answer_seconds':dict(zip(names,seconds))},'scope':'Fresh4096paired180inputs,threefixedmodels. BPfirst. Explicitgraphpriorandfinitecohortscope; no sourcecohortreuse in plotted accuracy.'};(out/'figure-manifest.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()

"""Public-data-only result and complete incremental timing display."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import study as S
from export_selective_fallback import manifest


def main():
    out=S.ROOT/'results/research/work';a=json.loads((out/'selective-fallback-summary.json').read_text());p=a['primary'];stats=json.loads((out/'selective-fallback-statistics-verification.json').read_text())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axes=plt.subplots(1,3,figsize=(15.5,5.1),constrained_layout=True)
    ax=axes[0];keys=['native','always_lambda1','selective_fallback'];fail=[16384-a['joint_success'][k] for k in keys];colors=['#227c9d','#999999','#508a3b']
    ax.bar(range(3),fail,color=colors,width=.65)
    for i,v in enumerate(fail):ax.text(i,v+max(fail)*.025,str(v),ha='center')
    ax.set_xticks(range(3),['Native','Always λ=1','Graph-informed\nselective fallback']);ax.set_ylabel('Joint failures / 16,384 inputs');ax.set_ylim(0,max(1,max(fail)*1.23));ax.set_title('Frozen models, fresh 120×120 mazes',loc='left',fontweight='bold')
    ax=axes[1];x=p['observed_gain_pp'];lo=p['one_sided95_net_gain_lower_pp'];hi=p['one_sided95_net_gain_upper_pp'];two=stats['two_sided95_descriptive']
    ax.errorbar([x],[.2],xerr=[[x-lo],[hi-x]],fmt='o',color='#508a3b',lw=3,capsize=5);ax.errorbar([x],[-.12],xerr=[[x-two['net_lower_pp']],[two['net_upper_pp']-x]],fmt='o',color='#777777',lw=2,capsize=4)
    ax.axvline(.05,color='#c23b23',ls='--');ax.axvline(0,color='#999999',lw=.8);ax.set_ylim(-.6,.6);ax.set_yticks([]);ax.spines['left'].set_visible(False);ax.set_xlim(min(-.01,two['net_lower_pp']-.02),max(.12,two['net_upper_pp']+.02));ax.set_xlabel('Selective minus native (percentage points)');ax.set_title(f"{p['gains']} paired gains; {p['losses']} losses",loc='left',fontweight='bold')
    ax.text(.5,.84,f'Frozen one-sided 95% bounds\n[{lo:.5f}, {hi:.5f}] pp',transform=ax.transAxes,ha='center',fontsize=9,color='#355e28');ax.text(.5,.19,f"Descriptive two-sided 95% interval\n[{two['net_lower_pp']:.5f}, {two['net_upper_pp']:.5f}] pp",transform=ax.transAxes,ha='center',fontsize=9,color='#666666')
    ax=axes[2];rows=a['timing'];bottom=np.zeros(3);groups=[('validator','Validator','#227c9d'),('queue','Gather / queue','#75b4c6'),('fallback_setup','Batch setup','#e3ae49'),('fallback_answer','Fallback solve','#508a3b'),('fallback_output','Output transfer','#9bbd7f')]
    for key,label,color in groups:
        y=np.array([100*r['cost_seconds'][key]/r['cost_seconds']['native_answer'] for r in rows]);ax.bar(range(3),y,bottom=bottom,label=label,color=color);bottom+=y
    ax.axhline(2,color='#c23b23',ls='--',label='Frozen ≤2% target');ax.set_xticks(range(3),['Repeat 1','Repeat 2','Repeat 3']);ax.set_ylabel('Incremental cost / native answer time (%)');ax.set_ylim(0,max(2.7,float(bottom.max())*1.15));ax.set_title(f"Median overhead: {p['median_answer_overhead_percent']:.3f}%",loc='left',fontweight='bold');ax.legend(fontsize=8,loc='upper left')
    fig.suptitle('Selective fallback uses graph knowledge and defers rejected inputs for batching',fontsize=15,fontweight='bold')
    fig.supxlabel(f"16,384 fresh120×120 loop20 inputs; three fixed models; native u=3; λ=1 only after exact graph-feasibility rejection.\nNo shortest-path oracle. Validator, batching/setup and output costs are charged; rejected cases wait until cohort end.\nFrozen conjunction decision: {p['decision']}. Component timings exclude data loading/evidence serialization; no online latency claim.",fontsize=9)
    for suffix in ['png','pdf','svg']:fig.savefig(out/f'selective-fallback.{suffix}',dpi=180)
    manifest(out)


if __name__=='__main__':main()

"""Public-data-only fixed-window accuracy, overhead and latency tradeoff."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import study as S
from export_window_fallback import manifest


def main():
    out=S.ROOT/'results/research/work';a=json.loads((out/'window-fallback-summary.json').read_text());p=a['primary'];statistics=json.loads((out/'window-fallback-statistics-verification.json').read_text());two=statistics['two_sided95_descriptive'];colors={'bounded1024':'#b65b2d','cohort_end':'#508a3b'};labels={'bounded1024':'Fixed 1,024 window','cohort_end':'Cohort-end queue'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axes=plt.subplots(2,2,figsize=(12.8,8.1),constrained_layout=True)
    ax=axes[0,0];keys=['native','bounded1024','cohort_end'];fail=[16384-a['joint_success'][k] for k in keys]
    ax.bar(range(3),fail,color=['#227c9d',colors['bounded1024'],colors['cohort_end']],width=.65)
    for i,v in enumerate(fail):ax.text(i,v+max(fail)*.025,str(v),ha='center')
    ax.set_xticks(range(3),['Native','Fixed 1,024','Cohort end']);ax.set_ylim(0,max(1,max(fail)*1.2));ax.set_ylabel('Joint failures / 16,384 inputs');ax.set_title(f"{p['gains']} paired gains; {p['losses']} losses",loc='left',fontweight='bold')
    ax=axes[0,1]
    for policy in colors:
        values=[r['answer_overhead_percent'] for r in a['timing'] if r['policy']==policy];ax.plot([1,2,3],values,'o-',label=f"{labels[policy]} (median {np.median(values):.2f}%)",color=colors[policy])
    ax.axhline(2,color='#c23b23',ls='--',label='Frozen ≤2% target');ax.set_xticks([1,2,3]);ax.set_xlabel('Complete timing repetition');ax.set_ylabel('Incremental cost / native answer (%)');ax.set_title('Validator, setup and output costs included',loc='left',fontweight='bold');ax.legend(fontsize=9)
    for ax,key,title in [(axes[1,0],'decoder_delay_seconds','Rejected-input delay: component decoder clock'),(axes[1,1],'wall_delay_seconds','Rejected-input delay: actual measured wall clock')]:
        for policy in colors:
            rows=[r for r in a['latency_summary'] if r['policy']==policy];med=[r[key]['p50'] for r in rows];high=[r[key]['p95'] for r in rows];ax.plot([1,2,3],med,'o-',color=colors[policy],label=labels[policy]+' median');ax.plot([1,2,3],high,'s--',color=colors[policy],label=labels[policy]+' 95th percentile')
        ax.set_xticks([1,2,3]);ax.set_xlabel('Complete timing repetition');ax.set_ylabel('Seconds after native output is available');ax.set_title(title,loc='left',fontweight='bold');ax.legend(fontsize=8,loc='center right')
    fig.suptitle('Bounded fallback batching: measure the cost of reducing rejection delay',fontsize=15,fontweight='bold')
    fig.supxlabel(f"16,384 fresh120×120 loop20 inputs; three fixed models; graph-informed λ=1 fallback; no shortestness oracle.\nFrozen conjunction: {p['decision']} (accuracy: {p['accuracy_decision']}; ≤2% overhead: {p['runtime_decision']}).\nSeparately one-sided95% gain bounds [{p['one_sided95_net_gain_lower_pp']:.5f}, {p['one_sided95_net_gain_upper_pp']:.5f}]pp; descriptive two-sided95% [{two['net_lower_pp']:.5f}, {two['net_upper_pp']:.5f}]pp.\nRaw wall delay includes intervening evidence work; component clock excludes it. Fixed1,024 means native inputs per model/GPU, not a millisecond SLA.",fontsize=9)
    for suffix in ['png','pdf','svg']:fig.savefig(out/f'window-fallback.{suffix}',dpi=180)
    manifest(out)


if __name__=='__main__':main()

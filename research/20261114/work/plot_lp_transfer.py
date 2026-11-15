"""Public-only paired ranking/quality/timing figure for fixed LPnormmatching."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import study as S
from export_lp_transfer import manifest


def main():
    out=S.ROOT/'results/research/work';a=json.loads((out/'lp-transfer-summary.json').read_text());arms=['pl','lp_scaled','unchanged'];labels=['BP AdamW\ncontinued PL','LP primal\nmatched max coefficient','Unchanged\ncheckpoint'];colors=['#227c9d','#508a3b','#858585'];plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axes=plt.subplots(2,2,figsize=(12.2,8.0),constrained_layout=True)
    for ax,key,title in [(axes[0,0],'joint_label_below_both','Label energy below both sparse and dense negatives'),(axes[0,1],'joint_success','Native-decoder valid shortest paths')]:
        values=[a[key][arm] for arm in arms];ax.bar(range(3),np.array(values)/4096*100,color=colors,width=.65)
        for i,v in enumerate(values):ax.text(i,v/4096*100+2,f'{v:,} / 4,096',ha='center',fontsize=9)
        ax.set_xticks(range(3),labels);ax.set_ylim(0,112);ax.set_ylabel('Joint success of all three models (%)');ax.set_title(title,loc='left',fontweight='bold')
    for ax,key,denominator,title,ylabel in [(axes[1,0],'timeouts',12288,'Fixed-cap failures','Timeout model-inputs / 12,288'),(axes[1,1],'answer_seconds',1,'Complete native answer cost','Summed synchronized GPU answer seconds')]:
        values=[sum(r[key] for r in a['rows'] if r['arm']==arm) for arm in arms];ax.bar(range(3),values,color=colors,width=.65)
        for i,v in enumerate(values):ax.text(i,v+max(1,max(values))*.025,f'{v:,}' if key=='timeouts' else f'{v:.2f} s',ha='center')
        ax.set_xticks(range(3),labels);ax.set_ylim(0,max(1,max(values))*1.18);ax.set_ylabel(ylabel);ax.set_title(title,loc='left',fontweight='bold')
    fig.suptitle('Source energy constraints and native inference after norm matching',fontweight='bold',fontsize=15)
    fig.supxlabel(f"4,096 fresh 120×120 loop20 inputs; three fixed readouts; native u=3, η=0.5, cap 8,192; no fallback or retuning.\nAll 1,698,987 training inequalities remain positive after one uniform scaling and float32 cast.\nRanking gain ≥90 pp: {a['ranking_primary']['decision']}; solve noninferiority ≤0.1 pp: {a['accuracy_primary']['decision']}; conjunction: {a['overall_decision']}.\nLP and BP use different objectives and compute budgets. These are two negative comparisons, not a global-energy certificate.",fontsize=9)
    for suffix in ['png','pdf','svg']:fig.savefig(out/f'lp-transfer.{suffix}',dpi=180)
    manifest(out)


if __name__=='__main__':main()

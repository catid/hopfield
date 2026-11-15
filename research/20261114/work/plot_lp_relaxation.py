"""Public-only H2z accuracy, nonconvergence, fractional states and runtime."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import study as S
from export_lp_relaxation import manifest


def main(out):
    a=json.loads((out/'lp-relaxation-summary.json').read_text());arms=['pl','lp_native','lp_lambda4'];labels=['BP AdamW\ncontinued PL','LP primal\nnative λ=0','LP primal\nfixed λ=4'];colors=['#227c9d','#858585','#8d4d9e'];plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    ax=axes[0,0];values=[a['joint_success'][arm] for arm in arms];ax.bar(range(3),np.array(values)/4096*100,color=colors,width=.65)
    for i,v in enumerate(values):ax.text(i,v/4096*100+2,f'{v:,} / 4,096',ha='center')
    ax.set_xticks(range(3),labels);ax.set_ylim(0,112);ax.set_ylabel('Joint success of all three models (%)');ax.set_title('Valid shortest paths',loc='left',fontweight='bold')
    settings=[(axes[0,1],'timeouts','Fixed-cap failures','Timeout model-inputs / 12,288',False),(axes[1,0],'mean_near_half_free','Fractional states near one half','Mean free cells with 0.45 < p < 0.55',True),(axes[1,1],'answer_seconds','Complete native answer cost','Summed synchronized answer seconds',False)]
    for ax,key,title,ylabel,average in settings:
        values=[sum(r[key] for r in a['rows'] if r['arm']==arm)/(48 if average else 1) for arm in arms];ax.bar(range(3),values,color=colors,width=.65)
        for i,v in enumerate(values):ax.text(i,v+max(1,max(values))*.025,f'{v:,.0f}' if key=='timeouts' else f'{v:.2f}',ha='center')
        ax.set_xticks(range(3),labels);ax.set_ylim(0,max(1,max(values))*1.18);ax.set_ylabel(ylabel);ax.set_title(title,loc='left',fontweight='bold')
    fig.suptitle('Binary energies preserved; continuous half-states penalized',fontweight='bold',fontsize=15)
    fig.supxlabel(f"4,096 fresh 120×120 loop20 inputs; three fixed readouts; u=3, η=0.5, cap 8,192; no graph fallback or retuning.\nFixed intervention: E(p) + 4Σ p(1−p) over free cells. Every binary energy and flip certificate is unchanged.\nSolve gain ≥90 pp versus LP native: {a['gain_primary']['decision']}; NI ≤0.1 pp versus BP PL: {a['noninferiority_primary']['decision']}; conjunction: {a['overall_decision']}.\nLP and BP have different training objectives and compute budgets. Counts of fixed-model outcomes are not independent input counts.",fontsize=9)
    for suffix in ['png','pdf','svg']:fig.savefig(out/f'lp-relaxation.{suffix}',dpi=180)
    manifest(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=S.ROOT/'results/research/work');main(parser.parse_args().output_dir)

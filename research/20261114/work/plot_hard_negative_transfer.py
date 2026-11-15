"""Public-only H2ac figure: solve quality, wrong certificates and cost."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import study as S
import argparse
from pathlib import Path


def main(out):
    a=json.loads((out/'hard-negative-transfer-summary.json').read_text());arms=['pl','lp_binary','hard_negative_binary'];labels=['BP AdamW\ncontinued PL','Earlier LP\nfour-color binary','Hard-negative LP\nfour-color binary'];colors=['#227c9d','#858585','#d08b22'];plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    ax=axes[0,0];values=[a['joint_success'][arm] for arm in arms];ax.bar(range(3),np.array(values)/4096*100,color=colors,width=.65)
    for i,v in enumerate(values):ax.text(i,v/4096*100+2,f'{v:,} / 4,096',ha='center')
    ax.set_xticks(range(3),labels);ax.set_ylim(0,112);ax.set_ylabel('Joint success of all three models (%)');ax.set_title('Valid shortest paths',loc='left',fontweight='bold')
    for ax,key,title,ylabel in [(axes[0,1],'wrong_certified','Wrong certified stops','Model-inputs / 12,288'),(axes[1,0],'timeouts','Declared-cap timeouts','Model-inputs / 12,288'),(axes[1,1],'answer_seconds','Complete answer cost','Summed synchronized answer seconds')]:
        values=[sum(r[key] for r in a['rows'] if r['arm']==arm) for arm in arms];ax.bar(range(3),values,color=colors,width=.65)
        for i,v in enumerate(values):ax.text(i,v+max(1,max(values))*.025,f'{v:,.0f}' if key!='answer_seconds' else f'{v:.2f}',ha='center')
        ax.set_xticks(range(3),labels);ax.set_ylim(0,max(1,max(values))*1.18);ax.set_ylabel(ylabel);ax.set_title(title,loc='left',fontweight='bold')
    fig.suptitle('One source-only hard-negative round improves fresh-maze solving',fontweight='bold',fontsize=14)
    fig.supxlabel(f"4,096 fresh 120×120 loop20 inputs; three fixed readouts; original binary energy and independent graph evaluation.\nNative: float32, u=3, η=0.5, cap 8,192. Binary: float64, all-open start, four colors, cap 512 sweeps, no continuous prefix.\nGain ≥10 pp versus earlier LP binary: {a['gain_primary']['decision']}; NI ≤0.1 pp versus BP PL: {a['noninferiority_primary']['decision']}; conjunction: {a['overall_decision']}.\nBinary cost includes cache construction. LP and BP have different training objectives/budgets; no graph fallback or target tuning.",fontsize=9)
    for suffix in ['png','pdf','svg']:fig.savefig(out/f'hard-negative-transfer.{suffix}',dpi=180)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);main(p.parse_args().directory)

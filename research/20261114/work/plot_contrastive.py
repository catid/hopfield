"""Public-only plot: ranking and solve quality remain distinct endpoints."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import study as S
from export_contrastive import manifest


def main():
    out=S.ROOT/'results/research/work';a=json.loads((out/'contrastive-training-summary.json').read_text());arms=['pl','contrastive','unchanged'];labels=['BP AdamW\ncontinued PL','BP AdamW\nPL + contrastive','Unchanged\ncheckpoint'];colors=['#227c9d','#b65b2d','#858585'];plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axes=plt.subplots(2,2,figsize=(12.2,8.0),constrained_layout=True)
    for ax,key,title in [(axes[0,0],'joint_label_ranked_below_endpoints','Pairwise objective ranking'),(axes[0,1],'joint_success','Native-decoder valid shortest paths')]:
        values=[a[key][arm] for arm in arms];ax.bar(range(3),np.array(values)/4096*100,color=colors,width=.65)
        for i,v in enumerate(values):ax.text(i,v/4096*100+2,f'{v:,} / 4,096',ha='center',fontsize=9)
        ax.set_xticks(range(3),labels);ax.set_ylim(0,112);ax.set_ylabel('Joint success of all three models (%)');ax.set_title(title,loc='left',fontweight='bold')
    ax=axes[1,0]
    for arm,color in zip(arms[:2],colors[:2]):
        traces=[json.loads((out/f'contrastive-training/fits/s{s}-{arm}/trace.json').read_text()) for s in range(3)];mean=np.mean([[r['mean_normalized_label_minus_endpoint'] for r in tr] for tr in traces],axis=0);ax.plot(np.arange(1,401),mean,color=color,label='Continued PL' if arm=='pl' else 'PL + contrastive')
    ax.axhline(0,color='#555',ls='--',lw=1);ax.set_xlabel('Fixed training update');ax.set_ylabel('Training normalized E(label) − E(endpoints)');ax.set_title('Training minibatches; mean over fixed models',loc='left',fontweight='bold');ax.legend()
    ax=axes[1,1];values=[sum(r['answer_seconds'] for r in a['rows'] if r['arm']==arm) for arm in arms];ax.bar(range(3),values,color=colors,width=.65)
    for i,v in enumerate(values):ax.text(i,v+max(values)*.025,f'{v:.2f} s',ha='center')
    ax.set_xticks(range(3),labels);ax.set_ylim(0,max(values)*1.18);ax.set_ylabel('Summed synchronized GPU answer seconds');ax.set_title('Full frozen cap; all failures retained',loc='left',fontweight='bold')
    fig.suptitle('Correcting a pairwise energy ranking does not guarantee decoder quality',fontweight='bold',fontsize=15)
    fig.supxlabel(f"1,000 original training mazes; 23 answer parameters; 400 matched updates/fit; three fixed original-model starts.\n4,096 fresh120×120 loop20 inputs; native u=3, η=0.5, cap8,192; hidden parameters/readouts frozen.\nRanking ≥90pp: {a['ranking_primary']['decision']}; solve noninferiority ≤0.1pp: {a['accuracy_primary']['decision']}; conjunction: {a['overall_decision']}.\nPairwise ranking is not a global-minimum certificate. GPU times are component sums, not parallel wall latency.",fontsize=9)
    for suffix in ['png','pdf','svg']:fig.savefig(out/f'contrastive-training.{suffix}',dpi=180)
    manifest(out)


if __name__=='__main__':main()

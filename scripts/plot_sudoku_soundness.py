"""Render exact support certificates and one public zero-energy counterexample."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT/'results/research/foureyes/logical_soundness'


def main():
    proof = json.loads((PUB/'certificates.json').read_text())
    parent_path = ROOT/'results/research/foureyes/noisy_pl_long/summary.json'
    parent = json.loads(parent_path.read_text())
    assert json.loads((PUB/'verification.json').read_text())['status'] == 'passed'
    groups = proof['grouped'];names = ['BP AdamW', 'Paper SGD\n32 updates', 'Paper SGD\n256 updates', 'Restricted\ncounts', 'Chernoff\ncounts']
    complete = [sum(r['missing'] == 0 for r in parent['source_audit']['models'] if r['arm'] == g['arm']) for g in groups]
    sound = [g['sound'] for g in groups];x = np.arange(5)
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios':[1.5, 1]})
    fig.subplots_adjust(left=.07, right=.98, top=.79, bottom=.23, wspace=.24)
    ax = axes[0];ax.bar(x-.18, complete, .35, color='#8197a8', label='Contains every true edge')
    ax.bar(x+.18, sound, .35, color='#12856e', label='Proved zero-stop sound')
    for pos, values in [(x-.18, complete), (x+.18, sound)]:
        for i, value in zip(pos, values):ax.text(i, value+1, str(value), ha='center', fontsize=10)
    ax.set_xticks(x, names);ax.set_ylim(0, 83);ax.set_ylabel('Models / 64 source banks');ax.set_title('Missing edges can be logically redundant', loc='left', fontweight='bold')
    ax.legend(frameon=False, loc='upper left', ncol=2, fontsize=10)
    witness = next(w for w in proof['witnesses'] if w['job'] == 'm008')
    board = np.array(witness['board']).reshape(9, 9)+1;free = np.array(witness['free']).reshape(9, 9)
    ax = axes[1];ax.set_xlim(0, 9);ax.set_ylim(9, 0);ax.set_aspect('equal');ax.axis('off')
    duplicates = {(r,c) for c in range(9) for r in range(9) if Counter(board[:, c])[board[r,c]] > 1}
    for r in range(9):
        for c in range(9):
            ax.add_patch(plt.Rectangle((c,r), 1, 1, facecolor='#fde7e7' if (r,c) in duplicates else '#eeeeee' if not free[r,c] else 'white', edgecolor='none'))
            ax.text(c+.5, r+.5, str(board[r,c]), ha='center', va='center', color='#ad2828' if (r,c) in duplicates else '#263746', fontweight='bold' if not free[r,c] else 'normal', fontsize=12)
    for k in range(10):
        ax.plot([k,k],[0,9], color='#263746', lw=1.7 if k % 3 == 0 else .45)
        ax.plot([0,9],[k,k], color='#263746', lw=1.7 if k % 3 == 0 else .45)
    ax.set_title('One unsound AdamW graph\nZero learned conflicts; invalid columns', loc='left', fontweight='bold', fontsize=12)
    fig.suptitle('Recovering every edge is stronger than proving safe zero-energy stopping', x=.07, ha='left', fontsize=17, fontweight='bold')
    fig.text(.07, .86, '228 sound models; 92 with explicit counterexamples. All 320 logical model cells classified.', fontsize=12)
    fig.text(.07, .065, 'Post-hoc exact proof using supplied Sudoku geometry; no new fitting or search. Domain: integer digits 0–8.\n32 additional models certified by counting identities. Extra constraints can still exclude valid completions.\nRight: official puzzle 260010, AdamW bank 6; red marks repeated column digits. Bold gray cells are supplied clues.', fontsize=10, color='#52616e')
    out = ROOT/'assets/figures';out.mkdir(exist_ok=True)
    for ext in ['png','svg']:fig.savefig(out/f'sudoku-logical-soundness.{ext}', dpi=180)
    inputs = [PUB/'certificates.json', PUB/'verification.json', parent_path]
    receipt = dict(renderer=str(Path(__file__).relative_to(ROOT)), inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (out/'sudoku-logical-soundness-provenance.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':main()

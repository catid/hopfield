"""Prepare isolated H2ab public artifacts without changing shared summaries."""
import shutil
import study as S
from mine_hard_negatives import PP,DEST


def main():
    out=DEST/'public-preview';out.mkdir(exist_ok=True);shutil.copyfile(PP,out/'protocol.json')
    for name in ['solver-summary.json','mining-audit.json','mining-gpu0.json','mining-gpu1.json']:shutil.copyfile(DEST/name,out/name)
    for seed in range(3):
        for pattern in ['mined-seed{}.npz','mined-seed{}.json','negatives-seed{}.npz','negative-map-seed{}.json']:name=pattern.format(seed);shutil.copyfile(DEST/name,out/name)
        shutil.copyfile(S.OUT/f'training_feasibility/initial-seed{seed}.npz',out/f'initial-seed{seed}.npz');folder=out/f'seed{seed}';folder.mkdir(exist_ok=True)
        for path in (DEST/f'seed{seed}').iterdir():
            if path.is_file():shutil.copyfile(path,folder/path.name)


if __name__=='__main__':main()

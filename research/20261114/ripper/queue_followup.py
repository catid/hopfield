"""One GPU lane for two predeclared H1 follow-up studies."""
from pathlib import Path
import json
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[3]
lane=int(sys.argv[1])
out=root/'artifacts/research/20261114/ripper'
rows=[]
for study in ['confirmation','lowdata']:
    log=(out/f'{study}-lane{lane}.log').open('w')
    argv=[str(root/'upstream/.venv/bin/python'),'-u',str(Path(__file__).with_name('followup.py')),
          '--study',study,'--gpu',str(lane),'--lane',str(lane)]
    start=time.time()
    p=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,cwd=root)
    exitcode=p.wait();log.close()
    rows.append({'study':study,'pid':p.pid,'argv':argv,'exitcode':exitcode,'start_unix':start,'end_unix':time.time()})
    (out/f'followup-queue{lane}.json').write_text(json.dumps(rows,indent=2)+'\n')
    if exitcode:sys.exit(exitcode)

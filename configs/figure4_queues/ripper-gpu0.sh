# Auditable command list; use figure4_queue.py worker for provenance, lock, and resume.
# curve-pl-s9
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method pl --seed 9
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-pl-s9
# curve-pl-s15
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method pl --seed 15
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-pl-s15
# curve-bptt1-s5
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt1 --seed 5
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-bptt1-s5
# curve-bptt1-s7
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt1 --seed 7
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-bptt1-s7
# curve-bptt1-s9
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt1 --seed 9
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-bptt1-s9
# curve-bptt1-s11
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt1 --seed 11
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-bptt1-s11
# curve-bptt1-s13
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt1 --seed 13
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-bptt1-s13
# curve-bptt1-s15
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt1 --seed 15
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-bptt1-s15
# curve-bptt1-s17
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt1 --seed 17
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-bptt1-s17
# curve-bptt1-s19
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt1 --seed 19
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-bptt1-s19
# curve-bptt-s0
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt --seed 0 --long
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/long/sudoku-bptt-s0
# curve-bptt-s3
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt --seed 3 --long
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/long/sudoku-bptt-s3
# curve-bptt-s7
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt --seed 7 --long
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/long/sudoku-bptt-s7
# curve-bptt-s13
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt --seed 13 --long
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/long/sudoku-bptt-s13
# curve-bptt-s19
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt --seed 19 --long
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/long/sudoku-bptt-s19

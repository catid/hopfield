# Auditable command list; use figure4_queue.py worker for provenance, lock, and resume.
# curve-pl-s3
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method pl --seed 3
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-pl-s3
# curve-pl-s4
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method pl --seed 4
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-pl-s4
# curve-pl-s6
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method pl --seed 6
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-pl-s6
# curve-pl-s10
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method pl --seed 10
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-pl-s10
# curve-pl-s16
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method pl --seed 16
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-pl-s16
# curve-mpf-s1
/home/catid/hopfield/upstream/.venv/bin/python -u training_methods.py train --task sudoku --method mpf --lr 60.0 --l1 1.0 --seed 1 --probe-per-octave 8 --probe-every-until 64 --out runs/ALT/dense/sudoku-mpf-s1
# curve-cd-s1
/home/catid/hopfield/upstream/.venv/bin/python -u training_methods.py train --task sudoku --method cd --lr 6.0 --knob 324 --l1 1.0 --reg 0.0 --seed 1 --probe-per-octave 8 --probe-every-until 64 --out runs/ALT/dense/sudoku-cd-s1
# curve-bptt1-s4
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt1 --seed 4
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-bptt1-s4
# curve-bptt1-s6
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt1 --seed 6
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-bptt1-s6
# curve-bptt1-s8
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt1 --seed 8
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-bptt1-s8
# curve-bptt1-s10
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt1 --seed 10
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-bptt1-s10
# curve-bptt1-s12
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt1 --seed 12
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-bptt1-s12
# curve-bptt1-s14
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt1 --seed 14
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-bptt1-s14
# curve-bptt1-s16
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt1 --seed 16
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-bptt1-s16
# curve-bptt1-s18
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt1 --seed 18
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-bptt1-s18
# curve-bptt-s5
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt --seed 5 --long
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/long/sudoku-bptt-s5
# curve-bptt-s11
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt --seed 11 --long
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/long/sudoku-bptt-s11
# curve-bptt-s17
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt --seed 17 --long
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/long/sudoku-bptt-s17

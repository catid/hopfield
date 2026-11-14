# Auditable command list; use figure4_queue.py worker for provenance, lock, and resume.
# curve-pl-s1
/home/catid/hopfield/upstream/.venv/bin/python -u training_methods.py train --task sudoku --method pl --lr 540.0 --l1 1.0 --seed 1 --probe-per-octave 8 --probe-every-until 64 --out runs/ALT/dense/sudoku-pl-s1
# curve-pl-s7
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method pl --seed 7
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-pl-s7
# curve-pl-s13
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method pl --seed 13
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-pl-s13
# curve-pl-s19
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method pl --seed 19
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-pl-s19
# curve-mpf-s4
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method mpf --seed 4
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-mpf-s4
# curve-mpf-s6
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method mpf --seed 6
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-mpf-s6
# curve-mpf-s8
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method mpf --seed 8
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-mpf-s8
# curve-mpf-s10
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method mpf --seed 10
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-mpf-s10
# curve-mpf-s12
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method mpf --seed 12
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-mpf-s12
# curve-mpf-s14
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method mpf --seed 14
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-mpf-s14
# curve-mpf-s16
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method mpf --seed 16
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-mpf-s16
# curve-mpf-s18
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method mpf --seed 18
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-mpf-s18
# curve-bptt1-s1
/home/catid/hopfield/upstream/.venv/bin/python -u training_methods.py train --task sudoku --method bptt1 --lr 60.0 --knob 10 --l1 0.01 --seed 1 --probe-per-octave 8 --probe-every-until 64 --out runs/ALT/dense/sudoku-bptt1-s1
# curve-bptt-s2
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt --seed 2 --long
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/long/sudoku-bptt-s2
# curve-bptt-s6
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt --seed 6 --long
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/long/sudoku-bptt-s6
# curve-bptt-s12
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt --seed 12 --long
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/long/sudoku-bptt-s12
# curve-bptt-s18
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt --seed 18 --long
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/long/sudoku-bptt-s18

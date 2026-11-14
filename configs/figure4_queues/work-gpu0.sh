# Auditable command list; use figure4_queue.py worker for provenance, lock, and resume.
# curve-pl-s2
/home/catid/hopfield/upstream/.venv/bin/python -u training_methods.py train --task sudoku --method pl --lr 540.0 --l1 1.0 --seed 2 --probe-per-octave 8 --probe-every-until 64 --out runs/ALT/dense/sudoku-pl-s2
# curve-pl-s12
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method pl --seed 12
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-pl-s12
# curve-pl-s18
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method pl --seed 18
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-pl-s18
# curve-mpf-s0
/home/catid/hopfield/upstream/.venv/bin/python -u training_methods.py train --task sudoku --method mpf --lr 60.0 --l1 1.0 --seed 0 --probe-per-octave 8 --probe-every-until 64 --out runs/ALT/dense/sudoku-mpf-s0
# curve-mpf-s3
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method mpf --seed 3
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-mpf-s3
# curve-mpf-s5
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method mpf --seed 5
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-mpf-s5
# curve-mpf-s7
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method mpf --seed 7
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-mpf-s7
# curve-mpf-s9
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method mpf --seed 9
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-mpf-s9
# curve-mpf-s11
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method mpf --seed 11
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-mpf-s11
# curve-mpf-s13
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method mpf --seed 13
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-mpf-s13
# curve-mpf-s15
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method mpf --seed 15
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-mpf-s15
# curve-mpf-s17
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method mpf --seed 17
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-mpf-s17
# curve-mpf-s19
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method mpf --seed 19
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-mpf-s19
# curve-cd-s0
/home/catid/hopfield/upstream/.venv/bin/python -u training_methods.py train --task sudoku --method cd --lr 6.0 --knob 324 --l1 1.0 --reg 0.0 --seed 0 --probe-per-octave 8 --probe-every-until 64 --out runs/ALT/dense/sudoku-cd-s0
# curve-bptt1-s2
/home/catid/hopfield/upstream/.venv/bin/python -u training_methods.py train --task sudoku --method bptt1 --lr 60.0 --knob 10 --l1 0.01 --seed 2 --probe-per-octave 8 --probe-every-until 64 --out runs/ALT/dense/sudoku-bptt1-s2
# curve-bptt-s10
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt --seed 10 --long
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/long/sudoku-bptt-s10
# curve-bptt-s16
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt --seed 16 --long
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/long/sudoku-bptt-s16

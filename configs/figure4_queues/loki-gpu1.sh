# Auditable command list; use figure4_queue.py worker for provenance, lock, and resume.
# curve-pl-s0
/home/catid/hopfield/upstream/.venv/bin/python -u training_methods.py train --task sudoku --method pl --lr 540.0 --l1 1.0 --seed 0 --probe-per-octave 8 --probe-every-until 64 --out runs/ALT/dense/sudoku-pl-s0
# curve-pl-s11
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method pl --seed 11
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-pl-s11
# curve-pl-s17
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method pl --seed 17
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-pl-s17
# curve-mpf-s2
/home/catid/hopfield/upstream/.venv/bin/python -u training_methods.py train --task sudoku --method mpf --lr 60.0 --l1 1.0 --seed 2 --probe-per-octave 8 --probe-every-until 64 --out runs/ALT/dense/sudoku-mpf-s2
# curve-cd-s2
/home/catid/hopfield/upstream/.venv/bin/python -u training_methods.py train --task sudoku --method cd --lr 6.0 --knob 324 --l1 1.0 --reg 0.0 --seed 2 --probe-per-octave 8 --probe-every-until 64 --out runs/ALT/dense/sudoku-cd-s2
# curve-cd-s3
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method cd --seed 3
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-cd-s3
# curve-cd-s5
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method cd --seed 5
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-cd-s5
# curve-cd-s7
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method cd --seed 7
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-cd-s7
# curve-cd-s9
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method cd --seed 9
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-cd-s9
# curve-cd-s11
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method cd --seed 11
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-cd-s11
# curve-cd-s13
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method cd --seed 13
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-cd-s13
# curve-cd-s15
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method cd --seed 15
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-cd-s15
# curve-cd-s17
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method cd --seed 17
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-cd-s17
# curve-cd-s19
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method cd --seed 19
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-cd-s19
# curve-bptt1-s0
/home/catid/hopfield/upstream/.venv/bin/python -u training_methods.py train --task sudoku --method bptt1 --lr 60.0 --knob 10 --l1 0.01 --seed 0 --probe-per-octave 8 --probe-every-until 64 --out runs/ALT/dense/sudoku-bptt1-s0
# curve-bptt-s9
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt --seed 9 --long
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/long/sudoku-bptt-s9
# curve-bptt-s15
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt --seed 15 --long
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/long/sudoku-bptt-s15

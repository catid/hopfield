# Auditable command list; use figure4_queue.py worker for provenance, lock, and resume.
# curve-pl-s5
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method pl --seed 5
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-pl-s5
# curve-pl-s8
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method pl --seed 8
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-pl-s8
# curve-pl-s14
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method pl --seed 14
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-pl-s14
# curve-cd-s4
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method cd --seed 4
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-cd-s4
# curve-cd-s6
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method cd --seed 6
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-cd-s6
# curve-cd-s8
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method cd --seed 8
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-cd-s8
# curve-cd-s10
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method cd --seed 10
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-cd-s10
# curve-cd-s12
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method cd --seed 12
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-cd-s12
# curve-cd-s14
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method cd --seed 14
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-cd-s14
# curve-cd-s16
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method cd --seed 16
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-cd-s16
# curve-cd-s18
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method cd --seed 18
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-cd-s18
# curve-bptt1-s3
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt1 --seed 3
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/many/sudoku-bptt1-s3
# curve-bptt-s1
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt --seed 1 --long
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/long/sudoku-bptt-s1
# curve-bptt-s4
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt --seed 4 --long
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/long/sudoku-bptt-s4
# curve-bptt-s8
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt --seed 8 --long
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/long/sudoku-bptt-s8
# curve-bptt-s14
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py train --task sudoku --method bptt --seed 14 --long
/home/catid/hopfield/upstream/.venv/bin/python -u training_curves.py probe --runs runs/ALT/long/sudoku-bptt-s14

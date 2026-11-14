"""CPU-only structural check of the optional kernel dispatch optimization."""
from pathlib import Path
import ast
import datetime
import hashlib
import json


ROOT = Path(__file__).resolve().parents[1]


def main():
    original = ROOT / "dataeff_source/cram/cram_sudoku_general.py"
    wrapper = ROOT / "scripts/sharded_sudoku_moves.py"
    a = next(n for n in ast.parse(original.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name == "_moves_general")
    b = next(n for n in ast.parse(wrapper.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name == "_moves_general_offset")
    assert b.args.args[-1].arg == "RNG_OFFSET"
    assert ast.unparse(b.args.args[-1].annotation) == "tl.constexpr"
    b.args.args.pop()
    b.name = a.name
    assignments = [n for n in ast.walk(b) if isinstance(n, ast.Assign)
                   and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id == "base"]
    assert len(assignments) == 1
    assert ast.unparse(assignments[0].value) == "((pid + RNG_OFFSET) * 512).to(tl.int32)"
    assert sum(isinstance(n, ast.Name) and n.id == "RNG_OFFSET" for n in ast.walk(b)) == 1
    assignments[0].value = ast.parse("(pid * 512).to(tl.int32)", mode="eval").body
    assert ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)
    result = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "passed",
        "source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "wrapper_sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
        "assertion": "Kernel AST is identical after removing only the RNG_OFFSET argument and the RNG base offset.",
        "scope": "Structural source comparison only; numerical parity and performance require GPU measurements.",
    }
    out = ROOT / "artifacts/orchestration_review/sharded-kernel-source-audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

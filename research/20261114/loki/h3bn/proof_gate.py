"""Missing, mismatched or nonfeasible compiled proofs are fatal, never fallback."""
from common import read,sha


def require_ready_proof(path,digest,candidates):
    if not path.is_file():raise RuntimeError('Missing compiled proof; whole-study incomplete')
    if sha(path)!=digest:raise RuntimeError('Compiled proof digest mismatch; whole-study incomplete')
    proof=read(path)
    if proof.get('candidates')!=candidates or proof.get('depth_limit')!=4 or proof.get('feasible') is not True:raise RuntimeError('Invalid or capped compiled proof; whole-study incomplete')
    return proof

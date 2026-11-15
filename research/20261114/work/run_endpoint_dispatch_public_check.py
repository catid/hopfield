"""Run portable endpoint-dispatch final-state check with heavyweight/production imports denied."""
import builtins
import hashlib
import json
from pathlib import Path
import sys
original=builtins.__import__
blocked={'endpoint_dispatch_study','endpoint_dispatch_generator','endpoint_dispatch_policy','endpoint_dispatch_batch','endpoint_dispatch_pipeline','audit_endpoint_dispatch','verify_endpoint_dispatch_inputs','analyze_endpoint_dispatch_bins','endpoint_shift_study','endpoint_shift_generator','audit_endpoint_shift','verify_endpoint_shift_inputs','analyze_endpoint_shift_bins','torch','scipy','study','bp_first_fresh_statistics','bp_first_fresh_study','bp_first_fresh_pipeline','bp_first_fresh_batch','bp_first_geometry_policy','bp_first_lp96_core','audit_bp_first_fresh','audit_bp_first_fresh_states','audit_bp_first_fresh_replays','audit_bp_first_fresh_costs','far_placement_study','far_placement_generator','audit_far_placement','verify_far_placement_inputs','analyze_far_placement_bins'}
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in blocked:raise ImportError('Blocked independent-checker import: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from verify_endpoint_dispatch_public import verify
path=Path(sys.argv[1]);verify(path)
(path/'portable-import-guard.json').write_text(json.dumps({'status':'passed','blocked_imports':sorted(blocked),'verification_sha256':hashlib.sha256((path/'public-verification.json').read_bytes()).hexdigest(),'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')

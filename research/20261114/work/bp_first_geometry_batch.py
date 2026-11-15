"""Candidate complete H2bn answer segment, awaiting review and freeze."""
import time
import numpy as np
import study as S
import larger_relaxation as L
from factorial_transfer import hidden
from bp_first_lp96_core import solve as lp96
from bp_first_geometry_policy import routes, scatter


def run(mode, host_tokens, models, doubles, watchdog=None, after_answer=None):
    if mode not in ['LP96', 'BP_geometry_LP96']:
        raise ValueError('Unknown arm')
    torch, _, _ = S.imports()
    torch.cuda.synchronize()
    answer_started = time.perf_counter()
    components = {}
    def start():
        torch.cuda.synchronize()
        return time.perf_counter()
    def finish(name, began):
        torch.cuda.synchronize()
        components[name] = time.perf_counter() - began
    began = start()
    t = torch.tensor(host_tokens.astype(np.int64), device='cuda')
    finish('input_transfer', began)
    began = start()
    a, hs, good = hidden(models['bp' if mode == 'BP_geometry_LP96' else 'lp'], t)
    hidden_good = good.cpu().numpy()
    finish('hidden_readout_and_flags', began)
    bp_cpu = None
    bp_native = None
    subset = None
    subset_tokens = None
    subset_couplings = None
    core = None
    def cpu_final(out, caps, model, tt, aa, prefix):
        began = start()
        cpu = {name: (out[name].bool() if name == 'prediction' else out[name]).cpu().numpy()
               for name in ['prediction', 'iterations', 'stopped', 'finite']}
        cpu['native_cap'] = caps['native_cap'].copy()
        cpu['block_cap'] = caps['block_cap'].copy()
        finish(prefix + '_answer_return', began)
        began = start()
        cpu['own_margin'] = model.flip_delta(tt, out['prediction'].double(), aa.double()).masked_fill(
            tt != 1, float('inf')).flatten(1).amin(1).cpu().numpy()
        finish(prefix + '_original_margin_and_return', began)
        return cpu
    if mode == 'LP96':
        began = start()
        core = lp96(doubles['lp'], t, a, watchdog)
        finish('LP96_complete_prefix', began)
        final = cpu_final(core['outputs'][7], core['blocks'][7], doubles['lp'], t, a, 'LP96')
        began = start()
        final['chosen_model'] = np.ones(len(t), np.int8)
        finish('model_provenance', began)
        geometry = np.zeros(0, bool)
        indices = np.zeros(0, np.int64)
    else:
        began = start()
        bp_native = L.solve(models['bp'], t, a, torch.full_like(a[:, 0], 3.), 8192, 0.)
        finish('BP_native_solve', began)
        bp_out = {key: bp_native[i] for i, key in enumerate(['prediction', 'iterations', 'stopped', 'finite'])}
        began = start()
        bp_caps = {'native_cap': (~bp_native[2] & bp_native[3]).cpu().numpy(),
                   'block_cap': np.zeros(len(t), bool)}
        finish('BP_cap_flags', began)
        bp_cpu = cpu_final(bp_out, bp_caps, doubles['bp'], t, a, 'BP')
        began = start()
        geometry, indices = routes(host_tokens, bp_cpu['prediction'])
        finish('geometry_screen_and_indices', began)
        began = start()
        if len(indices):
            ix = torch.tensor(indices, device=t.device)
            subset_tokens, subset_couplings = t[ix], a[ix]
        finish('LP_subset_gather', began)
        if len(indices):
            began = start()
            core = lp96(doubles['lp'], subset_tokens, subset_couplings, watchdog)
            finish('LP96_complete_prefix', began)
            subset = cpu_final(core['outputs'][7], core['blocks'][7], doubles['lp'],
                               subset_tokens, subset_couplings, 'LP96')
        else:
            components.update(LP96_complete_prefix=0., LP96_answer_return=0., LP96_original_margin_and_return=0.)
            began = start()
            subset = {name: value[:0].copy() for name, value in bp_cpu.items()}
            finish('empty_subset_receipt_arrays', began)
        began = start()
        final = scatter(bp_cpu, subset, indices)
        finish('subset_scatter_and_provenance', began)
    began = start()
    clamps = ((host_tokens == 1) | (final['prediction'] == (host_tokens >= 2))).all((1, 2))
    final['accepted'] = (hidden_good & final['finite'] & final['stopped'] & clamps
                         & ~final['native_cap'] & ~final['block_cap']
                         & np.isfinite(final['own_margin']) & (final['own_margin'] >= -1e-6))
    final['hidden_good'] = hidden_good
    final['clamps_ok'] = clamps
    finish('final_numerical_acceptance', began)
    torch.cuda.synchronize()
    answer_seconds = time.perf_counter() - answer_started
    summed = sum(components.values())
    if summed > answer_seconds + 1e-9:
        raise AssertionError('Overlapping answer components')
    barrier_wait = after_answer() if after_answer is not None else 0.
    # All returns needed for an actual answer and its acceptance precede this
    # boundary. Scientific copies, trace packing and writing follow it.
    diagnostic_started = time.perf_counter()
    audit = {'couplings': a.cpu().numpy(), 'hidden_steps': hs.cpu().numpy()}
    if bp_native is not None:
        audit['bp_final_logits'] = bp_native[4].cpu().numpy()
    if core is not None:
        for name in ['prediction', 'iterations', 'stopped', 'finite']:
            audit['LP_' + name] = np.stack([core['outputs'][v][name].cpu().numpy() for v in range(1, 8)])
        for name in ['block_moves', 'post_block_sweeps', 'native_cap', 'block_cap', 'initial_simple_path']:
            audit['LP_' + name] = np.stack([core['blocks'][v][name] for v in range(2, 8)])
        audit['LP_binary_flip_count'] = core['prefix']['flip_count'].cpu().numpy()
        audit['LP_binary_cache_margin'] = core['prefix']['minimum_flip_margin'].cpu().numpy()
    torch.cuda.synchronize()
    diagnostic_seconds = time.perf_counter() - diagnostic_started
    return {'answer': final, 'bp_prefix': bp_cpu, 'lp_subset': subset,
            'geometry_valid': geometry, 'subset_indices': indices,
            'core': core, 'audit': audit,
            'timing': {'mode': mode, 'answer_seconds': answer_seconds,
                'components': components, 'unattributed_in_segment_seconds': answer_seconds - summed,
                'answer_started_monotonic': answer_started,
                'answer_finished_monotonic': answer_started + answer_seconds,
                'post_answer_barrier_wait_seconds': barrier_wait,
                'diagnostic_started_monotonic': diagnostic_started,
                'diagnostic_seconds': diagnostic_seconds, 'actual_subset_inputs': len(indices),
                'n': len(t), 'all_required_operations_inside_answer_segment': True}}

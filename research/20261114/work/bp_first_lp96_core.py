"""Candidate H2bn LP96-only dependency chain, awaiting review and freeze.

The algorithm mirrors frozen confirmation96_batch.py arms 1 through 7.
It performs no BP solve or intermediate-arm CPU answer copies. All internal
states and traces remain available to the caller for source-parity audits.
"""
import numpy as np
import study as S
import binary_coordinate as C
import bridge_veto_moves as VETO
import bridge_lexical_moves as LEX
import bridge_lexical_moves as BOUND
import bridge_terminal_moves as TERM


def solve(model_double, tokens, couplings, watchdog=None):
    torch, _, _ = S.imports()
    n = len(tokens)
    if not 1 <= n <= 32:
        raise ValueError('LP96 source batch must contain 1 to 32 inputs')
    cpu_tokens = tokens.cpu().numpy()
    outputs, blocks = {}, {}
    cache = C.BinaryQuadratic(model_double, tokens, couplings.double())
    prefix = C.solve(cache)
    outputs[1] = {'prediction': prefix['prediction'], 'iterations': prefix['sweeps'],
                  'stopped': prefix['stopped'], 'finite': prefix['finite']}
    if watchdog:
        watchdog()
    for v in range(2, 8):
        parent = v - 1
        base = outputs[parent]
        prediction = base['prediction'].clone()
        stopped = base['stopped'].cpu().numpy().copy()
        finite = base['finite'].cpu().numpy().copy()
        moves = np.zeros(n, np.int64)
        post_sweeps = np.zeros(n, np.int64)
        native_cap = ((~base['stopped'] & base['finite']).cpu().numpy()
                      if v == 2 else blocks[parent]['native_cap'].copy())
        block_cap = np.zeros(n, bool) if v == 2 else blocks[parent]['block_cap'].copy()
        simple = np.zeros(n, bool)
        raw = None
        initial_terminal = np.zeros(n, bool)
        terminal_records = []
        if v <= 3:
            if v == 3:
                simple = np.array([VETO.simple_path(t, b) for t, b in
                                   zip(cpu_tokens, prediction.cpu().numpy().astype(bool))])
            eligible = stopped & finite & ~simple
        else:
            margins = cache.delta(prediction).masked_fill(~cache.free, float('inf')).flatten(1).amin(1).cpu().numpy()
            stable = finite & ~native_cap & (margins >= -1e-8)
            if v == 4:
                initial_terminal, terminal_records = TERM.terminal_checks(
                    cpu_tokens, prediction.cpu().numpy().astype(bool), block_cap & stable)
                stopped[initial_terminal] = True
                block_cap[initial_terminal] = False
                eligible = blocks[2]['block_cap'] & block_cap & stable
            else:
                eligible = block_cap & stable
        indices = np.flatnonzero(eligible)
        if len(indices):
            if len(indices) == n:
                selected_cache, initial = cache, prediction
            else:
                ix = torch.tensor(indices, device=tokens.device)
                selected_cache = C.BinaryQuadratic(model_double, tokens[ix], couplings[ix].double())
                initial = prediction[ix]
            if v == 2:
                raw = VETO.solve(selected_cache, initial, watchdog)
            elif v == 3:
                raw = LEX.solve(selected_cache, initial, watchdog)
            else:
                raw = BOUND.solve(selected_cache, initial, watchdog,
                                  max_rounds=16 if v in [4, 6] else 32)
            if v >= 4:
                raw = TERM.certify_final(cpu_tokens[indices], raw)
            prediction[indices] = raw['prediction']
            stopped[indices] = raw['stopped']
            finite[indices] = raw['finite']
            moves[indices] = raw['block_moves']
            post_sweeps[indices] = raw['native_sweeps']
            native_cap[indices] = raw['native_cap']
            block_cap[indices] = raw['block_cap']
        outputs[v] = {'prediction': prediction,
                      'iterations': base['iterations'] + torch.tensor(post_sweeps, device=tokens.device),
                      'stopped': torch.tensor(stopped, device=tokens.device),
                      'finite': torch.tensor(finite, device=tokens.device)}
        blocks[v] = {'parent_arm': parent, 'valid_prefix_indices': indices,
                     'initial_simple_path': simple, 'initial_terminal': initial_terminal,
                     'initial_terminal_records': terminal_records, 'block_moves': moves,
                     'post_block_sweeps': post_sweeps, 'native_cap': native_cap,
                     'block_cap': block_cap, 'raw': raw}
        if watchdog:
            watchdog()
    return {'outputs': outputs, 'prefix': prefix, 'blocks': blocks}

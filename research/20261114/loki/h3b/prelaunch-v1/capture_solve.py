"""Pinned N-Queens search plus a read-only budget-prefix snapshot.

Derived from Apache-2.0 CRAM commit62bbe830c7d6f87f2ba18f30a28039258b986454.
Only snapshot allocations/copies and return formatting differ; moves, RNG,
stopping, compaction and block boundaries are unchanged.
"""
from cram.cram_nqueens import *

@torch.no_grad()
def solve_with_prefix(model, givens, *, chains=1, temperature=.3, cap=20000, seed=0, device='cpu', margin=1e-3, block=1 << 16,
          stop='energy', prefix_cap=20000):
    """Focused heat-bath Gibbs on the learned energy, `chains` independent chains per row of `givens` ((r, c) bool).

    A chain starts from the board holding only the given queens; given cells are never moved. Each move picks a cell
    uniformly among the free cells whose current state has the higher field (flipping it lowers the energy), or among
    all free cells when there is none; then it draws the cell's state from exp(-field / T), T = temperature * unit.
    Stopping test, both fitted from the training solutions inside the fit: `stop='energy'`: E <= tau + margin (tau: the
    highest training-solution energy); `stop='local'`: every free cell holds the state with the lower field by at
    least delta - margin (delta: the smallest such margin over every cell of every training solution), i.e. the board
    is a fixed point of the learned conditionals at least as sharp as every training solution (it reads the focus
    differences the move computes anyway, so it costs no arithmetic). A chain that reaches `cap` moves returns the
    lowest-energy board it visited. Only the learned energy steers the search.
    Flop (see move_flop, init_flop): every move of every chain, counted per chain; RNG, comparisons and indexing are
    not counted. Returns boards (r, chains, c) bool, stopped (r, chains), energy (r, chains), moves, flop (r, chains)."""
    dev = torch.device(device)
    S = model['S'].to(dev)
    B = model['B'].to(dev)
    c = S.shape[0]
    T = temperature * model['unit']
    thresh = model['tau'] + margin
    if stop not in ('energy', 'local'):
        raise ValueError(stop)
    delta = model.get('delta', 0.) - margin

    def passes(x, g, e, free):
        if stop == 'energy':
            return e <= thresh
        own = g.gather(2, x[..., None])[..., 0]
        other = g.gather(2, (1 - x)[..., None])[..., 0]
        return ((other - own >= delta) | ~free).all(1)
    Sg = S.permute(1, 3, 0, 2).contiguous()  # (j, s, i, a): the field change of every cell when cell j takes state s
    gen = torch.Generator(device=dev).manual_seed(seed)
    giv = torch.as_tensor(givens, device=dev).bool().repeat_interleave(chains, 0)  # (b, c), chain-major per row
    total = giv.shape[0]
    out_x = torch.zeros(total, c, dtype=torch.bool, device=dev)
    out_e = torch.zeros(total, device=dev)
    out_stop = torch.zeros(total, dtype=torch.bool, device=dev)
    out_moves = torch.zeros(total, dtype=torch.long, device=dev)
    px, pe, ps, pm = out_x.clone(), out_e.clone(), out_stop.clone(), out_moves.clone()
    for lo in range(0, total, block):
        captured = False
        ids = torch.arange(lo, min(lo + block, total), device=dev)
        x = giv[ids].long()
        free = ~giv[ids]
        g = fields(S, B, x)
        e = 0.5 * g.gather(2, x[..., None])[..., 0].sum(1) + 0.5 * B[torch.arange(c, device=dev), x].sum(1)
        best_x, best_e = x.clone(), e.clone()
        moves = torch.zeros(len(ids), dtype=torch.long, device=dev)
        stop_now = passes(x, g, e, free)
        live = ~stop_now
        out_x[ids[stop_now]] = x[stop_now].bool()
        out_e[ids[stop_now]] = e[stop_now]
        out_stop[ids[stop_now]] = True
        keep = live
        ids, x, free, g, e, best_x, best_e, moves = (t[keep] for t in (ids, x, free, g, e, best_x, best_e, moves))
        step = 0
        while len(ids) and step < cap:
            a = torch.arange(len(ids), device=dev)
            own = g.gather(2, x[..., None])[..., 0]
            other = g.gather(2, (1 - x)[..., None])[..., 0]
            unhappy = (own > other) & free
            score = torch.rand(x.shape, generator=gen, device=dev)
            pool = torch.where(unhappy.any(1, keepdim=True), unhappy, free)
            cell = torch.where(pool, score, torch.full_like(score, -1.)).argmax(1)
            local = g[a, cell]  # (b, a)
            gumbel = -torch.log(-torch.log(torch.rand(local.shape, generator=gen, device=dev).clamp_min(1e-30)))
            new = (-local / T + gumbel).argmax(1)
            old = x[a, cell]
            e = e + local[a, new] - local[a, old]
            g = g + Sg[cell, new] - Sg[cell, old]
            x[a, cell] = new
            moves += 1
            better = e < best_e
            best_x[better], best_e[better] = x[better], e[better]
            step += 1
            done = passes(x, g, e, free)
            if step == prefix_cap:
                hi = min(lo + block, total)
                px[lo:hi], pe[lo:hi], ps[lo:hi], pm[lo:hi] = out_x[lo:hi], out_e[lo:hi], out_stop[lo:hi], out_moves[lo:hi]
                px[ids] = torch.where(done[:, None], x, best_x).bool()
                pe[ids] = torch.where(done, e, best_e)
                ps[ids], pm[ids] = done, moves
                captured = True
            if done.any() or step == cap:
                fin = done if step < cap else torch.ones_like(done)
                res_x = torch.where(done[:, None], x, best_x)
                res_e = torch.where(done, e, best_e)
                out_x[ids[fin]] = res_x[fin].bool()
                out_e[ids[fin]] = res_e[fin]
                out_stop[ids[fin]] = done[fin]
                out_moves[ids[fin]] = moves[fin]
                keep = ~fin
                ids, x, free, g, e, best_x, best_e, moves = (t[keep] for t in (ids, x, free, g, e, best_x, best_e,
                                                                               moves))
        if not captured:
            # All chains in this block finished before the prefix budget.
            hi = min(lo + block, total)
            px[lo:hi], pe[lo:hi], ps[lo:hi], pm[lo:hi] = out_x[lo:hi], out_e[lo:hi], out_stop[lo:hi], out_moves[lo:hi]
    r = len(givens)
    moves = out_moves.view(r, chains).cpu().numpy()
    flop = init_flop(c) + moves * move_flop(c)
    final = (out_x.view(r, chains, c).cpu().numpy(), out_stop.view(r, chains).cpu().numpy(),
            out_e.view(r, chains).cpu().numpy(), moves, flop.astype(np.float64))
    prefix_moves = pm.view(r, chains).cpu().numpy()
    prefix = (px.view(r, chains, c).cpu().numpy(), ps.view(r, chains).cpu().numpy(),
              pe.view(r, chains).cpu().numpy(), prefix_moves,
              (init_flop(c) + prefix_moves * move_flop(c)).astype(np.float64))
    return prefix, final

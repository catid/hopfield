# A binary-energy-preserving continuous extension

This is a derivation for the frozen `GraphLatentEnergyMU` implementation, not a change to the trained model or hidden readout. The relevant source is [the answer energy, field and certificate](https://github.com/Prefrontal-AI/CRAM/blob/62bbe830c7d6f87f2ba18f30a28039258b986454/cram/cram_maze_hard_graph.py) and [the self-link support and damped logit update](https://github.com/Prefrontal-AI/CRAM/blob/62bbe830c7d6f87f2ba18f30a28039258b986454/cram/maze_h3.py).

Let `p` denote the continuous answer and `F = {i: token_i = 1}` the free cells. The model clamps walls to zero and endpoints to one. With fixed hidden readout, define

\[
E_\lambda(p)=E_0(p)+\lambda\sum_{i\in F}p_i(1-p_i).
\]

For every binary board, each added term is exactly zero. Consequently the energy of **every** binary board, every relative binary energy, and every true free-cell bit-flip cost are unchanged. The original `flip_delta` method is therefore the correct certificate for every value of λ. This transformation cannot repair a binary energy ordering that favors disconnected endpoints.

The added term is a compensated diagonal interaction,

\[
\lambda p_i-\lambda p_i^2.
\]

The source's answer field is the negative derivative of its non-entropy energy. Its modified free-cell field is therefore

\[
f_\lambda(p)_i=f_0(p)_i+\lambda(2p_i-1),
\qquad i\in F.
\]

Clamped fields remain ±10,000. The existing update becomes

\[
u^{k+1}=u^k+\eta\left[f_\lambda(\sigma(u^k))-u^k\right].
\]

The diagonal continuous Hessian changes by `−2λ` on free cells. Positive λ raises the energy of intermediate probabilities while leaving binary corners fixed; negative λ does the opposite. The native implementation's entropy uses an ε inside logarithms, while its logit update uses `u`; the added derivative above is exact independently of that unchanged numerical convention. Derivative checks subtract the exact source entropy to compare its non-entropy field directly.

Several superficially similar edits would be incorrect:

- Adding only `−λp²` changes binary energies by `−λp`; the linear compensation is necessary.
- Changing the scalar `kappa[0]` changes the self coefficient by an attention-dependent amount `δkappa * a_i0`. The learned self attention varies across cells. A single global open-cell bias cannot generally compensate all those changes.
- Adding a center term to `mu_kernel` without changing its assumed off-diagonal flip formula would produce a false certificate contribution. The explicit wrapper changes the continuous field and energy while retaining the original certificate.

The [implementation](binary_preserving.py) changes no trained tensor, readout or clamping rule. Before any scientific intervention, [validation](validate_binary_preserving.py) exhaustively checked two small graphs under all three frozen models and λ = −2, −1, 0, 1, 2: 2,880 binary-energy checks and 19,200 directed bit-flip checks. Augmented binary energies and true flip costs were bitwise unchanged. Native flip costs matched direct energy differences within 1.78e−14. Continuous autograd error was at most 1.07e−14 and central finite-difference error 7.22e−9. Model state hashes and clamped fields were unchanged. These finite checks validate the source-specific implementation; the binary invariance itself follows algebraically from `p(1−p)=0` at both binary values.

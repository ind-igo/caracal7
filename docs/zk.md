# Zero knowledge for the prover

A design, not a build. It says what the proof leaks, step by step of the box in `docs/protocol.md`,
what hides each leak, what it costs, and what has to be proved. The commitment layer follows
zo0k (Chiesa, Fenzi, Weissenberg 2026, `~/notes/raw/papers/zo0k.md`): zero-knowledge encodings
that pad a message with random symbols, a fold that commutes with the padding, masked sumcheck
rounds and a masked clear vector. The openings at the points are outside zo0k's model, because
our verifier must learn true evaluations for a quadratic identity; they get polynomial masks that
vanish on the grid, the STARK remedy, fitted to this prover's degree budget and to its running
claim. Honest-verifier zero knowledge is the target; Fiat-Shamir gives the rest. Nothing here
lowers soundness: every change adds message coefficients to codes the ledger already bounds, or
committed columns and claims of the kinds the ledger already charges.

Six earlier drafts were reviewed (Codex; the sixth also by Opus), and this one once more (section 8); each review found real errors, and the design is the
residue. The first draft put random rows on the grid (3.1 says why not). The second had masks
with `F` coefficients, a per-column point list and clear-line reads that broke the small-grid
identity. The third carried the mask products in an unbound column and cancelled the quotient
masks away from the DEEP point. The fourth gave the mask claim per-point coefficients and put
line coefficients under a functional of their own; both break the running claim, whose one
functional per point serves every column. The fifth bound the quotient-mask correction in a tree
committed after the DEEP point, which lets the prover choose the correction and proves nothing.
The sixth let extension slots stand for plain monomials, which evaluate outside `F` on the grid
and would let a prover commit `F2`-valued cells, and used low-degree masks that the
Frobenius-real column basis cannot hold. Three constraints shaped the result: **a committed column is opened at every point of the list
with the same functional as every other column**, so a mask can only be a fixed random vector
and nothing committed after `z` can bind anything; therefore the quotient masks must cancel in
the identity by themselves (3.4), which the decomposition's own freedom allows; and **every
extension slot's polynomial is either a multiple of a vanishing polynomial or `F`-valued on the
grid**, so that whatever a prover puts in any slot, every column's restriction to `H` stays in
`F` (2.1).

## 1. What leaks today

| step | region | what the verifier learns |
|---|---|---|
| 3 | `Z2` lines in the clear | every chain-end pair product and wiring product, `h2` elements per line |
| 5 | `Q3` in the clear | the small-grid quotient, `2 h2` elements |
| 7 | openings `c(z_p)` | `20 P` `F`-linear functionals of every column: every column at every point, quotient coordinates twentyfold |
| 9, 10 | the corner `(e1, e2)` | the last chain's last row of every column; each wiring line's own product |
| 15c, `l = 1` | opened leaves of `W`, `Z`, `Q` | `s_1` symbols of every column's codeword |
| 15c, `l > 1` and 17 | opened rows of `C_{l-1}` | `s_{l-1}` symbols of each of the eight folded columns |
| 15e | round polynomials | two functionals of the current folded vector per round |
| 16 | clear vector | the folded vector itself |
| 1, 3, 5 | Merkle roots | a deterministic function of the columns: a low-entropy witness can be confirmed offline |

The openings are the largest leak and the hardest one. Everything below the line is what zo0k
was built for.

## 2. The commitment layer (zo0k)

### 2.1 Level-1 rows: random symbols in every column's message

`Enc` becomes a zero-knowledge encoding: the packed message of a column, `N/4` symbols of `F4`,
is extended at the top of the coefficient index by `t` uniformly random `F4` symbols, then
Reed-Solomon encoded as before. The map from the `t` random coefficients to any `t` evaluations
at distinct points of `D` is a diagonal times a Vandermonde matrix, invertible; so any `t` opened
symbols are uniform and independent of the message (zo0k Proposition 3.19, perfect). The roots
become hiding for the same reason: every leaf depends on the random symbols.

`t` must exceed everything the verifier learns about that codeword, not only the `s_1` opened
symbols: the round polynomials of level 1 draw their masking from the same randomness (2.3), so
the entropy left after the openings must cover them. `t = s_1 + 16` per codeword.

**The extension.** This is the mechanism the rest of the design reuses. A column's message is
`stored_ext(c) = (stored(c), extra coefficients, random symbols)`. An extra coefficient is a slot
of one of two kinds. An `F`-slot holds one byte and stands for a polynomial `phi`; the running
claim's functional on it is `phi(z_p)`. An `E`-slot holds an `E` value in 32 bytes (twenty used,
one per element of the `F`-basis `b_j v^k` of `E`; the byte index is five binary digits) and
stands for a polynomial `phi`; the functional on its bytes is `phi(z_p)` times the basis element.
Both are tensor-structured in the binary digits and a vector on the odd digit, like everything
the units carry. The functional is zero on the random symbols. So the polynomial a column stands
for is `c(X) + sum_slots (slot value) phi_slot(X)`, whatever the slots hold. The level-1
consistency functionals extend by the code's own formula, `coord_tau(b_j s^i)` for `i >= N/4`.
Every column shares one extension layout, zero where a column has nothing to say; `y_1 = sum_c
beta_c stored_ext(c)` is longer by that layout and the extra slots ride into the tail as rows of
`Mat(y_1)`. The layout is its own binary index, the byte digit lowest, then the coefficient
index of each slot family, padded to a multiple of the digits the tail folds and placed above
`N` (which the folded digits divide), so a slot family's functional is a product over its bits
and the units of `pcs/tensor.mojo` gain one product per family and point with no odd digit; the
random symbols `t`, `t_l` round up to the same multiple.

**The slot rule.** The grid `H` lies in `F2`, and a column's values on `H` are in `F` (the
Frobenius-real form of spec 9.1) only because the mixed basis represents exactly those
polynomials. A slot standing for a plain monomial such as `X1^{h1} X2` would evaluate to an `F2`
value on `H`, and a prover who filled it in a witness column would commit `F2`-valued cells, which
the workload lowerings (P3) do not allow for. So every slot's `phi` is one of two kinds: a
multiple of `Z_{H1}` or `Z_{H2}`, which vanishes on `H`; or a **real monomial**
`X1^{2^{a1-1} a} X2^{2^{a2-1} b}`, whose values on `H` lie in `F` because `x_l^{2^{a_l-1}}` has odd
order dividing 126. Every slot below is of one of those kinds, so any column's restriction to
`H` is `c|_H` plus an `F`-valued function, and every check stays a polynomial identity over the
witness field; a prover filling foreign slots pays only the Schwartz-Zippel degree, which section
6 charges at the layout's maximum. Random polynomials that must act on `H` (the `theta` of 3.4,
the `chi` of 3.6) are therefore spanned by real monomials; the space of those below the column
bidegree has `4 m1 m2` dimensions, and the slots add as many more as needed.

Cost: the layout is the witness masks (3.2, `3 d` `E`-slots), the quotient columns' top rows and
masks (3.3, 3.4: `2 h2 + 4 (h1 + d) + 3 d1 d2 + d_theta` `F`-slots), `Q3`'s mask (3.6) and `4 t`
bytes of random symbols. With `d` the argument count of 3.2 (about 60 on the passport, 30 on ECDSA, 80 on Keccak,
whose 25 read shifts double in the second-order arguments) and `d1 d2` about 100: 15 KB per
column on the passport (2.6 percent of `N`), 5 KB on ECDSA (6 percent), 10 KB on Keccak (40
percent of its small `N`, which is the statement to measure first). The code's rate rises by the layout over `4 L`; the level-1
domain and query count are recomputed, and a grid whose domain has no slack takes the next
domain (Keccak's `64 x 384` does: 26,880 to 32,256 symbols).

### 2.2 Tail rows: fresh random rows at every committed level

Every tail level is a code switch: `Mat(y_l)` is re-encoded under a new code over `E`. The prover
appends `t_l` fresh uniformly random rows of eight `E` elements before encoding, so every level's
oracle is a zero-knowledge encoding with `t_l >= s_l + margin`. The old randomness rides inside
`Mat(y_l)` as ordinary rows; there is no separate mask oracle as in zo0k's Construction 2.15,
because the Ligerito fold carries the whole vector and the tensor-unit verifier prices the extra
rows as a longer row index and nothing else.

The fresh rows are never folded away: a row of level `l` is one element of `y_{l+1}`, so the
clear vector grows by `t_ell + t_{ell-1} / 8 + ...`, about `t + t/8`: on ECDSA from 162 elements to
about 310, 3 KB more, and the last level's domain and query count are recomputed for the longer
message. `t_l = s_l + 16`.

### 2.3 Round polynomials: masked by the random rows already in the vector

A round sends a degree-2 polynomial `s_d` with `s_d(0) + s_d(1)` fixed by the claim: two free
coefficients. zo0k masks them with per-round mask polynomials under a small zero-knowledge code
and folds the mask claims into the next relation (Construction 2.10). In the tail those masks are
already present. The vector being folded contains uniformly random rows (the level-1 random
symbols at `l = 1`, the fresh rows of `C_{l-1}` after), and the batched functional of step 15d is
nonzero on them: the consistency functionals `g_q` are the code's generator rows and do not vanish
there. A random row with weight pair `(w_0, w_1)` in the current digit contributes `m(X) w(X)` to
the round polynomial with `m` uniform linear: the multiples of `w`. Two rows with non-proportional
weight pairs span every degree-2 polynomial, so the rounds are uniform given the claim.

This is obligation Z3, and its rank condition is on the randomness left after the openings: the
opened symbols of a codeword fix `s` of its `t` random coefficients, and the round masks must come
from the other `t - s`. Hence the margins in 2.1 and 2.2. If Z3 fails at some level, zo0k's
construction is the fallback: three mask rows per level, `mu~` sent, `epsilon` sampled, the claim
masked as `epsilon T + mu~`, at the same cost. Characteristic 127 satisfies its `char != 2`.

### 2.4 Clear vector: one random column and an `epsilon`

At the last committed level `ell` the prover adds a ninth column `g` to `Mat(y_ell)`, a uniform
vector of the clear length, committed in `C_ell` before `rbar_ell`. After the rounds the prover
sends `T_g = <g, w~>`, the verifier samples `epsilon` nonzero in `E` (rejection, like the field
bytes), and step 16 sends `h = y_{ell+1} + epsilon g`. Step 17 checks, at every opened position,
`Enc(h)[s] = sum_a rbar[a] X[s, a] + epsilon X[s, g]`, and `<h, w~> = T + epsilon T_g`. `h` is
uniform; `T_g` is determined by `h`, `T` and `epsilon`. The opened rows of `C_ell` are hidden by its fresh rows (2.2), as at every level.

Soundness: the last consistency check now tests a combination of nine columns with the tensor
scalars and `epsilon`; the ledger's commitment bound gains one `epsilon` term of the same kind as
its tensor-challenge terms (a far `g` is caught over `epsilon`), charged like the others.

Cost: one `E` per opened row at the last level, one `E` for `T_g`, no change to the clear length.

## 3. The openings at the points

### 3.1 Why random rows on the grid do not work here

The STARK remedy for evaluation leaks is extra random rows in the trace that every constraint
skips. On this grid a row is a chain and a constraint skips it only if the family carries a
factor that vanishes there. The residual's bidegree bound, below `(2 h1, 2 h2)`, is exactly what a
degree-2 family with one linear gate `(X1 - e1)` or `(X2 - e2)` fills; there is no room for a
selector on quadratic terms, which is why grouped families use exclusive columns instead of
selectors and why booleanity and the lookup relations are global. Random cells break the
exclusive-column trick and booleanity outright; a selector on the accumulator transition is
degree 3; gates of degree above one need a residual grid above `2 h2`, which for these grids means
`4 h2` and doubles the LDE and residual, forty percent of a wide prove. Beyond degree, the point
list defeats grid rows: at `(1, z2)` only one cell per random chain contributes, and the shifted
points `(omega1^k z1, z2)` share `z2`, so a random row at one `x1` masks one point of a column,
not the several it is read at. Lookup record columns cannot hold random bytes, because the
sorted copy must contain them.

### 3.2 Polynomial masks in the extension

A column carries small random polynomials that vanish on the grid:

    c'(X1, X2) = c(X1, X2) + Z_{H1}(X1) rho_c(X2) + Z_{H2}(X2) (sigma_c(X1) + X2 tau_c(X1)),

`rho_c`, `sigma_c`, `tau_c` of degree below `d` with coefficients in `E`. On `H` every mask
vanishes, so every family holds for `c'` exactly as for `c`, no gating. The coefficients are
`E`-slots whose `phi` is `Z_{H1}(X1) X2^j`, `Z_{H2}(X2) X1^j`, `Z_{H2}(X2) X2 X1^j`: the vanishing
factor is inside the functional, so `c'` is what the verifier evaluates, and the ordinary slots
keep holding the Frobenius-real `c` (an `E`-coefficient mask cannot be folded into the mixed
basis). `c'` has bidegree `(h1, h2 + 1)`.

Why `E` coefficients: a mask with `F` coefficients evaluates at a grid coordinate to an `F` value,
one byte of entropy against the twenty of an opening at `(1, z2)`; and even at a generic point its
value spans only `d` of the twenty `F` dimensions of `E`. With `E` coefficients one coefficient
hides one opening. Why `tau`: `(1, z2)` and `(1, omega2 z2)` have the same `Z_{H2}` value and the
same `sigma(1)`, so without an `X2`-dependent term they would share a mask and the difference of
the two openings would leak. `tau` costs one more degree in `X2`, paid in 3.3.

What the masks give, per column and point:

- a generic point `(z1, z2)`: `Z_{H1}(z1) rho(z2) + Z_{H2}(z2) (sigma(z1) + z2 tau(z1))`;
- the shifted points `(omega1^k z1, z2)`: the same `rho(z2)`, different `sigma`, `tau` values
  (Vandermonde in `z1` once `d` exceeds the number of shifts);
- the next-chain point `(z1, omega2 z2)`: `rho(omega2 z2)` and `omega2 z2 tau(z1)` differ;
- the chain lines `(1, z2)`, `(e1, z2)`, `(1, omega2 z2)`, zero rows `(b, z2)`: `Z_{H1}` vanishes,
  the mask is `Z_{H2}(z2) (sigma(a) + z2 tau(a))` at the line's `a`, distinct per line and per `z2`;
- a restriction line `(z1, 1)`: `Z_{H1}(z1) rho(1)`; the small-grid point of 3.6: both terms;
- the corner `(e1, e2)`: no mask. The design requires the first and the last chain idle (no
  family live there, every cell the public filler or a restriction's public line) and every
  family that reads the next chain masked off at `x2 = 1` (the group mask does this for grouped
  families; an ungrouped family with a next-chain read makes chain 2 idle as well): then the
  corner openings carry no witness, and the residual restricted to chain 1 or `e2` is a fixed
  function of public cells and the masks `rho(1)`, `rho(e2)`, which 3.4 needs.

These masks are for the witness and accumulator columns. The quotient columns carry only the
cancelling masks of 3.4, the lines only their `lambda` (3.6) and `Q3` only its `chi`, because any
mask that does not vanish at the small-grid point or on the lines the small grid reads would
enter those identities.

Every column is opened at every point, as today, and (3.4) the residual's value at every listed
point is revealed too, which reads the columns at the points' own shifts. A column's mask has
to separate every argument that occurs: `d` is the number of distinct `x1` values among the
list's points and their read shifts, and among the chain lines and their shifts, plus the same
for `x2`; at a grid argument `a` the mask has two dimensions, `sigma(a)` and `tau(a)`, one per
distinct `x2` it is paired with. About 30 on ECDSA, 60 on the passport, 80 on Keccak; the mask
holds `3 d` `E` coefficients, `96 d` bytes.

Pinned openings must not be masked: `Z_k(1, z2) = 1`, `R_k(1, z2) = start`, zero rows
`c(b, z2) = 0`, restricted columns on their lines. The prover zeroes the mask there
(`sigma(1) = tau(1) = 0` for accumulators, `sigma(b) = tau(b) = 0` for zero-row columns,
`rho(1) = 0` for restricted columns), one linear condition each, and the verifier checks nothing
new because the value it expects is what it gets.

### 3.3 The residual with masked columns

Substituting `c'` for `c` in a quadratic term `a b`:

    a' b' = a b + Z1 (a rho_b + b rho_a) + Z2 (a s_b + b s_a)
              + Z1^2 rho_a rho_b + Z2^2 s_a s_b + Z1 Z2 (rho_a s_b + rho_b s_a),

`Z1 = Z_{H1}(X1)`, `Z2 = Z_{H2}(X2)`, `s = sigma + X2 tau`, shifts applied to the arguments.
Summed over the entries with their coefficients, gates and `alpha` powers, `R'` is `R` plus
terms that are all multiples of `Z1` or `Z2`, so `R'` vanishes on `H` exactly when `R` does, and
`R'` has bidegree `(2 h1 + 1, 2 h2 + 3)`: `Z1^2 rho rho` reaches `X1`-degree `2 h1` plus a gate;
`Z2 (a s_b)` reaches `X2`-degree `2 h2` with no gate at all, because `tau` adds a degree, and
`Z2^2 s s` reaches `2 h2 + 2` plus a gate.

A polynomial vanishing on `H1 x H2` of bidegree `(n1, n2)` is `Z1 U + Z2 W` with `U` of bidegree
`(n1 - h1, n2)` and `W` of bidegree `(n1, n2 - h2)`; today `n = (2 h1 - 1, 2 h2 - 1)` and
`U = A + X2^{h2} B`, `W = Q2`, all three of the column shape. The division that produces it (by `Z1` in `X1`, the remainder by `Z2`) gives `U` bidegree
`(n1 - h1, n2)` and `W` bidegree `(h1, n2 - h2)`. With the masks `U` has bidegree `(h1 + 1,
2 h2 + 3)` and `W` `(h1, h2 + 3)`: the same three quotient columns with extra top rows in their
extensions, held as vanishing slots so the slot rule holds: `Z_{H1}(X1) X2^j` and
`Z_{H1}(X1) X1 X2^j` for `A` and `B` (`2 h2` `F`-slots per coordinate column each),
`Z_{H2}(X2) X2^k X1^i`, `k = 0..3`, for `B` (`4 (h1 + 2)`) and for `Q2` (`4 h1`). The identity
is then exactly

    R' = (A' + X2^{h2} B') Z1 + Q2' Z2.

How the prover computes it. The ordinary reads stay `F`-valued: the residual pass computes `R`
on `G` from the unmasked LDE as today, in the fp32-lane kernel, and adds the mask terms in a second
pass that is linear in the columns. Grouped per column and read shift,
`Z1 D1 + Z2 D2 = sum_a a(shifted x) Lambda_a(x)`, where `Lambda_a` collects the mask polynomials
of every entry that reads `a` with that shift, weighted by the entry's coefficient, gate and
`alpha` power: an `E`-valued function that is `rho`-part in `x2` only and `sigma`, `tau`-part in
`x1` only, times the entry's gate, so it is tabulated on `G1` and `G2`, not on `G`: up to six
small tables per column and shift, one per gate kind and mask kind. The LDE's values on `G` are in `F2`, so this is one `F2 x E` product per column, read shift and
grid point: about two thirds of the base residual's work, which has one `F2 x F2 x E` product per
entry and grid point and about four entries per column-shift. The
mask-only products `Z1^2 rho rho + ...` are polynomials of bidegree below `(2, 2 d)` and so on,
computed from coefficients, negligible. The top coefficients of `R'` are read off the same
polynomials in coefficient form (the columns' top `X1` and `X2` coefficient rows, which the prover
holds before `k_to_stored`, times the mask coefficients; every entry contributes, gated or not):
`c_top(X2)` at `X1^{2 h1}`, `c_top'` at `X1^{2 h1 + 1}`, `d_k(X1)` at `X2^{2 h2 + k}`. The prover
subtracts `Z1^2 (c_top + X1 c_top') + Z2^2 sum_k X2^k d_k`, which removes exactly those
coefficients, and what is left, `R''`, has bidegree below `(2 h1, 2 h2)`, vanishes on `H`, is
exact pointwise on `G` and is decomposed as today. The subtracted terms are `Z1 . Z1 (c_top + X1
c_top')` and `Z2 . Z2 sum_k X2^k d_k`: the top-row slots of `U` and `W` above, added in coefficient
form.

Step 12 becomes: from the masked openings compute `R'(z)`; abort unless
`R'(z) = (A'(z) + z2^{h2} B'(z)) Z1(z1) + Q2'(z) Z2(z2)`, the quotient openings masked as in 3.4.

### 3.4 Quotient openings: masks that cancel in the identity

`A'`, `B'`, `Q2'` are opened at `z` (three values, one identity) and at the other points (no
identity); their evaluations are functions of the witness. Nothing committed can carry a
correction for a mask (a column is opened everywhere with one functional, and a column committed
after `z` is free), so the masks must cancel in the identity by themselves. The decomposition
`R' = Z1 U + Z2 W`, `U = A' + X2^{h2} B'`, `W = Q2'`, has exactly that freedom: for any `phi`,
`(U + Z2 phi, W - Z1 phi)` is another decomposition, and inside `U` the split into `A'` and `B'`
is free by `(A' + X2^{h2} theta, B' - theta)`. The prover commits, per coordinate column `i`,

    A''_i = A'_i + Z2 phi_i + X2^{h2} theta_i,   B''_i = B'_i + Z2 psi_i - theta_i,
    Q2''_i = Q2'_i - Z1 (phi_i + X2^{h2} psi_i),

with `phi_i, psi_i` uniformly random `F` polynomials of bidegree `(d1, d2)` in vanishing
`F`-slots (`Z_{H2}(X2) X1^a X2^b` on `A`, `B`; `Z_{H1}(X1) X1^a X2^b` and
`Z_{H1}(X1) X2^{h2+b} X1^a` on `Q2`), and `theta_i` a uniformly random combination of real
monomials: `- theta_i` in `B`'s ordinary coefficients where the monomial is below the column
bidegree and in real `F`-slots above it, `X2^{h2} theta_i` in real `F`-slots on `A`. The identity
is unchanged,

    R' = (A'' + X2^{h2} B'') Z1 + Q2'' Z2,

with no correction term. Twenty coordinate columns with independent masks make the single-value
openings (3.5) uniform in `E`: a real monomial evaluates in `F` at a grid point, but twenty of
them under the basis elements span `E`.

What the verifier sees, per point:

- At every point: three values whose masks `(Z2 phi + z_p2^{h2} theta, Z2 psi - theta,
  -Z1 (phi + z_p2^{h2} psi))` span the identity's kernel, rank 2 in three unknowns, because the
  identity is a polynomial identity and holds at every point, not only where the verifier checks
  it. So at `z` the view is uniform on the identity's affine subspace, and at every other point
  the view is uniform plus one value, `R'(z_p)`, the residual at that point. `R'(z_p)` is a fixed
  polynomial in the masked columns' values at the point's own read shifts (second-order
  arguments such as `(omega1^{k+k'} z1, z2)` and `(z1, omega2^2 z2)`), which are not in the list
  and are uniform through fresh mask dimensions when `d` counts them (3.2); a fixed function of
  values the simulator samples anyway. Joint uniformity of the masks over the list needs `d1`
  above the number of distinct `x1` arguments (the shifts share `z2`, so only the `X1`-degree
  separates them) and `d2` above the `x2` ones; obligation Z2.
- On a chain line `(a, z2)`, `a` a grid value: `Z1 = 0`, so `Q2''` is unmasked and equals
  `W(a, z2) = R'(a, z2) / Z_{H2}(z2)`: the residual restricted to the line `x1 = a`, a fixed
  polynomial in the masked columns' values at `(omega1^k a, z2)` and `(a, omega2 z2)`, which are
  uniform through `sigma`, `tau` (3.2). A fixed function of uniform inputs is simulatable, and
  those inputs are the same masked values the simulator already produces for the openings.
- On a chain `(z1, b)`, `b` in `{1, e2}` (restriction lines): `Z2 = 0`, so `A'' + B'' = U(z1, b)
  = R'(z1, b) / Z_{H1}(z1)`, the residual on chain `b`, which reads the columns at
  `(omega1^k z1, b)` where the only mask is `Z_{H1}(z1) rho(b)`, the same for every shift. This
  is where the idle chains are needed: on an idle chain the cells are filler that does not
  depend on the witness (zeros, random bits or bytes of the column's kind, table rows), so the
  residual there is a fixed function of filler and masks. `A''` and `B''` separately are masked
  by `theta`.
- At the corner `(e1, e2)`: `A'' + B''` as above; `Q2''` is `W(e1, e2)`, the `X2`-derivative of
  `R'` at the corner divided by `Z_{H2}'(e2)`, a fixed function of the columns' corner cells
  (filler) and their `X2`-derivatives there, each uniform through `sigma(e1) + e2 tau(e1)`.

The idle chains are a frontend obligation, not a profile flag: `pad_trace` fills trailing chains
of a group, so a statement declares chains 1 and `e2` outside every group, and the lookup's
sorted copy, which is the prover's, must place table rows there (its first and last cells are
the table's smallest and largest rows whatever the witness, but a whole chain of them holds
only if the table filler is large enough; the dummy rule of `docs/milestone-3-lookup.md`). What
the verifier learns on those chains is then a function of filler, masks and public data; the
per-workload check is Z5.

Soundness: `A''`, `B''`, `Q2''` are quotient columns like any other; a valid decomposition is a
valid decomposition. Cost: about `3 d1 d2 + d_theta` `F`-slots per coordinate column, a few
hundred bytes.

### 3.5 Single-value openings of `E`-valued columns

An `E`-valued column (`Z_k`, `R_k`, `A`, `B`, `Q2`, the lines of 3.6) is committed as twenty
coordinate columns and was opened as twenty `E` values per point, four hundred `F` functionals
for the twenty the identity needs. The claim `A(z) = sum_i b_i A_i(z)`, `b_i` the basis elements, is a claim on the
virtual column `sum_i b_i stored(A_i)`, a fixed vector, tested by the fold with one `beta` under the
shared functional; the verifier needs nothing else. Openings of `E`-valued columns become one `E`
value per point: on the passport the `Z` and `Q` openings drop from 200 columns to 10. Done
before any zero knowledge (build order 1, `docs/profile.md` has the sizes): the prover's
`compact_openings` and `expand_beta`, the verifier's `_expand_beta`, and its residual reading a
coordinate column `Z_t` as the one value at `t = 0` and zero at `t > 0`, since an accumulator's
entries are twenty copies that differ only in the basis factor. Without it the masks would have
to hide twenty coordinate openings per point.

### 3.6 The small-grid lines

A line of `h2` witness-derived elements has no free degree of freedom, so in a zero-knowledge
proof `Z2` and `Q3` are committed, not sent. A `Z2_k` becomes an ordinary `E`-valued column of
the `Z` tree, constant along `x1` (a polynomial in `X2` alone is a polynomial of the column
shape), with one mask `Z_{H2}(X2) lambda(X2)`, `lambda` of degree below `d` in `E`, in extension
slots with `phi = Z_{H2}(X2) X2^j`, and none of the grid masks of 3.2. Its opening at any point is
`L(z_p2) + Z_{H2}(z_p2) lambda(z_p2)`, so the small grid reads `L(z2)` and `L(omega2 z2)` at
`(1, z2)` and `(1, omega2 z2)`, masked by two independent `lambda` values, and the closing reads
`L(e2)` at the corner, unmasked because `Z_{H2}(e2) = 0`. Constancy along `x1` is not implied
by the commitment (a prover could put different slices under the two reads and the closing),
so each line gets the transition family `L(omega1 X1, X2) - L(X1, X2) = 0` on the grid, twenty
linear entries.

The small-grid residual with masked reads. Every read the small grid makes is at a grid `x1`
(`(1, .)` or `(e1, .)`) or is a line, so every mask on it is a multiple of `Z_{H2}(X2)`; the
masked residual is `R2' = R2 + Z_{H2} D` with every mask term, of any degree in the masks, inside
`D`, and `Q3' = R2' / Z_{H2}` exactly. Its degree is the largest over the small-grid families of
`(family degree) (h2 + d) - h2`: below `2 h2 + 3 d` for the chain-end families
`Z2 Z(e1, .) N(e1, .)`, and `(S + 1)(h2 + d) - h2` for the wiring line below.

**`Q3` as one column.** A polynomial in `X2` of degree below `m h2` is stored as the column
`Q(X1, X2) = sum_{i<m} X1^i Q3_i(X2)`, `Q3 = sum_i X2^{i h2} Q3_i`, of bidegree `(m, h2)`, which fits
the column shape for `m <= h1`; opened at the point `(z*, z2)` with `z* = z2^{h2}` it gives
`Q3(z2)`. So `Q3'` is one `E`-valued column of the `Q` tree, carrying no mask of 3.2 (any would enter
the identity), and the point list gains `(z*, z2)`, a new point kind for `required_points`, the
verifier's reads and the units (today a point is `z` with coordinates fixed or shifted); every
column is opened there too, masked as in 3.2. Whatever column the prover commits, `Q(z*, X2)` is
a polynomial in `X2`, so the identity `R2'(z2) = Q(z*, z2) Z_{H2}(z2)` is a univariate identity
and needs no other guard; its degree is charged at the layout's maximum, about `(h1 + 3) h2`. The stacking has a freedom of its own
that masks `Q3`'s openings at the other points: `Q + (X1 - X2^{h2}) chi` has the same values on
the curve `x1 = x2^{h2}`. The slot rule constrains `chi`: on `H` the factor is `x1 - 1`, in `F2`,
so `chi` is `(X1^{127} - 1) psi + Z_{H2}(X2) kappa` with `psi`, `kappa` random combinations of real
monomials; then the mask is `F`-valued on `H` (`(x1 - 1)(conj(x1) - 1)` is real), vanishes on the
curve, and at every other point of the list is nonzero through `psi` where `z_p1` is off the grid
and through `kappa` on the chain lines `(1, z2)`. Its monomials are real `F`-slots per coordinate
column, a few hundred.

**Starts and closings.** `Z2_k(1) = 1` becomes the small-grid family
`(Z2_k - 1) Z_{H2}(X2) / (X2 - 1)`, which `Q3` divides exactly when the start holds (the factor is
`h2` at `X2 = 1`, nonzero because 127 divides no admissible `h2`); the verifier evaluates the
public factor as `(z2^{h2} - 1)/(z2 - 1)`. The closing of a grand product reads
`Z2_k(e2)`, `Z_k(e1, e2)` and the corner factors: with the last chain idle, `Z_k` is constant
along it and equal to its pinned start, the corner factors are filler, and `Z2_k(e2)` is the
public constant (1 or the table constant).

**Wiring.** The wiring products are separate lines `Z2_g` because a line's recurrence may have
degree at most three on the small grid today, and only their joint product at `e2` is pinned;
each `Z2_g(e2)` on its own is a witness-dependent ratio and would leak. With `Q3` stored as
above its degree is bound by `h1 h2`, not `2 h2`, so the wiring products become **one line**
`Z2_w` with the recurrence `Z2_w(omega2 X2) prod_s D_s = Z2_w(X2) prod_s N_s` over all `S` wired
slots (ECDSA has eleven in six products; degree `S + 1`), `Q3` of degree below
`(S + 1)(h2 + d) - h2`, `m = S + 2` pieces at most, and its closing `Z2_w(e2)` is the pinned joint
value. A zero factor `D_s` on one chain would leave the merged recurrence `0 = 0` from that
chain on; the public factors keep step 10's zero check, and a witness factor
`w + gamma_w + beta_w sigma` is zero with probability `1/|E|` over `gamma_w`, charged to the
ledger's `wire_zero_factors`. The small-grid stage computes `Q3` on `(S + 2)/2` cosets instead of
one, milliseconds.

Cost: twenty `F` columns per line (one per grand product, one for wiring) and twenty for `Q3`, in
encode, Merkle and level-1 rows, one `E` per point each in the openings; one more point. On the
passport, whose three lines are all wiring: one line and `Q3`, 40 columns on 485, +16 KB of
level-1 rows and +6 KB for the extra point (single-value openings), against 403 KB of clear
lines. On ECDSA the same 40 columns, +12 KB of rows against 92 KB of lines.

## 4. The protocol after the change

Relative to the box of `docs/protocol.md`:

- Step 1: every column's message is `stored_ext`: the column, its mask coefficients, the shared
  extension layout, `t` random symbols. The first and the last chain are idle.
- Step 3: `Z2` lines are masked columns of the `Z` tree, constant along `x1` by a family; one
  wiring line.
- Step 5: the residual with masks; `A''`, `B''`, `Q2''` with top rows and the cancelling masks
  inside; `Q3` as one stacked column.
- Step 6: `z`; the point list gains `(z2^{h2}, z2)`.
- Step 7: openings of the masked columns, every column at every point, one value per `E`-valued
  column.
- Steps 9 to 13: boundaries at pinned openings as today; the residual and small-grid identities
  on the masked openings; `Z2_k(1) = 1` as a small-grid family; the closings at `e2` against the
  pinned constants.
- Step 14: the running claim over the columns and the points; the functionals on the extension
  slots as in 2.1, 3.2, 3.3, 3.4, 3.6.
- Step 15a: `Mat(y_l)` gains `t_l` fresh random rows; at `l = ell` also the column `g`.
- Step 16: `T_g`, then `epsilon`, then `h`.
- Step 17: the consistency check includes `epsilon X[s, g]`; the claim check includes
  `epsilon T_g`.
- Section 4: domain separators for `T_g` and `h`; `epsilon` after `T_g`.

## 5. Cost

Prover: the extensions add two to eighteen percent to the encode (2.1, largest on the smallest
grid, where the domain also grows a step); the residual's mask pass about two thirds of the
residual stage; twenty to sixty more columns (the lines, `Q3`), up to ten percent of the encode
on the wide statements; the constancy families. About +10 percent on the hash statements and
+15 on the wide ones. Verifier: the units gain one product per slot family and point, about double today's
sixteen per point, one more point, and the mask terms at `z`. Proof:
the clear lines go (400 KB on the passport, 92 KB on ECDSA, 36 KB of `Q3` on Poseidon), the
`E`-column openings shrink twentyfold (3.5: 200 KB on the passport, 73 KB on ECDSA, 32 KB on
Keccak, 23 KB on SHA-256), the tail's clear vector grows by 3 KB, the level-1 rows grow with the
higher rate (more queries) and the new columns. Net: every statement's proof is smaller than today, the wide ones
by a third or more.

## 6. What has to be proved

Soundness: the level-1 code at the higher rate and the tail codes with `t_l` more rows, every
count in `docs/soundness.md` re-derived. The extensions hold message coefficients with tensor
functionals: P4's units argument covers them. The virtual-column claims (3.5) are claims on fixed
combinations of committed columns under the shared functional, tested with their own `beta`: P4
again. The `epsilon` term of 2.4. The masked residual identity is a polynomial identity checked
at a random `z`, Schwartz-Zippel at the layout's maximum bidegree, about `(2 h1 + 2 d, 2 h2 + 2 d)`
whatever slots a prover fills (every slot is a vanishing multiple or a real monomial, so the
restriction to `H` stays `F`-valued); its right-hand side is a multiple of the vanishing polynomials by
construction, so `R'` vanishing on `H` follows as today, `R'` and `R` agree on `H` and are
`F`-valued there, and P2's argument goes through. The small-grid identity likewise with the
stacked `Q3` at the layout's maximum, about `(h1 + 3) h2`, and the `small_grid` term recharged at
that degree;
the wiring line's degree-`(S + 1)` family sits under it.

Zero knowledge, each an own obligation with a numerical check before the proof:

- **Z1, openings.** For a column with the point list and mask degree `d`, the map from the mask
  coefficients (minus the pinned ones) to the mask values at the `P + 1` points has full rank over
  `F` for uniform `z`, except with probability of order `d h / |E|`. Vandermonde structure in `z1`
  and `z2`, and `1, z, ..., z^{19}` spanning `E` over `F`. Check: rank on every benchmark statement.
- **Z2, quotients.** The three quotient openings are uniform on the identity's subspace at `z`
  and jointly uniform over the generic points (rank of the `phi`, `psi`, `theta` masks over the
  list, with `d1` above the shift count); on the grid lines and at the corner the unmasked
  values are fixed functions of masked column values and public cells, as listed in 3.4, and
  the simulator computes them from the same sampled masks it uses for the openings. Check: the
  rank computations, including the masked values on the lines the residual reads.
- **Z3, rounds.** At every level, the batched functional restricted to the randomness left after
  the openings has rank at least two per round, over the batching scalars. Check: rank per round
  on every benchmark. Fallback: zo0k's mask rows and `epsilon` per level.
- **Z4, composition.** The simulator: sample the challenges, the masks, the `r_i`, the random
  symbols, the fresh rows and `g`; produce openings uniform subject to the pinned values and the
  identities (Z1, Z2), rows uniform (2.1, 2.2), rounds uniform (Z3), `h` uniform (2.4), `T_g` from the
  checks, the line values of 3.4 from the masks. zo0k's composition lemma (their 2.6, distinguishers with at
  most `t` queries) level by level, plus the grid part. Honest-verifier; Fiat-Shamir gives the rest.
- **Z5, the idle chains.** Everything read at the corner is filler or a pinned accumulator end,
  the closings compare against public constants only, and the residual on chains 1 and `e2`
  reads only idle cells and masks. Check: the compiled statement rejects, under the `zk`
  profile, a live first or last chain and a family that reads the next chain without a mask at
  `x2 = 1`.

## 7. Build order

1. Single-value `E` openings (3.5). Done. No zero knowledge yet; a proof-size win on its own, and the
   virtual-column claim is what 3.6 needs.
2. The extension mechanism (2.1 without the random symbols): `stored_ext`, the layout, the
   extended functionals and consistency rows, the tail's longer row index, and the level-1 rate
   and query recomputation with it (the message is longer from this step on).
3. The idle chains and the frontend rule (3.2, Z5), then the lines as columns with their
   constancy family, `Q3` as one column with the new point kind, and the wiring merge (3.6): the
   start family, the multi-coset small grid, the verifier's reads. Proof bytes per
   `docs/profile.md`.
4. The masks and the masked residual (3.2, 3.3, 3.4): mask coefficients, the mask pass, the top
   rows, the cancelling quotient masks. Z1 and Z2 rank checks land as tests.
5. The random symbols, the tail's fresh rows and the masked clear vector (2.1 to 2.4), the tail
   rates and query counts. Z3's rank check.
6. The simulator write-up (Z4), the ledger refresh, a `zk` profile flag next to `CLIENT`, off by
   default until the write-up holds.

Each step lands with its checks as tests and its numbers in `docs/profile.md`.

## 8. Open issues from the last review

The seventh review found no structural break but five points the build must settle first:

1. **Mask degree against the chain length.** Keccak needs about 80 distinct `x1` arguments and
   has `h1 = 64`, so a `sigma` of degree 80 pushes `Z2 c sigma` to `X1`-degree 142, past the
   two extra rows of 3.3. Either `U` takes more rows (the division allows any number; each row
   is `h2` slots per coordinate column) or the `x1` arguments are split between `sigma` and a
   second mask in `X1 X2`.
2. **A third dimension at grid arguments.** At `x1 = e1` a column has two mask dimensions,
   `sigma(e1)` and `tau(e1)`, and a column read at `(e1, z2)`, at the next chain and in the corner
   derivative gives the verifier three functionals of them. One more term, `Z_{H2} X2^2 upsilon(X1)`,
   gives the third dimension at the cost of one more `X2`-row in 3.3.
3. **`chi`'s slots.** `(X1 - X2^{h2})((X1^{127} - 1) psi + Z_{H2} kappa)` is `F`-valued on `H`
   only as a whole; its monomials, held in separate slots, are not, so the slot rule of 2.1 is
   violated by a prover who fills them independently. `chi` needs a construction from real
   monomials and vanishing multiples alone, or `Q3`'s openings away from its point need another
   hiding argument.
4. **Real monomials do not separate every shift.** `X1^{2^{a1-1} a}` takes the same value at
   shifts that differ by `h1 / 2^{a1-1}` (18 on the passport: shifts 14, 32 and 122 collide),
   so a real-monomial mask (`theta`, `chi`) has fewer independent values over the shifted points
   than points. `theta` acts only where `Z_{H2} = 0`, single points, and is unaffected; `chi` is,
   and it joins item 3.
5. **Keccak's domain.** The layout of 2.1 does not fit the next domain either at rate 1/4
   (`32,256` symbols leave 7,680 bytes of slack, the witness masks alone); Keccak's `64 x 384`
   needs `40,320` or a smaller `d`. The encode and rate numbers of section 5 for the hash
   statements are to be measured, not taken from the estimates.

The obligations Z1 to Z5 stand; items 2 to 4 sharpen Z2, items 1 and 5 the cost.

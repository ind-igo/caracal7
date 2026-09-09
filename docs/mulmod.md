# Arithmetic mod p: the product as a polynomial identity, additions, the canonical check

The polynomial-mulmod spec on the builder (`relations/mulmod.mojo`): `a b = r`, then `r` folded to
`f < 2^257`, congruent mod secp256k1's `p`, with no product ever committed (items 1 to 8); additions and
subtractions as a bit-level second column group; the canonical check `f < p`. A circuit is a list of chains
of those four kinds. Read the spec for the argument; this doc records what the code does where the spec left
a choice, one hole the spec's zero row closes, the selector columns the fold needs, and the wiring.

## Layout (`q = 4`, 144 rows)

Row `x` of a chain holds weight slots `4 (142 - x) + j`, `j = 0..3`, so a Horner scan with scale `zeta^4`
over rows 0 to 142 evaluates a bit column at `zeta` (position weight `zeta^j`). Row 143 holds no weight and
no accumulator reads it. Every bit is stored at the slot of its own weight:

| columns | count | contents |
|---|---:|---|
| `a{t}{j}` | 12 | bit `i` of `a` at slot `i`, in the columns of piece `t = min(i // 88, 2)` |
| `b{j}` | 4 | bit `i` of `b` at slot `i` |
| `c{t}{m}{j}` | 84 | bit `m` of the coefficient of `A_t B` at weight `s`, at slot `s + m`; `c = 64 b6 + 32 b5 + v`, `b6 b5 = 0` |
| `r{j}` | 4 | the product bit at slot `w` |
| `y{k}{j}` | 20 | the carry out of slot `w` (five bits), stored at slot `w` |
| `h{j}` | 4 | the high half of `r`: bit `256 + i` at slot `i` |
| `o{j}`, `z{k}{j}` | 4 + 12 | the first fold `lo + sum_s (hi << s)` and its carries (three bits) |
| `g{j}` | 4 | the high half of `o` at slot `i` |
| `f{j}`, `v{k}{j}` | 4 + 12 | the second fold, the chain's output, and its carries |
| `ax{j}`, `ay{j}`, `as{j}` | 12 | the addition chain `x + y = s + q p` |
| `aq1`, `aq2` | 2 | `q = q1 + 2 q2`, chain-constant |
| `ac{k}{j}` | 12 | the addition's signed carry `c0 + 2 c1 - 4 cn` in `[-4, 3]`, stored at its slot |
| `lo` | public | 1 on rows 79 to 142 (weights below 256), 0 elsewhere; the same on every chain |
| `bd` | public | 1 on weights below `WIDTH = 260`: every chain value's bound |
| `cp` | public | 1 on weights below 264: the copy's rows, the high half of a product of two bounded values |
| `cn` | public | 1 on the canonical-check chains |
| `pb{j}` | public | the bits of `p` at their slots, the same on every chain |
| `s{t}` | public | 1 on the 22 rows of piece `t` (the last piece: weights 176 to 259) |

Accumulators `ra`, `ha`, `rb`, `rf`, `fx`, `fy`, `fs`, `rc` (`R_A`, the plain fingerprint of `a`, `R_B`, the
plain fingerprints of `f`, `x`, `y`, `s`, and `R_C`), all with scale
`zeta^4`; `zeta = gamma`, `rho = delta`, and `rho^t zeta^k` for `k <= 9` are 27 derivation rows. `R_A =
sum_t rho^t A_t(zeta)` with `A_t` at global weights, so `R_C = zeta^6 R_A R_B` needs no `zeta^{s_t}` factors:
the coefficient bit's weight is `rho^t 2^m zeta^{j + 6 - m}`, exponents 0 to 9. The chain-end family is
`zeta^6 R_A R_B - R_C`.

## Families

- `carry{j}` (4, linear, 32 terms): the 21 coefficient bits at slot `w` plus the carry out of slot `w - 1`
  equal `r_w + 2 carry(w)`. Position 0 reads the carry from the next row's position 3 (`k1 = 1`). The
  ripple reads the pile directly; the spec's level-1 certificate (the third review called it unnecessary)
  is skipped, 16 columns fewer.
- `alias{t}{j}` (12, quadratic): `b6 b5 = 0` for the coefficient at each position; bit 6 sits six slots up,
  bit 5 five, cyclic reads of one or two rows up.
- `hhi{j}`, `ghi{j}` (8, quadratic): `h = cp * r@80`, `g = cp * o@80`: the copy reads 64 rows up (256
  weights), cyclic, and the selector zeroes it outside its rows.
- `ofold{j}`, `ffold{j}` (8, 15 terms): `lo * src_w` plus the seven copy bits at slots `w - s`, `s` in
  `{0, 4, 6, 7, 8, 9, 32}` (`2^256 = 2^32 + 977 mod p`; the reads are same-row or 1, 2, 3, 8 rows below),
  plus the carry in, equal `out_w + 2 carry(w)` with a 3-bit carry.
- `add{j}` (4, 15 terms): `x + y + carry in = s + q1 p + q2 (2 p) + 2 carry out`, `2 p` the bits of `p`
  read one slot down. The carry is signed: both sides are sums of bit vectors, so the running difference
  can go negative (`x = 2, y = 0, s = 1, q p = 1` at weight 0 needs carry `-1`). Three carry bits, zero
  rows on position 3 of the idle row like the product's; the top carry is forced to zero by magnitudes
  (every value below `2^260`, `q p < 2^258`).
- `{ax,ay,as}bd{j}` (12): `v = bd v`, the value is zero above weight 260.
- `aq{1,2}const` (2): `q` equals itself one row down, cyclic, so it is constant along the chain.
- `aq{1,2}canon` (2): `cn q = 0`.
- `piece{t}{j}` (12): `a = s_t a`, a piece's columns are zero outside its rows. Without it the plain
  fingerprint `ha` binds only the sum over the three pieces, so a prover could pile every bit of `a` into
  one piece: a convolution coefficient can then reach 127, which the identity over F_127 reads as zero
  (found by the Codex review with a concrete operand). The public factor on `ra` used to pin the pieces
  column by column; wiring through `ha` lost that. Pieces are 88 bits so that they are row-aligned.
- Booleanity on the 190 bit columns.

## Additions, subtractions, the canonical check

One chain kind serves all three, because `x + y = s + q p` is symmetric in which value is free. An addition
takes `x`, `y` and outputs `s`, with `q` the largest in 0..3 keeping `s >= 0`, so `s < p`. A subtraction
`x - y` is the same chain read backwards: the minuend is wired to `s`, `y` to `y`, and the output is `x`,
with `q` the smallest making it non-negative, so the output is below `p`. `q` goes to 3 because a public
input below `2^257` can exceed `2 p`. The canonical check of `f` is the chain with `s` the public constant
`p - 1` (a public factor the verifier fingerprints from the constant), `y` the witness `p - 1 - f`, and `q`
forced to zero by the mask `cn`: then `f + y = p - 1` with `y >= 0` as bits is `f < p`. Without the mask a
prover would take `q = 1` and `y = 2 p - 1 - f`; the test does that and is rejected.

The bound `bd` matters for soundness of the circuit as a whole: a product's fold copies the high half of `r`
from weights 256 to 263, which is exact only for `r < 2^520`, so every operand must stay below `2^260`.
Product outputs are forced below `2^257` by their ripple, public values by their encoding, and addition
values by `bd`; the last piece of `a` holds 84 bits, the others 88, so coefficients stay below 95.

## The fold and the selector

The spec binds the copies of the high half to `r` with two fingerprints, `F_hi` "for the high half" and
`F_copies`. A Horner accumulator has row-uniform coefficients, so nothing in the IR fingerprints half a
column: a copy bound by a fingerprint of all of `r`, or by a cyclic transition, is `r` rotated, and the
rotated low half lands in the piles. The one thing that knows a row's weight is a public column, so the fold
uses one: `lo` is the selector of the low-half rows, the same 144 values on every chain (`m = 1`, a dense
block; `m = h2` once the statement knows the grid). Then the copy is `h = cp * r@80` and the pile reads
`lo * r_w`: `h` is exactly the high half at slots 0 to 259 and zero elsewhere, the pile is exactly
`lo_w + sum_s hi_{w - s}`, and `o = lo + hi (2^32 + 977) < 2^294`. The same selectors fold `o` into `f`.
Two column sets and two selectors replace the spec's four copies and two accumulators. The copy selector
is one row wider than `lo` because operands are folded values below `2^257`, so a product can pass
`2^512`; the two must differ, since a bit counted in both the low part and the copy is counted twice.

The folds need no zero rows: their piles are zero above weight 291 (the selector and the copies are), so
every carry above the live weights is forced to zero, the carry into the idle row included; and the idle
row's pile is zero (`lo` is zero there and the wrapped copy reads land on selector-zeroed rows), so it
cannot carry into weight 0. Only the first ripple, whose idle-row pile is 21 free coefficient bits, needs
them. Soundness of the result: the first ripple forces `r = a b` exactly (both sums are below `2^512`, so
the top carry is zero); the selector families force `h` and `g` to be the high halves; each fold is an
integer identity per slot, and folding preserves the residue mod `p`. `f` is below `2^256 + 2^71 < 2 p`,
the spec's non-canonical operand range.

## Wiring: a circuit of chains

`mulmod_statement(circuit=...)` takes one `Chain(kind, a, b)` per chain, each operand `-1` (a public value)
or the index of an earlier chain whose output it is. Slots are `ha`, `rb`, `rf`, `fx`, `fy`, `fs` (six,
three wiring products); an operand from chain `k` is a wire to `k`'s output slot, a public operand a public
factor on its slot, and every output nobody consumes a public factor. The output fingerprint is the plain one only: the spec's
piece-weighted output would need three more selectors (piece boundaries are not row-aligned), so the
consumer fingerprints its own `a` plainly with `ha` instead, one accumulator either way. Ordering chains so
that edges become next-chain reads, the spec's open item, saves nothing here: chain-end families cannot
read the next chain, and slots are grid-wide, so the clear grand products cost `2 e h2` bytes per product
whatever the edges. Public inputs are the circuit bytes (count, then kind and two references per chain),
then the values, 33 bytes each in statement order: the static `public_data` derives the per-chain mask
`cn` and the factor list from them. The circuit in the public inputs is redundant with the compiled
wiring; it exists because `Workload.public_data` sees only public inputs. It is not merely redundant: a
header that differs from the compiled circuit but yields the same factor count would turn a canonical
check into an addition (its `cn` mask off, its constant replaced by a value from the inputs). So the
statement pins its circuit bytes (`Statement.pin`, `Shape.pinned`, hashed in the prefix), and the verifier
refuses public inputs that do not start with them.

## The zero row

The spec says row 0 is a zero row "certified by the chain-start opening every column has". This IR opened
only Z columns at the chain boundaries, and without the check the instance is unsound: the last row is
read by the ripple (its slot 3 carries into weight 0) but ingested by nobody, so a prover can put a pile
of two there, carry a one into weight 0, and satisfy every family with `r = a b + 1`. The test does exactly
that, and shows the statement without zero rows accepts it. `Statement.zero(column, FIX_ONE | FIX_E)` is the fix: a `ZERO` record on the shape, and the verifier
checks the column's opening at `(coordinate, z2)` is zero, which forces the row to zero on every chain
(a polynomial in `X2` of degree below `h2` vanishing at random `z2`). The mulmod zeroes the five carry
columns of position 3 on the last row; every column already has that opening, so it costs nothing.

## Public factors

The derived public data is the eight public blocks (`h2 x 144` bytes each), then each factor's ingest
columns in statement order: an `a` operand in its pieces (12 columns), any other value plain (4 columns),
the canonical check's constant `p - 1`, 144 bytes per column. The verifier's fingerprint is `horner_chain_end`. The last byte of a value
is 0 or 1: the fingerprint reads 257 bits, and a higher claim would verify against an honest trace.

## Not done

The chain-end families that would make edges to the next chain free, and `m = h2` on the constant public
columns (the statement does not know the grid). Everything above the arithmetic, the curve and the ECDSA
composition, is the next layer.

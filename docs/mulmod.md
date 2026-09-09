# The 256-bit product as a polynomial identity

The polynomial-mulmod spec, items 1 to 8, on the builder (`relations/mulmod.mojo`): `a b = r`, then `r`
folded to `f < 2^257`, congruent mod secp256k1's `p`, with no product ever committed. Read the spec for the
argument; this doc records what the code does where the spec left a choice, one hole the spec's zero row
closes, the selector columns the fold needs, and the wiring of chains into a circuit.

## Layout (`q = 4`, 144 rows)

Row `x` of a chain holds weight slots `4 (142 - x) + j`, `j = 0..3`, so a Horner scan with scale `zeta^4`
over rows 0 to 142 evaluates a bit column at `zeta` (position weight `zeta^j`). Row 143 holds no weight and
no accumulator reads it. Every bit is stored at the slot of its own weight:

| columns | count | contents |
|---|---:|---|
| `a{t}{j}` | 12 | bit `i` of `a` at slot `i`, in the columns of piece `t = i // 86` |
| `b{j}` | 4 | bit `i` of `b` at slot `i` |
| `c{t}{m}{j}` | 84 | bit `m` of the coefficient of `A_t B` at weight `s`, at slot `s + m`; `c = 64 b6 + 32 b5 + v`, `b6 b5 = 0` |
| `r{j}` | 4 | the product bit at slot `w` |
| `y{k}{j}` | 20 | the carry out of slot `w` (five bits), stored at slot `w` |
| `h{j}` | 4 | the high half of `r`: bit `256 + i` at slot `i` |
| `o{j}`, `z{k}{j}` | 4 + 12 | the first fold `lo + sum_s (hi << s)` and its carries (three bits) |
| `g{j}` | 4 | the high half of `o` at slot `i` |
| `f{j}`, `v{k}{j}` | 4 + 12 | the second fold, the chain's output, and its carries |
| `lo` | public | 1 on rows 79 to 142 (weights below 256), 0 elsewhere; the same on every chain |
| `cp` | public | 1 on rows 78 to 142 (weights below 260): the copy's rows, one more than `lo` for a product below `2^516` |

Accumulators `ra`, `ha`, `rb`, `rc`, `rf` (`R_A`, the plain fingerprint of `a`, `R_B`, `R_C`, and the plain
fingerprint of `f`), all with scale
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
- Booleanity on the 164 bit columns.

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

`mulmod_statement(circuit=...)` takes one `(a, b)` per chain, each `-1` (a public operand) or the index of
an earlier chain whose output it is. Slots are `ha`, `rb`, `rf` (three, two wiring products); an operand
from chain `k` is `wire(slot, j, rf, k)`, a public operand a public factor on its slot, and every output
nobody consumes a public factor on `rf`. The output fingerprint is the plain one only: the spec's
piece-weighted output would need three more selectors (piece boundaries are not row-aligned), so the
consumer fingerprints its own `a` plainly with `ha` instead, one accumulator either way. Ordering chains so
that edges become next-chain reads, the spec's open item, saves nothing here: chain-end families cannot
read the next chain, and slots are grid-wide, so the clear grand products cost `2 e h2` bytes per product
whatever the edges. Public inputs are tagged values, a tag byte (`a`, `b`, or output) then 33 bytes, in
statement order, so the static `public_data` knows which ingest columns to derive.

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

The derived public data is the two selector blocks (`h2 x 144` bytes each), then each tagged value's
ingest columns in order: an `a` operand in its pieces (12 columns), a `b` operand or an output plain (4
columns), 144 bytes per column. The verifier's fingerprint is `horner_chain_end`. The last byte of a value
is 0 or 1: the fingerprint reads 257 bits, and a higher claim would verify against an honest trace.

## Not done

The canonical check (`f < p`) is a separate small instance per the spec; additions and subtractions mod
`p` are a second column group on the same grid.

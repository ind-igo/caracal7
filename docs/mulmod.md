# Arithmetic mod p: the product as a polynomial identity, additions, the canonical check

The polynomial-mulmod spec on the builder (`workloads/mulmod.mojo`): `a b = r`, then `r` folded to
`f < 2^257`, congruent mod secp256k1's `p`, with no product ever committed (items 1 to 8), or reduced mod
P-256's `p` in one word pass (the P-256 section); signed three-operand additions mod `p` or mod `n` on
two add lanes per chain, with the canonical check, the guard and equality as masked additions. A circuit is a list of ops, each a product or an addition. Read the spec for the argument; this doc records what the code does where the spec left
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
| `l{L}{x,y,z,s}{j}` | 32 | add lane `L` (two per chain): `x + sy y + sz z = s + q m` |
| `l{L}q{k}` | 8 | `q = q0 + 2 q1 + 4 q2 - 2 q3` in `[-2, 7]`, chain-constant |
| `l{L}c{k}{j}` | 32 | the lane's signed carry `c0 + 2 c1 + 4 c2 - 8 c3` in `[-8, 7]`, stored at its slot |
| `lo` | public | 1 on rows 79 to 142 (weights below 256), 0 elsewhere; the same on every chain |
| `bd` | public | 1 on weights below `WIDTH = 260`: every chain value's bound |
| `cp` | public | 1 on weights below 264: the copy's rows, the high half of a product of two bounded values |
| `l{L}sy`, `l{L}sz` | public | the lane's operand signs per chain: 0, 1 or `-1` (126) |
| `l{L}qz`, `l{L}sm` | public | per chain, 1 forces the lane's `q` to zero (canonical check, guard) or its `s` (equality) |
| `pb{j}` | public | the bits of the chain's modulus at their slots: `p`, or `n` on a mod-`n` chain |
| `s{t}` | public | 1 on the 22 rows of piece `t` (the last piece: weights 176 to 259) |

Accumulators `ra`, `ha`, `rb`, `rf`, `rc` (`R_A`, the plain fingerprint of `a`, `R_B`, the plain
fingerprint of `f`, `R_C`) and `f{L}{x,y,z,s}` (the plain fingerprints of each lane's values), 13 in all with scale
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
- `l{L}add{j}` (4 per lane, 20 terms): `x + sy y + sz z + carry in = s + q m + 2 carry out`; `q m` is
  `q0 m + q1 (2 m) + q2 (4 m) - q3 (2 m)`, the bits of `m` read 0, 1, 2 and 1 slots down. The signs are
  public columns multiplied in (`Term(coef, sign, value)`), so one lane serves `x + y`, `x - y`,
  `x - y - z` and every other sign pattern. The carry is signed: both sides are sums of bit vectors, so the
  running difference can go negative (`x = 2, y = 0, s = 1, q p = 1` at weight 0 needs carry `-1`). Four
  carry bits, zero rows on position 3 of the idle row like the product's; the top carry is forced to zero
  by magnitudes (every value below `2^260`, `|q m| < 2^259`).
- `l{L}{x,y,z,s}bd{j}` (16 per lane) and `bbd{j}` (4): `v = bd v`, the value is zero above weight 260.
  `b` needs it since a hint operand (below) has no factor or wire to bound it; `a` is bounded by its pieces.
- `l{L}q{k}const` (4 per lane): `q` equals itself one row down, cyclic, so it is constant along the chain.
- `l{L}q{k}z` (4 per lane): `qz q = 0`. `l{L}s{j}m` (4 per lane): `sm s = 0`.
- `piece{t}{j}` (12): `a = s_t a`, a piece's columns are zero outside its rows. Without it the plain
  fingerprint `ha` binds only the sum over the three pieces, so a prover could pile every bit of `a` into
  one piece: a convolution coefficient can then reach 127, which the identity over F_127 reads as zero
  (found by the Codex review with a concrete operand). The public factor on `ra` used to pin the pieces
  column by column; wiring through `ha` lost that. Pieces are 88 bits so that they are row-aligned.
- Booleanity on the 236 bit columns.

## Add ops: additions, subtractions, the canonical check, the guard, equality

An add op is `x + sy y + sz z = s + q m` with `sy`, `sz` in `{-1, 0, 1}`, `m` the chain's modulus (`p`, or
`n` for the scalar arithmetic of ECDSA), and `q` in `[-2, 7]`, four bits `q0 + 2 q1 + 4 q2 - 2 q3`. Honest
operands are products below `2 p` or canonical values, and a circuit subtracts only canonical values, so
the sum lies in `(-2 m, 6 m)` and the floor quotient in `[-2, 5]`. The op enforces only the bound
`s < 2^260`, not `s < m`: a lane with a larger sum is not unsound, it is a circuit the host refuses. `s` is either the
op's output (`OUT`, reduced, the honest prover picks the floor quotient), a public constant (`PUB`, the op
is a check: the operands are congruent to the constant), or masked to zero (`NIL`, `sm = 1`: the operands
are congruent to zero, the `eq` op). One operand may be `FREE`: a witness the host solves for, allowed
only with `qz = 1` and a public `s`, so that the op is an integer identity in which the free operand's
existence as bits is the claim:

- `canon(x)`: `x + y = p - 1`, `y` free, `q = 0`: `x < p`. A prover without the mask would take `q = 1`
  and `y = 2 p - 1 - x`; the test does that and is rejected.
- `guard(x)`: `x - y = 1`, `y` free, `q = 0`: `x >= 1`. After `canon`, `1 <= x < p`, the guard on the
  curve's `dx` that makes the slope well-defined.

Every value below `2^260` is what the op proves about its operands; only `canon` makes a value canonical.
Outputs are canonical when honest but the circuit enforces only the bound. The bound `bd` matters for
soundness of the circuit as a whole: a product's fold copies the high half of `r` from weights 256 to 263,
which is exact only for `r < 2^520`, so every operand must stay below `2^260`. Product outputs are forced
below `2^257` by their ripple, public values by their encoding, and add-lane values by `bd`; the last piece
of `a` holds 84 bits, the others 88, so coefficients stay below 95.

The modulus is per chain: `pb{j}` are dense public blocks whose rows on a chain hold the bits of that
chain's modulus, so a chain holds add ops of one modulus, and the product lane, which reads `pb` nowhere,
is indifferent. The signs and masks are dense public blocks too, one value per chain and lane.

## The fold and the selector

The spec binds the copies of the high half to `r` with two fingerprints, `F_hi` "for the high half" and
`F_copies`. A Horner accumulator has row-uniform coefficients, so nothing in the IR fingerprints half a
column: a copy bound by a fingerprint of all of `r`, or by a cyclic transition, is `r` rotated, and the
rotated low half lands in the piles. The one thing that knows a row's weight is a public column, so the fold
uses one: `lo` is the selector of the low-half rows, the same 144 values on every chain (`m = h2` on the
ECDSA workloads: one chain of public data, `h1` products for the verifier per point; `m = 1` by default). Then the copy is `h = cp * r@80` and the pile reads
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

## The P-256 pass

`mulmod_statement(curve=CURVE_P256)` keeps the product lane and the add lanes and replaces the two folds.
P-256's `p = 2^256 - 2^224 + 2^192 + 2^96 - 1` has no small fold (`2^256 = 2^224 - 2^192 - 2^96 + 1`
takes only 32 bits off a value), so the pass is FIPS 186-4 D.2.3 on 32-bit words: product word `i`,
`i = 8..16`, is congruent to a signed pattern on the words 0 to 7 (`_p256_words`: the pattern of `2^256`,
then each next word shifts the last and folds its overflow through the same pattern; word 16 covers
products of two operands below `2^260`). Output word `k` reads product word `k + d` with coefficient
`_p256_coef(d, k)`, in `[-4, 5]`, for 16 offsets `d` (0 to 14 and 16).

| columns | count | contents |
|---|---:|---|
| `f{j}` | 4 | the output bit at slot `w` |
| `fq{k}` | 4 | `q = q0 + 2 q1 + 4 q2 - 8 q3` in `[-8, 7]`, chain-constant |
| `fc{k}{j}` | 16 | the signed carry `c0 + 2 c1 + 4 c2 - 8 c3`, stored at its slot |
| `w{d}` | public, 16 | on the 8 rows of output word `k` (weights below 256) the coefficient of product word `k + d`, `-1` as 126; 0 above and on the idle row |
| `fp{j}` | public, 4 | the bits of P-256's `p` at their slots on every chain: the fold's modulus (`pb` holds `n` on a mod-n chain, and the first trace of the full circuit failed exactly there) |

`h`, `o`, `g`, `z`, `v`, `lo` and `cp` do not exist on this statement; `bd`, `pb`, `s{t}` and the lanes
are as above.

- `ffold{j}` (4, 29 terms): `sum_d w_d r@(8 d rows up) - sum_k sign_k fq_k fp@(k slots down) + carry in
  = f + 2 carry out`. The read of product word `k + d` is `8 d` rows up, cyclic; the `2^k` of `q`'s bit
  `k` comes from reading `p` `k` slots down, so the term's coefficient is the sign alone (the first
  version multiplied the weight in twice). Zero rows on `fc{k}3` of the idle row, like the add lane.
- `fq{k}const` (4): `q` equals itself one row down.

Soundness. `r = a b` exactly and below `2^520` by the product ripple. A read of product word `i` from the
rows of output word `k` lands on rows `142 - 8 i - (0..7) >= 7`: no selector-gated read wraps around the
chain, and the rows of word 16 above weight 519 read zero bits. Per slot the residual's magnitude is at
most 14 (the largest sum of `|coefficient|` over a word, word 7 with word 16's 5) + 8 (carry in) + 4 (`q
p`'s four bit terms) + 1 + 16 (carry out) = 43 < 127, so a family vanishing in `F_127` is the integer
identity at that slot. Summing `2^w` times the slot identities over the 576 slots: the zero row makes the
carry into weight 0 and the carry out of slot 575 the same zero bits, so `sum_w 2^w f_w = S - q p`
exactly, whatever bits the prover wrote, and `f` is congruent to `a b mod p`. `f` is nonnegative there,
and every consumer bounds it (`bd`, the pieces, or a public factor's 257-bit claim), so the next product
is again below `2^520`. The zero rows are not load-bearing here: without them a carry `c` in `[-8, -1]`
around the ring gives `f = S - q p + c (1 - 2^576)`, a value of 576 bits that every consumer's bound
rejects (the product ripple's zero rows are the ones that matter); they are kept so that the argument
does not lean on the consumer, and they cost nothing. The add lanes see canonical or bounded operands as before.

Completeness. For operands below `2^256` (public values are required below `2^256` in this mode,
`_input`; outputs and free operands are canonical; a hint is bounded below `2^260` by the host and
canonical when honest) the product is below `2^512`, word 16 is zero, `S` lies in `(-4 2^256, 7 2^256)`,
`q = floor(S / p)` in `[-5, 7]`, `f = a b mod p < p`, the per-slot piles lie in `[-7, 8]` and the carries
in `[-7, 7]`. The trace writer (`_p256_chain`) refuses a carry outside `[-8, 7]`.

Cost against secp256k1's folds: 16 fewer bit columns (220 W against 236 on ECDSA), 18 more public
columns (36 against 18; 24 of them chain-constant, so `m = h2` leaves 12 dense), and 11 more opening points (23
against 12: the word offsets are distinct cyclic reads, and every point opens every column, `points x
opened columns x 20 B` of proof). The nine-pass alternative (the four-term identity applied until the
value is short) would have shared its four offsets across passes but cost nine `f + carry` column sets;
rejected.

## Wiring: a circuit of ops

`mulmod_statement(circuit=..., curve=...)` takes a list of `Op`: `mul(x, y)` or `add(x, y, sy, z, sz, s, qz, mod)`
(with `sub`, `eq`, `canon`, `guard` as shorthands), each operand `PUB` (a public value), `FREE`, `NIL`
(no `z`), `hint(h)` (witness `h` of the workload's hint list: the prover supplies it, every occurrence
after the first is wired to the first, and nothing else binds it; a curve slope, say, which `l dx = dy`
then forces) or the index of an earlier op whose output it is. Placement is automatic: the `k`-th product
takes chain `k`'s product lane; add ops fill the two add lanes of chains 0, 1, ... in order, opening a new
chain when the lanes are full or the modulus changes. Slots are `ha`, `rb`, `rf` and the four values of
each lane (11, nine wiring products); an operand from op `k` is a wire to `k`'s output slot, a public
operand a public factor on its slot, a public `s` a factor on the `s` slot, a hint occurrence a wire to
its first occurrence, and every output nobody consumes a public factor. The output fingerprint is the plain one only: the spec's piece-weighted output
would need three more selectors (piece boundaries are not row-aligned), so the consumer fingerprints its
own `a` plainly with `ha` instead, one accumulator either way. Ordering chains so that edges become
next-chain reads, the spec's open item, saves nothing here: chain-end families cannot read the next chain,
and slots are grid-wide, so the clear grand products cost `2 e h2` bytes per product whatever the edges.
Public inputs are the circuit bytes (count, then 13 bytes per op: kind, four 16-bit references, two signs,
`qz`, the modulus), then the values, 33 bytes each in factor order (per op its public operands `x`, `y`,
`z`, then a public `s`; then the unconsumed outputs): the static `public_data` derives the per-chain sign,
mask and modulus blocks and the factor list from them. The circuit in the public inputs is redundant with
the compiled wiring; it exists because `Workload.public_data` sees only public inputs. It is not merely
redundant: a header that differs from the compiled circuit but yields the same factor count would turn a
canonical check into an addition (its `qz` mask off, its constant replaced by a value from the inputs). So
the statement pins its circuit bytes (`Statement.pin`, `Shape.pinned`, hashed in the prefix), and the
verifier refuses public inputs that do not start with them. A workload whose circuit is fixed in code
(`EcdsaK1`) passes `pin=False` and derives its public data from the circuit it rebuilds, with no header.

Host values are `Big` integers (`workloads/bigint.mojo`, sign and 32-bit limbs): `circuit_values` walks
the ops, takes public operands from the inputs, computes each output or free operand and the quotient,
and refuses an instance whose checks do not hold; `circuit_trace` writes each op on its chain and lane.

## The zero row

The spec says row 0 is a zero row "certified by the chain-start opening every column has". This IR opened
only Z columns at the chain boundaries, and without the check the instance is unsound: the last row is
read by the ripple (its slot 3 carries into weight 0) but ingested by nobody, so a prover can put a pile
of two there, carry a one into weight 0, and satisfy every family with `r = a b + 1`. The test does exactly
that, and shows the statement without zero rows accepts it. `Statement.zero(column, FIX_E)` is the fix: a `ZERO` record on the shape, and the verifier
checks the column's opening at `(coordinate, z2)` is zero, which forces the row to zero on every chain
(a polynomial in `X2` of degree below `h2` vanishing at random `z2`). The mulmod zeroes the five carry
columns of position 3 on the last row; every column already has that opening, so it costs nothing.

## Public factors

The derived public data is the 18 public blocks (`h2 x 144` bytes each), then each factor's ingest
columns in statement order: an `a` operand in its pieces (12 columns), any other value plain (4 columns),
a check's constant such as `p - 1` or `1`, 144 bytes per column. The verifier's fingerprint is `horner_chain_end`. The last byte of a value
is 0 or 1: the fingerprint reads 257 bits, and a higher claim would verify against an honest trace.

## Not done

The chain-end families that would make edges to the next chain free. The curve and the ECDSA composition
are the next layer: `docs/ecdsa.md`.

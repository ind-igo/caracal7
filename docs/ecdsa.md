# ECDSA on the mulmod circuits: design

One secp256k1 signature verification as a fixed circuit of the chains in `workloads/mulmod.mojo`.
Numbers are counts, not measurements.

## 1. The statement

Public inputs `(r, s, e, Q)`. Verifier-side, outside the circuit and before anything else: `0 < r, s < n`,
`Q` has canonical coordinates, is on the curve and is not the identity (prime order, cofactor 1, so nothing
more); then `u1 = e s^-1`, `u2 = r s^-1 mod n`. The circuit proves

    x(u1 G + u2 Q) = r + q n,  q in {0, 1},  with x canonical (below p).

That is the ECDSA predicate. Everything the verifier derives from the public inputs (`u1`, `u2`, their
digits, the selected table points, the blinding constants) is public data of the fixed statement: the
artifact, `sigma` and the public-factor descriptors do not change per signature; only the factor values do.

## 2. Public scalars: the digits select public points through public factors

`u1` and `u2` are public, so which multiple of `G` or `Q` each step adds is public. A step's operand slot is
wired, once, to a public-factor slot; per signature, `public_data` fills that factor with the fingerprint of
the selected point's coordinate. No witness digits, no selection gadget, no scalar-field arithmetic, no
signature-dependent wiring. The verifier computes the points itself; the line is that the doublings of the
variable base stay in the circuit, and the verifier's curve work per signature is a constant number of
operations plus one fixed multiple of the blinding point (section 3). The fingerprints dominate the
verifier anyway: about 200 of them at 256 `E` products each, against a few thousand field products for the
curve work.

- Straus-Shamir over four GLV halves (replacing a `Q` side of two addends per 4-bit window and
  a fixed-base `G` side of 32 8-bit additions): `u2 = k1 + k2 lambda`, `u1 = k3 + k4 lambda mod n`, signed
  magnitudes below `2^128` (libsecp256k1's split), bases `Q`, `phi(Q) = (beta x_Q, y_Q)`, `G`, `phi(G)`, the
  sign of a half absorbed into its base (`-T = (x, p - y)`). 22 windows of 6 bits; window `w` adds the one
  public point `P_w = sum_i d_{i,w} T_i`, which the verifier builds from four tables of the multiples `T ..
  65 T` (about 260 host additions, plus 3 per window). 126 doublings (the accumulator starts at `16 B`, so
  window 21 needs none) and 22 additions.
- Recoding of an odd magnitude `k` in base `2^b` (`b = 6`, 22 digits): for the low 21 digits, `d = (k mod
  2^(b+1)) - 2^b`, `k <- (k - d) / 2^b`, which keeps `k` odd; digit 21 is what remains, in `[-1, 3]` for
  halves below `2^128`. Every odd `|k| < 2^(22 b)` has exactly this representation (Codex checked the
  small lengths exhaustively). Without the terminal rule the recurrence never ends: `k = 1` emits `-63`
  forever.
- Parity: the recoding takes an odd value, so a half `k` (signed, after the GLV split) is recoded as `k - c`
  with `c = 1` for even `k` and `c = 2` for odd `k`, and window 0 adds `(d_0 + c) T` instead of `d_0 T`.
  `k = 0` and `k = 1` both recode `-1` (terminal digit `-1`, the low digits `63`). The sign of the base is
  the GLV half's sign only; skewing the magnitude and then moving the sign into the base would reverse the
  skew and make zero contribute `-2 T`. The window-0 entries are `(d_0 + c) T` for `d_0 + c` in `[-62, 65]`; the
  verifier builds `T .. 65 T` per base.
- If `P_w` is the identity (possible only for a `Q` that is a small-ratio multiple of `G`), the step adds `B`
  instead and the closing constant subtracts `2^(6 w) B` more: `t = sum 2^(6 w)` over such windows, the
  closing constant `-(2^130 + t) B`.
- Schedule: `acc = 16 B`; window 21; for `w = 20..0`: six doublings, one addition; then the closing constant;
  the last addition computes only `x`.

## 3. The blinding point and the denominator guard

Affine addition fails when the inputs are equal or opposite, and the party who chooses the statement can
force that. Curve validation does not prove knowledge of `log_G Q`, and public-key recovery makes a valid
signature for any chosen `R`: pick `R`, set `r = x(R) mod n`, any `s`, any `e`, then `Q = (s R - e G) / r`.
With a fixed `B` the attacker sets `R = -2^129 B`: the honest accumulator before the closing step is
`-2^128 B`, equal to the closing addend, `dx = dy = 0`, any slope passes, and the trace is both unprovable for
the honest prover and forgeable for a dishonest one (Codex built the instance on secp256k1). The
intermediate steps are not reachable this way (the digits depend on `r`, which depends on `R`), the closing
step is.

So `B` is per instance: `B = hash_to_curve(Q, r, s, e)` (try-and-increment on a hash of the canonical
encoding). Then `R`, and every partial sum, is fixed before `B` is known, and an exceptional step is a
relation between `log B` and public scalars: negligible in the random-oracle model, for the honest prover
(completeness) and for any statement chooser. The verifier computes `16 B` and `2^130 B` (130 doublings)
and the closing constant; that is the "one fixed multiple" of section 2.

Soundness does not rest on that argument. Each addition carries the guard `dx != 0 mod p`: CANON on `dx`
(`dx < p`) and `dx = 1 + y` for a witness `y` bounded by `bd`, with the quotient masked to zero like CANON's
so the identity is exact over the integers (`dx >= 1`); two add-lane ops and no MUL. Without the mask `dx =
0, y = p - 1, q = 1` passes. With `dx` nonzero, `l dx = dy` forces `l mod p`, and `x3, y3` follow; by induction from `16 B` and
the public addends every point is forced. Doublings need `y != 0`: every point in the chain is on the curve
(chord and tangent from on-curve inputs), and the group has odd order, so no finite point has `y = 0`. The
guard could be dropped under the random-oracle argument (the prover cannot choose the points, so `dx` is
the honest value); it costs 194 add-lane ops and buys unconditional soundness, so it stays.

## 4. Group operations

Affine, the slope `l` a witness hint. Honest values: add-lane outputs canonical (`q` chosen so `0 <= s <
p`), products below `2^257` (the fold's ripple), public values canonical, the slope canonical. Enforced
values: below `2^260` (`bd`); the add family is an exact integer identity for any such values, so
congruence mod `p` holds against any prover, and the honest ranges only matter for completeness (the
quotient `q` in `[-2, 6]` fits the 4-bit signed encoding).

- addition `(x1, y1) + (x2, y2)`, `(x2, y2)` public: `dx = x2 - x1`, CANON `dx`, `dx = 1 + y`, `dy = y2 - y1`,
  MUL `l dx`, EQ `l dx = dy`, MUL `l l`, `x3 = l^2 - x1 - x2`, `x1 - x3`, MUL `l (x1 - x3)`, `y3 = l (x1 -
  x3) - y1`: 3 MUL, 8 add-lane ops.
- doubling: MUL `x x`, `3 x^2 = x^2 + x^2 + x^2`, `2 y`, MUL `l (2 y)`, EQ `l (2 y) = 3 x^2`, MUL `l l`, `x3 =
  l^2 - x - x`, `x - x3`, MUL, `y3`: 4 MUL, 6 add-lane ops.
- closing: CANON `x_R`, then `x_R - r = q n` on the chain whose modulus block holds `n`: 2 add-lane ops. `x_R
  < p < 2 n` and `0 < r < n` force `q` into `{0, 1}`; `r + n >= p` just makes `q = 1` impossible.

Counts: 126 doublings, `22 + 1 = 23` additions (the last without `x1 - x3` and `y3`): `504 + 68 = 572` MUL;
`126 x 6 + 23 x 8 - 2 + 2 = 940` add-lane ops, 470 chains' worth with two lanes. The MUL count sets the chain
count: `h2 = 576 = 2^6 9`, grid `144 x 576`, 82,944 rows, 4 idle MUL lanes. (Before Straus-Shamir: 786 MUL,
1,520 add-lane ops, `144 x 896`.) The RS domain does not shrink with the grid: 20,736 symbols per column
still take the 4 cosets of order 161,280 (rate 1/31), so the RS stages, the Merkle work and the tail are
unchanged; the grid stages are 36% smaller. The spec's row for ECDSA is `144 x 1344` at 1,300
mulmods; the difference is the fixed-base comb (no doublings for `G`) and the public odd-multiple table of
`Q` (no table chains). Public factors: 23 additions x 2 coordinates, `16 B`, the closing constant, `r`,
the constants `1`, `p - 1`, `n`: about 50 points, 18 public columns (11 are wiring slots: 3 on the MUL lane,
4 per add lane).

## 5. Builder changes

1. **Two add lanes per chain.** The add columns (`ax, ay, az, as, aq*, ac*`) and their families duplicated
   with a lane suffix, each lane with its own four accumulators (`fx, fy, fz, fs`) and four slots. 5 MUL
   accumulators + 8 = 13 of the 16 the Z-tree leaf allows. `Chain` becomes one MUL op and two add ops, any
   idle.
2. **Three-operand add op.** `x + sy y + sz z = s + q p`, `sy, sz` in `{-1, 0, +1}` per-chain public
   columns, `q = b0 + 2 b1 + 4 b2 - 2 b3` in `[-2, 7]`, chain-constant, `q p` as the modulus block read `k` slots down
   for `2^k p`. Carries signed 4-bit `[-8, 7]`: with inputs below `2^260` the per-weight pile is in
   `[-6, 4]` and the family residual in `[-28, 27]`, away from any nonzero multiple of 127 (Codex's
   derivation; re-derive in the doc when implemented). `bd` on all four operands; zero rows for the
   carries on the idle row as today. ADD, SUB, CANON are the cases `sz = 0`; CANON keeps `q` masked to zero.
3. **EQ op.** The add op with `s` masked to zero by a per-chain public column and no output wire: `x + sy y +
   sz z = q p`, congruence of non-canonical values in one op.
4. **Per-chain modulus.** `pb{j}` holds `n` on the closing chain. Both lanes of that chain see `n`, so the
   other lane idles there.
5. **Public data per signature.** A host module `ecdsa.mojo` with secp256k1 field and curve arithmetic
   (64-bit limbs; the bit-list helpers are too slow for 132 doublings), the GLV split, the recoding, the
   tables of `Q`, hash-to-curve for `B`, and the factor values. Reported as verifier work: `s^-1` once,
   `u1, u2`, the split, about 20 curve operations for the `Q` tables, 132 doublings and 3 additions for the
   `B` constants, about 200 fingerprints. The `G` tables are compile-time data.
6. **No `Statement.pin` change, no driver change.** The circuit is fixed; `public_inputs` are
   `(r, s, e, Q)` and `public_data` is the static function of them, like the other workloads.

Proof size and time will not follow the old per-mulmod estimate: two widened add lanes are about 236 W +
208 Z + 48 Q columns against 190 + 128 + 48, and six wiring products against three. Measure.

## 6. Open

- A MUL op certifying `a b + c d = s mod p` (the second product with `d = p - 1` turns an EQ into a
  product with an output: a doubling is 3 MUL and 3 add-lane ops, an addition 3 MUL and 6) would make
  446 MUL, `h2 = 448`, 516 add-lane ops. Rejected: the RS domain does not shrink with it.
  `domain_for` needs `L >= 32 rows`, and `N / 4 = 16,128` rows still take the 4 cosets of 161,280 (2
  cosets would be rate 1/20; the domain halves only below 10,080 rows, `h2 <= 280`). Against a 22%
  smaller grid the second operand set costs about 100 W columns (pieces, `b'`, a second coefficient
  set: a shared set would put coefficients above 95, outside the certificate's range in F_127) and 3
  accumulators (48 Z columns), so the RS stages, the openings and the proof grow by about 30%. It flips
  if the rate rule ever accepts 1/20.
- secp256r1: section 8.
- The hash-to-curve for `B`: which hash, and whether the encoding of `(Q, r, s, e)` it takes is the
  public-input byte string as is.

## 7. Implemented

`workloads/ecurve.mojo`: the host arithmetic shared by the curves (`Curve[F]` over a limb field `F`,
affine points on `Big`, Fermat inversions), `recode`/`skew`, the blinding point by try-and-increment on
Blake3 of the 160 public-input bytes (`r, s, e, x_Q, y_Q`, the encoding open in section 6, settled as the
public inputs as they are), and the `Walk` that emits the group operations as ops. `workloads/ecdsa_k1.mojo`:
the secp256k1 constants, the GLV split by the exact lattice basis, the four-base schedule and the `EcdsaK1`
workload: `walk` emits the fixed circuit once and, per signature, the public factor values in factor order
and the slope hints. The verifier's `public_data` runs the same walk from the public inputs. Tests:
`tests/test_ecdsa_k1.mojo` (the circuit and the prover round trip), `tests/test_ecdsa_k1_host.mojo`.

Deviations from sections 4 and 5:

- The slope enters as a **hint operand** (`hint(h)` in `mulmod.mojo`): a witness with no factor, wired
  between its three occurrences (`l dx`, `l l`, `l (x1 - x3)`), bounded by the piece selectors on `a` and a
  new `bbd` family on `b`. The op counts of section 4 assumed that and hold: 572 MUL, 940 add-lane ops,
  1,512 ops on 572 chains of the `144 x 576` grid (each public point's coordinate is a factor at every
  operand that reads it, `x2` twice per addition). `q` is `b0 + 2 b1 + 4 b2 - 2 b3` in `[-2, 7]`.
- No circuit header: `mulmod_statement(pin=False)`. The public inputs are the 160 bytes.
- The four digit tables (66 multiples each of `Q`, `phi(Q)`, `G`, `phi(G)`) are built per signature. The
  field is `workloads/fp_k1.mojo` (four 64-bit limbs, UInt128 products, the fold twice, Fermat inversion);
  `Curve` converts from `Big` at each field operation, so a curve operation is about 50 us and a live
  walk (the tables, the 130 doublings for the blinding constants, the chain) about 45 ms. `Big` keeps
  the scalars, with a shift-subtract division and a binary extended Euclid for `s^-1`. The `G` tables
  could be constants.
- The trace generator addresses columns by precomputed indices and gets the coefficient piles from the
  set bits of both operands: 165 ms for the 1,512 ops (was 2.5 s through string-keyed lookups).

Measured on the 16 GB Mac (Metal), `CLIENT.grid(144, 576)`, `e = 16`: proof 631,856 bytes; warm prove
648 ms, trace 150 ms, public data 45 ms, verify 118 ms (the tail levels about 30 of it). Before
Straus-Shamir on the `144 x 896` grid: 683,392 bytes, 4,503 ms, 1,358 ms;
the prover round trip test (two live walks, prove, verify) 10.2 s.

## 8. secp256r1 (P-256)

The csp-benchmarks generator signs on P-256 by default, and barretenberg, provekit, ligetron and
provekit-groth16 verify P-256, so a P-256 row is the same-curve comparison against those four; secp256k1
is the same-curve comparison against risc0 and jolt.

**Built** (`workloads/ecdsa_p256.mojo`, `workloads/fp_p256.mojo`, `tests/test_ecdsa_p256_host.mojo`): the
field on four 64-bit limbs in Montgomery form (`-p^-1 = 1 mod 2^64`, so the Montgomery factor of each
iteration is the low limb itself; the prime has no small fold), the curve with `a = p - 3`, and the walk: Straus-Shamir
over the two bases `Q` and `G` with the full scalars `u2`, `u1` (no endomorphism), 43 windows of 6 bits,
the same recoding, tables and blinding as secp256k1. A doubling costs one more add-lane op: the tangent
numerator `3 x^2 + a` is two three-operand ops (`2 x^2`, then `+ x^2 + a` with `a` a public factor). The
schedule is 252 doublings and 44 additions: 1,139 MUL and 2,116 add-lane ops, 1,139 chains, so the grid is
`144 x 1152` (the next legal size above 1,139; the add lanes need 1,058), 479 public factors, 296 hints.
The RFC 6979 A.2.5 signature verifies on the host, the live walk emits the fixed circuit, and its values
satisfy every op over P-256's `p` and `n` (`circuit_values` with the moduli passed in). One precondition
inherited from section 1: `e` must be below `n`. A FIPS 186 digest of 256 bits exceeds P-256's `n` with
probability about `2^-32` (against `2^-128` on secp256k1); such an input is rejected, not reduced.

**Not built: the product reduction.** The MUL lane of `mulmod.mojo` (polynomial-mulmod 7) folds a product
by `2^256 = 2^32 + 977 mod p`: seven shifted copies of the high half added to the low half, twice, with
3-bit carries, so the output is below `2^257`. Mod P-256, `2^256 = 2^224 - 2^192 - 2^96 + 1`: a pass takes only
about 32 bits off (the `2^224` term keeps most of the length), the terms are signed, and a product of
operands below `2^260` needs nine passes of four signed reads to get below `2^260`, against two of seven. `EcdsaP256` therefore has `circuit` and `public_inputs` but no `statement`,
`trace` or `public_data`. Two designs for the reduction, both a builder change with its own soundness
argument (the fold bounds of `docs/mulmod.md` are specific to the secp256k1 shifts):

- Word folds. Split the high half into 64-bit words; `2^(256 + 64 j) mod p` is a sparse signed pattern
  (four terms for `j = 0`, five for `j = 1`, more for the higher words). One pass adds every word's pattern
  at once: a public selector column per (word, shift, sign) says which rows read which copy, the carries
  are signed like the add lane's, and a public multiple of `p` (a `pb`-like block) is added so that the
  pile is nonnegative. The patterns have 4, 5, 5, 7 and 8 terms (29 in all) and the signed sum is below
  `2^290` in magnitude; a second pass on its top 34 bits (the four-term identity) and the offset bring
  the output below about `2^260`, so `FOLDED` grows by two or three bits (products of such operands still
  fit the chain's live slots; re-derive the pile and carry ranges). About 30 read terms per slot against
  8 today, on the same two column sets.
- Uniform passes. Nine passes of the four-term identity, each the shape of today's fold families with
  signed carries: no selector per pattern, but nine `dst + copy + carry` column sets against two. Too wide.

The first is the one to build. Both leave the add lanes as they are: the modulus blocks `pb{j}` already
hold any 256-bit modulus, so `p` and `n` of P-256 go where secp256k1's do; on the host, `modulus` in
`mulmod.mojo` (the moduli of the trace and the public data) is still secp256k1's and becomes a parameter
of the statement.

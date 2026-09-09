# ECDSA on the mulmod circuits: design

One secp256k1 signature verification as a fixed circuit of the chains in `workloads/mulmod.mojo`. Written
2026-09-09 after three Codex reviews; the first draft had a signature-dependent circuit, a witness point
`R`, slope-only additions and a fixed blinding point, and all four are gone. Numbers are counts, not measurements.

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

- `Q` side: GLV split `u2 = k1 + k2 lambda mod n` with signed magnitudes below `2^128` (libsecp256k1's
  split), bases `Q` and `phi(Q) = (beta x_Q, y_Q)`, the sign of a half absorbed into the base (`-T = (x, p -
  y)`). 32 windows of 4 bits: 124 doublings (the accumulator starts at `16 B`, so window 31 needs none) and 64
  additions.
- `G` side: fixed base, 8-bit signed odd digits, tables `d 2^(8 w) G` for `w = 0..31`: 4,096 points of
  constants (256 KiB) plus the window-0 correction entries. 32 additions, no doublings. The `G` additions
  come after the last doubling, so nothing scales them.
- Recoding of an odd magnitude `k` in base `2^b` (`b = 4` or `8`): for the low 31 digits, `d = (k mod 2^(b+1))
  - 2^b`, `k <- (k - d) / 2^b`, which keeps `k` odd; digit 31 is what remains. Every odd `|k| < 2^(32 b)`
  has exactly this representation (Codex checked the small lengths exhaustively). Without the terminal
  rule the recurrence never ends: `k = 1` emits `-15` forever.
- Parity: the recoding takes an odd value, so a half `k` (signed, after the GLV split) is recoded as `k - c`
  with `c = 1` for even `k` and `c = 2` for odd `k`, and window 0 adds `(d_0 + c) T` instead of `d_0 T`.
  `k = 0` and `k = 1` both recode `-1` (terminal digit `-1`, the low digits `15`). The sign of the base is
  the GLV half's sign only; skewing the magnitude and then moving the sign into the base would reverse the
  skew and make zero contribute `-2 T`. The window-0 entries are `(d_0 + c) T` for `d_0 + c` in `[-14, 17]`
  (`Q` side) or `[-254, 257]` (`G` side); the verifier builds `T .. 17 T` for `Q` (one doubling, fifteen
  additions) and `phi` of them (seventeen field products), or only the selected entry.
- If a window-0 addend is the identity (`d_0 + c = 0`), the step adds `B` instead, and the closing constant
  subtracts one more `B`. All such steps come after the last doubling, so the correction is exactly `t B`,
  `t <= 3`: four closing constants `-(2^128 + t) B`.
- Schedule: `acc = 16 B`; window 31 of the `Q` side (two additions); for `w = 30..0`: four doublings, two
  additions; then the 32 `G` additions; then the closing constant; the last addition computes only `x`.

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
(completeness) and for any statement chooser. The verifier computes `16 B` and `2^128 B` (132 doublings)
and the four closing constants; that is the "one fixed multiple" of section 2.

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

Counts: 124 doublings, `64 + 32 + 1 = 97` additions (the last without `x1 - x3` and `y3`): `496 + 290 =
786` MUL; `124 x 6 + 97 x 8 - 2 + 2 = 1,520` add-lane ops, 760 chains' worth with two lanes. The MUL count
sets the chain count: `h2 = 896 = 2^7 7`, grid `144 x 896`, 129,024 rows, about 110 idle MUL lanes. The spec's row for ECDSA is `144 x 1344` at 1,300
mulmods; the difference is the fixed-base comb (no doublings for `G`) and the public odd-multiple table of
`Q` (no table chains). Public factors: 97 additions x 2 coordinates, `16 B`, the closing constant, `r`,
the constants `1`, `p - 1`, `n`: about 200 fingerprints, one coset of the 18 at `h2 = 896` (11 are wiring
slots: 3 on the MUL lane, 4 per add lane).

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

- 16-bit windows for `G` (16 additions, about 32 MiB of constants) if the 32 `G` additions ever matter; they
  are about 12% of the MUL count.
- secp256r1: `a = -3` (one more op per doubling), no endomorphism (256 doublings), a different fold.
- The hash-to-curve for `B`: which hash, and whether the encoding of `(Q, r, s, e)` it takes is the
  public-input byte string as is.

## 7. Implemented (2026-09-09)

`workloads/ecdsa.mojo`: the host arithmetic (`Curve`, affine points on `Big`, Fermat inversions, the fold
reduction), the GLV split by the exact lattice basis, `recode`/`skew`, the blinding point by
try-and-increment on Blake3 of the 160 public-input bytes (`r, s, e, x_Q, y_Q`, the encoding open in
section 6, settled as the public inputs as they are), and the `Ecdsa` workload: `walk` emits the fixed
circuit once and, per signature, the public factor values in factor order and the slope hints. The
verifier's `public_data` runs the same walk from the public inputs. Test: `tests/test_ecdsa.mojo`.

Deviations from sections 4 and 5:

- The slope enters as a **hint operand** (`hint(h)` in `mulmod.mojo`): a witness with no factor, wired
  between its three occurrences (`l dx`, `l l`, `l (x1 - x3)`), bounded by the piece selectors on `a` and a
  new `bbd` family on `b`. The op counts of section 4 assumed that and hold: 786 MUL, 1,520 add-lane ops,
  2,306 ops on 786 chains of the `144 x 896` grid, 221 hints, 492 public factors (each public point's
  coordinate is a factor at every operand that reads it, `x2` twice per addition; the count of section 4
  was per point, not per read). `q` is `b0 + 2 b1 + 4 b2 - 2 b3` in `[-2, 7]`.
- No circuit header: `mulmod_statement(pin=False)`. The public inputs are the 160 bytes.
- The fixed-base tables are not constants yet: `walk` computes `2^(8 w) G` and the 32 selected multiples per
  signature (about 800 curve operations, ponytail-marked). With the 132 doublings for the blinding constants
  and the `Q` tables, a live walk is about 1,000 affine operations at a few milliseconds each.

Measured on the 16 GB Mac (Metal), `CLIENT.grid(144, 896)`, `e = 16`: proof 683,392 bytes; the prover
round trip test (two live walks, prove, verify) 30.5 s; the verifier's proof work under 0.4 s (the profile
lines in the test output), the rest of its time the walk and the 492 fingerprints.

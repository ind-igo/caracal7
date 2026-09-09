# The 256-bit product as a polynomial identity

The polynomial-mulmod spec, items 1 to 6, on the builder (`relations/mulmod.mojo`): `a b = r` with no product
ever committed. Read the spec for the argument; this doc records what the code does where the spec left a
choice, and one hole the spec's zero row closes.

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

Accumulators `ra`, `rb`, `rc`, `rr` (`R_A`, `R_B`, `R_C`, and the plain fingerprint of `r`), all with scale
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
- Booleanity on the 124 bit columns.

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

`a`, `b`, `r` are public factors on `ra`, `rb`, `rr` (three slots, two wiring products). The derived public
data is the factors' ingest columns in order: `a` in its pieces (12 columns), `b`, `r` (4 each), 144 bytes
per column. The verifier's fingerprint is `horner_chain_end`, 20 x 143 E products.

## Not done

The fold (spec 7) and wiring between chains (spec 8): a chain's inputs equated to other chains' outputs, and
the chain ordering that turns wires into next-chain reads. The grid is one chain per product; the ECDSA
composition decides how chains share the grid.

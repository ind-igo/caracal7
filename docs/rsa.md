# RSA-2048 verify

One RSA-2048 signature verification (`workloads/rsa.mojo`): `s^e = m mod n` with `e = 65537`, so 16
squarings and one multiplication, 17 modmuls, each the integer identity `a b = q n + r` with `q` a witness.
The modulus is an arbitrary public number, so there is no fold like secp256k1's; a modmul is two raw
products and a subtraction. The chain is the mulmod chain's 144 rows (`docs/mulmod.md`), 172 columns
wide: no fold, no add lanes, a raw product per chain. One verify is 1745 chains; the bench grid is
144 x 2016, since no legal grid lies between 1344 and 2016 chains (`a2 <= 7`). The signature `s` and the modulus `n` can be witness: `s_wired` wires the occurrences of each
limb of `s` to each other, `n_wired` takes the limbs of `n` from wires the caller adds (`n_chains`).

## Limb products summed along chains

A product `a b` of two 2048-bit numbers is a block of 256-bit limb products `a_i b_j` on a `limbs x limbs`
block of chains, and the running sums along the chains replace addition ops: every read of the sum lane
shifts forward on axis 2, each chain adds its own lo half, the hi halves from below and the running lo from
the diagonal. So a rectangle modmul is 128 products and 16 carry chains with no add op and no wire, and the identity
`a b = q n + r` closes on a compare lane over the 16 head chains with a `2^258` bias and a carry read between
heads, wired only for `lo(t_qn)` and `r_p`. Four wiring slots per chain (`ha`, `rb`, `cy`, `cz`).

Why this and not Karatsuba block products: additions cost wires, and wires are the scarce resource. A
three-operand add lane takes four wiring slots and 5,184 cells; Karatsuba's recombination needs about
340 adds per modmul, three times the wire budget before a single product is wired. Running sums need no
wires. The product cost is quadratic in bits, so a different limb count, lazy reduction or Montgomery do
not help, and 17 modmuls is minimal for `e = 65537`.

## Squaring symmetry

Sixteen of the seventeen modmuls square their input, and `a_i a_j` with `i > j` repeats `a_j a_i`. The
`AB` block of a squaring holds the products with `i <= j` only, `lo(r)` and the passed `hi(r)` weighted 2
off the diagonal: 37 chains for 72 at eight limbs.

The rectangle's running sums read at two constant strides (the diagonal at `limbs`, the chain below at 1)
because every anti-diagonal has the same length. A triangle's diagonals do not, so no chain order gives
constant strides. The layout is one tail chain, then the diagonals from the top with `i` ascending, and
every read is a forward stride from a small set (at most `ceil(limbs / 2) + 1` values); each stride in use
is a public mask (`cl{k}`, `hm{k}`, `rm{k}` carrying the weight, `sq{k}`, `mp{k}` for the compare carry)
and a term of the family. The square `(i, i)` passes its hi to `(i, i + 1)`, the last chain of the next
diagonal, in place of the dropped `(i + 1, i)`; the tail takes the last square's hi and heads limb
`2 limbs - 1`.

One role table, `_block_cells`, computes every chain's role and strides from the `(i, j)` order (a stride
is a lookup, checked forward); the statement's families and masks, the trace and the public data all read
that table, and the rectangle is the same code with its own `(i, j)` order.

Widths: `t` is below `2^259` (two doubled halves), so the hi read covers weights 256..259 (the `hm` row
masks) and `bu` keeps `t` below `2^260`; a tail's `t` is bound below `2^256` by `bu` itself, so the top limb
cannot hide a carry. The signed 4-bit ripple carry holds the largest pile (8 plus the carry in).

## Rectangles without tails

A rectangle block (`q n` of every modmul, `a b` of the last) was `(limbs + 1) x limbs` chains: the chain
`(limbs, j)` held `hi(t) + hi(r)` of the column top and passed it to the next column's top through its
`lo(t)`, a uniformity device from the first version. With per-stride masks a second hi read is one more
mask, not a new lane: the top row reads `hi(t)` and `hi(r)` of the column to its left directly (`hi2`,
stride `limbs`, the existing `hm`/`rm` masks), so a rectangle is `limbs^2` product chains. The `QN` block
drops its top tail too, and the `AB` head of limb `2 limbs - 1` subtracts the `QN` corner's `hi(t) + hi(r)`
by a forward stride read (`qt{k}`, `qr{k}`, strides 37 and 65 at eight limbs) in place of a wired `lo(t)`.
The `AB` rectangle of the last modmul keeps its top tail: the compare lane lives on a chain per limb.

Per verify `16 x 37 + 65 + 17 x 64 = 1745` chains. The proof grows about 1.5 percent from the new shift
points (`hi2` at `(UP, 8)`, the corner reads at `(UP, 37)` and `(UP, 65)`); the chains are the constrained
resource for the passport fold (`docs/passport.md`), so the corner read stays.

Karatsuba on the `q n` block, assessed and deferred: one level (three 4 x 4 rectangles for one 8 x 8) would
save 16 chains per modmul, 272 per verify. The recombination is real work: the sums `q_k + q_(k+4)` have to
be bound (an idle u-lane ripple with `cy` and `cz` as wired copies), the signed recombination needs a bias
and about ten stride masks on the middle block's heads, the 257-bit limbs widen the `hi(r)` reads, the
compare lane's bias changes. Build it when a statement needs the chains.

## Public data

The public columns are the role table's masks, one term per distinct value and weight range
(`rsa_public_data`, the term form of `docs/public-columns.md`); the verifier derives them from the public
inputs. The fixture is OpenSSL's (genrsa 2048, dgst -sha256 -sign; `m` the PKCS#1 v1.5 encoding, checked
as `pow(s, e, n)`).

## Tests

`test_rsa`: a 512-bit instance (2 limbs, `e = 3`, 17 chains on a 48-chain grid): every bit family holds on
the trace, the proof verifies, a wrong `m` is refused on both sides. The 2048-bit fixture is `bench_rsa`'s. The
geometry (every read forward, the identity telescoping to `a b = q n + r`, every weight of the corner's `t`
and `r` read, the bias margin on the last head) was modelled for 1 to 255 limbs before the layout was fixed.

# Public columns and restrictions (statement-layer decisions 4 and 5)

Plan for the two public-input mechanisms Keccak and ECDSA need before a frontend can exist:
public columns with a closed form, and public restrictions on a chain line. Public factors in
the wiring grand product (ECDSA constants) are wiring, a later doc.

## The idea

A public column is a function of row position the verifier can evaluate at a point in
O(coefficients) work. On the prover it is nothing but a column of values on the residual grid G
that family entries read. It is never committed, encoded, hashed as data, or opened. The whole
feature is one asymmetry: the prover has values, the verifier has a closed form, and the family
entries do not know the difference.

The simplification that makes it cheap: the LDE stage already turns coefficient blocks
(column, k2, k1, 2) into values on G with a forward transform. A public column of the spec's
form, a polynomial in (X1, X2^m), is such a block, dense in X1 and strided by m in X2. So the
prover never evaluates a form. The frontend hands it coefficient bytes, the LDE stage
transforms them like a witness column, and the residual loader reads them by column index. No
new kernel. The verifier evaluates the same block at a point with Horner.

## Decisions

- **Coefficients are per-proof host data derived from public inputs, on both sides.** The
  message changes per proof while the artifact does not. The frontend owns a function from
  public inputs to coefficient blocks and restriction polynomials; the prover and the verifier
  both call it. The prover core takes the blocks through `load_public` beside `load_trace`; the
  verifier takes them as an argument beside the public inputs it already hashes. Soundness rests
  on the verifier deriving them itself from the hashed public inputs. The prover and the
  verifier never see a "form", only blocks and their dimensions. The spec's form language
  (`poly`, products, `restriction`) is the compiler's concern, not the prover's.
- **One index space.** Columns on the LDE buffer are witness, then Z coordinates, then public,
  so an entry reading a public column is a normal entry and `_read` does not change. The builder
  rejects a quadratic entry with two public reads (pointless) and the spec's degree rule holds by
  construction: public times witness is degree 2, which the IR admits. Shape checks the block
  fits the grid: `m >= 1`, `d2 >= 1`, and `m (d2 - 1) < h2`. Without that bound a block with an
  `X2^h2` term is truncated by the coefficient buffer while the verifier's Horner keeps it, and
  the two sides evaluate different polynomials.
- **Challenge-weighted columns are not public columns.** The statement layer's `zeta^t` and lane
  weights (statement-layer 8) depend on challenges sampled after W is committed and take values
  in E. They cannot be derived from public inputs before `prove` and do not fit the F2 LDE path.
  They need their own plan (an E-valued term the verifier evaluates from the transcript).
- **Dense blocks in version one.** A block is (d2, h1) coefficients, row j the coefficient
  vector of X2^(m j). The spec's product form (chain selector times interpolant) is expanded by
  the frontend into one block. The verifier pays the dense evaluation; the factored form is a
  verifier-only change to the block description if the measurement asks for it (below).

## Data shapes

- **Shape.** `publics: List[PublicSpec]` with `{m, d2}` per public column (d1 = h1 always: dense in
  X1 costs nothing on the prover and the frontend pads); `restrictions: List[Restriction]` with
  `{column, coordinate (FIX_ONE or FIX_E on axis 2), degree}`. `columns_p = len(publics)`. Both
  lists enter the prefix as integers. The LDE buffer grows to `columns_w + columns_z + columns_p`;
  the openings, trees, and proof do not.
- **Prover.** `pub_coeff` arena region, `columns_p x N x 2` bytes of F2 coefficients. The
  frontend supplies F2 coefficients: the interpolant of F-valued (even bit-valued) grid points has
  F2 coefficients, because the grid twiddles are F2-valued (caracal-prover 9.1). `load_public(ctx, prover, blocks)` uploads; `prove` issues a third `lde`
  call after the Z one, into the LDE buffer past the Z columns. The `ltmp` scratch already sizes
  by the larger of the witness and Z column counts; it takes `columns_p` into that max.
- **Points.** A restriction at (z1, e2) is the point `(0, FIX_E)`; `shift_points` gains it when
  the shape has a restriction on that line. `point_coord` already handles a fixed coordinate per
  axis. One more point costs `columns x e` proof bytes and one evaluation query.
- **Verifier.** Takes `public: PublicData` = the blocks and the restriction polynomials, derived
  by the caller. Where `residual_at`'s reads are gathered, a column index at or past
  `columns_w + columns_z` evaluates its block at the read's point instead of taking an opening;
  cached per (column, point). Restriction check: the opening of `column` at the line point equals
  the degree < h1 polynomial at z1. Both use one `eval_block(coeffs, m, d2, x1, x2)` with Horner
  in X1 then in X2^m.

## Sizes

| item | Keccak-256 of 2048 B, 64 x 384, 142 columns (statement-layer 8) | ECDSA, 144 x 1344 |
|---|---|---|
| public columns | 17, one per absorb lane | few; challenge weights excluded (above) |
| coefficient bytes per proof, host | 17 x 24,576 x 2 = 836 KB | small |
| prover LDE growth | 17 over 142 columns, ~12% | negligible |
| verifier terms at z, dense | 17 x 24,576 = 418K E mults, ~40 ms host | small |
| verifier terms at z, factored (spec) | ~20K | same |
| proof growth | two points (first state, digest), 2 x 190 x 16 = 6.1 KB | one point |
| kernels changed | none | none |

The dense verifier evaluation is the one ceiling. Version one accepts ~40 ms on the host for
Keccak-2048; the factored form (a block as a product of two smaller blocks) is a
`PublicSpec` variant the verifier multiplies, added when measured verifier time matters.

## Tests

- `test_residual`: a family with a public read (`c_k - c0 * pub`) matches the host residual at
  a point, with the public column's values on G from a host evaluation of the block.
- `test_prover`: the synthetic instance with one public column (a random `(X1, X2^4)` block) and
  one restriction (a witness column's last chain against its interpolant) accepts; the same
  proof with a changed public input rejects at the residual identity; a changed restriction
  polynomial rejects at the restriction check.

## Work

Shape and layout ~60 lines; verifier evaluation and restriction ~80; prover upload and the LDE
call ~30; synthetic and tests ~120; the size of the second lookup commit. Order: Shape and the
LDE call with the residual test; the verifier and the prover test; the synthetic restriction.

## Status (2026-09-07)

Implemented as planned, no kernel changes. `PUB` and `RES` records in ir.mojo; `Shape` takes
`publics` and `restrictions`, validates the degree rule and the column ranges, and puts both in
the prefix; `shift_points` adds the restriction line; the prover has `pub_coeff` and
`load_public` (blocks expanded to full coefficient tables through `expand_blocks`) and a third
`lde` call; the verifier takes `public` (blocks, then restriction polynomials), evaluates public
reads with `eval_block` cached per (column, point), and checks each restriction with `eval_line`.
The synthetic instance has a public column periodic along axis 2 (m = 4, d2 = h2 / 4) and a
restriction of c1 to its last-chain interpolant; the host interpolation helpers live in
synthetic.mojo until the builder owns them. Tests: test_residual (public read on G, DEEP identity,
`eval_block` against the dense Horner, the off-row coefficients are zero) and test_prover (accept;
a changed coefficient fails the residual identity; a changed polynomial fails the restriction;
wrong size; a block past the degree rule is rejected by Shape).

## Not in this doc

Public factors in the wiring grand product (ECDSA constants), and the wiring itself. Row groups
and names: the statement builder above the IR, which is where the frontend interface lives.

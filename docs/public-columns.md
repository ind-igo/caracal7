# Public columns and restrictions

The two public-input mechanisms of the statement layer: public columns with a closed form, and public
restrictions on a chain line. Public factors in the wiring grand product (ECDSA constants) are wiring.

## The idea

A public column is a function of row position the verifier can evaluate at a point itself, in one
product per value of its public data. On the prover it is nothing but a column of values on the residual grid `G` that family entries
read. It is never committed, encoded, hashed as data, or opened. The whole feature is one asymmetry: the
prover has values, the verifier has a description it evaluates itself, and the family entries do not
know the difference.

The LDE stage already turns coefficient blocks into values on `G` with a forward transform, so a public
column takes the witness column's path on the device: the prover tiles the column's public data to `H`,
runs the trace's `idft2` on it (`load_public`) and a third `lde` call puts it past the Z columns of the
LDE buffer. Columns on that buffer are witness, then Z coordinates, then public, so an entry reading a
public column is a normal entry. No kernel knows the column is public.

## Decisions

- **Public data is derived from the public inputs on both sides.** The frontend owns a function from
  public inputs to public data; the prover and the verifier both call it, and the verifier hashes the
  public inputs. Soundness rests on the verifier deriving the data itself.
- **One index space.** The builder rejects a quadratic entry with two public reads (pointless), and public
  times witness is degree 2, which the IR admits.
- **Challenge-weighted columns are not public columns.** The `zeta^t` lane weights depend on challenges
  sampled after W is committed and take values in `E`; they are terms the verifier evaluates from the
  transcript, not columns.
- **Values, not coefficient blocks.** The first version handed the prover coefficient blocks of a
  polynomial in `(X1, X2^m)`; the dense host interpolation was the whole end-to-end cost (23 s prover
  side and 25 s verifier at Keccak-2048, against a 0.5 s prove). The public data of a column is a period
  of its values, `(h2 / m, h1)` F bytes, the same on both sides; the record is `m` alone. The verifier
  interpolates the period barycentrically at the point (`eval_values`, the period's interpolant on
  `<omega2^m>` at `x2^m`): one product per public value, linear in the message.
- **Term form for structured columns.** On the passport statements the values form was half the verify:
  153 columns of 290,304 values, 48.6 MB to derive and interpolate. The public columns of a row group are
  structured (a row pattern over the four slots on all chains of the group, on the live chains, or on one
  chain; the round constants on the sixteen chain classes of a block), so a column's public data can be a
  **term list**: `PubTerm` is a row vector (`h1` F values) and a chain list (indices in the column's
  period), and the column is the sum over its terms of `row(x1) [x2 in chains]`. `pack_terms` encodes it,
  marked by a first byte `TERMS` (255, which no F value is); a dense period (first byte below 127) is still
  accepted. `Layout.selector` and `Layout.mask` are one term each; `sha256_group_public` and
  `rsa_public_data` build terms directly; Keccak, SHA-256, Poseidon, mulmod, ECDSA and the synthetic
  instance stay dense. Soundness is unchanged: the verifier evaluates the same polynomial, the interpolant
  of a sum of products being the sum of the products' interpolants.

## Shape and code

- **Shape.** `publics`, the `PUB` records (`m` per public column), and `restrictions`, the `RES` records
  (`{column, coordinate (FIX_ONE or FIX_E on axis 2), degree}`), both byte lists of `ir.mojo`. `columns_p = len(publics)`. Both
  lists enter the prefix. The LDE buffer grows to `columns_w + columns_z + columns_p`; the openings,
  trees and proof do not.
- **Points.** A restriction at `(z1, e2)` is the point `(0, FIX_E)`; `shift_points` adds it when the
  shape has a restriction on that line. One more point costs `columns x e` proof bytes.
- **Verifier.** `column_offsets` parses and validates the public data once (bounds, F values, chains
  below the period) and returns each column's offset. Where `residual_at`'s reads are gathered, a column
  index at or past `columns_w + columns_z` evaluates its public data at the read's point instead of
  taking an opening, cached per (column, point). The Lagrange values of both axes are cached too, per
  point on axis 1 and per point and `m` on axis 2 (`h1 + h2 / m` inversions each, once): a dense column
  is `eval_values_with` on them, a term column `eval_terms` (the row against the axis-1 values times the
  sum of the axis-2 values over the chains). A column whose values are the same on every chain is
  declared with `m = h2`: one chain of public data, and `h1` products per point (the mulmod builder's
  chain-constant columns take `m` from the workload's grid). The group checks walk terms (exact:
  one term of ones on the mask's chains in order; else every chain in the mask). A restriction check:
  the opening of `column` at the line point equals the degree `< h1` polynomial at `z1` (`eval_line`).
- **Prover.** `load_public` takes the whole public data and tiles either form to `H` (`tile_values`);
  then `idft2` and the LDE. Nothing on the device changed for the term form.

## Tests

- `test_residual`: a family with a public read (`c_k - c0 * pub`) matches the host residual at a point,
  with the public column's values on `G` from the host.
- `test_prover`: the synthetic instance with one public column (periodic along axis 2, `m = 4`) and one
  restriction (a witness column's last chain against its interpolant) accepts; the same proof with a
  changed public input rejects at the residual identity; a changed restriction polynomial rejects at the
  restriction check; a wrong size is rejected.

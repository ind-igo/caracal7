# The prover as a protocol

This is the interactive protocol the code implements, written the way a paper writes one: the
objects, then every message, challenge and check in transcript order, then what each check
certifies and why the protocol has this shape. It describes `prover.mojo` and `verifier.mojo` as
they are, not the spec's intent; where the two differ the code wins and the difference is noted.
The soundness ledger (`docs/soundness.md`) charges error terms to the steps below by their names.
Measured stage times and proof bytes per workload are in `docs/profile.md`.

Notation. `F = F127`, `F2 = F[i]`, `F4 = F2[j]`, `E = F4[u] / (u^5 - g)`, twenty coordinates,
`|E| = 127^20`. `H = H1 x H2` is the grid, `h_l = |H_l| = 2^{a_l} m_l`, `N = h1 h2`, `omega_l`
generates `H_l`, `e_l = omega_l^{-1}`, `G_l` the group of order `2 h_l` containing `H_l`.
`Z_{H_l}(X) = X^{h_l} - 1`. A column is a function `H -> F`; it is identified with its polynomial of
bidegree below `(h1, h2)`. An `E`-valued column (an accumulator, a quotient) is committed as its
twenty coordinate columns `c_t` and opened as the one value `sum_t b_t c_t(z)`, `b_t` the basis
elements; "column" in steps 7, 8 and 14 means a witness column or an `E`-valued column, so
`stored(c)` of an `E`-valued column is `sum_t b_t stored(c_t)`, `N` values of `E` fixed by the
commitment. `stored(c)` of a witness column is the column in the mixed basis of spec 9.1, `N` bytes. `Enc`
is the level-1 code: pack four stored bytes into one `F4` symbol and Reed-Solomon encode over `F4`
on the domain `D`, `|D| = L`, a union of one to four cosets of a subgroup of `F4*`; a column whose
symbols exceed the rate is split into `n_cw` codewords. `Merkle` is a Blake3 tree over rows of
symbols. `<., .>` is the `E`-bilinear pairing of a length-`N` vector with a functional.

## 1. Setup

`pp <- Setup(statement)`. Everything below is fixed before the first message and serialized into the
transcript prefix, so both parties hold it.

- **Parameters.** The fields, the grid `(a1, m1, a2, m2)`, the level-1 domain `D` and its rate, the
  query count `s_1` of level 1 and `s_l` of every tail level (each from its own rate), the tail
  schedule (domains `D_l`, rows `n_l`, codewords, the three digits folded per level, the clear
  length), and the grinding bits (20).
- **The compiled statement** (`relations/statement.compile`, `proof.Shape`): the column lists of
  the three trees (`W` witness, `Z` accumulators, `Q` quotients) and the public columns; the entry
  table (one entry per term of a relation: the two columns it reads, their offsets, the coefficient
  and the gate; several entries share a family index, which is the relation's `alpha` power); the
  accumulators (kind, columns, start value, table); the wiring lines `sigma` and the public
  factors; the restrictions; the zero rows; the lookup tables; the point list; the challenge
  derivation table.
- **The point list** `z_1 .. z_P` (`relations/ir.required_points`, deduplicated): the DEEP point
  `z`; the shifted points the entries read, `(omega1^k z1, z2)` and the next-chain point
  `(z1, omega2 z2)`, only when some entry needs them; with accumulators, the chain lines `(1, z2)`,
  `(e1, z2)`, `(1, omega2 z2)` and the corner `(e1, e2)`; and the restriction lines the statement
  declares, such as `(z1, 1)`. Every point is `z` with zero, one or two coordinates replaced by fixed
  grid values or shifted by a root of unity, so the list is a function of `z`. The corner `(1, 1)` of
  spec section 3 is not a point: `Z(1, z2) = 1` at random `z2` covers it.
- **Public inputs** `io` and the public data both sides derive from them (`docs/public-columns.md`):
  one period of every public column and the polynomial of every restriction. The verifier derives
  them; it never hashes the derived bytes, only `io`.
- **The prefix**: the serialized protocol version, parameters, compiled descriptors (some as their
  digests), tables and `io`. Both transcripts absorb it first.

## 2. The protocol

`b <- <P(w), V(io)>(pp)`. `P` holds the witness columns. Every challenge is uniform in `E` unless
stated; every "abort" outputs `b = 0`. Messages marked "with accumulators" exist only when the
statement has accumulators (the hash workloads have none; then there is no `Z` tree, no `Z2`, no
`Q3`, and level 1 opens two trees). The implemented protocol is the Fiat-Shamir compilation of this
one with Blake3 (section 4); the message and challenge order is the code's.

```
 1. P: commit the witness. For every column c in W compute stored(c) and Enc(stored(c));
       C_W <- Merkle(rows of the codewords); send C_W.
 2. V: sample beta, delta, gamma in E (stage 1); derive the statement's fixed functions of
       them; if the statement has wiring, sample beta_w, gamma_w in E.
 3. P, with accumulators. For every grand product k (permutation, lookup) an E-valued column Z_k
       with Z_k(1, x2) = 1 and Z_k(omega1 x1, x2) D_k(x1, x2) = Z_k(x1, x2) N_k(x1, x2) along each
       chain, and a line Z2_k: H2 -> E carrying the chain-end pairs, Z2_k(1) = 1,
       Z2_k(omega2 x2) D_end = Z2_k(x2) Z_k(e1, x2) N_k(e1, x2). For every Horner fingerprint k a
       column R_k with R_k(1, x2) = start_k and R_k(omega1 x1, x2) = scale R_k(x1, x2) - ingest
       (the ingest a fixed or selector-gated linear form of the row's columns, sign folded in).
       For every wiring product g a line Z2_g alone, Z2_g(1) = 1, Z2_g(omega2 x2) D_g = Z2_g(x2) N_g,
       its factors (w + gamma_w + beta_w id) and (w + gamma_w + beta_w sigma) on the chain-end
       values w of the wired slots. C_Z <- Merkle(Enc of the coordinate columns of every Z_k and
       R_k); send C_Z and every Z2 line in the clear (h2 elements each).
 4. V: sample alpha in E.
 5. P: the quotients. R = sum_j alpha^j R_j on the residual grid G = G1 x G2, every grid family j
       (pointwise; transition, gated by (X1 - e1) or by (X2 - e2) per entry; cyclic reads); A, B,
       Q2 of bidegree below (h1, h2) with R = (A + X2^{h2} B) Z_{H1}(X1) + Q2 Z_{H2}(X2). With
       accumulators: R2 = sum_j alpha^j R2_j on a coset of G2, the chain-end families (the
       accumulator and wiring closures gated by (X2 - e2), the other chain-end terms gated when
       their descriptor says so), Q3 = R2 / Z_{H2} of degree below 2 h2. C_Q <- Merkle(Enc of the
       A, B, Q2 coordinate columns); send C_Q and, with accumulators, Q3 as its 2 h2 values on G2.
 6. V: sample z = (z1, z2) in E^2; this fixes the point list z_1 .. z_P.
 7. P: send the openings alpha_{c,p} = c(z_p) for every column c of W, Z, Q (one value for an
       E-valued column) and every point p.
 8. V: sample beta_c in E per opened column and gamma_p in E per point; a coordinate column c_t of an
       E-valued column c takes beta_c b_t, so sum over the stored columns below is the sum over the
       opened ones.
 9. V, with accumulators: boundaries. Z_k(1, z2) = 1 and R_k(1, z2) = start_k from the openings at
       (1, z2); for every grand product: Z2_k(1) = 1 and Z2_k(e2) Z_k(e1, e2) N_k(e1, e2) = D_k(e1, e2)
       (for a lookup, the table constant) from the openings at (e1, e2). For every zero row:
       c(b, z2) = 0. Abort on any failure.
10. V, with wiring. Every Z2_g(1) = 1, and jointly over the products g, their slots and the public
       factors f with fingerprints v_f computed from io,
         prod_g Z2_g(e2) prod_slots (w + gamma_w + beta_w id) prod_f (v_f + gamma_w + beta_w id_f)
         = prod_slots (w + gamma_w + beta_w sigma) prod_f (v_f + gamma_w + beta_w sigma_f),
       w read at (e1, e2); abort if a public factor or its selector is zero, or on inequality.
11. V, with accumulators: small grid. Abort if z2 lies in G2 (an interpolation node). Interpolate
       every Z2 line at z2 and omega2 z2 and Q3 at z2 from the clear vectors; assemble R2(z2) from
       them, the openings at (e1, z2) and (1, omega2 z2), and the chain-end terms; abort unless
       R2(z2) = Q3(z2) (z2^{h2} - 1).
12. V: residual. Evaluate every grid family at its points from the openings and the public
       columns' closed forms; abort unless
         sum_j alpha^j R_j(z) = (A(z) + z2^{h2} B(z)) (z1^{h1} - 1) + Q2(z) (z2^{h2} - 1).
13. V: restrictions. For every restriction, the opening of its column on its line equals the
       public polynomial at z1; abort otherwise.
14. Both: the running claim. y_1 <- sum_c beta_c stored(c) in E^N (P holds it). w~ <- sum_p gamma_p
       w_{z_p}, the functional with <stored(c), w_{z_p}> = c(z_p); V holds it as a list of tensor
       units (products over the binary digits times a vector on the odd digit, sixteen per point).
       T <- sum_{c,p} beta_c gamma_p alpha_{c,p}. Claim: <y_1, w~> = T.
15. For every committed tail level l = 1 .. ell:
    a. P: Mat(y_l) has 8 columns (the three lowest unfolded binary digits) and n_l rows; a level
          whose rows exceed its rate is split into n_cw,l codewords on the top digits; encode each
          column of each codeword with Reed-Solomon over E on D_l; C_l <- Merkle(rows of
          8 n_cw,l elements); send C_l.
    b. P: an 8-byte grinding nonce for this level's query set (about 2^20 trials on average);
          V: check its 20 leading zero bits, then sample s positions in the previous domain (D and
          s_1 for l = 1, D_{l-1} and s_{l-1} after).
    c. P: send the rows of the previous level at S with Merkle multiproofs: for l = 1 the leaves
          of C_W, C_Z (with accumulators) and C_Q (4 n_cw bytes per column); after, the rows of
          C_{l-1}.
    d. V: check the multiproofs; compute the expected symbols of y_l at every opened position and
          codeword from the opened rows (l = 1: per codeword cw and F4 coordinate tau,
          sum_c beta_c coord_tau(X[s, c, cw]); after: per codeword, sum_a rbar_{l-1}[a] X[s, cw, a]);
          sample batching scalars b_0, b_1, .. in E; T <- b_0 T + sum_q b_q v_q; w~ <- b_0 w~ +
          sum_q b_q g_q, g_q the functional of the code's generator row at that position and
          codeword (units again).
    e. Three rounds d = 1, 2, 3: P sends s_d, a degree-2 polynomial (3 elements); V aborts
          unless s_d(0) + s_d(1) = T; V samples r_d in E; T <- s_d(r_d); V folds every unit at
          r_d. rbar_l <- (1 - r_1, r_1) x (1 - r_2, r_2) x (1 - r_3, r_3); P: y_{l+1} <- Mat(y_l) rbar_l.
16. P: send y_{ell+1} in the clear (the clear length in elements; y_1 itself, N elements, when
       the schedule commits no tail level).
17. P: a grinding nonce; V: sample s_ell positions in D_ell (level 1's domain and count when
       ell = 0); P opens the rows of C_ell (or of the three trees) at them; V aborts unless, at every
       opened position and codeword, the encoding of y_{ell+1} equals the folded opened row
       sum_a rbar_ell[a] X[s, cw, a] (at ell = 0: the level-1 symbol of the opened leaves), and
       unless <y_{ell+1}, w~> = T evaluated from the units and the clear vector.
18. V: output b = 1.
```

Two details the box compresses. The Frobenius-real form: a witness column has coefficients in
`F2`, and `stored(c)` holds one `F` byte per degree of freedom, so the functional `w_z` is a short
sum of tensor products per point, sixteen in the verifier's unit form (`pcs/tensor.mojo`), not one;
the units carry this. The `E (x) F4` alphabet: a level-1 symbol of `y` is four `E` values, one per
`F4` coordinate, and the batching in 15d is coordinate-wise; multiplying into `E` would be 4-to-1
and unsound (spec 9.1, "Alphabet").

## 3. What each step certifies

Read the box backwards. Steps 15 to 17 are Ligerito (Ligero with a recursive tail): they certify
that the committed rows are close to codewords and that the claimed `alpha_{c,p}` are the
evaluations of the committed columns at the points, with error the query terms and the field terms
of `docs/soundness.md`. Given that, steps 9 to 13 are identities between
the committed columns. Most are checked at a random coordinate of `z` with Schwartz-Zippel error;
the closing products of steps 9 and 10 are equalities at the fixed corner `(e1, e2)`, and their
error comes from the stage-1 and wiring challenges the fingerprints were built with. The ledger's
"Relation numerators" table names each one:

| step | identity | certifies | ledger name |
|---|---|---|---|
| 12 | `R = (A + X2^{h2} B) Z_{H1} + Q2 Z_{H2}` at `z` | every grid family vanishes on `H`: pointwise relations everywhere, transition relations on every within-chain pair, the accumulator and Horner recurrences included | `grid_identity`, `alpha_batch` |
| 11 | `R2 = Q3 Z_{H2}` at `z2` | every chain-end family vanishes on `H2`: the chain-end pairs of every accumulator and wiring product, and the chain-end terms in which a Horner endpoint must equal what the statement says it is | `small_grid`, `alpha_batch`, `horner_identities` |
| 9 | `Z_k(1, X2) = 1`, `R_k(1, X2) = start` and the zero rows at `z2` | no chain is rescaled or restarted; the rows that must be zero are | `starts_and_zero_rows` |
| 9 | closing products at `(e1, e2)` | the grand product over the whole trace is 1 (permutation) or the table constant (lookup) | outside this ledger's scope: the CSP workloads have no lookup or permutation accumulator |
| 10 | joint wiring product at `(e1, e2)` | the copy constraint: the multiset of `(value, id)` over the wired slots and public factors equals the multiset of `(value, sigma)` | `wire_product`, `wire_fingerprints`, `wire_zero_factors` |
| 13 | line restrictions at `z1` | a column agrees with a public polynomial on a whole line | `restrictions` |

The composition is: fix the adversary's columns at each commit barrier (`W` before stage 1, `Z`
and `Z2` after stage 1 and the wiring challenges, `Q` and `Q3` after `alpha`); every later challenge
is independent of the columns fixed before it; the identities of steps 9 to 13 hold for the
decoded columns except on the union of the bad events; the decoded columns then satisfy every
family of the statement on `H`; and the workload lowering (SHA-256, Keccak, RSA and the rest)
makes those families equivalent to the claimed computation. The ledger sums the bad-event
probabilities; it does not take the largest term.

The five open obligations of the ledger, in this picture:

- **P1** is step 15d at `l = 1`: the packed `F4` alphabet admits the scalar Reed-Solomon
  guarantee in the block metric (spec 12.1, unpublished; own proof).
- **P2** is steps 9 to 13: the two-grid identities imply every intended relation, including the
  chain-end pairs, under the adaptive choice of `Z` after stage 1 and of `Q` after `alpha`.
- **P3** is the workload lowering behind the entry table of the setup: carries, selectors,
  canonical values, idle rows, hints, and the nonzero-polynomial argument for Horner and wiring
  fingerprints.
- **P4** is steps 14 to 17: the tensor-unit verifier is the analyzed fold protocol, with the mixed
  digits, the four-coordinate first batch and the last clear check.
- **P5** is section 4: the Fiat-Shamir compilation.

## 4. Fiat-Shamir

The transcript is Blake3 with a domain separator per message kind: prefix, `W` root, `Z` root and
`Z2`, `Q` root and `Q3`, openings, tail root, tail round, clear vector, grinding nonce. The prover
runs it on the device, so the host never sees a challenge; the verifier reruns it on the host. A
field element is its `e` coordinate bytes, one `F127` byte each; sampling masks each byte to seven
bits and rejects 127; positions are uniform integers below the domain size, by rejection. Every
opened level carries one 8-byte grinding nonce whose hash must start with 20 zero bits: about
`2^20` trials for the prover, one hash for the verifier. The verifier rejects any proof byte read as
a field coordinate that is not canonical before it does arithmetic on it, authenticated rows
included (ledger I1). The prefix binds the parameters, the compiled descriptors, the tables and the
public inputs, so a challenge is a function of the statement and every earlier absorbed message;
Merkle multiproofs are checked against the absorbed roots and not absorbed themselves.

## 5. Why this shape

**One quotient argument, no sumcheck over the trace.** The residual `R` is computed on the doubled
grid `G` as one pass over the columns' low-degree extensions, and the quotients are divisions by
`Z_{H1}` and `Z_{H2}`, which are constants on the cosets. The transforms, the Reed-Solomon encode,
the openings and the fold are dense linear maps over `F127` bytes with `E` scalars; the residual
multiplies pairs of byte-valued reads before its `E` weights; only the accumulators do `E` by `E`
products and inversions, along chains. That is the design bet: byte-valued witnesses make the
linear passes int8 and fp16 matrix work on every GPU. A sumcheck over the trace would move the
rounds after the first into `E`, where a product costs about 25 `F4` products, and would need
either a hypercube grid or a mixed-radix round for the odd parts `m1`, `m2`.

**A grid of chains, not a hypercube.** Multiplicative subgroups give shifts for free: a transition
reads `(omega1 x1, x2)`, and the opening at `(omega1 z1, z2)` is a point like any other. Chains
are the unit of parallel work and of accumulation; the small grid `H2` carries what crosses chain
ends, one line per grand product, sent in the clear because it is `h2` elements.

**Accumulators in product form.** Permutations, lookups and wiring are grand products, not
fraction sums, because in characteristic 127 a fraction sum counts multiplicities modulo 127. The
same fact rules out LogUp and one-hot sums over more than 127 entries. Records are fingerprinted by
a fixed injective embedding into `E`; Horner chains ingest challenge-weighted, selector-gated terms,
and the ledger's `horner_identities` and `wire_fingerprints` terms are where a false fingerprint
must show as a nonzero polynomial in the challenges.

**Ligerito as the commitment.** Level 1 opens `s_1` rows of width `4 n_cw` bytes per column across
the three trees; the rows above are folded by the tail, three binary digits per level, one `E`
matrix of eight columns per level, so rows cost only through the tail. The odd digit `m1 m2` is
never folded: it rides in the row index and in the clear vector. With at least one committed tail
level the verifier holds the query as tensor units and folds them per round, so it never holds a
length-`N` vector; its cost is units times levels plus the clear vector. Where the bytes go per
workload is measured in `docs/profile.md`: the openings at every point times every column, the
`Z2` and `Q3` lines and the level-1 rows are the three large regions.

**What the protocol is not.** It is not zero-knowledge: the opened rows and the openings leak
witness bytes (`docs/passport.md`). It has no recursion or aggregation, so the proof is for
an off-chain verifier. And every identity is one in characteristic 127, so integers are proved in
bits and small limbs; a dense integer dot product has no cheap relation here.

## 6. Where the steps live

| box step | prover | verifier |
|---|---|---|
| 1, 3, 5 commits | `Prover._commit`: `encode` (`idft2` for a trace, `k_to_stored`, `pack`, `rs_encode`; the `Q` tree enters at its coefficients) and `merkle` (`pcs/encode`, `pcs/merkle`) | roots read from the proof |
| 3 accumulators | `relations/accumulate` | `_boundaries`, `_wiring` (steps 9, 10) |
| 5 quotients | `relations/residual` (`lde`, `residual`, `quotient`), `relations/smallgrid` | `_residual`, `_small_grid` (steps 11, 12) |
| 13 restrictions | | `_restrictions` |
| 7 openings | `pcs/open` (`point_tables`, `class_weights` and `open_stage1`, `open_stage2`, or `build_queries` and `open_direct`; `compact_openings`, `expand_beta`, `fold`) | `_opening`, `_PublicReads`, `_expand_beta` |
| 14 to 17 tail | `Prover._tail_level`, `_open_previous`, `pcs/tail` | `_Tail` on `pcs/tensor` units, `_open_previous`, `pcs/merkle.check_multiproof` |
| section 4 transcript | `core/transcript` on the device | `HostTranscript` |

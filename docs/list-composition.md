# Packed and recursive list argument

Status: **own proof, needs review**. This note checks vault spec 12.3 and the
verifier's challenge order. It does not prove the field implementation,
basis maps, workload constraints, or Fiat-Shamir compilation. P1-P5 remain
open in [soundness.md](soundness.md).


The ledger is unchanged. With its existing grinding model, the DKT26
comparison still gives 106.41 conditional bits for ECDSA and 105.67-106.41
for the 16 recorded workload cases. No new PCS error term was found in the
scope below.

## Exact scope

First use an interactive protocol with independent uniform challenges and
binding full-word oracles. Each root fixes all its rows, including unqueried
rows. A successful Merkle check alone does not prove this ideal model; that
transfer remains in P5.

The level-1 word used in the algebraic argument is F4-valued, one symbol
per column and split at each row. Tail words are E-valued. This alphabet
condition is necessary for descent. The verifier checks canonical bytes
only in opened leaves, so do not assume it checked hidden rows. In the ideal
full-word model, complete each raw word by replacing each coordinate byte
at least 127 by zero. Keep all other bytes. This fixed, coordinatewise map
gives words in the required fields and commutes with projecting to earlier
roots or subsets of columns. Define all lists from these completed words.
On every queried row of an accepting proof, canonical validation makes the
completion equal to the raw payload. A malformed queried row rejects.
Thus acceptance for the raw words implies the required sampled equations
for the completed words. This reduction adds no probability term. It does
not claim that the root authenticates a different word or that hidden rows
are canonical. Original root labels and transcript bytes stay unchanged.
This is an analysis word for statement soundness, not a proximity claim
about malformed raw words. Clear proof-field vectors are checked when read.

Assume these algebraic identities for every message, including adversarial
messages. Packing is an invertible F-linear map from trace columns to F4
polynomial coefficients. Its E-linear extension uses `R=E tensor_F F4`, with
four E coordinates per symbol. Encoding and opening functionals act on these
same messages. At each tail, the eight polynomial columns and the message
entries use the same order; the fold uses three weights `(1-r_j,r_j)`.
Sumcheck and clear checks evaluate the stated functionals. Proving that the
shipped maps satisfy these identities remains part of P1/P4.

At each level, the scalar RS code has n distinct domain points and dimension
`1<K<n`. Choose `eta>0` with `sqrt(K/n)+eta<1`, and set exactly
`t=floor(n*(1-sqrt(K/n)-eta))` and `T=n-t`. Thus
`T/n>=sqrt(K/n)+eta` and `T>K-1`. Merely requiring `T>K-1` would not
justify the list or query bounds below. The ledger rejects dimension one.
Distance counts joint rows across all columns and splits. A root list contains
all messages within t rows of that fixed word. Its bound is
`B=ceil(1/(2*eta*sqrt(K/n)))`. A final clear message is a singleton, not a
received-word list.

## Packed descent and initial opening

Since 4 divides 20, E contains F4 and R splits into four copies of E. These
copies use the four conjugate domains `sigma_j(D)`, with the same row labels.
They need not be the same RS code. The four-component union remains charged.

The DKT26 scalar input is **Theorem 5.12, p. 53**, in every characteristic.
**Corollary 7.2, p. 66**, gives affine MCA error `a/(Q-1)`, independent of
the number of coefficients. **Corollary 7.7, pp. 68-69**, supplies its
interleaved form. Here `Q=|E|` and a is the scalar exceptional count. Source
details are in the [DKT26 section](soundness.md#dkt26-johnson-mca-comparison).

Fix a folded message agreeing with the packed received word on a joint set A
of at least T rows. Off the four MCA events, component j supplies witness
polynomials for every input column and split. Its full agreement set can be
larger than A. Restrict the witnesses to A; equality of the four full
component sets is not needed.

Undo `sigma_j` on a witness and its domain. It agrees with F4 values on at
least K points of D. For `tau in Gal(E/F4)`, that witness and its
coefficientwise image under tau agree on those points. Their difference has
degree below K, so it is zero. Its coefficients are in F4. Witnesses from
different components have the same values on A; interpolation makes them
equal. They define one joint candidate `X_tilde`. Interpolation in each
component also gives the message identity `y=X_tilde*beta`. No unique nearby
codeword is needed. Undoing embeddings on witnesses does not require beta
to be fixed by those embeddings.

Distinct joint tuples agree at at most K-1 rows. If b candidates each use T
agreement rows, counting pairwise overlaps and applying sum of squares gives

```
b * (T^2 - n*(K-1)) <= n * (T-(K-1)).
```

The positive denominator gives the stated list bound. This also applies to
subsets of columns and multiple splits, without multiplying column-list
sizes. The executable still rejects splits; its scope is unchanged.

Before z, every column list is fixed. Two distinct trace candidates differ
by a nonzero polynomial of total degree at most `h1+h2-2`. Bind thus costs
`columns*B*(B-1)*(h1+h2-2)/(2Q)`. Off Bind, the answer at z matches at most
one candidate per column. This pins a possible choice; it does not prove
existence or joint closeness.

Openings precede beta and gamma. For a joint candidate with a wrong opening,
`sum beta_c*gamma_p*(V_c(z_p)-answer_c,p)` is a nonzero polynomial of total
degree at most two. Union over the fixed joint list costs `2B/Q`. Off this
event, a close folded message satisfying the running claim lifts to a
candidate with every opening correct. Off Bind, it is the pinned candidate.
Both allowances remain charged.

## Recursive extraction

Number the initial joint root as 0 and the tail roots as 1 through r. Let
`Lambda_i` be the root-i list and B_i its bound. A tail member is its full
message before the three-bit contraction. A member of `Lambda_(i+1)` is a
possible proposed message for the level-i fold. Use a singleton for the
final clear message.

The following order occurs in `verifier.mojo`:

| Object fixed or checked | Next challenge | Code boundary |
|---|---|---|
| W root | Stage-1 and wiring challenges | `verify` |
| Z root and Z2 | Alpha | `verify` |
| Q root and Q3 | z | `verify` |
| Opening vector | Independent beta/gamma | `verify` |
| Next tail root or final clear vector | Previous-level positions | `_Tail.level`, `_Tail.clear`, `_open_previous` |
| Authenticated rows and running claim | Independent batch coefficients | `_Tail.level` |
| Each quadratic round message | That round's challenge | `_Tail.level` |

Sumcheck bytes are read in one block, but each round is absorbed before its
own challenge. These absorbs determine the challenge order. `Shape` requires
opening point zero to be the unshifted z. `_level1_symbol` folds all four
coordinates separately, with one beta per column across splits.
`_tail_symbol` uses the three product weights. These source observations do
not prove the assumed basis identities.

Define bad events at their preceding transcript prefixes. Bound each event
conditional on that prefix, then average and take their union. Do not
condition these bounds on future acceptance or future extraction.

For a level-i query event, `Lambda_(i+1)` is already fixed. Each member whose
proposed encoding differs from the folded word on more than t_i rows passes
all s_i queries with probability at most `miss_i^s_i`, where
`miss_i=sqrt(K_i/n_i)+eta`. Union over that list costs
`B_(i+1)*miss_i^s_i`, or `miss_i^s_i` for the clear message. Sampling is with
replacement. Sorting distinct positions for a multiproof does not change
the event: all original draws must pass.

Fix a tail member Y. Its discrepancies in the queried equations and prior
running claim are fixed before the batch coefficients. If any is nonzero,
a uniform linear batch hides it with probability at most `1/Q`. The ledger
retains the larger allowance `A_batch_i/Q`. Union over the root list costs
`B_i*A_batch_i/Q`.

After batching, the honest partial-sum polynomial for this fixed Y has
degree at most two per round: both message and query weights are multilinear
in the folded bit. If the incoming claim is false, a round message whose
values at zero and one sum to that claim differs from the honest polynomial.
The difference has at most two roots. Three fresh challenges cost at most
`6/Q`; union over Y costs `6B_i/Q`. Later prover messages can depend on
earlier challenges. Each message must precede its own fresh challenge.

For a fixed root, full agreement-set MCA covers every close output at once.
DKT26 **Corollary 7.8, pp. 69-70**, charges `3a_i/Q` for the three shared
fold challenges. The root precedes all three. Intervening prover messages
do not change their independent uniform distribution in the ideal protocol.
There is no extra list factor on this MCA event.

Work backward outside all these bad events. The clear message satisfies its
running check directly. The previous query check makes it close to the
folded root. MCA lifts it to a member Y of that root's joint list. The
sumcheck and batch bounds imply that Y satisfies the earlier running claim
and queried equations. Y can be selected after those queries, but belongs
to the list fixed before them. Repeat at the previous root. At level 0,
packed MCA and the opening argument yield a joint trace candidate with all
claimed openings correct. This finite induction does not select a member
before its queries or multiply list sizes across levels.

## Relation and cryptographic boundaries

The final candidate projects into the W list fixed before stage 1, the W/Z
list fixed before alpha, and the W/Z/Q list fixed before z. Each has size at
most B_0. A bound for one candidate at its relevant prefix can be multiplied
by B_0, without a product of list sizes across prefixes.

The required relation contract has two separate parts. At each challenge
barrier j, define a bad predicate `E_j(V,chi_j)` from the preceding transcript,
one candidate projection V fixed there, and that barrier's fresh challenge.
Prove `Pr[E_j(V,chi_j) | prefix] <= A_j/Q`. This event must not depend on
future verifier challenges or future acceptance. Union over V costs at most
`B_0*A_j/Q`.

Separately prove coverage: outside the PCS events, any accepted false
statement implies some `E_j(V,chi_j)` for a projection of the extracted
candidate. When an earlier event uses existence of later witnesses, require
those witnesses to satisfy the downstream algebraic identities exactly.
Failures of those identities that pass a later random evaluation belong to
that later barrier's event, not to the earlier one. The coverage proof must
allow all later adaptive choices of Z, Z2, Q, Q3, and opening answers.
Z2 is fixed with Z before alpha; Q3 is fixed with Q before z. They are clear
polynomial vectors in the prefix, not extra list choices at those barriers.

One cannot fix a later Z or Q and pretend it preceded stage 1 or alpha. In
particular, the Horner endpoint bound needs recurrence identities to force
the endpoint determined by W. This review does not construct the required
relation events or prove coverage. Those are P2/P3 contracts. The list union
is valid conditional on them, not a substitute for them.

Under those contracts, the interactive union bound is

```
sum_i M_i * miss_i^s_i
  + (A_gap + A_bind + 2B_0 + B_0*A_rel
     + sum_(tail roots i) B_i*(A_batch_i+6)) / Q,
```

where `M_i=B_(i+1)` or one for the clear message. This probability has no
grinding discount. The executable divides the query term by `2^grind_bits`
under a separate work model. This review does not prove that reduction,
hash binding, Fiat-Shamir security, or a quantum bound; these remain in P5.

Next establish the packing, encoding, opening, and fold identities assumed
above for every admitted message and supported shape. Existing `test_open`,
`test_tensor`, and `test_tail` checks support these identities on fixtures;
they are not universal proofs. Then prove the fixed-projection relation
contracts, starting with Horner recurrence and endpoints. The separate Bend development now checks the abstract scalar, affine,
interleaved, tensor, and packed-list transfers. Its current field and
fixed-coordinate assumptions still need a connection to the compact
E-valued column openings. See [security-assurance.md](security-assurance.md).

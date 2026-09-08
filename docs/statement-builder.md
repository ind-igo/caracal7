# Statement builder (statement-layer 2, 3, 9; spec 10 open item "the compiler")

Plan for the common layer above the byte IR: the thing a frontend (Keccak, ECDSA, memory) talks
to, and the thing that produces what the prover and verifier consume today: `Shape`, the family
bytes, the accumulator descriptors, the tables, the point list. Public columns are planned in
`docs/public-columns.md` and are an input here, not a part of it.

## The idea

Today a workload is written as `Families.add` calls with numeric column indices, a `Shape`, a
column-major trace, and advice bytes. Nothing names a column, says what a byte means, groups
columns by row kind, or derives the opening points from what the checks read. The spec's artifact
(statement-layer 2) is that missing layer. The builder is a Mojo API that constructs the
artifact, runs the spec's check list, and compiles it to the bytes the prover already reads. It
is not a language and not a compiler in the spec's sense: no parser, no file format, no lowering
of high-degree expressions. A frontend that needs a helper column declares it.

## Decisions

- **The artifact is the bytes we already hash, plus a names table.** `compile` returns the
  `Shape`, the family bytes, and a `Layout` (name to column index, group to column range, the
  advice and public block order). The prefix already hashes the family table, the accumulator
  descriptors, the tables, and (public-columns) the public specs and restrictions; names and
  groups are prover-side conveniences and stay out of the hash. Serialization waits for a reader
  outside Mojo.
- **One `Statement` type, one `compile`.** Columns, reads, families, accumulators, lookups,
  public columns, and restrictions are added by name; `compile` resolves names, runs the checks,
  emits `Families` calls, and derives the point list. The existing `Families` builder stays the
  emitter: the accumulator and lookup forms already generate their descriptor plus transition
  entries from one call, which is the "one definition per relation" rule (decisions.md, design
  guidance).
- **Column kinds are builder types.** `bit`, `limb6`, `byte` (any F value), `E` (a Z block of e
  coordinate columns). The kind says what certificate the column needs (a bit gets `c (c - 1)`,
  a limb gets its range certificate once) and what an operation may assume. The prover core never
  sees kinds.
- **Neutral padding is emitted by the builder.** Statement-layer decision 1 makes the padding of
  an idle chain a function of the group's family kinds. The `Layout` carries, per group, the
  padding rule (zero; Z = 1 with zero records; table-ordered filler; memory init), and a
  `pad_trace` helper applies it, so no frontend writes its own.
- **The accumulator form is the spec's.** `{start, ingest, scale, end}` with the grand product
  (KIND_PERM, KIND_LOOKUP) as the first instances. The general form (Horner and identity
  accumulators for polynomial mulmod) is the second Z kind; the builder API takes the spec's
  record now so a frontend written against it does not change when the kind lands. Merged chains
  (`docs/milestone-3-lookup.md`, "Merged chains, deferred") are a builder transformation on the
  same record.
- **Opening points come from the checks.** `compile` collects the read shifts, the accumulator
  points, the restriction line, and drops the fixed points nothing reads. `Shape` takes the point
  list instead of deriving it; the verifier checks the list against the families the same way it
  checks `shift_points` today.

## API (as built, `relations/statement.mojo`)

```
var st = Statement()
st.col("a0", BIT, group="keccak")              # W column; kinds BIT, LIMB6, BYTE; the index is the declaration order
st.acc("z", KIND_PERM, num=[...], den=[...])   # Z block; KIND_LOOKUP takes table=st.table(rows)
st.pub("m", m=4)                               # public column, block (d2, h1); d2 = 0 means h2 / m
st.restrict("s0", FIX_E)                       # a public line on the last chain; count = 0 means h1 coefficients
st.horner("ra", [Term(1, st.read("a", k1=1))], scale=2, start=0)   # the second Z kind: R(next) = scale R + sum coef chal read; scale, chal are element indices
st.chain_end("mul", [Term(1, st.read("ra"), st.read("rb")), Term(-1, st.read("rc"))], gated=False)   # a family on H2 over accumulators at (e1, X2)
var sa = st.slot("ra")                         # a wiring slot: the accumulator's chain-end values; slots pair into products
st.wire(sa, 3, sb, 7)                          # slot sa on chain 3 equals slot sb on chain 7
st.public_factor("r", "ra", sb, 0)             # a public value fingerprinted by "ra", equal to slot sb on chain 0; its ingest columns follow the restriction lines in the public data
var r = st.read("a0", k1=3)                    # cyclic read (omega1^3 x1, x2); k2=1 is the next-chain read
st.family("chi", [Term(1, r, st.read("a1")), Term(-1, st.read("a2"))], GATE_NONE)
var el = st.derived(CHAL_MUL, 3, 1)            # a challenge derivation row; the element index for Term.chal
var c = st.compile[p]()                        # Compiled: shape, families bytes, layout (take_shape() for the Prover)
```

A `Term` is `coef * element(chal) * b_basis * b_basis2 * a * b` with `b` optional; a public read is a read of
a `pub` name. `family` emits one entry per term through `Families.add`, sharing the family index (its position
among `family` and `acc` calls); `gate` is `mult`. Kinds emit their certificates after the user's families: a
BIT column its Booleanity family, a LIMB6 column a lookup into the [64] table through a `<name>.sorted` column
the builder appends to W. Groups are labels for `pad_trace(layout, trace, group, live_rows)`, which fills idle
rows with a table row for lookup record columns and zero elsewhere. `advice(layout, trace)` derives the sort
indices, `public_block(layout, i, vals)` a block from values, `restriction_line(layout, i, vals)` a line from the
h1 chain values (`chain_values(layout, trace, i)` reads them from the trace on the prover's side). Skipped from the sketch: a `group()` call with a pad rule (the rule is a function of the kinds), `Chal`
and `Line` wrappers (element indices and FIX_* do the job).

## Workload (`workload.mojo`)

A frontend is a struct that conforms to `Workload`: `statement()`, `trace[p](layout)` (live rows filled,
idle rows padded), `public_inputs[p]()` (what the verifier is given; the prefix hashes it), and the static
`public_data[p](layout, public_inputs)` (every public block, then every restriction line, from the public
inputs alone). `prove_workload[p, H](ctx, w)` and `verify_workload[p, H](proof, w, public_inputs)` are the
one path from a workload to a proof and back: compile, trace, advice, public data, load, prove; compile,
public data, verify. `public_data` is static so it cannot touch the trace: that is where "the derivation
becomes code on both sides" (verifier.mojo) lives. `Synthetic` is the first instance; Keccak is the second.
Tests that need a wrong trace or a wrong advice call the pieces directly.

## Checks (statement-layer 2, as code in `compile`)

- Every family is degree at most 2 in committed and public columns; after its gate it stays in
  `(2 h1 - 1, 2 h2 - 2)`: a quadratic entry may carry the axis-1 gate, only a linear entry the
  axis-2 gate (already in `Families.add`).
- A cyclic read has `k1` in `[0, h1)` and never crosses a chain end (it wraps by construction on
  the chain axis; the check is that a transition, not a cyclic read, is used where the row after
  the last must not wrap).
- `k2 = 1` only in a linear family with the axis-2 gate.
- Every column's live rows are covered by a family or a boundary: a column read by nothing and
  constrained by nothing is an error, not a warning.
- Kinds: a `bit` column has its Booleanity family; a `limb6` column has its range certificate
  (six bit columns or a lookup into the `[64]` table); an `E` column is a Z block owned by one
  accumulator.
- Accumulators: record columns are W columns; lookup f and s are distinct; the table is canonical
  bytes of the record width with two distinct consecutive entries (already in `Shape`).
- Public: `m >= 1`, `d2 >= 1`, `m (d2 - 1) < h2`; a quadratic term has at most one public read
  (public-columns doc).
- Points: the derived list contains every shift any entry reads and the lines any restriction or
  accumulator boundary opens.

## IR gaps (checked 2026-09-06 against ir.mojo, residual.mojo, accumulate.mojo, proof.mojo)

| spec feature | IR today | builder needs |
|---|---|---|
| cyclic read at any k1 | `dj1 = 2 k1` on G, wraps in `_read`, `shift_points` adds the point | nothing |
| next-chain read, linear, axis-2 gate | `k2 = 1`, `mult = 2`, `add` rejects a quadratic | nothing |
| gates, F constants, basis factors, quadratic terms | entry fields | nothing |
| challenge expressions | derivation table on `Shape.chals` (op, a, b rows; `standard_chals` is the two accumulator rows); `k_derive_chals` and `derived_chals` walk it; `chal` is a u8 index into the list | nothing (done 2026-09-07) |
| opening points from the checks | `Shape.point_list` is an input, hashed; `required_points` is the set it must contain (boundary points only with accumulators); the verifier finds boundary points by lookup | nothing (done 2026-09-07) |
| public columns, restrictions | planned (`docs/public-columns.md`) | that plan first |
| accumulator `{start, ingest, scale, end}` | KIND_HORNER (done 2026-09-08): `horner`, ingest terms are the transition's own entries | nothing |
| chain-end families into Q3 | `chain_end` terms (Shape.ends) next to the (W) pairs, alpha powers by family index (done 2026-09-08) | nothing |
| wiring, public factors | `slot`, `wire`, `public_factor` (done 2026-09-09): Shape.wires, sigma, pubf; two fresh challenges; one joint boundary | chain ordering for next-chain edges, with ECDSA |
| memory | absent, `TODO(memory)` slots | later |
| names, groups, kinds, padding | no IR concept | builder only, no IR change |
| shared reads collapsed into one kappa | `ponytail:` note in ir.mojo, every term its own entry | builder pass; a perf item once Keccak's family list is measured |

Keccak needs nothing from the "later" rows: its state moves round to round as a next-chain
linear copy, its bit certificates are pointwise and cyclic families, its public inputs are the
public-columns plan.

## Tests

- `test_builder`: the synthetic families rebuilt through `Statement` compile to the same family
  bytes and descriptors as `synthetic_families` (byte equality), and the padding helper fills an
  idle chain that the prover then accepts.
- Every check above has a failing case: an unconstrained column, a quadratic with the axis-2
  gate, a bit column with no Booleanity family, a restriction degree at h1.
- `test_prover`: a statement with a dropped fixed point proves and verifies; the verifier rejects
  a point list that omits a read.

## Work order

1. (done) `Shape` takes the point list; the challenge derivation table. Two small IR changes, tests
   in test_prover and test_accumulate. Do these before the builder so it has nothing to work
   around.
2. (done 2026-09-07) `Statement`, `Layout`, `compile`, the checks, `pad_trace`, in `relations/statement.mojo`;
   test_builder rebuilds the hand-written synthetic list by name byte for byte.
3. (done 2026-09-07) `synthetic_statement` is the one path; every test and bench compiles it. The hand-written
   `Families` list lives in test_builder as the oracle.
4. (done 2026-09-07) Keccak-256 on the builder (statement-layer 8, 9; `docs/keccak.md`): the first real
   family list, 284 families and 777 entries at 28 points, one implementation swept over the message length.

## Not in this doc

Public columns (their own doc, done first). The second Z kind, merged chains, wiring, and memory:
each gets a short doc when its workload is next.

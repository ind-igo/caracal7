# Milestone 3: Herder lookup (spec 6.3)

Plan for the lookup argument. Memory (spec 6.4) is deferred; see the last section.

## Decisions

- **Lookup only.** The client profile has no memory (configuration.md §3). Every place memory will slot in carries a `TODO(memory)` comment that says what goes there and why it waits.
- **Advice index.** The frontend supplies one u32 per record row per lookup instance: the position in the table of the tuple that row holds. The frontend knows it for free at trace generation (a range check of `v` has index `v`; a Keccak chi lookup has the index built from the input bits). Recovering it later means matching tuples against the table on device, a hash map for a fact that was already in hand. The list is prover-only: never committed, never hashed, never sent. Soundness does not depend on it. A wrong index gives a sorted copy in the wrong order or with a record not in the table, and the product identity against `C_T` fails. It can only make an honest proof fail, never make a false proof pass. Cost: four bytes per row per instance on the upload, nothing in the proof.
- **Table on `Shape`.** The table bytes are part of the artifact and go into the transcript prefix hash, so prover and verifier compute the same `C_T`.

## Data shapes

- **Descriptor.** The 40-byte ACC descriptor has two free bytes. Byte 38 becomes `kind` (0 accumulator, 1 lookup; `TODO(memory)`: 2 memory). Byte 39 is the table id. For a lookup the `num` columns are the record columns `f` and the `den` columns are the sorted copy `s`, both of width `w`. Nothing else in the descriptor changes.
- **Shape.** Gets `tables: List[List[UInt8]]`, each a flat `(K, w, e)` byte block. `prefix_bytes` hashes each table after the families.
- **Challenge codes.** The stage-1 element list grows from 3 to `CHALS = 5`: beta, delta, gamma sampled, then `(1+beta)` and `(1+beta)·delta` derived on both sides (a one-thread kernel in the prover, `derived_chals` in the verifier). Entry codes 4 and 5 index them; `kappa_of` and `k_fold_alpha` do not change.

## Kernels

- **`relations/sort.mojo`, counting sort.** Three launches: histogram (atomic add per row into K bins), one-thread exclusive scan over K bins, scatter (atomic fetch-add on a per-bin cursor; copies the w columns of row i to its slot). Equal records are identical bytes, so the order inside a bin does not matter. The scatter writes `s` into the W trace columns before encode. An index outside the table is dropped by both kernels: the advice is untrusted input, and the dropped record leaves a sorted copy the verifier rejects. Atomics go through `unsafe_bitcast[Int32]` on the byte base. The one-thread scan is a `ponytail:` ceiling; K is at most a few thousand on the client grid.
- **Factor kernel, `accumulate.mojo`, kind 1 branch.** `N(x) = (1+beta)(delta + fp(f)(x))`; `D(x) = (1+beta)delta + fp(s)(x) + beta·fp(s)(row+1)` with row+1 cyclic. In row-major order (row = x2·h1 + x1) row+1 is the next row both inside a chain and across the chain end, so `d_end` already holds the cross-chain factor and the small grid is unchanged. This closes the `ponytail:` note in `smallgrid.mojo`. The last row's D wraps to row 0 rather than being set to 1: no check reads it (the transition is gated at x1 = e1, the small grid at x2 = e2, and the boundary has no D), but the `d_end` line must be the same degree < h2 polynomial the verifier evaluates from the openings at (e1, z2) and (1, omega2 z2), and a sentinel would change it.
- **`Families.lookup(...)` builder, `ir.mojo`.** Emits the (L) transition entries with the two new challenge codes and the shifted point for `s` at the next row.

## Verifier

- Computes `C_T = prod_j (1+beta)(delta + fp(t_j)) / prod_{j<K−1} ((1+beta)delta + fp(t_j) + beta·fp(t_{j+1}))`. Rejects if any factor is zero, and rejects a table without two distinct consecutive entries: with no cross term, f = (t, u, .., u) against s = (u, .., u) meets the identity for any u. Soundness otherwise: the cross terms are irreducible linear forms in (beta, delta) distinct from the `(delta + a)` factors, so by unique factorization every run value of s touches a table break and the multiset of f equals the multiset of s, which lies in the table.
- Boundary rule for kind 1: `Z2(e2)·Z(e1,e2)·N(e1,e2) = C_T`, in place of the `C = 1` check.
- `_factor_at` gets a kind 1 branch that reads `s` at the point `(1, omega2·z2)` for the chain-end factor. `TODO(memory)`: rule 6 for memory sits next to it.

## Tests

- `test_sort`: device sort against a host sort; all K bins hit; one empty bin.
- `test_accumulate`: L factors against a scalar host reference at three rows, the last row included.
- `test_prover`: synthetic lookup instance accepts; rejects when one record is not in the table; rejects when two bins of the advice index are swapped.

## Arena

The sort adds `4N` bytes per lookup for the index, `4(K+1)` for bins, `4K` for cursors, three lines in the planner. The per-stage `bytes_for(p, shape)` stays deferred; nothing forced it.

## Commits

1. This doc, decisions entry, sort kernels and test (1ee3b13). Codex review: the last-row D must wrap, not be 1 (found in parallel while writing the factor kernel); a table without two distinct entries admits a false lookup; an out-of-range advice index wrote outside the bins region. All three fixed in the next commit.
2. Descriptor kind, derived challenges, lookup builder, factor branch, Shape tables, prefix hash, verifier constant and factor branch, synthetic instance, tests.

## Status

Shipped. The prover (`prove` step 2) sorts every lookup descriptor's records into its s columns before W is committed; `load_advice` uploads the index lists. Tests: `test_sort`, the lookup case in `test_accumulate` (device factors against the host, boundary equals `C_T`), and `test_prove_and_verify_with_lookup` (accept; a record outside the table rejected; swapped advice rejected; a table of the wrong width rejected at `Shape`).

## Not checked by the prover

The dummy rule (every table tuple appears at least once among filler rows) is trace generation. The synthetic instance does it; the Keccak frontend will do it.

## Memory, deferred

Spec 6.4 needs: a stable multi-pass radix sort on `(addr, ts)` keys in place of the counting sort; descriptor kind 2 with the address, timestamp, and value columns; the adjacency families (same address: value carried, timestamp increasing; new address: initial value) as residual entries; the timestamp counter; boundary rule 6 in the verifier; an arena region for the radix passes. Each slot in the code is marked `TODO(memory)`.

# Milestone 3: Herder lookup (spec 6.3)

Plan for the lookup argument. Memory (spec 6.4) is deferred; see the last section.

## Decisions

- **Lookup only.** The client profile has no memory (configuration.md §3). Every place memory will slot in carries a `TODO(memory)` comment that says what goes there and why it waits.
- **Advice index.** The frontend supplies one u32 per record row per lookup instance: the position in the table of the tuple that row holds. The frontend knows it for free at trace generation (a range check of `v` has index `v`; a Keccak chi lookup has the index built from the input bits). Recovering it later means matching tuples against the table on device, a hash map for a fact that was already in hand. The list is prover-only: never committed, never hashed, never sent. Soundness does not depend on it. A wrong index gives a sorted copy in the wrong order or with a record not in the table, and the product identity against `C_T` fails. It can only make an honest proof fail, never make a false proof pass. Cost: four bytes per row per instance on the upload, nothing in the proof.
- **Table on `Shape`.** The table bytes are part of the artifact and go into the transcript prefix hash, so prover and verifier compute the same `C_T`.

## Data shapes

- **Descriptor.** The 40-byte ACC descriptor has two free bytes. Byte 38 becomes `kind` (0 accumulator, 1 lookup; `TODO(memory)`: 2 memory). Byte 39 is the table id. For a lookup the `num` columns are the record columns `f` and the `den` columns are the sorted copy `s`, both of width `w`. Nothing else in the descriptor changes.
- **Shape.** Gets `tables: List[List[UInt8]]`, each a flat `(K, w, e)` byte block. `prefix_bytes` hashes each table after the families.
- **Challenge codes.** Two new codes in the IR entry: 4 for `(1+beta)` and 5 for `(1+beta)·delta`. `kappa_of` and `k_fold_alpha` learn them. This is the whole change to the residual pipeline.

## Kernels

- **`relations/sort.mojo`, counting sort.** Three launches: histogram (atomic add per row into K bins), one-thread exclusive scan over K bins, scatter (atomic fetch-add on a per-bin cursor; copies the w columns of row i to its slot). Equal records are identical bytes, so the order inside a bin does not matter. The scatter writes `s` into the W trace columns before encode. Atomics go through `unsafe_bitcast[Int32]` on the byte base. The one-thread scan is a `ponytail:` ceiling; K is at most a few thousand on the client grid.
- **Factor kernel, `accumulate.mojo`, kind 1 branch.** `N(x) = (1+beta)(delta + fp(f)(x))`; `D(x) = (1+beta)delta + fp(s)(x) + beta·fp(s)(row+1)`, and `D = 1` at row `N−1`. In row-major order (row = x2·h1 + x1) row+1 is the next row both inside a chain and across the chain end, so `d_end` already holds the cross-chain factor and the small grid is unchanged. This closes the `ponytail:` note in `smallgrid.mojo`.
- **`Families.lookup(...)` builder, `ir.mojo`.** Emits the (L) transition entries with the two new challenge codes and the shifted point for `s` at the next row.

## Verifier

- Computes `C_T = prod_j (1+beta)(delta + fp(t_j)) / prod_{j<K−1} ((1+beta)delta + fp(t_j) + beta·fp(t_{j+1}))`. Rejects if any factor is zero.
- Boundary rule for kind 1: `Z2(e2)·Z(e1,e2)·N(e1,e2) = C_T`, in place of the `C = 1` check.
- `_factor_at` gets a kind 1 branch that reads `s` at the point `(1, omega2·z2)` for the chain-end factor. `TODO(memory)`: rule 6 for memory sits next to it.

## Tests

- `test_sort`: device sort against a host sort; all K bins hit; one empty bin.
- `test_accumulate`: L factors against a scalar host reference at three rows, the last row included.
- `test_prover`: synthetic lookup instance accepts; rejects when one record is not in the table; rejects when two bins of the advice index are swapped.

## Arena

The sort adds `4N` bytes for the index, `4(K+1)` for bins, `4K` for cursors. If the bump layout does not take the three regions cleanly, this is the moment for the deferred per-stage `bytes_for(p, shape)`.

## Order of commits

Each commit is followed by a background Codex review.

1. This doc, decisions entry, sort kernels and test.
2. Descriptor kind, challenge codes, lookup builder, factor branch, accumulate test.
3. Shape tables, prefix hash, verifier `C_T` and L branch, synthetic instance, prover tests.

## Not checked by the prover

The dummy rule (every table tuple appears at least once among filler rows) is trace generation. The synthetic instance does it; the Keccak frontend will do it.

## Memory, deferred

Spec 6.4 needs: a stable multi-pass radix sort on `(addr, ts)` keys in place of the counting sort; descriptor kind 2 with the address, timestamp, and value columns; the adjacency families (same address: value carried, timestamp increasing; new address: initial value) as residual entries; the timestamp counter; boundary rule 6 in the verifier; an arena region for the radix passes. Each slot in the code is marked `TODO(memory)`.

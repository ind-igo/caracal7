# Soundness work

This file covers soundness work only; the build and test entry points are in `README.md`.

Before a security analysis or a parameter change, read `docs/soundness.md`: the numbers, what the
ledger charges, and the open obligations P1-P5. Recompute the ledger
(`uv run mojo run --Werror -I src bench/bench_soundness.mojo`) after any change to the field, a
grid, a rate, a query count or a workload. Interactive bits and work bits are different models;
never present either as a certified security level. The headline is the DKT26 joint-list row; the
Haböck row is the conservative baseline and stays printed. A new bound needs a public source with theorem
number and page, or is marked "own proof, needs review".

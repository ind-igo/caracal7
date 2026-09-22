# Soundness work

This file covers soundness work only; the build and test entry points are in `README.md`
(`uv sync`, `./run_tests.sh`).

Before a security analysis or a parameter change, read `docs/soundness.md` (the ledger: what the
executable charges and the open obligations P1-P5, J1-J3) and `docs/security-assurance.md` (the
checkpoint: the covered cases and what still blocks the implemented claim).

Rules:

- The main Hab25 ledger result is the conservative baseline. The DKT26 old-list and joint-list
  numbers are alternative conditional comparisons the executable also prints; the linear MCA
  number is a separate research calculation the executable does not print. Keep all of them
  labelled as such and never present any as a certified security level.
- Recompute the full ledger (`uv run mojo run --Werror -I src bench/bench_soundness.mojo`) after
  any change to the field, a grid, a rate, a query count or a workload, and compare the compiled
  dimensions and columns with the recorded input before reusing a number.
- A new bound needs a public source with theorem number and page, or is marked
  "own proof, needs review".

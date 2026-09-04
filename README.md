# caracal7-prover

Mojo implementation of **Caracal**, a uniform prover over F127 with Ligerito as the commitment layer. Target: the ethproofs client-side suite on an M1, proof under 500 KB, prover under 400 ms, more than 100 bits.

The spec lives in the notes vault: `wiki/projects/caracal7/specs/` (`sorted-copy-prover.md` is the main page, `statement-layer.md` the frontend, `polynomial-mulmod.md` the multiplication relation). This repo does not restate it. Implementation decisions go in `docs/decisions.md`.

## Layers

| layer | name | where |
|---|---|---|
| 4 | frontend | statement to IR program |
| 3 | relations | Herder (lookup, permutation, memory), bit certificates, mulmod, copy, public columns |
| 2 | Caracal IR and its check | families, accumulators, chains, small grid, quotients |
| 1 | Ligerito | Reed–Solomon columns in Blake3 Merkle trees, opened at P points |

## Build order

1. Ligerito on synthetic columns: F127 and tower E, two-pass encoder (L = 80,640), Blake3 tree with 1,024-byte leaves, commit, open, fold, tail, verifier, transcript.
2. Caracal IR: families, fused residual pass, Q1/Q2, Z tree, chain ends, small grid, Q3.
3. Herder memory instance.
4. Keccak-128 through the frontend.

## Layout

```
src/caracal7/   package, one module per layer piece
tests/          test_*.mojo, each a TestSuite runner
docs/           decisions.md
```

## Use

```
pixi install
pixi run test
```

Mojo only. No Python scaffolding. CPU first for correctness; GPU kernels later behind fixed buffer interfaces.

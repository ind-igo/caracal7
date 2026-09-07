# Keccak-256 workload (statement-layer 8, 9; step 4 of the build order)

The second `Workload` and the first real family list: 142 bit columns, 284 families, 777 entries, 28 opening
points (the spec counts 34 with the accumulator boundary points this statement does not have). `relations/keccak.mojo`.

## The idea

A row is one bit position `z` of the whole 25-lane state in one round; a chain is one round, so `h1 = 64`.
Chains run permutation-major, round-fastest. Every column is a bit and every relation is an XOR or an AND of
bits at degree 2. The three things a round moves are the three things the IR already has: rho rotates a lane
by `r`, so the output at `z` reads the input at `z - r`, a cyclic read at `64 - r`; pi is a renaming and costs
nothing; the state moves to the next round as a next-chain linear copy with the axis-2 gate. Nothing in the
layout is Keccak-specific beyond the family list, which is why it is the first measurement of the residual at
hundreds of families.

## Decisions

- **Message length is the only knob.** `Keccak(message)` is one implementation; `b = len // 136 + 1` blocks
  take the last `24 b` chains of whatever grid `CLIENT.grid(64, chains)` derives. The bench sweeps 128 B
  (64 x 24), 1024 B (64 x 192), 2048 B (64 x 384). None needs the codeword split: 64 x 384 is 24,576 rows,
  6,144 symbols per column, inside the level-1 domain at rate 1/32.
- **Idle chains come first and hold the zero state.** The round with no constant fixes zero (theta and chi of
  zero are zero), the first chain is restricted to zero anyway, and the digest restriction wants the last chain
  live. So the builder's zero padding is exact and no `pad_trace` call is needed: `keccak_trace` writes every
  chain from the same round function with zero public words before the live ones.
- **Iota rides in the public column of the next chain.** The constant does not commute through theta, so `x`
  stays the pre-iota chi output; `RC[r - 1]` is applied in chain `r` through `e = a xor pub` on lane 0, where an
  absorb chain also carries the message block (and `RC[23]` of the previous permutation). `RC[23]` of the last
  permutation folds into the digest restriction instead. 17 public columns, one per absorb lane, dense
  `(h2, 64)` blocks (`m = 1`): the spec's factored form is a verifier-side optimization for later.
- **Public inputs are the message then the digest.** `public_data` derives the 17 blocks from the message and
  the 29 restriction lines (25 zero lines of one coefficient for the first chain's state, 4 digest lanes of 64
  coefficients on the last chain) from the digest. The verifier trusts nothing else.
- **Booleanity on all 142 columns, per the spec.** Every column is a polynomial function of the zero start and
  the public bits through the relations, so the 142 Booleanity families are redundant for soundness; they are
  kept because the builder's BIT kind emits them and the spec's count includes them. Dropping them (BYTE kind)
  removes 284 of 777 entries and is the first lever once the residual is measured.

## Column order (declaration order = trace order)

`a0..a24` state in, `e0..e16` after absorb, per `x`: `p{x}1 p{x}2 p{x}3 c{x}` the parity chain, `d0..d4`,
`t0..t24` after theta, `n0..n24` the chi helper `(1 - b[x+1]) b[x+2]`, `x0..x24` chi out. Lane `l = x + 5 y`.

## Tests

`test_keccak`: the reference digest against the empty and "abc" vectors; the trace against every compiled
entry on the host (gates and public reads included); the prover round trip at 64 x 24 with a changed digest
byte rejected.

## Open

- (done, decisions.md perf pass 1) `build_queries` at `P = 28` was the largest prover stage at 2048 B; per-point
  tables took it from 293 to 12 ms. `encode W` and `open` lead now; the residual at 777 entries is not on
  the critical path.
- The dense public block: 17 x N x 2 bytes of host interpolation per proof on both sides (`interpolate_grid`
  is O(N (h1 + h2))), and 17 x 28 dense block evaluations in the verifier. This is the verify time the bench
  prints (22 s at 2048 B) and a known deviation from statement-layer 6 ("nothing is O(N)"); the spec's
  factored form or a host FFT fixes it on the host side only.
- Shared reads collapsed into one kappa (statement-builder.md, IR gaps): `t` columns are read at 24 offsets
  by 75 entries.

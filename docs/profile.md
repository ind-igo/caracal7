# Where the time and the bytes go

Prover stages, verifier steps and proof bytes per workload, measured on one run of every benchmark
statement. The protocol the stages implement is `docs/protocol.md`; the stage names are the
prover's own profile labels (`prove(profile=True)` synchronizes after every stage), the verifier's
are its step marks, and the byte regions are the proof layout of `proof.mojo` walked region by
region. Refresh with the same driver when a stage changes; the numbers in `README.md` are the
Criterion harness's and move separately.

Machine: Apple M1 Pro 16 GB, load average about 30 from other work, after the factored
openings and the three-coset LDE, warm prover (the third prove of a prepared
session), `CLIENT` profile (`E = F_127^20`, Johnson regime, 20 grinding
bits, tail rate 1/8). The hash inputs are 2048 bytes; Poseidon is 16 elements; the RSA, SOD, DSC
and passport fixtures are the bench files' (2048-bit keys, e = 65537).

| | sha256 | keccak | poseidon | ecdsa | rsa-2048 | sod | dsc | passport |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| grid | 32 x 2688 | 64 x 384 | 64 x 896 | 144 x 576 | 144 x 2016 | 144 x 2688 | 144 x 2688 | 144 x 4032 |
| columns W/Z/Q | 43/0/60 | 142/0/60 | 78/80/60 | 236/260/60 | 172/120/60 | 285/140/60 | 252/140/60 | 285/140/60 |
| opened columns | 46 | 145 | 85 | 252 | 181 | 295 | 262 | 295 |
| opening points | 20 | 28 | 28 | 12 | 18 | 52 | 50 | 53 |
| level-1 queries | 76 | 104 | 88 | 74 | 103 | 80 | 80 | 103 |
| clear vector, E elements | 21 | 6 | 14 | 162 | 567 | 756 | 756 | 1134 |

## Prover stages, ms

The profiled run synchronizes after every stage, so the stage sum is a few percent above the
unprofiled warm prove on the small statements and within noise on the large ones.

| stage | sha256 | keccak | poseidon | ecdsa | rsa-2048 | sod | dsc | passport |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| encode W | 8 | 5 | 7 | 38 | 90 | 228 | 199 | 310 |
| merkle W | 2 | 1 | 2 | 5 | 8 | 22 | 18 | 22 |
| accumulate | 0 | 0 | 1 | 10 | 18 | 32 | 28 | 46 |
| encode Z | 0 | 0 | 8 | 42 | 62 | 120 | 119 | 160 |
| merkle Z | 0 | 0 | 1 | 5 | 7 | 21 | 13 | 16 |
| small grid | 0 | 0 | 1 | 1 | 2 | 12 | 3 | 3 |
| lde | 4 | 4 | 14 | 39 | 100 | 263 | 201 | 388 |
| residual | 5 | 4 | 11 | 43 | 99 | 325 | 220 | 486 |
| quotient | 6 | 2 | 5 | 15 | 52 | 72 | 71 | 108 |
| encode Q | 7 | 2 | 4 | 9 | 25 | 41 | 42 | 55 |
| merkle Q | 3 | 1 | 2 | 3 | 5 | 10 | 9 | 10 |
| build_queries | 1 | 1 | 2 | 1 | 7 | 1 | 1 | 3 |
| open | 5 | 4 | 8 | 15 | 38 | 133 | 125 | 195 |
| fold | 1 | 0 | 1 | 4 | 8 | 14 | 13 | 22 |
| tail levels, transcript, finish | 29 | 19 | 28 | 30 | 77 | 118 | 98 | 141 |
| sum of stages | 71 | 43 | 95 | 260 | 598 | 1412 | 1160 | 1965 |
| warm prove, unprofiled | 81 | 58 | 106 | 261 | 593 | 1378 | 1160 | 1957 |

Reading it: on the hash statements the encoder and the tail are the budget and the whole prove is
under 110 ms. On the wide statements (RSA and up) the residual pass and the LDE that feeds it are
the largest pair, about 45 percent of the passport, then `encode W` and `open`. The openings are
factored by the classes of points that share `z2` (`open.point_classes`: the SOD's 52 points fall
into 14 classes, so the contraction runs over 28 weight rows instead of 52 points) where that wins,
and are the direct GEMM on RSA and Poseidon, whose points fall into almost as many classes. The tail
is 5 to 15 percent everywhere; the Merkle trees are under 3 percent.

## Verifier steps, ms

| step | sha256 | keccak | poseidon | ecdsa | rsa-2048 | sod | dsc | passport |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| transcript and openings | 0 | 0 | 0 | 0 | 0 | 3 | 13 | 4 |
| boundaries | 0 | 0 | 0 | 70 | 160 | 15 | 199 | 240 |
| small grid | 0 | 0 | 1 | 14 | 18 | 108 | 37 | 68 |
| residual at z | 15 | 9 | 10 | 20 | 3 | 12 | 7 | 28 |
| restrictions | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| running claim | 4 | 6 | 6 | 2 | 8 | 22 | 17 | 57 |
| tail levels | 11 | 14 | 12 | 19 | 28 | 38 | 24 | 85 |
| clear vector | 4 | 2 | 3 | 16 | 69 | 290 | 156 | 284 |
| total | 39 | 44 | 43 | 160 | 325 | 578 | 506 | 862 |

Reading it: the hash statements verify in the residual and the tail. The RSA-based statements
verify in two places that are not the commitment at all: the boundaries (the wiring product and
the public factors' fingerprints, one Horner scan per factor over the public data) and the clear
vector (units times clear length; the clear vector is the odd part `m1 m2` of the grid times the
unfolded digits, 1134 elements on 4032). The SOD's 108 ms small grid is its many chain-end terms.
These are the open verify-time items of `docs/passport.md`.

## Proof bytes

Measured after the single-value openings of `E`-valued columns (`docs/zk.md` 3.5): an accumulator
or a quotient piece is one opened column, not twenty. The multiproof bytes vary by a few hundred
between runs with the sampled positions.

| region | sha256 | keccak | poseidon | ecdsa | rsa-2048 | sod | dsc | passport |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| header and public inputs | 2,088 | 2,088 | 73 | 168 | 778 | 374 | 526 | 826 |
| W root | 32 | 32 | 32 | 32 | 32 | 32 | 32 | 32 |
| Z root and Z2 lines | 0 | 0 | 32 | 69,152 | 80,672 | 161,312 | 161,312 | 241,952 |
| Q root and Q3 | 32 | 32 | 35,872 | 23,072 | 80,672 | 107,552 | 107,552 | 161,312 |
| openings (points x opened columns x 20 B) | 18,400 | 81,200 | 47,600 | 60,480 | 65,160 | 306,800 | 262,000 | 312,700 |
| level-1 rows and multiproofs | 81,856 | 130,352 | 153,012 | 235,956 | 251,348 | 249,428 | 238,772 | 314,304 |
| tail levels: roots, rounds, rows, multiproofs | 92,000 | 72,224 | 84,384 | 77,760 | 90,400 | 89,536 | 89,184 | 97,760 |
| clear vector | 420 | 120 | 280 | 3,240 | 11,340 | 15,120 | 15,120 | 22,680 |
| total | 194,828 | 286,048 | 321,285 | 469,860 | 580,402 | 930,154 | 874,498 | 1,151,566 |

Before the change the openings were 41,200 / 113,120 / 122,080 / 133,440 / 126,720 / 504,400 /
452,000 / 514,100 bytes in the same order, and the totals 216,956 / 317,712 / 396,213 / 544,324 /
642,314 / 1,123,434 / 1,063,602 / 1,355,046.

Reading it: the level-1 rows are the largest region on every statement but the SOD, DSC and
passport, as the layout rule of spec 9.5 says. On those three the openings still lead: every opened
column at every point, and they declare 50 or more points (the cyclic-read offsets of the limb
lanes), so 295 opened columns times 53 points is 313 KB of the passport's 1.15 MB, with the `Z2` lines
another 242 KB (three product lines times 4032 rows times 20 bytes) and `Q3` 161 KB. The tail levels
are about 90 KB on every statement: a level's rows are 160 bytes, so the tail's size is the query
count times the levels, independent of the statement. The levers on the large proofs are therefore
the point count and the witness column count, then `h2` through the clear lines, and the level-1
rate only after those.

## How to refresh

`uv run mojo run --Werror -I src -I bench bench/bench_breakdown.mojo` builds each workload from the
bench fixtures, proves three times, prints the stage profile of a fourth prove, verifies with the
step profile, and walks the proof bytes with `ProofReader` in the verifier's order. It takes about
a minute on the M1 Pro.

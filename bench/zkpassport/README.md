# zkPassport baseline on the same machine

The comparison for `docs/passport.md`: zkPassport's Noir circuits proved with Barretenberg on the same M1
Pro as `bench_sod` and `bench_dsc`. Their phone numbers are not comparable to Mac numbers; this recipe runs
both stacks on one machine.

## Recipe

Versions from their CI: Noir 1.0.0-beta.22, bb 5.0.0, Node 24 or later. Install the toolchain beside the
system one (bbup appends a PATH line to `~/.zshrc`; remove it):

```
git clone --depth 1 https://github.com/zkpassport/circuits.git zkp
NARGO_HOME=$PWD/tc/nargo noirup -v 1.0.0-beta.22
BB_PATH=$PWD/tc/bb bbup -v 5.0.0
cd zkp && npm ci
for p in sig_check_dsc_tbs_700_rsa_pkcs_2048_sha256 sig_check_id_data_tbs_700_rsa_pkcs_2048_sha256 \
         data_check_integrity_sa_sha256_dg_sha256 disclose_bytes compare_age compare_expiry outer_count_6; do
  nargo compile --package $p
done
git apply ../circuits-timing.patch          # times `bb prove`, runs `bb verify`, prints proof bytes
cp ../bench.test.ts src/ts/tests/
TZ=UTC npx jest src/ts/tests/bench.test.ts --runInBand --verbose 2>&1 | grep BENCH
```

`bench.test.ts` is their `outer.test.ts` with a CSCA of RSA-2048 SHA-256 (theirs: 4096 SHA-512), the
`compare_expiry` circuit in place of the nationality inclusion check, and a manifest of the six compiled
circuits (the fixture manifest holds other builds). The passport is their generated fixture: DSC RSA-2048
SHA-256, SOD SHA-256, a TD3 MRZ.

## Statement

Six subproofs and one outer proof. The outer proof verifies the six recursively (a Honk verifier for each
inside a 4.6 M-gate circuit) and is what a verifier receives.

| circuit | what it proves | gates (bb) |
| --- | --- | ---: |
| sig_check_dsc | SHA-256 of the DSC certificate body, RSA-2048 verify by the CSCA key, Merkle path of the CSCA in the registry, commitment to the DSC key | 113,987 |
| sig_check_id_data | SHA-256 of the signed attributes, RSA-2048 verify by the DSC key, commitment chain | 93,211 |
| data_check_integrity | SHA-256 of DG1 and of the security object against the signed attributes, commitment chain | 79,338 |
| disclose_bytes | the disclosed MRZ bytes under a mask, nullifier | 73,722 |
| compare_expiry | expiry date after a date | 82,198 |
| compare_age | age at least 18 | 80,069 |
| outer_count_6 | recursive verification of the six | 4,557,976 |

The caracal7 statement covers the first four and the two comparisons: `sod` (SHA of DG1, of the security
object, of the signed attributes, RSA verify, DSC key commitment, disclosed window, nullifier) and `dsc`
(SHA of the body, RSA verify by the CSCA, the committed key inside the body), with the registry, the
predicates and the padding check on the verifier's side (`workloads/csca.mojo`, `workloads/mrz.mojo`).
Their `disclose_bytes` masks bytes inside the proof; ours discloses a fixed 32-byte window. Their
integrity check also hashes DG2 (the photo); ours does not.

## Results (Apple M1 Pro, 16 GB, load average 12: the machine was in use)

Two runs, `bb prove` wall time per circuit, ZK on (`noir-recursive` target), `bb verify` after each.

| proof | prove | verify | proof size |
| --- | ---: | ---: | ---: |
| sig_check_dsc | 1.14 s, 1.22 s | 0.07 s | 14.7 KB |
| sig_check_id_data | 0.96 s, 0.92 s | 0.06 s | 14.7 KB |
| data_check_integrity | 0.92 s, 0.86 s | 0.06 s | 14.7 KB |
| disclose_bytes | 0.88 s, 0.73 s | 0.06 s | 14.7 KB |
| compare_expiry | 0.88 s, 0.81 s | 0.07 s | 14.7 KB |
| compare_age | 0.86 s, 0.86 s | 0.07 s | 14.7 KB |
| six subproofs, sum | 5.6 s, 5.4 s | 0.4 s | 88 KB |
| outer_count_6 | 39.7 s, 39.8 s | 0.09 s | 14.7 KB |
| **passport, all** | **45 s** | | |

`bb write_vk` per circuit: 0.4 to 0.6 s (4.9 s cold for the first). Witness generation in Node is not
counted on either side.

caracal7 on the same machine in the same hour (`bench_sod`, `bench_dsc`, 144 x 2688, warm):

| proof | prove | verify | proof size |
| --- | ---: | ---: | ---: |
| sod | 2.0 s | 0.7 s | 1.11 MB |
| dsc | 1.6 s | 0.8 s | 1.04 MB |
| **passport, both** | **3.6 s** | **1.5 s** | **2.15 MB** |

On the idle machine earlier the same day: sod 1.67 s, dsc 1.38 s, a passport 3.0 s prove, 1.3 s verify.

## Reading

- Prove: caracal7's passport is 12x faster than zkPassport's outer proof and about 1.5x faster than the sum of
  their six subproofs (which no verifier accepts without the outer, or six verifications and a linking
  argument).
- Verify: theirs is 0.09 s for the outer proof against our 1.5 s; proof size 14.7 KB against 2.15 MB.
  Ligero-style proofs are large and their verifier reads the opened columns.
- Zero knowledge: their proofs hide the witness; ours do not (the opened rows leak witness bytes; a masking
  layer is future work).
- Their subproofs are Honk proofs of 74 K to 114 K gates each, about 0.9 s each: the fixed cost of a Honk
  proof on this machine is most of it. The outer proof's 4.6 M gates are the six recursive verifiers.

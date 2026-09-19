# Passport demo: state and next steps

Plan and background: notes vault `wiki/projects/caracal7/passport-demo.md`. This file is the pick-up point.

## State (2026-09-19)

- `workloads/rsa.mojo`: RSA-2048 verify (e = 65537, 17 modmuls) in one proof, squaring symmetry done
  (decisions.md "RSA squaring symmetry"): 1888 chains. The signature s and the modulus n can be witness
  (`s_wired`: the occurrences of each limb of s are wired to each other; `n_wired`: the limbs of n come from
  wires the caller adds, `n_chains`).
- `workloads/sod.mojo`: SHA(DG1) -> SHA(LDS security object) -> SHA(signed attributes) -> RSA verify with s and
  n witness, plus the commitment SHA(n || r) to the DSC key (r 32 random bytes), one proof, 2068 chains on
  144 x 2688 (decisions.md "DSC certificate check"). The verifier gets the lengths, the offsets, the PKCS#1
  padding limbs and the commitment digest.
- `workloads/dsc.mojo`: the DSC certificate check: SHA(certificate body) -> RSA verify against the public CSCA
  key with s witness; the body's 32-byte windows at the key's offset are wired to the commitment's windows, so
  the DSC key inside the body is the committed one; the commitment digest is public. 2146 chains on 144 x 2688.
- A passport is the two proofs with one commitment digest in both public inputs. The DSC key, both signatures,
  the body and r never reach the verifier. The prover is not zero-knowledge (the spec leaves it out of scope):
  witness means "not given", the proof's opened rows still leak witness bytes until a masking layer exists.
- Public columns in term form (docs/public-columns.md): the verifier's public data is under 1 MB per proof.
- Numbers (M1 Pro, warm, `bench/bench_sod.mojo`, `bench/bench_dsc.mojo`): SOD 1.54 s prove, 1.08 MB, 0.53 s
  verify; DSC 1.38 s prove, 1.04 MB, 0.72 s verify. A passport: 2.9 s prove, 2.1 MB, 1.25 s verify. Grids of
  144 x 2688 (the 2016 grid holds neither proof: the commitment group adds 81 chains).

## Why two proofs

One proof would hold the SOD, the body hash and the second RSA verify: about 4200 chains, above the 4032 grid,
and that grid allows three wiring slots plus the factors (F2* has 16128 endpoints) where the RSA lane uses four
and the fingerprints one. Two proofs on 2688 need no slot merge; the link between them is the commitment,
which the DSC key needs anyway: with n public, the DSC check would be a proof about public data that the
verifier checks itself in microseconds.

## Next, in order

1. Done: public columns in term form. Verify time halved; the verifier's public data is under 1 MB.
2. Done: DSC certificate check as a second proof, the DSC key hidden under a commitment.
3. Predicates and nullifier: age, expiry, nationality from the MRZ bytes on the DG1 group (the window
   mechanism of `sha256g` gives any 32-byte window of a message as a fingerprint); nullifier as one hash.
4. CSCA registry: the CSCA key is a public input today; a Merkle path or a public list.
5. zkPassport baseline on the same M1: build the Noir circuits, run Barretenberg, sum the subproofs.
   Do not quote their phone numbers against Mac numbers.

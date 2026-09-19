# Passport demo: state and next steps

Plan and background: notes vault `wiki/projects/caracal7/passport-demo.md`. This file is the pick-up point.

## State (2026-09-19)

- `workloads/rsa.mojo`: RSA-2048 verify (e = 65537, 17 modmuls) in one proof, squaring symmetry done
  (decisions.md "RSA squaring symmetry"): 1888 chains. The signature s and the modulus n can be witness
  (`s_wired`: the occurrences of each limb of s are wired to each other; `n_wired`: the limbs of n come from
  wires the caller adds, `n_chains`).
- `workloads/sod.mojo`: SHA(DG1) -> SHA(LDS security object) -> SHA(signed attributes) -> RSA verify with s and
  n witness, plus the commitment SHA(n || r) to the DSC key (r 32 random bytes), the nullifier
  SHA(SHA(LDS security object) || scope) and a disclosed 32-byte window of DG1, one proof, 2101 chains on
  144 x 2688 (decisions.md "DSC certificate check", "Predicates and nullifier"). The verifier gets the lengths,
  the offsets, the PKCS#1 padding limbs, the commitment digest, the window, the scope and the nullifier.
- `workloads/mrz.mojo`: the verifier's predicates on the window: nationality, birth date, sex and expiry of a
  TD3 MRZ, age and validity on a date. The default window starts at the nationality (DG1 byte 59): the
  document number stays hidden, the optional data after the expiry is disclosed. The predicates run on the
  verifier's side on disclosed bytes: a hidden-date comparison inside the proof waits for the masking layer,
  since the opened rows leak the witness anyway.
- `workloads/dsc.mojo`: the DSC certificate check: SHA(certificate body) -> RSA verify against the public CSCA
  key with s witness; the body's 32-byte windows at the key's offset are wired to the commitment's windows, so
  the DSC key inside the body is the committed one; the commitment digest is public. 2146 chains on 144 x 2688.
- `workloads/csca.mojo`: the verifier's check of the public inputs, beside `verify` on each proof.
  `passport_check(sod_inputs, dsc_inputs, registry)`: the DSC proof's CSCA key and exponent are in the
  registry (a text file of key ids, the SHA-256 of the big-endian modulus, and exponents), the padding limbs
  of both m are the PKCS#1 v1.5 SHA-256 encoding (the prover supplies them; unpinned, s^e = m holds for any
  upper limbs), and both proofs carry one commitment digest. No chains: a wrong value breaks the trust in the
  proof, not the proof.
- A passport is the two proofs with one commitment digest in both public inputs. The DSC key, both signatures,
  the body and r never reach the verifier. The prover is not zero-knowledge (the spec leaves it out of scope):
  witness means "not given", the proof's opened rows still leak witness bytes until a masking layer exists.
- Public columns in term form (docs/public-columns.md): the verifier's public data is under 1 MB per proof.
- Numbers (M1 Pro, warm, `bench/bench_sod.mojo`, `bench/bench_dsc.mojo`): SOD 1.67 s prove, 1.11 MB, 0.56 s
  verify (measured on a loaded machine; the statement before the nullifier and the window ran at 1.60 s in the
  same run); DSC 1.38 s prove, 1.04 MB, 0.72 s verify. A passport: 3.0 s prove, 2.15 MB, 1.3 s verify. Grids
  of 144 x 2688 (the 2016 grid holds neither proof: the commitment group adds 81 chains).

## Why two proofs

One proof would hold the SOD, the body hash and the second RSA verify: about 4200 chains, above the 4032 grid
(the largest with accumulators; deferred item 1 below). Two proofs on 2688 fit; the link between them is the commitment,
which the DSC key needs anyway: with n public, the DSC check would be a proof about public data that the
verifier checks itself in microseconds.

## Next, in order

1. Done: public columns in term form. Verify time halved; the verifier's public data is under 1 MB.
2. Done: DSC certificate check as a second proof, the DSC key hidden under a commitment.
3. Done: predicates and nullifier. The DG1 window is a public factor of its fingerprint; the nullifier is
   one SHA-256 group (33 chains) whose first window is the LDS digest and whose second is the public scope.
4. Done: CSCA registry as a public list on the verifier's side, with the padding and commitment checks of the
   public inputs in `passport_check`. A Merkle path inside the proof would hide the signing CSCA, but the
   disclosed window names the country; it waits for the masking layer.
5. Done: zkPassport baseline on the same M1 (`bench/zkpassport/README.md`): six Honk subproofs of 0.7 to
   1.2 s each (5.5 s in sum) and the recursive outer proof at 40 s; 14.7 KB per proof, 0.06 to 0.09 s to
   verify, zero-knowledge. caracal7 in the same hour on the same loaded machine: 3.6 s for both proofs
   (3.0 s idle), 2.15 MB, 1.5 s verify, not zero-knowledge.
6. Next candidates: the masking layer (zero knowledge; the baseline's one clear advantage besides proof
   size), or the deferred optimizations below (one proof instead of two first).

## Deferred optimizations

None of these cut security. In order of payoff:

1. One proof instead of two. About 4200 chains at real sizes (the bench fixture's 650-byte certificate gives
   4085), 5 percent over 4032, the largest grid with accumulators (the small grid needs a coset of G2, of
   order 2 h2, inside F2*). Three prover limits meet here, none of them passport-specific. The wiring ids
   lived in F2*, where 4032 allowed three slots plus the factors and the RSA lane uses four; since
   decisions.md "Wiring ids in F4" they are elements of F4* and the slot count is free. The axis-2 plans run the odd part 63 as one radix stage and the quotient's coset
   inverses as dense GEMMs at K = 2 h2: the 8064 grid costs 6x per row (decisions.md "Measured: SHA-256 cost
   per compression block"), 4032 is unmeasured. The RSA lane is 89 percent of the chains; a public modulus
   does not cut its bit products, the limb split (item 4) is the lever. Order: measure a 144 x 4032 grid
   against 2688 per row; then the fold, only if the per-row cost stays near 1 and the lane
   shrinks by 5 percent. A union of both statements as extra columns on one 2688 grid gains nothing: the
   opened columns double with the columns.
2. Verify time. The fingerprint sums run one row at a time; a batched Horner over all groups is a small
   change worth maybe 20 to 30 percent of the 0.56 s. Cheap, revisit second.
3. The commitment group SHA(n || r), 81 chains in both proofs. A commitment to the digest of n would be
   smaller but the DSC proof needs n in limbs, so it needs a hash-to-limbs wire. Not obvious; leave.
4. RSA limb split. 8 limbs and 17 modmuls is the tuned point; a 16-limb split with a Karatsuba row may
   cut 10 to 15 percent of the RSA chains. Untested.
5. Grid slack. The 2688 grid wastes about 590 chains in the SOD proof; a 2304 grid (h2 = 2^8 * 9) needs the
   slot rule to take seven cosets. Look after item 1, which changes the grid anyway.
6. Proof size. 1.1 MB per proof is the Ligero opening; it scales with the square root of the trace. A
   smaller proof needs another commitment, out of scope for F127.

# Passport statements

A passport check is the SOD (the document security object signed by the document signer certificate,
the DSC) and the DSC itself signed by the country's CSCA. Both are SHA-256 hashes and RSA-2048
verifies, so the statements are row groups of `sha256g` (`docs/sha256.md`) and the RSA verify
(`docs/rsa.md`) on the 144-row chain. The prover is not zero-knowledge: witness means "not given to the
verifier", and the proof's opened rows still leak witness bytes until the masking layer of `docs/zk.md`
exists.

## The workloads

- `workloads/sod.mojo`: SHA(DG1) -> SHA(LDS security object) -> SHA(signed attributes) -> RSA verify
  with the signature `s` and the modulus `n` witness, plus the commitment SHA(n || r) to the DSC key
  (`r` 32 random bytes), the nullifier SHA(SHA(LDS security object) || scope) and a disclosed 32-byte
  window of DG1, one proof of 1958 chains on 144 x 2688 (the bench grid; it fits 144 x 2016 too, where it
  proves about 25 percent faster at the same proof size). The verifier gets the lengths, the offsets, the PKCS#1 padding
  limbs, the commitment digest, the window, the scope and the nullifier.
- `workloads/mrz.mojo`: the verifier's predicates on the window: nationality, birth date, sex and expiry
  of a TD3 MRZ, age and validity on a date. The default window starts at the nationality (DG1 byte 59):
  the document number stays hidden, the optional data after the expiry is disclosed. The predicates run
  on the verifier's side on disclosed bytes; a hidden-date comparison inside the proof waits for the
  masking layer, since the opened rows leak the witness anyway.
- `workloads/dsc.mojo`: SHA(certificate body) -> RSA verify against the public CSCA key with `s`
  witness; the body's 32-byte windows at the key's offset are wired to the commitment's windows, so the
  DSC key inside the body is the committed one, and the commitment digest is public. 2003 chains with the
  bench fixture's 650-byte body on 144 x 2688.
- `workloads/passport.mojo`: the whole passport in one proof on 144 x 4032: the SOD's three groups, the
  nullifier group, the certificate body group and two RSA verifies on one column set (`rsa_columns` once,
  `rsa_instance` per chain base). The body's 32-byte windows are wired to the SOD verify's `q n` products,
  so the DSC key is a wire and there is no commitment group and no `r`. 3799 chains with the bench
  fixture's body, about 3900 at real sizes.
- `workloads/csca.mojo`: the verifier's check of the public inputs, beside `verify` on each proof.
  `passport_check(sod_inputs, dsc_inputs, registry)`: the DSC proof's CSCA key and exponent are in the
  registry (a text file of key ids, the SHA-256 of the big-endian modulus, and exponents), the padding
  limbs of both `m` are the PKCS#1 v1.5 SHA-256 encoding (the prover supplies them; unpinned, `s^e = m`
  holds for any upper limbs), and both proofs carry one commitment digest. No chains: a wrong value
  breaks the trust in the proof, not the proof.

A passport is one proof (`passport_check_one` on its public inputs), or the two proofs with one
commitment digest in both public inputs (`passport_check`). The DSC key, both signatures, the body and
`r` never reach the verifier. The public columns are in term form (`docs/public-columns.md`), so the
verifier's public data is under 1 MB per proof. The numbers are in `README.md`.

## One proof or two

One proof holds the SOD, the body hash and the second RSA verify on 144 x 4032, the largest grid with
accumulators (the small grid needs a coset of `G2`, of order `2 h2`, inside `F2*`). Against the two
proofs on 2688 it proves 20 percent faster, is 38 percent smaller and verifies in the same time (the
boundaries and the clear vector grow with the grid, the second proof's fixed costs are gone); it drops the
two commitment groups and the random bytes, since the DSC key is a wire from the body to the SOD verify.
Two things made it fit: wiring ids in `F4*` instead of `F2*`, which frees the slot count, and the RSA
rectangles without their tail chains (`docs/rsa.md`), which took the RSA lane from 1888 to 1745 chains
per verify. A union of both statements as extra columns on one 2688 grid gains nothing: the opened
columns double with the columns.

The two-proof form stays for a prover limited to 2688 chains; its link is the commitment, which the DSC
key needs there: with `n` public, the DSC check would be a proof about public data that the verifier
checks itself in microseconds.

## Baseline

zkPassport's Noir circuits for the same passport, proved with Barretenberg on the same M1
(`bench/zkpassport/README.md`): six Honk subproofs of 0.7 to 1.2 s each and a recursive outer proof of
40 s; 14.7 KB per proof, 0.06 to 0.09 s to verify, zero-knowledge. Zero knowledge is the baseline's one
clear advantage besides proof size; it is the next step here (`docs/zk.md`).

## Open

None of these cut security. In order of payoff:

1. Verify time. The SOD verify is the clear vector (about 300 units against 756 clear entries) and the
   small grid; the DSC verify's boundaries (the body's fingerprint sums, one row at a time; a batched
   Horner over all groups is a small change). The clear check could materialize the unit sum once per
   r-free index instead of per unit.
2. The commitment group SHA(n || r), 81 chains in both proofs of the two-proof form. A commitment to the
   digest of `n` would be smaller, but the DSC proof needs `n` in limbs, so it needs a hash-to-limbs wire.
3. One level of Karatsuba on the `q n` block, 272 chains per verify (`docs/rsa.md`).
4. Proof size (`docs/profile.md`). On the passport the three large regions are the openings (every column
   at every point), the Z2 and Q3 lines and the level-1 opened rows. The levers are the point count (the
   limb lanes' cyclic-read offsets), the column count and `h2`, then the level-1 rate; not the commitment.

# Passport demo: state and next steps

Plan and background: notes vault `wiki/projects/caracal7/passport-demo.md`. This file is the pick-up point.

## State (2026-09-19)

- `workloads/rsa.mojo`: RSA-2048 verify (e = 65537, 17 modmuls) in one proof, squaring symmetry done
  (decisions.md "RSA squaring symmetry"): 1888 chains on 144 x 2016.
- `workloads/sod.mojo`: SHA(DG1) -> SHA(LDS security object) -> SHA(signed attributes) -> RSA verify, one
  proof, 1987 chains on 144 x 2016 (decisions.md "Passport SOD in one proof").
- Public columns in term form (docs/public-columns.md, "term form"): the verifier's public data is 823 KB for
  the SOD (was 48.6 MB), derived in 17 ms.
- Numbers (M1 Pro, warm): RSA 0.59 s prove, 627 KB, 0.59 s verify; SOD 1.7 s prove, 1.03 MB, 1.14 s verify.
  The verify is now the tail levels (0.6 s of the SOD's), then the running claim, the boundaries and the clear
  vector. A passport with the certificate check is about 2.5 s.

## Next, in order (about four to five sessions to a solid zkPassport benchmark)

1. Done: public columns in term form (the product form). Verify time halved; the verifier's public data is
   under 1 MB.
2. DSC certificate check. A second RSA verify is about 3900 chains. The next legal grid, 4032, holds four
   F2* cosets (16128 endpoints / 4032), and one RSA verify takes four slots plus one coset for factors.
   Decide: second proof, or reduce the wiring slots to three.
3. Predicates and nullifier: age, expiry, nationality from the MRZ bytes on the DG1 group; nullifier as
   one hash.
4. zkPassport baseline on the same M1: build the Noir circuits, run Barretenberg, sum the subproofs.
   Do not quote their phone numbers against Mac numbers.

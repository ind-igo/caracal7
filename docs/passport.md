# Passport demo: state and next steps

Plan and background: notes vault `wiki/projects/caracal7/passport-demo.md`. This file is the pick-up point.

## State (2026-09-18)

- `workloads/rsa.mojo`: RSA-2048 verify (e = 65537, 17 modmuls) in one proof, squaring symmetry done
  (decisions.md "RSA squaring symmetry"): 1888 chains on 144 x 2016.
- `workloads/sod.mojo`: SHA(DG1) -> SHA(LDS security object) -> SHA(signed attributes) -> RSA verify, one
  proof, 1987 chains on 144 x 2016 (decisions.md "Passport SOD in one proof").
- Numbers (M1 Pro, warm): RSA 0.59 s prove, 627 KB, 1.1 s verify; SOD 1.7 s prove, 1.03 MB, 2.1 s verify.
  About half of each verify is public-data derivation. A passport with the certificate check is about 3.5 s.

## Next, in order (about four to five sessions to a solid zkPassport benchmark)

1. Product-form public columns. Both verifiers derive full public columns (SOD: 176 columns, 59 MB) and
   spend about half their time on it. Verifier only, no statement change. Target: verify time halves,
   verifier memory in kilobytes. Step 5 (portable verifier) needs it.
2. DSC certificate check. A second RSA verify is about 3900 chains. The next legal grid, 4032, holds four
   F2* cosets (16128 endpoints / 4032), and one RSA verify takes four slots plus one coset for factors.
   Decide: second proof, or reduce the wiring slots to three.
3. Predicates and nullifier: age, expiry, nationality from the MRZ bytes on the DG1 group; nullifier as
   one hash.
4. zkPassport baseline on the same M1: build the Noir circuits, run Barretenberg, sum the subproofs.
   Do not quote their phone numbers against Mac numbers.

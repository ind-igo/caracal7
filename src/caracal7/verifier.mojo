"""Verifier: a host program, separate from the prover (design section 6). The seven steps of
statement-layer section 6 in transcript order; milestone 1 runs steps 1, 2, 5, 7 on the Ligerito
checks. Steps 3, 4, 6 (small grid, public columns, boundaries) arrive with the Z tree and the frontend.

Every step is a host function over proof bytes and a host `H: Hash`; none of them exist yet."""

from caracal7.params import Params
from caracal7.hash import Hash
from caracal7.proof import Shape, ProofReader, VERSION, prefix_bytes
from caracal7.transcript import HostTranscript, DS_PREFIX, DS_TREE_W, DS_TREE_Q, DS_OPENINGS


def verify[p: Params, H: Hash](var proof_bytes: List[UInt8], shape: Shape, public_inputs: List[UInt8], families: List[UInt8]) raises -> Bool:
    var r = ProofReader(proof_bytes^)
    var t = HostTranscript[p, H]()

    # step 1: header, prefix, W root, stage-1 challenges
    if r.u32() != Int(VERSION):
        raise Error("bad version")
    var pub = r.prefixed()
    if pub != public_inputs:
        raise Error("public inputs differ")
    var prefix = prefix_bytes[p](shape, public_inputs, families)
    t.absorb(DS_PREFIX, prefix)
    var root_w = r.take(H.DIGEST)
    t.absorb(DS_TREE_W, root_w)
    var stage1 = t.elements(4)                      # beta_1, delta, gamma, alpha

    # step 2: Q root -> z
    var root_q = r.take(H.DIGEST)
    t.absorb(DS_TREE_Q, root_q)
    var z = t.elements(2)

    # openings -> beta, gamma
    var openings = r.take(shape.points * shape.columns() * p.e)
    t.absorb(DS_OPENINGS, openings)
    var beta_gamma = t.elements(shape.columns() + shape.points)

    raise Error("not implemented: verifier step 5 (residual identity at z)")
    # step 5: residual identity at z from the openings (residual.residual_at): R(z) = (A + z2^h2 B)(z1^h1 - 1) + Q2 (z2^h2 - 1)
    # step 7: per level: root, S on the previous level, multiproofs against the roots (check_multiproof),
    #         expected symbols, three sumcheck checks s_i(0) + s_i(1) = s_{i-1}(r_{i-1}); clear vector:
    #         last consistency rows and <y_ell, w~_ell> directly
    # r.done(); return True

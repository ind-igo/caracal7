"""Verifier: a host program, separate from the prover (design section 6). The seven steps of
statement-layer section 6 in transcript order; milestone 1 runs steps 1, 2, 5, 7 on the Ligerito
checks. Steps 3, 4, 6 (small grid, public columns, boundaries) arrive with the Z tree and the frontend.

Every step is a host function over proof bytes and a host `H: Hash`; none of them exist yet."""

from caracal7.params import Params
from caracal7.hash import Hash
from caracal7.proof import Shape, ProofReader, VERSION, prefix_bytes
from caracal7.transcript import HostTranscript, DS_PREFIX, DS_TREE_W, DS_TREE_Q, DS_OPENINGS
from caracal7.field import F2, E, f_add, f_sub, ext_mul, ext_pow, ext_embed
from caracal7.tables import Domains
from caracal7.residual import ENTRY, NONE, entry, shift_points, point_index, residual_at


def verify[p: Params, H: Hash](var proof_bytes: List[UInt8], shape: Shape, public_inputs: List[UInt8], mut families: List[UInt8]) raises -> Bool:
    if len(families) != shape.entries * ENTRY:
        raise Error("family table does not match shape.entries")
    var r = ProofReader(proof_bytes^)
    var t = HostTranscript[p, H]()

    # step 1: header, prefix, W root, stage-1 challenges
    if r.u32() != Int(VERSION):
        raise Error("bad version")
    var pub = r.prefixed()
    if pub != public_inputs:
        raise Error("public inputs differ")
    var prefix = prefix_bytes[p, H](shape, public_inputs, families)
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

    # step 5: residual identity at z from the openings: R(z) = (A + z2^h2 B)(z1^h1 - 1) + Q2 (z2^h2 - 1)
    var d = Domains.__init__[p]()
    var pts = shift_points(families)
    var reads = List[E]()
    for k in range(shape.entries):
        var en = entry(families, k)
        reads.append(_opening[p](openings, shape, point_index(pts, en.dj1_a, en.dj2_a), en.col_a))
        reads.append(E(0) if en.col_b == NONE else _opening[p](openings, shape, point_index(pts, en.dj1_b, en.dj2_b), en.col_b))
    var z1 = _e[p](z, 0)
    var z2 = _e[p](z, 1)
    var rz = residual_at(families, _e[p](stage1, 3), z1, z2, ext_pow[1](d.omega1, p.h1() - 1), ext_pow[1](d.omega2, p.h2() - 1), reads)
    var one = ext_embed[4](SIMD[DType.uint8, 1](1))
    var z2h = ext_pow[4](z2, p.h2())
    var qa = _quotient_at[p](openings, shape, 0)
    var qb = _quotient_at[p](openings, shape, 1)
    var q2 = _quotient_at[p](openings, shape, 2)
    var rhs = f_add(ext_mul[4](f_add(qa, ext_mul[4](z2h, qb)), f_sub(ext_pow[4](z1, p.h1()), one)),
                    ext_mul[4](q2, f_sub(z2h, one)))
    if rz != rhs:
        raise Error("residual identity fails at z")

    raise Error("not implemented: verifier step 7 (tail)")
    # step 7: per level: root, S on the previous level, multiproofs against the roots (check_multiproof),
    #         expected symbols, three sumcheck checks s_i(0) + s_i(1) = s_{i-1}(r_{i-1}); clear vector:
    #         last consistency rows and <y_ell, w~_ell> directly
    # r.done(); return True


def _e[p: Params](l: List[UInt8], i: Int) -> E:
    var v = E(0)
    for t in range(p.e):
        v[t] = l[i * p.e + t]
    return v


def _opening[p: Params](openings: List[UInt8], shape: Shape, point: Int, column: Int) -> E:
    return _e[p](openings, point * shape.columns() + column)


def _quotient_at[p: Params](openings: List[UInt8], shape: Shape, q: Int) -> E:
    """Q(z) = sum_tau e_tau <w_z, coord_tau(Q)> at point 0 from the quotient coordinate columns."""
    var acc = E(0)
    for tau in range(p.e):
        var basis = E(0)
        basis[tau] = 1
        acc = f_add(acc, ext_mul[4](basis, _opening[p](openings, shape, 0, shape.columns_w + q * p.e + tau)))
    return acc

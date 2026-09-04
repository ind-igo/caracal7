"""Verifier: a host program, separate from the prover (design section 6). The seven steps of
statement-layer section 6 in transcript order; milestone 1 runs steps 1, 2, 5, 7 on the Ligerito
checks. Steps 3, 4, 6 (small grid, public columns, boundaries) arrive with the Z tree and the frontend.

Step 7 exists for the clear-vector case (no committed tail level, spec 9.3 termination): the
level-1 multiproofs against both roots, consistency at every opened position with the E (x) F4
alphabet rule of 9.1, and the evaluation claim <y, sum_p gamma_p w_{z_p}> directly."""

from caracal7.params import Params
from caracal7.hash import Hash
from caracal7.proof import Shape, ProofReader, VERSION, prefix_bytes
from caracal7.transcript import HostTranscript, DS_PREFIX, DS_TREE_W, DS_TREE_Q, DS_OPENINGS, DS_CLEAR
from caracal7.field import F2, F4, E, f_add, f_sub, f_mul, ext_mul, ext_pow, ext_embed
from caracal7.tables import Domains
from caracal7.residual import ENTRY, NONE, entry, shift_points, point_index, residual_at
from caracal7.encode import pack_slot
from caracal7.open import slot_weight
from caracal7.merkle import check_multiproof, distinct_sorted


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

    # step 7, clear-vector case: y_2 is sent in the clear and level 1 is opened against it
    comptime assert p.m_cosets == 1, "leaf s is the point g^s"   # ponytail: coset twist with the encoder's
    if len(shape.tail) > 0:
        raise Error("not implemented: verifier tail levels")
    var y = r.take(shape.clear_length * p.e)
    t.absorb(DS_CLEAR, y)
    var positions = t.positions(p.queries(), p.L())
    var row_w = 4 * p.n_cw() * shape.columns_w
    var row_q = 4 * p.n_cw() * shape.columns_q
    var mp_w = r.prefixed()
    var rows_w = check_multiproof[H](root_w, p.L(), row_w, positions, mp_w)
    var mp_q = r.prefixed()
    var rows_q = check_multiproof[H](root_q, p.L(), row_q, positions, mp_q)
    r.done()

    # consistency: coord_tau(Enc(y)(s)) = sum_c beta_c coord_tau(X[s, c]) in E (x) F4, coordinate-wise
    var opened = distinct_sorted(positions)
    for i in range(len(opened)):
        var enc = encode_at[p](y, ext_pow[2](d.g, opened[i]))
        for tau in range(4):
            var rhs = E(0)
            for c in range(shape.columns()):
                var sym = rows_w[i * row_w + c * 4 + tau] if c < shape.columns_w else rows_q[i * row_q + (c - shape.columns_w) * 4 + tau]
                rhs = f_add(rhs, f_mul(_e[p](beta_gamma, c), E(sym)))
            if enc[tau] != rhs:
                raise Error("consistency fails at an opened position")

    # evaluation claim: <y, sum_p gamma_p w_{z_p}> = sum_{c,p} beta_c gamma_p alpha_{c,p}
    var lhs = E(0)
    var rhs = E(0)
    for pt in range(shape.points):
        var dj1 = Int(pts[pt * 4]) | Int(pts[pt * 4 + 1]) << 8
        var dj2 = Int(pts[pt * 4 + 2]) | Int(pts[pt * 4 + 3]) << 8
        var z1p = ext_mul[4](z1, ext_embed[4](ext_pow[1](d.g1, dj1)))
        var z2p = ext_mul[4](z2, ext_embed[4](ext_pow[1](d.g2, dj2)))
        var gamma = _e[p](beta_gamma, shape.columns() + pt)
        var acc = E(0)
        for slot in range(p.N()):
            acc = f_add(acc, ext_mul[4](slot_weight[p](slot, z1p, z2p, d.rho1, d.rho2), _e[p](y, slot)))
        lhs = f_add(lhs, ext_mul[4](gamma, acc))
        var claim = E(0)
        for c in range(shape.columns()):
            claim = f_add(claim, ext_mul[4](_e[p](beta_gamma, c), _opening[p](openings, shape, pt, c)))
        rhs = f_add(rhs, ext_mul[4](gamma, claim))
    if lhs != rhs:
        raise Error("evaluation claim fails")
    return True


def encode_at[p: Params](y: List[UInt8], pt: F4) -> InlineArray[E, 4]:
    """Enc(y)(pt) in E (x) F4 as four E coordinates: sum_i (sum_j y[slot(i, j)] b_j) pt^i, the F4 scalar
    pt^i acting on the coordinates by its 4 x 4 matrix (spec 9.1 alphabet rule)."""
    var acc = InlineArray[E, 4](fill=E(0))
    var pw = F4(1, 0, 0, 0)
    for i in range(p.N() // 4):
        for j in range(4):
            var bj = F4(0)
            bj[j] = 1
            var m = ext_mul[2](bj, pw)
            var v = _e[p](y, pack_slot[p](i, j))
            for tau in range(4):
                acc[tau] = f_add(acc[tau], f_mul(v, E(m[tau])))
        pw = ext_mul[2](pw, pt)
    return acc^


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

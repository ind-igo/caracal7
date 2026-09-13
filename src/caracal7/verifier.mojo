"""Verifier: a host program, separate from the prover (design section 6). The seven steps of
statement-layer section 6 in transcript order; milestone 1 runs steps 1, 2, 5, 7 on the Ligerito
checks; every step is a host function over proof bytes and a host `H: Hash`. Steps 3, 4, 6 (small
grid, public columns, boundaries) arrive with the Z tree and the frontend.

Step 7 is the tail of spec 9.3 on the host, directly: per committed level the root, the previous
level's multiproofs against the three roots, the expected symbols computed from the opened rows (the E (x) F4
alphabet rule of 9.1 at level 1, the r_bar-combined row later), the batched query materialized as a
vector, three sumcheck checks, and the fold of the query; then the clear vector, its consistency
at the last opened positions, and <y_ell, w~_ell>. ponytail: O(|y_l|) host work per level; the
tensor form of 9.3 when a verifier budget exists."""

from std.time import perf_counter_ns

from caracal7.core.params import Params
from caracal7.core.hash import Hash
from caracal7.proof import Shape, ProofReader, VERSION, prefix_bytes
from caracal7.relations.statement import Compiled
from caracal7.core.transcript import HostTranscript, DS_PREFIX, DS_TREE_W, DS_TREE_Z, DS_TREE_Q, DS_OPENINGS, DS_CLEAR, DS_TAIL_ROOT, DS_TAIL_ROUND
from caracal7.core.field import F2, F4, E, f_add, f_sub, f_mul, ext_mul, ext_pow, ext_embed, E_LEVEL, E_BYTES
from caracal7.core.tables import Domains, RsDomain, f2_primitive
from caracal7.pcs import pack_slot, check_multiproof, distinct_sorted, host_r3, rbar_at, tail_encode_at, quadratic_at
from caracal7.pcs.tensor import Unit, query_units, consistency_units, row_units, clear_value, f4_dual
from caracal7.relations import ENTRY, NONE, ACC, END, WIRE, PUBF, KIND_LOOKUP, KIND_HORNER, acc_z_col, acc_start, acc_kind, acc_table, PUB, RES, ZERO, POINT, FIX_ONE, FIX_E, required_points, entry, derived_chals, lookup_constant, horner_chain_end, point_index, point_coord, residual_at, interp_cyclic, eval_values, eval_line, value_bytes
from caracal7.core.bytes import get_u16, list_e, check_field_bytes


def verify[p: Params, H: Hash](var proof_bytes: List[UInt8], shape: Shape, public_inputs: Span[UInt8, _], mut families: List[UInt8],
                               public: List[UInt8] = List[UInt8](), profile: Bool = False) raises -> Bool:
    """The transcript in order: every proof value read and every challenge sampled is here; the checks
    between them are one helper per step of statement-layer section 6. `public` is the data both sides
    derive from the public inputs (docs/public-columns.md): one period of values of every public column,
    then the polynomial of every restriction; the verifier never hashes it.
    ponytail: soundness rests on the caller deriving `public` from `public_inputs` (which the prefix hashes);
    nothing here checks that. The statement builder is where the derivation becomes code on both sides."""
    _check_statement[p](shape, families, public_inputs, public)
    var r = ProofReader(proof_bytes^)
    var t = HostTranscript[p, H]()
    var tv = perf_counter_ns()

    # step 1: header, prefix, W root, stage-1 challenges
    if r.u32() != Int(VERSION):
        raise Error("bad version")
    var pub = r.prefixed()
    if Span(pub) != public_inputs:
        raise Error("public inputs differ")
    var prefix = prefix_bytes[p, H](shape, public_inputs, families)
    t.absorb(DS_PREFIX, prefix)
    var root_w = r.take(H.DIGEST)
    t.absorb(DS_TREE_W, root_w)
    var stage1 = t.elements(3)                      # beta, delta, gamma
    derived_chals(stage1, shape.chals)
    var wchal = List[UInt8]()
    if shape.wiring_products() > 0:
        wchal = t.elements(2)                       # beta_w, gamma_w of the copy constraint

    # step 2: Z root and Z2 -> alpha; Q root and Q3 -> z
    var root_z = List[UInt8]()
    var z2v = List[UInt8]()
    if shape.accumulators() > 0:                       # no Z tree without accumulators
        root_z = r.take(H.DIGEST)
        t.absorb(DS_TREE_Z, root_z)
        if shape.products() > 0:
            z2v = r.field_bytes(shape.products() * p.h2() * p.e)
            t.absorb(DS_TREE_Z, z2v)
    var alpha = list_e(t.elements(1), 0)
    var root_q = r.take(H.DIGEST)
    t.absorb(DS_TREE_Q, root_q)
    var q3 = List[UInt8]()
    if shape.accumulators() > 0:
        q3 = r.field_bytes(2 * p.h2() * p.e)
        t.absorb(DS_TREE_Q, q3)
    var z = t.elements(2)
    var z1 = list_e(z, 0)
    var z2 = list_e(z, 1)

    # openings -> beta per column, gamma per point
    var openings = r.field_bytes(shape.points * shape.columns() * p.e)
    t.absorb(DS_OPENINGS, openings)
    var beta_gamma = t.elements(shape.columns() + shape.points)
    var d = Domains.__init__[p]()
    _vmark(profile, "transcript and openings", tv)

    # steps 3 to 6 on the opened values
    _boundaries[p](shape, openings, z2v, stage1)
    _wiring[p](shape, families, openings, z2v, wchal, stage1, public, d)
    _vmark(profile, "boundaries", tv)
    _small_grid[p](shape, openings, z2v, q3, alpha, stage1, wchal, z2, d)
    _vmark(profile, "small grid", tv)
    _residual[p](shape, families, openings, public, alpha, stage1, z1, z2, d)
    _vmark(profile, "residual at z", tv)
    _restrictions[p](shape, openings, public, z1)
    _vmark(profile, "restrictions", tv)

    # step 7: the tail in tensor form, one committed level at a time, then the clear vector
    var tail = _Tail[p](shape, openings, beta_gamma, z1, z2, d)
    _vmark(profile, "running claim", tv)
    for i in range(len(shape.tail)):
        tail.level[H](r, t, shape, i, root_w, root_z, root_q, beta_gamma, d)
    _vmark(profile, "tail levels", tv)
    tail.clear[H](r, t, shape, root_w, root_z, root_q, beta_gamma, d)
    _vmark(profile, "clear vector", tv)
    return True


def verify[p: Params, H: Hash](var proof_bytes: List[UInt8], mut c: Compiled, public_inputs: Span[UInt8, _],
                               public: List[UInt8] = List[UInt8](), profile: Bool = False) raises -> Bool:
    """Against a compiled statement: its shape and family table travel together."""
    return verify[p, H](proof_bytes^, c.shape, public_inputs, c.families, public, profile)


def _check_statement[p: Params](shape: Shape, families: List[UInt8], public_inputs: Span[UInt8, _], public: List[UInt8]) raises:
    """Shape validated its own family table; bind this one to the opening list and the challenge count
    before any index is used, and the public inputs to the statement's pinned bytes."""
    if len(families) != shape.entries * ENTRY:
        raise Error("family table does not match the shape")
    if len(public_inputs) < len(shape.pinned) or public_inputs[0:len(shape.pinned)] != Span(shape.pinned):
        raise Error("public inputs do not start with the statement's pinned bytes")
    var need = required_points(families, shape.restrictions, shape.accumulators() > 0, shape.zeros)
    for i in range(len(need) // POINT):
        if point_index(shape.point_list, get_u16(need, i * POINT), get_u16(need, i * POINT + 2)) < 0:
            raise Error("family table reads a point outside the shape's opening list")
    for k in range(shape.entries):
        if entry(families, k).chal > shape.chal_count():
            raise Error("family table names a challenge element past the shape's derivation table")
    if len(public) != shape.public_bytes[p]():
        raise Error("public data has the wrong size")
    check_field_bytes(public)


def _one() -> E:
    return ext_embed[E_LEVEL](SIMD[DType.uint8, 1](1))


def _boundaries[p: Params](shape: Shape, openings: List[UInt8], z2v: List[UInt8], stage1: List[UInt8]) raises:
    """Steps 3 and 6, the accumulator boundaries (spec 7.1, 7.3 for (P)): Z(1, z2) = 1 from the opening at
    (1, z2); Z2(1) = 1; Z2(e2) Z(e1, e2) N(e1, e2) = D(e1, e2) from the openings at (e1, e2); a lookup closes
    on its table constant. The chain-end pairs (W) themselves are the small grid's R2 = Q3 (X2^h2 - 1).
    Then the zero rows (polynomial-mulmod 4): a column that is zero on row 0 or the last row of every chain
    opens to zero at (coordinate, z2), a polynomial in X2 of degree < h2 vanishing at random z2."""
    var one = _one()
    var at_start = point_index(shape.point_list, FIX_ONE, 0)     # (1, z2)
    var at_end = point_index(shape.point_list, FIX_E, FIX_E)     # (e1, e2)
    for k in range(shape.accumulators()):
        var z_col = acc_z_col(shape.accs, k)
        var pi = shape.product_of(k)                             # its Z2 line
        if pi < 0:                                               # Horner: 7.1 with the record's start; its chain ends are chain-end terms
            if _coords_at[p](openings, shape, at_start, z_col) != ext_embed[E_LEVEL](SIMD[DType.uint8, 1](acc_start(shape.accs, k))):
                raise Error("accumulator chain start is not its start value")
            continue
        if _coords_at[p](openings, shape, at_start, z_col) != one:
            raise Error("accumulator chain start is not 1")
        if list_e(z2v, pi * p.h2()) != one:
            raise Error("Z2(1) is not 1")
        var lhs = ext_mul[E_LEVEL](ext_mul[E_LEVEL](list_e(z2v, pi * p.h2() + p.h2() - 1), _coords_at[p](openings, shape, at_end, z_col)),
                             _factor_at[p](openings, shape, at_end, at_end, shape.accs, k, stage1, False))
        if acc_kind(shape.accs, k) == KIND_LOOKUP:
            # 6.3: the last row has no pair factor, so the product closes on the table constant
            var w = get_u16(shape.accs, k * ACC + 2)
            if lhs != lookup_constant(shape.tables[acc_table(shape.accs, k)], w, stage1):
                raise Error("lookup product is not the table constant")
        elif lhs != _factor_at[p](openings, shape, at_end, at_end, shape.accs, k, stage1, True):
            raise Error("accumulator grand product is not 1")
        # TODO(memory): boundary rule 6 of spec 6.4 (the memory accumulator's closing factor) goes here.
    for i in range(len(shape.zeros) // ZERO):
        if _opening[p](openings, shape, point_index(shape.point_list, get_u16(shape.zeros, i * ZERO + 2), 0), get_u16(shape.zeros, i * ZERO)) != E(0):
            raise Error("chain row is not zero")


def _wiring[p: Params](shape: Shape, families: List[UInt8], openings: List[UInt8], z2v: List[UInt8], wchal: List[UInt8],
                       stage1: List[UInt8], public: List[UInt8], d: Domains) raises:
    """The wiring products (accumulate.mojo): each starts at 1; jointly, prod_g Z_g(e2) N_g(e2) times the public
    factors' (v + beta_w id + gamma_w) equals prod_g D_g(e2) times their (v + beta_w sigma + gamma_w)."""
    if shape.wiring_products() == 0:
        return
    var one = _one()
    var at_end = point_index(shape.point_list, FIX_E, FIX_E)
    var kappa = f2_primitive()
    var e2f = ext_pow[1](d.omega2, p.h2() - 1)
    var bw = list_e(wchal, 0)
    var gw = list_e(wchal, 1)
    var lhs = one
    var rhs = one
    for g in range(shape.wiring_products()):
        var pi = shape.wiring_product(g)                         # its Z2 line
        if list_e(z2v, pi * p.h2()) != one:
            raise Error("wiring product does not start at 1")
        lhs = ext_mul[E_LEVEL](lhs, list_e(z2v, pi * p.h2() + p.h2() - 1))
        for sl in range(2):
            var col = get_u16(shape.wires, g * WIRE + 2 * sl)
            if col == NONE:
                continue
            var wg = f_add(_coords_at[p](openings, shape, at_end, col), gw)
            var sg = F2(shape.sigma[((2 * g + sl) * p.h2() + p.h2() - 1) * 2], shape.sigma[((2 * g + sl) * p.h2() + p.h2() - 1) * 2 + 1])
            lhs = ext_mul[E_LEVEL](lhs, f_add(wg, ext_mul[E_LEVEL](bw, ext_embed[E_LEVEL](ext_mul[1](ext_pow[1](kappa, 2 * g + sl), e2f)))))
            rhs = ext_mul[E_LEVEL](rhs, f_add(wg, ext_mul[E_LEVEL](bw, ext_embed[E_LEVEL](sg))))
    var off = shape.public_bytes[p]()
    for i in range(len(shape.pubf) // PUBF):
        off -= shape.factor_bytes[p](i)
    for i in range(len(shape.pubf) // PUBF):
        var n = shape.factor_bytes[p](i)
        var cols = List[UInt8](capacity=n)
        for t in range(n):
            cols.append(public[off + t])
        off += n
        var vg = f_add(horner_chain_end[p](families, shape.accs, get_u16(shape.pubf, i * PUBF), cols, stage1), gw)
        var f_id = f_add(vg, ext_mul[E_LEVEL](bw, ext_embed[E_LEVEL](F2(shape.pubf[i * PUBF + 2], shape.pubf[i * PUBF + 3]))))
        var f_sg = f_add(vg, ext_mul[E_LEVEL](bw, ext_embed[E_LEVEL](F2(shape.pubf[i * PUBF + 4], shape.pubf[i * PUBF + 5]))))
        if f_id.reduce_or() == 0 or f_sg.reduce_or() == 0:
            raise Error("zero public factor")
        lhs = ext_mul[E_LEVEL](lhs, f_id)
        rhs = ext_mul[E_LEVEL](rhs, f_sg)
    if lhs != rhs:
        raise Error("wiring grand product is not the public factor")


def _small_grid[p: Params](shape: Shape, openings: List[UInt8], z2v: List[UInt8], q3: List[UInt8], alpha: E, stage1: List[UInt8],
                           wchal: List[UInt8], z2: E, d: Domains) raises:
    """Step 3, the small grid (spec 7.4): R2(z2) = Q3(z2) (z2^h2 - 1), R2 from Z2 interpolated at z2 and
    omega2 z2, Q3 interpolated on G2, and the openings at (e1, z2)."""
    if shape.accumulators() == 0:
        return
    var one = _one()
    var at_e1 = point_index(shape.point_list, FIX_E, 0)          # (e1, z2)
    var at_next = point_index(shape.point_list, FIX_ONE, 2)      # (1, omega2 z2)
    var kappa = f2_primitive()
    var e2 = ext_embed[E_LEVEL](ext_pow[1](d.omega2, p.h2() - 1))
    var w2 = ext_embed[E_LEVEL](d.omega2)
    var r2 = E(0)
    for k in range(shape.accumulators()):
        var pi = shape.product_of(k)
        if pi < 0:
            continue
        var za = interp_cyclic(z2v, pi * p.h2(), p.h2(), d.omega2, z2)
        var zb = interp_cyclic(z2v, pi * p.h2(), p.h2(), d.omega2, ext_mul[E_LEVEL](z2, w2))
        var c = _coords_at[p](openings, shape, at_e1, acc_z_col(shape.accs, k))
        var n_z = _factor_at[p](openings, shape, at_e1, at_next, shape.accs, k, stage1, False)
        var d_z = _factor_at[p](openings, shape, at_e1, at_next, shape.accs, k, stage1, True)
        var term = ext_mul[E_LEVEL](f_sub(z2, e2), f_sub(ext_mul[E_LEVEL](zb, d_z), ext_mul[E_LEVEL](ext_mul[E_LEVEL](za, c), n_z)))
        r2 = f_add(r2, ext_mul[E_LEVEL](ext_pow[E_LEVEL](alpha, shape.family_of(k)), term))
    for g in range(shape.wiring_products()):       # (z2 - e2) (Z(omega2 z2) prod den - Z(z2) prod num) over the two slots
        var pi = shape.wiring_product(g)
        var za = interp_cyclic(z2v, pi * p.h2(), p.h2(), d.omega2, z2)
        var zb = interp_cyclic(z2v, pi * p.h2(), p.h2(), d.omega2, ext_mul[E_LEVEL](z2, w2))
        var nn = one
        var dd = one
        for sl in range(2):
            var col = get_u16(shape.wires, g * WIRE + 2 * sl)
            if col == NONE:
                continue
            var wg = f_add(_coords_at[p](openings, shape, at_e1, col), list_e(wchal, 1))
            var sig = interp_cyclic(shape.sigma, (2 * g + sl) * p.h2(), p.h2(), d.omega2, z2, 2)
            nn = ext_mul[E_LEVEL](nn, f_add(wg, ext_mul[E_LEVEL](ext_mul[E_LEVEL](list_e(wchal, 0), ext_embed[E_LEVEL](ext_pow[1](kappa, 2 * g + sl))), z2)))
            dd = ext_mul[E_LEVEL](dd, f_add(wg, ext_mul[E_LEVEL](list_e(wchal, 0), sig)))
        var term = ext_mul[E_LEVEL](f_sub(z2, e2), f_sub(ext_mul[E_LEVEL](zb, dd), ext_mul[E_LEVEL](za, nn)))
        r2 = f_add(r2, ext_mul[E_LEVEL](ext_pow[E_LEVEL](alpha, get_u16(shape.wires, g * WIRE + 4)), term))
    for i in range(len(shape.ends) // END):           # chain-end terms (smallgrid.mojo): coef chal alpha^family A [B] [(z2 - e2)]
        var v = ext_mul[E_LEVEL](ext_pow[E_LEVEL](alpha, get_u16(shape.ends, i * END + 4)), _coords_at[p](openings, shape, at_e1, get_u16(shape.ends, i * END)))
        v = f_mul(v, E(shape.ends[i * END + 6]))
        if shape.ends[i * END + 7] != 0:
            v = ext_mul[E_LEVEL](v, list_e(stage1, Int(shape.ends[i * END + 7]) - 1))
        if get_u16(shape.ends, i * END + 2) != NONE:
            v = ext_mul[E_LEVEL](v, _coords_at[p](openings, shape, at_e1, get_u16(shape.ends, i * END + 2)))
        if shape.ends[i * END + 8] != 0:
            v = ext_mul[E_LEVEL](v, f_sub(z2, e2))
        r2 = f_add(r2, v)
    var q3z = interp_cyclic(q3, 0, 2 * p.h2(), d.g2, z2)
    if r2 != ext_mul[E_LEVEL](q3z, f_sub(ext_pow[E_LEVEL](z2, p.h2()), one)):
        raise Error("small grid identity fails at z2")


def _residual[p: Params](shape: Shape, families: List[UInt8], openings: List[UInt8], public: List[UInt8], alpha: E,
                         stage1: List[UInt8], z1: E, z2: E, d: Domains) raises:
    """Step 5: the residual identity at z from the openings, R(z) = (A + z2^h2 B)(z1^h1 - 1) + Q2 (z2^h2 - 1)."""
    var one = _one()
    var preads = _PublicReads[p](shape, public, shape.point_list, z1, z2, d)
    var reads = List[E]()
    for k in range(shape.entries):
        var en = entry(families, k)
        reads.append(preads.read(openings, point_index(shape.point_list, en.dj1_a, en.dj2_a), en.col_a))
        reads.append(E(0) if en.col_b == NONE else preads.read(openings, point_index(shape.point_list, en.dj1_b, en.dj2_b), en.col_b))
    var rz = residual_at(families, alpha, stage1, z1, z2, ext_pow[1](d.omega1, p.h1() - 1), ext_pow[1](d.omega2, p.h2() - 1), reads)
    var z2h = ext_pow[E_LEVEL](z2, p.h2())
    var qa = _quotient_at[p](openings, shape, 0)
    var qb = _quotient_at[p](openings, shape, 1)
    var q2 = _quotient_at[p](openings, shape, 2)
    var rhs = f_add(ext_mul[E_LEVEL](f_add(qa, ext_mul[E_LEVEL](z2h, qb)), f_sub(ext_pow[E_LEVEL](z1, p.h1()), one)),
                    ext_mul[E_LEVEL](q2, f_sub(z2h, one)))
    if rz != rhs:
        raise Error("residual identity fails at z")


def _restrictions[p: Params](shape: Shape, openings: List[UInt8], public: List[UInt8], z1: E) raises:
    """Restrictions (docs/public-columns.md): the opening of the column on its line equals the public polynomial at z1."""
    var res_off = value_bytes(shape.publics, p.h1(), p.h2())
    for i in range(len(shape.restrictions) // RES):
        var col = get_u16(shape.restrictions, i * RES)
        var coord = get_u16(shape.restrictions, i * RES + 2)
        var count = get_u16(shape.restrictions, i * RES + 4)
        if _opening[p](openings, shape, point_index(shape.point_list, 0, coord), col) != eval_line(public, res_off, count, z1):
            raise Error("restriction fails")
        res_off += count * 2


struct _Tail[p: Params]:
    """Step 7, the tail in tensor form (pcs/tensor.mojo): the running claim and the state its levels fold.
    The claim starts as <y_2, sum_p gamma_p w_{z_p}> = sum beta_c gamma_p alpha_{c,p}; the query is a list
    of digit products, folded per level at the sumcheck challenges, and no vector of length N is ever held."""
    var units: List[Unit]           # the query as digit products
    var running: E                  # the running claim
    var folded: Int                 # binary digits folded so far
    var y_len: Int                  # rows of the current message
    var r_prev: List[UInt8]         # r of the last committed level, empty while that is level 1
    var roots: List[List[UInt8]]    # committed tail roots
    var doms: List[RsDomain]        # their domains
    var dual: InlineArray[F4, 4]

    def __init__(out self, shape: Shape, openings: List[UInt8], beta_gamma: List[UInt8], z1: E, z2: E, d: Domains) raises:
        comptime assert Self.p.n_cw() == 1, "one codeword per column: rows are (s, column, 4)"   # ponytail: split with the encoder's
        self.units = List[Unit]()
        self.running = E(0)
        self.folded = 0
        self.y_len = Self.p.N()
        self.r_prev = List[UInt8]()
        self.roots = List[List[UInt8]]()
        self.doms = List[RsDomain]()
        self.dual = f4_dual()
        ref pts = shape.point_list
        for pt in range(shape.points):
            var dj1 = Int(pts[pt * 4]) | Int(pts[pt * 4 + 1]) << 8
            var dj2 = Int(pts[pt * 4 + 2]) | Int(pts[pt * 4 + 3]) << 8
            var z1p = point_coord(z1, dj1, d.g1, Self.p.h1())
            var z2p = point_coord(z2, dj2, d.g2, Self.p.h2())
            var gamma = list_e(beta_gamma, shape.columns() + pt)
            query_units[Self.p](z1p, z2p, gamma, d.rho1, d.rho2, self.units)
            var claim = E(0)
            for c in range(shape.columns()):
                claim = f_add(claim, ext_mul[E_LEVEL](list_e(beta_gamma, c), _opening[Self.p](openings, shape, pt, c)))
            self.running = f_add(self.running, ext_mul[E_LEVEL](gamma, claim))

    def level[H: Hash](mut self, mut r: ProofReader, mut t: HostTranscript[Self.p, H], shape: Shape, i: Int,
                       root_w: List[UInt8], root_z: List[UInt8], root_q: List[UInt8], beta_gamma: List[UInt8], d: Domains) raises:
        """Committed level i: its root, the previous level's multiproofs, the expected symbols from the
        opened rows, the batching scalars, three sumcheck rounds, and the fold of the query."""
        comptime D = Self.p.a1 + Self.p.a2
        comptime M = Self.p.m1 * Self.p.m2
        comptime e = Self.p.e
        var lvl = shape.tail[i]
        var root = r.take(H.DIGEST)
        t.absorb(DS_TAIL_ROOT, root)
        var prev = _open_previous[Self.p, H](r, t, shape, i, root_w, root_z, root_q, self.roots)
        var count = Self.p.queries() if i == 0 else shape.tail[i - 1].queries
        var v_count = 4 * count if i == 0 else count
        # the expected symbols v (9.3) from the opened rows; a function of the transcript, so not sent nor absorbed
        var v = List[UInt8](capacity=v_count * e)
        for q in range(count):
            var idx = _index_of(prev.opened, prev.positions[q])
            if i == 0:
                for tau in range(4):
                    _push_e(v, _level1_symbol[Self.p](prev, shape, beta_gamma, idx, tau))
            else:
                _push_e(v, _tail_symbol(prev, idx, self.r_prev))
        var batch = t.elements(v_count + 1)
        var claim = ext_mul[E_LEVEL](list_e(batch, 0), self.running)
        for k in range(v_count):
            claim = f_add(claim, ext_mul[E_LEVEL](list_e(batch, 1 + k), list_e(v, k)))
        # w~ = batch_0 running + sum_q batch_q g_q, as units
        var b0 = list_e(batch, 0)
        for k in range(len(self.units)):
            self.units[k].scalar = ext_mul[E_LEVEL](self.units[k].scalar, b0)
        if i == 0:
            for q in range(count):
                var weights = InlineArray[E, 4](fill=E(0))
                for tau in range(4):
                    weights[tau] = list_e(batch, 1 + 4 * q + tau)
                consistency_units[Self.p](d.level1.point(prev.positions[q]), weights, self.dual, self.units)
        else:
            for q in range(count):
                row_units(self.doms[i - 1].point(prev.positions[q]), list_e(batch, 1 + q), self.folded, D, M, self.units)
        var rounds = r.field_bytes(9 * e)
        var r_l = List[UInt8]()
        for dgt in range(3):
            if f_add(list_e(rounds, 3 * dgt), list_e(rounds, 3 * dgt + 1)) != claim:
                raise Error("sumcheck fails at a tail level")
            var msg = List[UInt8](capacity=3 * e)
            for b in range(3 * e):
                msg.append(rounds[3 * dgt * e + b])
            t.absorb(DS_TAIL_ROUND, msg)
            var rd = t.elements(1)
            claim = quadratic_at(rounds, 3 * dgt, list_e(rd, 0))
            for k in range(len(self.units)):
                self.units[k].fold(self.folded + dgt, list_e(rd, 0))
            r_l.extend(rd^)
        self.folded += 3
        self.running = claim
        self.r_prev = r_l^
        self.y_len = lvl.rows
        self.roots.append(root^)
        self.doms.append(RsDomain(lvl.L // lvl.cosets, lvl.cosets))

    def clear[H: Hash](mut self, mut r: ProofReader, mut t: HostTranscript[Self.p, H], shape: Shape,
                       root_w: List[UInt8], root_z: List[UInt8], root_q: List[UInt8], beta_gamma: List[UInt8], d: Domains) raises:
        """The clear vector: consistency against the last committed level at its opened positions, then
        the evaluation claim directly."""
        comptime D = Self.p.a1 + Self.p.a2
        if shape.clear_length != self.y_len:
            raise Error("shape.clear_length does not match the tail schedule")
        var y = r.field_bytes(shape.clear_length * Self.p.e)
        t.absorb(DS_CLEAR, y)
        var last = _open_previous[Self.p, H](r, t, shape, len(shape.tail), root_w, root_z, root_q, self.roots)
        r.done()
        for idx in range(len(last.opened)):
            if len(shape.tail) == 0:
                var enc = encode_at[Self.p](y, d.level1.point(last.opened[idx]))
                for tau in range(4):
                    if enc[tau] != _level1_symbol[Self.p](last, shape, beta_gamma, idx, tau):
                        raise Error("consistency fails at an opened position")
            else:
                var dom = self.doms[len(self.doms) - 1]
                if tail_encode_at(y, self.y_len, dom.point(last.opened[idx])) != _tail_symbol(last, idx, self.r_prev):
                    raise Error("consistency fails at an opened position")
        if clear_value(self.units, y, self.folded, D) != self.running:
            raise Error("evaluation claim fails")


def _vmark(profile: Bool, name: String, mut t0: Int):
    """With profile on: print the ms since the last mark. Off: nothing."""
    if profile:
        var now = perf_counter_ns()
        print("  verify ", (now - t0) // 1000000, " ms  ", name)
        t0 = now


@fieldwise_init
struct Opened:
    """One level opened: the sampled positions, the distinct ascending list, and its rows
    (three row sets at level 1: witness, accumulator, and quotient trees)."""
    var positions: List[Int]
    var opened: List[Int]
    var rows_w: List[UInt8]
    var rows_z: List[UInt8]
    var rows_q: List[UInt8]
    var row_w: Int
    var row_z: Int
    var row_q: Int


def _open_previous[p: Params, H: Hash](mut r: ProofReader, mut t: HostTranscript[p, H], shape: Shape, i: Int,
                                       root_w: List[UInt8], root_z: List[UInt8], root_q: List[UInt8], roots: List[List[UInt8]]) raises -> Opened:
    """Sample S on the level before committed level i (level 1 when i == 0) and check its multiproof(s)."""
    if i == 0:
        var positions = t.positions(p.queries(), p.L())
        var row_w = 4 * shape.columns_w
        var row_z = 4 * shape.columns_z
        var row_q = 4 * shape.columns_q
        var mp_w = r.prefixed()
        var rows_w = check_multiproof[H](root_w, p.L(), row_w, positions, mp_w)
        var rows_z = List[UInt8]()
        if row_z > 0:
            var mp_z = r.prefixed()
            rows_z = check_multiproof[H](root_z, p.L(), row_z, positions, mp_z)
        var mp_q = r.prefixed()
        var rows_q = check_multiproof[H](root_q, p.L(), row_q, positions, mp_q)
        return Opened(positions=positions.copy(), opened=distinct_sorted(positions), rows_w=rows_w^, rows_z=rows_z^, rows_q=rows_q^,
                      row_w=row_w, row_z=row_z, row_q=row_q)
    var lvl = shape.tail[i - 1]
    var positions = t.positions(lvl.queries, lvl.L)
    var mp = r.prefixed()
    var rows = check_multiproof[H](roots[i - 1], lvl.L, 8 * p.e, positions, mp)
    return Opened(positions=positions.copy(), opened=distinct_sorted(positions), rows_w=rows^, rows_z=List[UInt8](), rows_q=List[UInt8](),
                  row_w=8 * p.e, row_z=0, row_q=0)


def _index_of(opened: List[Int], s: Int) raises -> Int:
    for i in range(len(opened)):
        if opened[i] == s:
            return i
    raise Error("sampled position was not opened")


def _level1_symbol[p: Params](o: Opened, shape: Shape, beta: Span[UInt8, _], idx: Int, tau: Int) -> E:
    """sum_c beta_c coord_tau(X[s, c]) over the three trees at opened row idx."""
    var acc = E(0)
    var wz = shape.columns_w + shape.columns_z
    for c in range(shape.columns()):
        var sym: UInt8
        if c < shape.columns_w:
            sym = o.rows_w[idx * o.row_w + c * 4 + tau]
        elif c < wz:
            sym = o.rows_z[idx * o.row_z + (c - shape.columns_w) * 4 + tau]
        else:
            sym = o.rows_q[idx * o.row_q + (c - wz) * 4 + tau]
        acc = f_add(acc, f_mul(list_e(beta, c), E(sym)))
    return acc


def _tail_symbol(o: Opened, idx: Int, r_prev: Span[UInt8, _]) -> E:
    """<X[s, :], r_bar> for a tail row of 8 E symbols."""
    var rr = host_r3(r_prev)
    var acc = E(0)
    for a in range(8):
        acc = f_add(acc, ext_mul[E_LEVEL](rbar_at(rr, a, 3), list_e(o.rows_w, idx * 8 + a)))
    return acc


def _push_e(mut l: List[UInt8], v: E):
    for t in range(E_BYTES):
        l.append(v[t])


def encode_at[p: Params](y: Span[UInt8, _], pt: F4) -> InlineArray[E, 4]:
    """Enc(y)(pt) in E (x) F4 as four E coordinates: sum_i (sum_j y[slot(i, j)] b_j) pt^i, the F4 scalar
    pt^i acting on the coordinates by its 4 x 4 matrix (spec 9.1 alphabet rule)."""
    var acc = InlineArray[E, 4](fill=E(0))
    var pw = F4(1, 0, 0, 0)
    for i in range(p.N() // 4):
        for j in range(4):
            var bj = F4(0)
            bj[j] = 1
            var m = ext_mul[2](bj, pw)
            var v = list_e(y, pack_slot[p](i, j))
            for tau in range(4):
                acc[tau] = f_add(acc[tau], f_mul(v, E(m[tau])))
        pw = ext_mul[2](pw, pt)
    return acc^



def _opening[p: Params](openings: Span[UInt8, _], shape: Shape, point: Int, column: Int) -> E:
    return list_e(openings, point * shape.columns() + column)


struct _PublicReads[p: Params]:
    """Reads for residual_at: an opening for a committed column, the values interpolated at the point for a
    public column (index at or past columns_w + columns_z), cached per (column, point)."""
    var base: Int               # columns_w + columns_z: the first public index
    var columns: Int            # shape.columns(), the openings stride
    var points: Int
    var publics: List[UInt8]
    var public: List[UInt8]
    var pts: List[UInt8]
    var z1: E
    var z2: E
    var g1: F2
    var g2: F2
    var w1: F2
    var w2: F2
    var cache: List[E]
    var valid: List[Bool]

    def __init__(out self, shape: Shape, public: List[UInt8], pts: List[UInt8], z1: E, z2: E, d: Domains):
        self.base = shape.columns_w + shape.columns_z
        self.columns = shape.columns()
        self.points = shape.points
        self.publics = shape.publics.copy()
        self.public = public.copy()
        self.pts = pts.copy()
        self.z1 = z1
        self.z2 = z2
        self.g1 = d.g1
        self.g2 = d.g2
        self.w1 = d.omega1
        self.w2 = d.omega2
        self.cache = List[E](length=shape.columns_p * shape.points, fill=E(0))
        self.valid = List[Bool](length=shape.columns_p * shape.points, fill=False)

    def read(mut self, openings: Span[UInt8, _], point: Int, column: Int) raises -> E:
        if column < self.base:
            return list_e(openings, point * self.columns + column)
        var i = column - self.base
        var slot = i * self.points + point
        if not self.valid[slot]:
            var off = 0
            for j in range(i):
                off += Self.p.h1() * (Self.p.h2() // get_u16(self.publics, j * PUB))
            var x1 = point_coord(self.z1, get_u16(self.pts, point * 4), self.g1, Self.p.h1())
            var x2 = point_coord(self.z2, get_u16(self.pts, point * 4 + 2), self.g2, Self.p.h2())
            self.cache[slot] = eval_values(self.public, off, get_u16(self.publics, i * PUB), Self.p.h1(), Self.p.h2(), self.w1, self.w2, x1, x2)
            self.valid[slot] = True
        return self.cache[slot]


def _fp_at[p: Params](openings: Span[UInt8, _], shape: Shape, point: Int, accs: Span[UInt8, _], k: Int, den: Bool) -> E:
    """fp of the num or den record of accumulator k at an opening point: sum_j b_j c_j(point)."""
    var v = E(0)
    for j in range(get_u16(accs, k * ACC + (4 if den else 2))):
        var b = E(0)
        b[j] = 1
        v = f_add(v, ext_mul[E_LEVEL](b, _opening[p](openings, shape, point, get_u16(accs, k * ACC + (22 if den else 6) + 2 * j))))
    return v


def _factor_at[p: Params](openings: Span[UInt8, _], shape: Shape, point: Int, next: Int, accs: Span[UInt8, _], k: Int,
                          chals: Span[UInt8, _], den: Bool) -> E:
    """N or D of accumulator k at an opening point by its kind (accumulate.mojo); `next` is the point one row
    on, which a lookup's D reads: (1, omega2 z2) for (e1, z2).
    TODO(memory): the KIND_MEMORY factor pair of spec 6.4 goes here."""
    var fp = _fp_at[p](openings, shape, point, accs, k, den)
    if acc_kind(accs, k) == KIND_LOOKUP:
        if den:
            return f_add(f_add(list_e(chals, 4), fp), ext_mul[E_LEVEL](list_e(chals, 0), _fp_at[p](openings, shape, next, accs, k, True)))
        return ext_mul[E_LEVEL](list_e(chals, 3), f_add(list_e(chals, 1), fp))
    return f_add(list_e(chals, 2), fp)


def _coords_at[p: Params](openings: Span[UInt8, _], shape: Shape, point: Int, col0: Int) -> E:
    """An E-valued column at a point from its e coordinate columns: sum_tau b_tau <w_z, coord_tau>."""
    var acc = E(0)
    for tau in range(p.e):
        var basis = E(0)
        basis[tau] = 1
        acc = f_add(acc, ext_mul[E_LEVEL](basis, _opening[p](openings, shape, point, col0 + tau)))
    return acc


def _quotient_at[p: Params](openings: Span[UInt8, _], shape: Shape, q: Int) -> E:
    """Q(z) for Q in (A, B, Q2) from the quotient coordinate columns at point 0."""
    return _coords_at[p](openings, shape, 0, shape.columns_w + shape.columns_z + q * p.e)

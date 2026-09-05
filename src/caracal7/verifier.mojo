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

from caracal7.params import Params
from caracal7.hash import Hash
from caracal7.proof import Shape, ProofReader, VERSION, prefix_bytes
from caracal7.transcript import HostTranscript, DS_PREFIX, DS_TREE_W, DS_TREE_Z, DS_TREE_Q, DS_OPENINGS, DS_CLEAR, DS_TAIL_ROOT, DS_TAIL_ROUND
from caracal7.field import F2, F4, E, f_add, f_sub, f_mul, ext_mul, ext_pow, ext_embed
from caracal7.tables import Domains, RsDomain
from caracal7.ir import ENTRY, NONE, entry, shift_points, point_index, point_coord, residual_at
from caracal7.encode import pack_slot, pack_index
from caracal7.open import slot_weight
from caracal7.merkle import check_multiproof, distinct_sorted
from caracal7.accumulate import ACC
from caracal7.bytes import get_u16, list_e
from caracal7.smallgrid import interp_cyclic
from caracal7.tail import e_mul_f4, host_r3, rbar_at, tail_encode_at, fold8_host, quadratic_at


def verify[p: Params, H: Hash](var proof_bytes: List[UInt8], shape: Shape, public_inputs: List[UInt8], mut families: List[UInt8]) raises -> Bool:
    if len(families) != shape.entries * ENTRY or len(shift_points(families)) != shape.points * 4:
        raise Error("family table does not match the shape")
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
    var stage1 = t.elements(3)                      # beta, delta, gamma

    # step 2: Z root and Z2 -> alpha; Q root -> z
    var root_z = List[UInt8]()
    var z2v = List[UInt8]()
    if shape.accumulators() > 0:                       # no Z tree without accumulators
        root_z = r.take(H.DIGEST)
        t.absorb(DS_TREE_Z, root_z)
        z2v = r.take(shape.accumulators() * p.h2() * p.e)
        t.absorb(DS_TREE_Z, z2v)
    var alpha = t.elements(1)
    var root_q = r.take(H.DIGEST)
    t.absorb(DS_TREE_Q, root_q)
    var q3 = List[UInt8]()
    if shape.accumulators() > 0:
        q3 = r.take(2 * p.h2() * p.e)
        t.absorb(DS_TREE_Q, q3)
    var z = t.elements(2)

    # openings -> beta, gamma
    var openings = r.take(shape.points * shape.columns() * p.e)
    t.absorb(DS_OPENINGS, openings)
    var beta_gamma = t.elements(shape.columns() + shape.points)

    var one = ext_embed[4](SIMD[DType.uint8, 1](1))
    # steps 3 and 6, the accumulator boundaries (spec 7.1, 7.3 for (P)): Z(1, z2) = 1 from the opening at
    # (1, z2) (point 2); Z2(1) = 1; Z2(e2) Z(e1, e2) N(e1, e2) = D(e1, e2) from the openings at (e1, e2)
    # (point 6). The chain-end pairs (W) themselves are the small grid's R2 = Q3 (X2^h2 - 1).
    var gamma1 = list_e(stage1, 2)
    for k in range(shape.accumulators()):
        var z_col = get_u16(shape.accs, k * ACC)
        if _coords_at[p](openings, shape, 2, z_col) != one:
            raise Error("accumulator chain start is not 1")
        if list_e(z2v, k * p.h2()) != one:
            raise Error("Z2(1) is not 1")
        var lhs = ext_mul[4](ext_mul[4](list_e(z2v, k * p.h2() + p.h2() - 1), _coords_at[p](openings, shape, 6, z_col)),
                             _factor_at[p](openings, shape, 6, shape.accs, k, gamma1, False))
        if lhs != _factor_at[p](openings, shape, 6, shape.accs, k, gamma1, True):
            raise Error("accumulator grand product is not 1")

    # step 5: residual identity at z from the openings: R(z) = (A + z2^h2 B)(z1^h1 - 1) + Q2 (z2^h2 - 1)
    var d = Domains.__init__[p]()
    var pts = shift_points(families)
    var reads = List[E]()
    for k in range(shape.entries):
        var en = entry(families, k)
        reads.append(_opening[p](openings, shape, point_index(pts, en.dj1_a, en.dj2_a), en.col_a))
        reads.append(E(0) if en.col_b == NONE else _opening[p](openings, shape, point_index(pts, en.dj1_b, en.dj2_b), en.col_b))
    var z1 = list_e(z, 0)
    var z2 = list_e(z, 1)

    # step 3, the small grid (spec 7.4): R2(z2) = Q3(z2) (z2^h2 - 1), R2 from Z2 interpolated at z2 and
    # omega2 z2, Q3 interpolated on G2, and the openings at (e1, z2) (point 3)
    if shape.accumulators() > 0:
        var e2 = ext_embed[4](ext_pow[1](d.omega2, p.h2() - 1))
        var w2 = ext_embed[4](d.omega2)
        var r2 = E(0)
        for k in range(shape.accumulators()):
            var za = interp_cyclic(z2v, k * p.h2(), p.h2(), d.omega2, z2)
            var zb = interp_cyclic(z2v, k * p.h2(), p.h2(), d.omega2, ext_mul[4](z2, w2))
            var c = _coords_at[p](openings, shape, 3, get_u16(shape.accs, k * ACC))
            var n_z = _factor_at[p](openings, shape, 3, shape.accs, k, gamma1, False)
            var d_z = _factor_at[p](openings, shape, 3, shape.accs, k, gamma1, True)
            var term = ext_mul[4](f_sub(z2, e2), f_sub(ext_mul[4](zb, d_z), ext_mul[4](ext_mul[4](za, c), n_z)))
            r2 = f_add(r2, ext_mul[4](ext_pow[4](list_e(alpha, 0), k), term))
        var q3z = interp_cyclic(q3, 0, 2 * p.h2(), d.g2, z2)
        if r2 != ext_mul[4](q3z, f_sub(ext_pow[4](z2, p.h2()), one)):
            raise Error("small grid identity fails at z2")
    var rz = residual_at(families, list_e(alpha, 0), stage1, z1, z2, ext_pow[1](d.omega1, p.h1() - 1), ext_pow[1](d.omega2, p.h2() - 1), reads)
    var z2h = ext_pow[4](z2, p.h2())
    var qa = _quotient_at[p](openings, shape, 0)
    var qb = _quotient_at[p](openings, shape, 1)
    var q2 = _quotient_at[p](openings, shape, 2)
    var rhs = f_add(ext_mul[4](f_add(qa, ext_mul[4](z2h, qb)), f_sub(ext_pow[4](z1, p.h1()), one)),
                    ext_mul[4](q2, f_sub(z2h, one)))
    if rz != rhs:
        raise Error("residual identity fails at z")

    # step 7: the tail. The running claim starts as <y_2, sum_p gamma_p w_{z_p}> = sum beta_c gamma_p alpha_{c,p}.
    comptime assert p.n_cw() == 1, "one codeword per column: rows are (s, column, 4)"   # ponytail: split with the encoder's
    var running = List[UInt8](length=p.N() * p.e, fill=0)
    var running_val = E(0)
    for pt in range(shape.points):
        var dj1 = Int(pts[pt * 4]) | Int(pts[pt * 4 + 1]) << 8
        var dj2 = Int(pts[pt * 4 + 2]) | Int(pts[pt * 4 + 3]) << 8
        var z1p = point_coord(z1, dj1, d.g1, p.h1())
        var z2p = point_coord(z2, dj2, d.g2, p.h2())
        var gamma = list_e(beta_gamma, shape.columns() + pt)
        for slot in range(p.N()):
            _add_e(running, slot, ext_mul[4](gamma, slot_weight[p](slot, z1p, z2p, d.rho1, d.rho2)))
        var claim = E(0)
        for c in range(shape.columns()):
            claim = f_add(claim, ext_mul[4](list_e(beta_gamma, c), _opening[p](openings, shape, pt, c)))
        running_val = f_add(running_val, ext_mul[4](gamma, claim))

    var y_len = p.N()
    var r_prev = List[UInt8]()                 # r of the last committed level, empty while that is level 1
    var roots = List[List[UInt8]]()
    var doms = List[RsDomain]()
    for i in range(len(shape.tail)):
        var lvl = shape.tail[i]
        var root = r.take(H.DIGEST)
        t.absorb(DS_TAIL_ROOT, root)
        var prev = _open_previous[p, H](r, t, shape, i, root_w, root_z, root_q, roots)
        var count = p.queries() if i == 0 else shape.tail[i - 1].queries
        var v_count = 4 * count if i == 0 else count
        # the expected symbols v (9.3) from the opened rows; a function of the transcript, so not sent nor absorbed
        var v = List[UInt8](capacity=v_count * p.e)
        for q in range(count):
            var idx = _index_of(prev.opened, prev.positions[q])
            if i == 0:
                for tau in range(4):
                    _push_e(v, _level1_symbol[p](prev, shape, beta_gamma, idx, tau))
            else:
                _push_e(v, _tail_symbol(prev, idx, r_prev))
        var batch = t.elements(v_count + 1)
        var claim = ext_mul[4](list_e(batch, 0), running_val)
        for k in range(v_count):
            claim = f_add(claim, ext_mul[4](list_e(batch, 1 + k), list_e(v, k)))
        var w_tilde = _materialize[p](i == 0, running, y_len, batch, prev, count, d.level1 if i == 0 else doms[i - 1])
        var rounds = r.take(9 * p.e)
        var r_l = List[UInt8]()
        for dgt in range(3):
            if f_add(list_e(rounds, 3 * dgt), list_e(rounds, 3 * dgt + 1)) != claim:
                raise Error("sumcheck fails at a tail level")
            var msg = List[UInt8](capacity=3 * p.e)
            for b in range(3 * p.e):
                msg.append(rounds[3 * dgt * p.e + b])
            t.absorb(DS_TAIL_ROUND, msg)
            var rd = t.elements(1)
            claim = quadratic_at(rounds, 3 * dgt, list_e(rd, 0))
            r_l.extend(rd^)
        running = fold8_host(w_tilde, lvl.rows, r_l)
        running_val = claim
        r_prev = r_l^
        y_len = lvl.rows
        roots.append(root^)
        doms.append(RsDomain(lvl.L // lvl.cosets, lvl.cosets))

    # the clear vector: consistency against the last committed level, then the evaluation claim directly
    if shape.clear_length != y_len:
        raise Error("shape.clear_length does not match the tail schedule")
    var y = r.take(shape.clear_length * p.e)
    t.absorb(DS_CLEAR, y)
    var last = _open_previous[p, H](r, t, shape, len(shape.tail), root_w, root_z, root_q, roots)
    r.done()
    for idx in range(len(last.opened)):
        if len(shape.tail) == 0:
            var enc = encode_at[p](y, d.level1.point(last.opened[idx]))
            for tau in range(4):
                if enc[tau] != _level1_symbol[p](last, shape, beta_gamma, idx, tau):
                    raise Error("consistency fails at an opened position")
        else:
            var dom = doms[len(doms) - 1]
            if tail_encode_at(y, y_len, dom.point(last.opened[idx])) != _tail_symbol(last, idx, r_prev):
                raise Error("consistency fails at an opened position")
    var lhs = E(0)
    for slot in range(y_len):
        lhs = f_add(lhs, ext_mul[4](list_e(running, slot), list_e(y, slot)))
    if lhs != running_val:
        raise Error("evaluation claim fails")
    return True


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


def _level1_symbol[p: Params](o: Opened, shape: Shape, beta: List[UInt8], idx: Int, tau: Int) -> E:
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


def _tail_symbol(o: Opened, idx: Int, r_prev: List[UInt8]) -> E:
    """<X[s, :], r_bar> for a tail row of 8 E symbols."""
    var rr = host_r3(r_prev)
    var acc = E(0)
    for a in range(8):
        acc = f_add(acc, ext_mul[4](rbar_at(rr, a, 3), list_e(o.rows_w, idx * 8 + a)))
    return acc


def _materialize[p: Params](level1: Bool, running: List[UInt8], length: Int, batch: List[UInt8], o: Opened, count: Int, dom: RsDomain) -> List[UInt8]:
    """w~ = batch_0 running + sum_q batch_q g_q as a vector (tail.mojo's kernels on the host)."""
    var w = List[UInt8](length=length * p.e, fill=0)
    var b0 = list_e(batch, 0)
    for n in range(length):
        _add_e(w, n, ext_mul[4](b0, list_e(running, n)))
    if level1:
        comptime K = p.N() // 4
        var pw_all = List[UInt8](length=count * K * 4, fill=0)     # pt_q^i for every q and i
        for q in range(count):
            var pt = dom.point(o.positions[q])
            var pw = F4(1, 0, 0, 0)
            for i in range(K):
                for c in range(4):
                    pw_all[(q * K + i) * 4 + c] = pw[c]
                pw = ext_mul[2](pw, pt)
        for slot in range(length):
            var i: Int
            var j: Int
            i, j = pack_index[p](slot)
            var bj = F4(0)
            bj[j] = 1
            var acc = E(0)
            for q in range(count):
                var at = (q * K + i) * 4
                var m = ext_mul[2](bj, F4(pw_all[at], pw_all[at + 1], pw_all[at + 2], pw_all[at + 3]))
                for tau in range(4):
                    acc = f_add(acc, f_mul(list_e(batch, 1 + 4 * q + tau), E(m[tau])))
            _add_e(w, slot, acc)
    else:
        for q in range(count):
            var pt = dom.point(o.positions[q])
            var bq = list_e(batch, 1 + q)
            var pw = F4(1, 0, 0, 0)
            for row in range(length):
                _add_e(w, row, e_mul_f4(bq, pw))
                pw = ext_mul[2](pw, pt)
    return w^


def _push_e(mut l: List[UInt8], v: E):
    for t in range(16):
        l.append(v[t])


def _add_e(mut l: List[UInt8], i: Int, v: E):
    var s = f_add(list_e(l, i), v)
    for t in range(16):
        l[i * 16 + t] = s[t]


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
            var v = list_e(y, pack_slot[p](i, j))
            for tau in range(4):
                acc[tau] = f_add(acc[tau], f_mul(v, E(m[tau])))
        pw = ext_mul[2](pw, pt)
    return acc^



def _opening[p: Params](openings: List[UInt8], shape: Shape, point: Int, column: Int) -> E:
    return list_e(openings, point * shape.columns() + column)


def _factor_at[p: Params](openings: List[UInt8], shape: Shape, point: Int, accs: List[UInt8], k: Int, gamma: E, den: Bool) -> E:
    """N or D of accumulator k at an opening point: gamma + sum_j b_j c_j(point) (accumulate.mojo)."""
    var v = gamma
    for j in range(get_u16(accs, k * ACC + (4 if den else 2))):
        var b = E(0)
        b[j] = 1
        v = f_add(v, ext_mul[4](b, _opening[p](openings, shape, point, get_u16(accs, k * ACC + (22 if den else 6) + 2 * j))))
    return v


def _coords_at[p: Params](openings: List[UInt8], shape: Shape, point: Int, col0: Int) -> E:
    """An E-valued column at a point from its e coordinate columns: sum_tau b_tau <w_z, coord_tau>."""
    var acc = E(0)
    for tau in range(p.e):
        var basis = E(0)
        basis[tau] = 1
        acc = f_add(acc, ext_mul[4](basis, _opening[p](openings, shape, point, col0 + tau)))
    return acc


def _quotient_at[p: Params](openings: List[UInt8], shape: Shape, q: Int) -> E:
    """Q(z) for Q in (A, B, Q2) from the quotient coordinate columns at point 0."""
    return _coords_at[p](openings, shape, 0, shape.columns_w + shape.columns_z + q * p.e)

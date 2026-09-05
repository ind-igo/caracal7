"""Tail levels (design section 4, spec 9.3): the level-l message y_l (slot, e) has index 8 row + a,
a in [8) the three lowest binary digits. Per level: encode Mat(y_l) (8 E-valued columns seen as 32
F4 columns) on the level's RS domain; the expected symbols v of the previous level's opened
positions; the batched query w~ = batch_0 running + sum_q batch_q g_q; three sumcheck rounds over
the digits; the fold y_{l+1} = Mat(y_l) r_bar and the folded query, the next level's running claim.

Functionals g_q of the previous level at leaf point pt_q:
    level 1 (E (x) F4 alphabet, 9.1)   g_{q,tau}[slot(i, j)] = coord_tau(b_j pt_q^i),  four per position
    tail level (E-linear)              g_q[row] = pt_q^row
Every kernel here is rung 1: one thread per output, no shared memory. The host helpers at the end
are the verifier's side of the same formulas.
"""

from std.math import ceildiv
from std.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext

from caracal7.field import F4, E, f_add, f_sub, f_mul, ext_mul, ext_pow, ext_embed
from caracal7.params import Params
from caracal7.tables import RsTables, RsDomain
from caracal7.encode import rs_encode_on, pack_index

comptime BLOCK = 256
comptime ROUND_THREADS = 1024      # partial sums of one sumcheck round
comptime DOM_BYTES = 20            # an RsDomain in the arena: g (4), then gamma4^k for k < 4


@always_inline
def _gid() -> Int:
    return Int(block_idx.x * block_dim.x + thread_idx.x)


@always_inline
def _e(base: Pointer[UInt8, MutAnyOrigin], off: Int) -> E:
    return base.unsafe_load[width=16](off)


@always_inline
def _f4(base: Pointer[UInt8, MutAnyOrigin], off: Int) -> F4:
    return base.unsafe_load[width=4](off)


@always_inline
def _u32(base: Pointer[UInt8, MutAnyOrigin], off: Int) -> Int:
    var v = base.unsafe_load[width=4](off)
    return Int(v[0]) | Int(v[1]) << 8 | Int(v[2]) << 16 | Int(v[3]) << 24


@always_inline
def e_mul_f4(v: E, w: F4) -> E:
    """E x F4: E is F4^4 in the tower, so each 4-byte slice is multiplied by w."""
    var out = E(0)
    comptime for k in range(4):
        var s = ext_mul[2](v.slice[4, offset=4 * k](), w)
        comptime for c in range(4):
            out[4 * k + c] = s[c]
    return out^


@always_inline
def one_e() -> E:
    var v = E(0)
    v[0] = 1
    return v


@always_inline
def rbar_at(r: InlineArray[E, 3], a: Int, digits: Int) -> E:
    """prod_{i < digits} (bit i of a ? r_i : 1 - r_i): the fold weight of digit value a."""
    var w = one_e()
    for i in range(digits):
        w = ext_mul[4](w, r[i] if (a >> i) & 1 else f_sub(one_e(), r[i]))
    return w


@always_inline
def _r3(base: Pointer[UInt8, MutAnyOrigin], r: Int) -> InlineArray[E, 3]:
    var out = InlineArray[E, 3](fill=E(0))
    for i in range(3):
        out[i] = _e(base, r + i * 16)
    return out^


def domain_bytes(dom: RsDomain) -> List[UInt8]:
    var out = List[UInt8](capacity=DOM_BYTES)
    for c in range(4):
        out.append(dom.g[c])
    for i in range(16):
        out.append(dom.gk[i])
    return out^


# ---- kernels ----

def k_points(base: Pointer[UInt8, MutAnyOrigin], positions: Int64, count: Int32, dom: Int64, L0: Int32, pts: Int64):
    """pts[q] = the point of leaf positions[q]: gamma4^(s // L0) g^(s mod L0)."""
    var q = _gid()
    if q >= Int(count):
        return
    var s = _u32(base, Int(positions) + 4 * q)
    var pt = ext_mul[2](_f4(base, Int(dom) + 4 + 4 * (s // Int(L0))), ext_pow[2](_f4(base, Int(dom)), s % Int(L0)))
    base.unsafe_store[width=4](Int(pts) + 4 * q, pt)


def k_running0[p: Params](base: Pointer[UInt8, MutAnyOrigin], w_z: Int64, gamma: Int64, P: Int32, dst: Int64):
    """The level-2 running query: sum_p gamma_p w_{z_p}."""
    comptime N = p.N()
    var slot = _gid()
    if slot >= N:
        return
    var acc = E(0)
    for pt in range(Int(P)):
        acc = f_add(acc, ext_mul[4](_e(base, Int(gamma) + pt * 16), _e(base, Int(w_z) + (pt * N + slot) * 16)))
    base.unsafe_store[width=16](Int(dst) + slot * 16, acc)


def k_expected_level1(base: Pointer[UInt8, MutAnyOrigin], positions: Int64, count: Int32, code_w: Int64, columns_w: Int32,
                      code_q: Int64, columns_q: Int32, beta: Int64, v: Int64):
    """v[4 q + tau] = coord_tau(Enc(y)(pt_q)) = sum_c beta_c X[s_q, c][tau]: the encoder is linear, so the
    symbol of the folded message is the beta-combination of the committed rows (the verifier's check)."""
    var gid = _gid()
    if gid >= 4 * Int(count):
        return
    var q = gid // 4
    var tau = gid % 4
    var s = _u32(base, Int(positions) + 4 * q)
    var acc = E(0)
    for c in range(Int(columns_w)):
        acc = f_add(acc, f_mul(_e(base, Int(beta) + c * 16), E(base[unsafe_offset=Int(code_w) + (s * Int(columns_w) + c) * 4 + tau])))
    for c in range(Int(columns_q)):
        acc = f_add(acc, f_mul(_e(base, Int(beta) + (Int(columns_w) + c) * 16), E(base[unsafe_offset=Int(code_q) + (s * Int(columns_q) + c) * 4 + tau])))
    base.unsafe_store[width=16](Int(v) + gid * 16, acc)


def k_expected_tail(base: Pointer[UInt8, MutAnyOrigin], positions: Int64, count: Int32, code: Int64, r: Int64, v: Int64):
    """v[q] = Enc(fold(y))(pt_q) = sum_a rbar[a] X[s_q, a] over the 8 committed E symbols of the row."""
    var q = _gid()
    if q >= Int(count):
        return
    var s = _u32(base, Int(positions) + 4 * q)
    var rr = _r3(base, Int(r))
    var acc = E(0)
    for a in range(8):
        acc = f_add(acc, ext_mul[4](rbar_at(rr, a, 3), _e(base, Int(code) + (s * 8 + a) * 16)))
    base.unsafe_store[width=16](Int(v) + q * 16, acc)


def k_materialize_level1[p: Params](base: Pointer[UInt8, MutAnyOrigin], running: Int64, batch: Int64, pts: Int64,
                                    count: Int32, w_tilde: Int64):
    """w~[slot] = batch_0 running[slot] + sum_{q, tau} batch_{1 + 4 q + tau} coord_tau(b_j pt_q^i), (i, j) = pack_index(slot)."""
    comptime N = p.N()
    var slot = _gid()
    if slot >= N:
        return
    var w = ext_mul[4](_e(base, Int(batch)), _e(base, Int(running) + slot * 16))
    var i: Int
    var j: Int
    i, j = pack_index[p](slot)
    var bj = F4(0)
    bj[j] = 1
    # ponytail: pt_q^i by a pow per (slot, q); a (q, i) power table when the tail shows in the profile
    for q in range(Int(count)):
        var m = ext_mul[2](bj, ext_pow[2](_f4(base, Int(pts) + 4 * q), i))
        comptime for tau in range(4):
            w = f_add(w, f_mul(_e(base, Int(batch) + (1 + 4 * q + tau) * 16), E(m[tau])))
    base.unsafe_store[width=16](Int(w_tilde) + slot * 16, w)


def k_materialize_tail(base: Pointer[UInt8, MutAnyOrigin], running: Int64, batch: Int64, pts: Int64,
                       count: Int32, rows: Int32, w_tilde: Int64):
    """w~[row] = batch_0 running[row] + sum_q batch_{1 + q} pt_q^row."""
    var row = _gid()
    if row >= Int(rows):
        return
    var w = ext_mul[4](_e(base, Int(batch)), _e(base, Int(running) + row * 16))
    for q in range(Int(count)):
        w = f_add(w, e_mul_f4(_e(base, Int(batch) + (1 + q) * 16), ext_pow[2](_f4(base, Int(pts) + 4 * q), row)))
    base.unsafe_store[width=16](Int(w_tilde) + row * 16, w)


def k_round_partial(base: Pointer[UInt8, MutAnyOrigin], w_tilde: Int64, y: Int64, length: Int32, d: Int32, r: Int64, partial: Int64):
    """Round d of the partial sumcheck over the three low digits: digits below d are bound to r_0 ..
    r_{d-1}, digit d is the variable b, everything above is summed. Thread t sums its share of the
    groups (row, digits above d) into partial[t] = (s(0), s(1), s(2))."""
    var t = _gid()
    if t >= ROUND_THREADS:
        return
    var dd = Int(d)
    var m = 1 << dd
    var rr = _r3(base, Int(r))
    var rb = InlineArray[E, 4](fill=E(0))
    for a in range(m):
        rb[a] = rbar_at(rr, a, dd)
    var acc0 = E(0)
    var acc1 = E(0)
    var acc2 = E(0)
    var groups = Int(length) // (2 * m)
    for grp in range(t, groups, ROUND_THREADS):
        var n0 = grp * 2 * m
        var y0 = E(0)
        var y1 = E(0)
        var w0 = E(0)
        var w1 = E(0)
        for a in range(m):
            y0 = f_add(y0, ext_mul[4](rb[a], _e(base, Int(y) + (n0 + a) * 16)))
            y1 = f_add(y1, ext_mul[4](rb[a], _e(base, Int(y) + (n0 + m + a) * 16)))
            w0 = f_add(w0, ext_mul[4](rb[a], _e(base, Int(w_tilde) + (n0 + a) * 16)))
            w1 = f_add(w1, ext_mul[4](rb[a], _e(base, Int(w_tilde) + (n0 + m + a) * 16)))
        acc0 = f_add(acc0, ext_mul[4](y0, w0))
        acc1 = f_add(acc1, ext_mul[4](y1, w1))
        acc2 = f_add(acc2, ext_mul[4](f_sub(f_add(y1, y1), y0), f_sub(f_add(w1, w1), w0)))
    base.unsafe_store[width=16](Int(partial) + (3 * t) * 16, acc0)
    base.unsafe_store[width=16](Int(partial) + (3 * t + 1) * 16, acc1)
    base.unsafe_store[width=16](Int(partial) + (3 * t + 2) * 16, acc2)


def k_round_sum(base: Pointer[UInt8, MutAnyOrigin], partial: Int64, dst: Int64):
    """One thread: the round message s = (s(0), s(1), s(2)) from the partial sums."""
    comptime for b in range(3):
        var acc = E(0)
        for t in range(ROUND_THREADS):
            acc = f_add(acc, _e(base, Int(partial) + (3 * t + b) * 16))
        base.unsafe_store[width=16](Int(dst) + b * 16, acc)


def k_fold8(base: Pointer[UInt8, MutAnyOrigin], src: Int64, rows: Int32, r: Int64, dst: Int64):
    """dst[row] = sum_{a < 8} rbar[a] src[8 row + a], rbar = (x) (1 - r_i, r_i)."""
    var row = _gid()
    if row >= Int(rows):
        return
    var rr = _r3(base, Int(r))
    var acc = E(0)
    for a in range(8):
        acc = f_add(acc, ext_mul[4](rbar_at(rr, a, 3), _e(base, Int(src) + (8 * row + a) * 16)))
    base.unsafe_store[width=16](Int(dst) + row * 16, acc)


# ---- host launchers ----

@always_inline
def _grid(n: Int) -> Int:
    return ceildiv(n, BLOCK)


def tail_encode(ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin],
                y: Int, rows: Int, L0: Int, m: Int, etmp: Int, code: Int, rs: RsTables) raises:
    """Mat(y) (rows, 8, e) -> code (m L0, 8, e): the RS encoder on 32 F4 columns, no inverse."""
    rs_encode_on(ctx, base, y, etmp, code, 32, rows, L0, m, rs)


def points(ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin], positions: Int, count: Int, dom: Int, L0: Int, pts: Int) raises:
    ctx.enqueue_function[k_points](base, Int64(positions), Int32(count), Int64(dom), Int32(L0), Int64(pts),
                                   grid_dim=_grid(count), block_dim=BLOCK)


def running0[p: Params](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin], w_z: Int, gamma: Int, P: Int, dst: Int) raises:
    ctx.enqueue_function[k_running0[p]](base, Int64(w_z), Int64(gamma), Int32(P), Int64(dst),
                                        grid_dim=_grid(p.N()), block_dim=BLOCK)


def expected_level1(ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin], positions: Int, count: Int,
                    code_w: Int, columns_w: Int, code_q: Int, columns_q: Int, beta: Int, v: Int) raises:
    """v (4 per opened level-1 position) from the committed rows of both trees and beta."""
    ctx.enqueue_function[k_expected_level1](base, Int64(positions), Int32(count), Int64(code_w), Int32(columns_w),
                                            Int64(code_q), Int32(columns_q), Int64(beta), Int64(v),
                                            grid_dim=_grid(4 * count), block_dim=BLOCK)


def expected_tail(ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin], positions: Int, count: Int, code: Int, r: Int, v: Int) raises:
    """v (1 per opened tail position) from the committed rows and the r of that level."""
    ctx.enqueue_function[k_expected_tail](base, Int64(positions), Int32(count), Int64(code), Int64(r), Int64(v),
                                          grid_dim=_grid(count), block_dim=BLOCK)


def tail_materialize[p: Params](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin], level1: Bool,
                                running: Int, batch: Int, pts: Int, count: Int, length: Int, w_tilde: Int) raises:
    if level1:
        ctx.enqueue_function[k_materialize_level1[p]](base, Int64(running), Int64(batch), Int64(pts), Int32(count), Int64(w_tilde),
                                                      grid_dim=_grid(p.N()), block_dim=BLOCK)
    else:
        ctx.enqueue_function[k_materialize_tail](base, Int64(running), Int64(batch), Int64(pts), Int32(count), Int32(length), Int64(w_tilde),
                                                 grid_dim=_grid(length), block_dim=BLOCK)


def tail_round(ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin],
               w_tilde: Int, y: Int, length: Int, digit: Int, r: Int, partial: Int, dst: Int) raises:
    """dst (3, e) = the round message of digit `digit` given r_0 .. r_{digit-1} at `r`."""
    ctx.enqueue_function[k_round_partial](base, Int64(w_tilde), Int64(y), Int32(length), Int32(digit), Int64(r), Int64(partial),
                                          grid_dim=_grid(ROUND_THREADS), block_dim=BLOCK)
    ctx.enqueue_function[k_round_sum](base, Int64(partial), Int64(dst), grid_dim=1, block_dim=1)


def tail_fold(ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin], src: Int, rows: Int, r: Int, dst: Int) raises:
    """dst = Mat(src) r_bar, for the message and for the query."""
    ctx.enqueue_function[k_fold8](base, Int64(src), Int32(rows), Int64(r), Int64(dst), grid_dim=_grid(rows), block_dim=BLOCK)


# ---- host side of the same formulas (verifier, tests) ----

def host_e(l: List[UInt8], i: Int) -> E:
    var v = E(0)
    for t in range(16):
        v[t] = l[i * 16 + t]
    return v


def host_r3(r: List[UInt8]) -> InlineArray[E, 3]:
    var out = InlineArray[E, 3](fill=E(0))
    for i in range(3):
        out[i] = host_e(r, i)
    return out^


def tail_encode_at(y: List[UInt8], rows: Int, pt: F4) -> E:
    """Enc(y)(pt) over E: sum_row y[row] pt^row."""
    var acc = E(0)
    var pw = F4(1, 0, 0, 0)
    for row in range(rows):
        acc = f_add(acc, e_mul_f4(host_e(y, row), pw))
        pw = ext_mul[2](pw, pt)
    return acc


def fold8_host(src: List[UInt8], rows: Int, r: List[UInt8]) -> List[UInt8]:
    var rr = host_r3(r)
    var out = List[UInt8](capacity=rows * 16)
    for row in range(rows):
        var acc = E(0)
        for a in range(8):
            acc = f_add(acc, ext_mul[4](rbar_at(rr, a, 3), host_e(src, 8 * row + a)))
        for t in range(16):
            out.append(acc[t])
    return out^


def quadratic_at(s: List[UInt8], off: Int, r: E) -> E:
    """The degree-2 polynomial with values s[off], s[off + 1], s[off + 2] at 0, 1, 2, evaluated at r:
    s0 (r - 1)(r - 2) / 2 - s1 r (r - 2) + s2 r (r - 1) / 2."""
    var one = one_e()
    var two = f_add(one, one)
    var half = E(0)
    half[0] = 64
    var r1 = f_sub(r, one)
    var r2 = f_sub(r, two)
    var t0 = ext_mul[4](ext_mul[4](host_e(s, off), ext_mul[4](r1, r2)), half)
    var t1 = ext_mul[4](host_e(s, off + 1), ext_mul[4](r, r2))
    var t2 = ext_mul[4](ext_mul[4](host_e(s, off + 2), ext_mul[4](r, r1)), half)
    return f_add(f_sub(t0, t1), t2)

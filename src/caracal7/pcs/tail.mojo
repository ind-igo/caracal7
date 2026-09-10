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
from std.gpu import global_idx, thread_idx
from max.gpu.host import DeviceContext

from caracal7.core.field import F4, E, E_WIDTH, f_add, f_sub, f_mul, ext_mul, ext_pow, ext_embed, ext_one
from caracal7.core.params import Params
from caracal7.core.tables import RsTables, RsDomain
from caracal7.pcs.encode import rs_encode_on, pack_index
from caracal7.core.backend import BACKEND
from caracal7.core.bytes import Base, Buf, u32, list_e
from caracal7.core.arena import Arena

comptime ROUND_THREADS = 16384     # partial sums of one sumcheck round (1024 left the GPU idle: 42 ms a round at N = 82,944)
comptime DOM_BYTES = 20            # an RsDomain in the arena: g (4), then gamma4^k for k < 4


@always_inline
def e_mul_f4(v: E, w: F4) -> E:
    """E x F4: E is F4^4 in the tower, so each 4-byte slice is multiplied by w."""
    var out = E(0)
    comptime for k in range(4):
        var s = ext_mul[2](v.slice[4, offset=4 * k](), w)
        comptime for c in range(4):
            out[4 * k + c] = s[c]
    return out


@always_inline
def rbar_at(r: InlineArray[E, 3], a: Int, digits: Int) -> E:
    """prod_{i < digits} (bit i of a ? r_i : 1 - r_i): the fold weight of digit value a."""
    var w = ext_one[4]()
    for i in range(digits):
        w = ext_mul[4](w, r[i] if (a >> i) & 1 else f_sub(ext_one[4](), r[i]))
    return w


@always_inline
def _r3(base: Base, r: Buf[16]) -> InlineArray[E, 3]:
    var out = InlineArray[E, 3](fill=E(0))
    for i in range(3):
        out[i] = r.load(base, i)
    return out^


def domain_bytes(dom: RsDomain) -> List[UInt8]:
    var out = List[UInt8](capacity=DOM_BYTES)
    for c in range(4):
        out.append(dom.g[c])
    for i in range(16):
        out.append(dom.gk[i])
    return out^


# ---- kernels ----

def k_points(base: Base, positions: Buf[4], count: Int32, dom: Buf[4], L0: Int32, pts: Buf[4]):
    """pts[q] = the point of leaf positions[q]: gamma4^(s // L0) g^(s mod L0)."""
    var q = global_idx.x
    if q >= Int(count):
        return
    var s = u32(base, positions.at(q))
    var pt = ext_mul[2](dom.load(base, 1 + s // Int(L0)), ext_pow[2](dom.load(base, 0), s % Int(L0)))
    pts.store(base, q, pt)


def k_running0[p: Params](base: Base, w_z: Buf[16], gamma: Buf[16], P: Int32, dst: Buf[16]):
    """The level-2 running query: sum_p gamma_p w_{z_p}."""
    comptime N = p.N()
    var slot = global_idx.x
    if slot >= N:
        return
    var acc = E(0)
    for pt in range(Int(P)):
        acc = f_add(acc, ext_mul[4](gamma.load(base, pt), w_z.load(base, slot * Int(P) + pt)))
    dst.store(base, slot, acc)


comptime POW_LO = 256              # pt^i = pt^(i & 255) pt^(256 (i >> 8)): the level-1 power table holds both factors


@always_inline
def power_table_len[p: Params]() -> Int:
    """Entries per query of the level-1 power table: the low factors, then the high ones."""
    return POW_LO + ceildiv(p.N() // 4, POW_LO)


def k_power_table[p: Params](base: Base, pts: Buf[4], count: Int32, ptab: Buf[4]):
    """ptab[q, k] = pt_q^k for k < POW_LO, then pt_q^(POW_LO (k - POW_LO))."""
    comptime TAB = power_table_len[p]()
    var gid = global_idx.x
    if gid >= Int(count) * TAB:
        return
    var k = gid % TAB
    var n = k if k < POW_LO else POW_LO * (k - POW_LO)
    ptab.store(base, gid, ext_pow[2](pts.load(base, gid // TAB), n))


def k_materialize_level1[p: Params](base: Base, running: Buf[16], batch: Buf[16], ptab: Buf[4],
                                    count: Int32, w_tilde: Buf[16]):
    """w~[slot] = batch_0 running[slot] + sum_{q, tau} batch_{1 + 4 q + tau} coord_tau(b_j pt_q^i), (i, j) = pack_index(slot)."""
    comptime N = p.N()
    comptime TAB = power_table_len[p]()
    var slot = global_idx.x
    if slot >= N:
        return
    var w = ext_mul[4](batch.load(base, 0), running.load(base, slot))
    var i: Int
    var j: Int
    i, j = pack_index[p](slot)
    var bj = F4(0)
    bj[j] = 1
    var lo = i % POW_LO
    var hi = POW_LO + i // POW_LO
    for q in range(Int(count)):
        var pw = ext_mul[2](ptab.load(base, q * TAB + lo), ptab.load(base, q * TAB + hi))
        var m = ext_mul[2](bj, pw)
        comptime for tau in range(4):
            w = f_add(w, f_mul(batch.load(base, 1 + 4 * q + tau), E(m[tau])))
    w_tilde.store(base, slot, w)


def k_materialize_tail(base: Base, running: Buf[16], batch: Buf[16], pts: Buf[4],
                       count: Int32, rows: Int32, w_tilde: Buf[16]):
    """w~[row] = batch_0 running[row] + sum_q batch_{1 + q} pt_q^row."""
    var row = global_idx.x
    if row >= Int(rows):
        return
    var w = ext_mul[4](batch.load(base, 0), running.load(base, row))
    for q in range(Int(count)):
        w = f_add(w, e_mul_f4(batch.load(base, 1 + q), ext_pow[2](pts.load(base, q), row)))
    w_tilde.store(base, row, w)


def k_round_partial(base: Base, w_tilde: Buf[16], y: Buf[16], length: Int32, d: Int32, r: Buf[16], partial: Buf[16]):
    """Round d of the partial sumcheck over the three low digits: digits below d are bound to r_0 ..
    r_{d-1}, digit d is the variable b, everything above is summed. Thread t sums its share of the
    groups (row, digits above d) into partial[t] = (s(0), s(1), s(2))."""
    var t = global_idx.x
    if t >= ROUND_THREADS:
        return
    var dd = Int(d)
    var m = 1 << dd
    var rr = _r3(base, r)
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
            y0 = f_add(y0, ext_mul[4](rb[a], y.load(base, n0 + a)))
            y1 = f_add(y1, ext_mul[4](rb[a], y.load(base, n0 + m + a)))
            w0 = f_add(w0, ext_mul[4](rb[a], w_tilde.load(base, n0 + a)))
            w1 = f_add(w1, ext_mul[4](rb[a], w_tilde.load(base, n0 + m + a)))
        acc0 = f_add(acc0, ext_mul[4](y0, w0))
        acc1 = f_add(acc1, ext_mul[4](y1, w1))
        acc2 = f_add(acc2, ext_mul[4](f_sub(f_add(y1, y1), y0), f_sub(f_add(w1, w1), w0)))
    partial.store(base, 3 * t, acc0)
    partial.store(base, 3 * t + 1, acc1)
    partial.store(base, 3 * t + 2, acc2)


def k_round_sum(base: Base, partial: Buf[16], dst: Buf[16]):
    """48 threads, one per (evaluation b, byte l): the round message s = (s(0), s(1), s(2)) from the partial sums."""
    var i = Int(thread_idx.x)                        # one block
    if i >= 3 * E_WIDTH:
        return
    var acc = SIMD[DType.uint8, 1](0)
    for t in range(ROUND_THREADS):
        acc = f_add(acc, base.unsafe_load[width=1](partial.at(3 * t) + i))
    base.unsafe_store(dst.at(0) + i, acc)


def k_fold8(base: Base, src: Buf[16], rows: Int32, r: Buf[16], dst: Buf[16]):
    """dst[row] = sum_{a < 8} rbar[a] src[8 row + a], rbar = (x) (1 - r_i, r_i)."""
    var row = global_idx.x
    if row >= Int(rows):
        return
    var rr = _r3(base, r)
    var acc = E(0)
    for a in range(8):
        acc = f_add(acc, ext_mul[4](rbar_at(rr, a, 3), src.load(base, 8 * row + a)))
    dst.store(base, row, acc)


# ---- host launchers ----

@always_inline
def _grid(n: Int) -> Int:
    return ceildiv(n, BACKEND.block)


def tail_encode(ctx: DeviceContext, arena: Arena,
                y: Int, rows: Int, L0: Int, m: Int, etmp: Int, code: Int, rs: RsTables) raises:
    """Mat(y) (rows, 8, e) -> code (m L0, 8, e): the RS encoder on 32 F4 columns, no inverse."""
    rs_encode_on(ctx, arena, y, etmp, code, 32, rows, L0, m, rs)


def points(ctx: DeviceContext, arena: Arena, positions: Int, count: Int, dom: Int, L0: Int, pts: Int) raises:
    ctx.enqueue_function[k_points](arena.buf, Buf[4](positions), Int32(count), Buf[4](dom), Int32(L0), Buf[4](pts),
                                   grid_dim=_grid(count), block_dim=BACKEND.block)


def running0[p: Params](ctx: DeviceContext, arena: Arena, w_z: Int, gamma: Int, P: Int, dst: Int) raises:
    ctx.enqueue_function[k_running0[p]](arena.buf, Buf[16](w_z), Buf[16](gamma), Int32(P), Buf[16](dst),
                                        grid_dim=_grid(p.N()), block_dim=BACKEND.block)


def tail_materialize[p: Params](ctx: DeviceContext, arena: Arena, level1: Bool,
                                running: Int, batch: Int, pts: Int, count: Int, length: Int, w_tilde: Int, ptab: Int) raises:
    """Level 1 needs `ptab`, (count, power_table_len) F4 of scratch."""
    if level1:
        ctx.enqueue_function[k_power_table[p]](arena.buf, Buf[4](pts), Int32(count), Buf[4](ptab),
                                               grid_dim=_grid(count * power_table_len[p]()), block_dim=BACKEND.block)
        ctx.enqueue_function[k_materialize_level1[p]](arena.buf, Buf[16](running), Buf[16](batch), Buf[4](ptab), Int32(count), Buf[16](w_tilde),
                                                      grid_dim=_grid(p.N()), block_dim=BACKEND.block)
    else:
        ctx.enqueue_function[k_materialize_tail](arena.buf, Buf[16](running), Buf[16](batch), Buf[4](pts), Int32(count), Int32(length), Buf[16](w_tilde),
                                                 grid_dim=_grid(length), block_dim=BACKEND.block)


def tail_round(ctx: DeviceContext, arena: Arena,
               w_tilde: Int, y: Int, length: Int, digit: Int, r: Int, partial: Int, dst: Int) raises:
    """dst (3, e) = the round message of digit `digit` given r_0 .. r_{digit-1} at `r`."""
    ctx.enqueue_function[k_round_partial](arena.buf, Buf[16](w_tilde), Buf[16](y), Int32(length), Int32(digit), Buf[16](r), Buf[16](partial),
                                          grid_dim=_grid(ROUND_THREADS), block_dim=BACKEND.block)
    ctx.enqueue_function[k_round_sum](arena.buf, Buf[16](partial), Buf[16](dst), grid_dim=1, block_dim=64)


def tail_fold(ctx: DeviceContext, arena: Arena, src: Int, rows: Int, r: Int, dst: Int) raises:
    """dst = Mat(src) r_bar, for the message and for the query."""
    ctx.enqueue_function[k_fold8](arena.buf, Buf[16](src), Int32(rows), Buf[16](r), Buf[16](dst), grid_dim=_grid(rows), block_dim=BACKEND.block)


# ---- host side of the same formulas (verifier, tests) ----


def host_r3(r: Span[UInt8, _]) -> InlineArray[E, 3]:
    var out = InlineArray[E, 3](fill=E(0))
    for i in range(3):
        out[i] = list_e(r, i)
    return out^


def tail_encode_at(y: Span[UInt8, _], rows: Int, pt: F4) -> E:
    """Enc(y)(pt) over E: sum_row y[row] pt^row."""
    var acc = E(0)
    var pw = F4(1, 0, 0, 0)
    for row in range(rows):
        acc = f_add(acc, e_mul_f4(list_e(y, row), pw))
        pw = ext_mul[2](pw, pt)
    return acc


def fold8_host(src: Span[UInt8, _], rows: Int, r: Span[UInt8, _]) -> List[UInt8]:
    var rr = host_r3(r)
    var out = List[UInt8](capacity=rows * 16)
    for row in range(rows):
        var acc = E(0)
        for a in range(8):
            acc = f_add(acc, ext_mul[4](rbar_at(rr, a, 3), list_e(src, 8 * row + a)))
        for t in range(16):
            out.append(acc[t])
    return out^


def quadratic_at(s: Span[UInt8, _], off: Int, r: E) -> E:
    """The degree-2 polynomial with values s[off], s[off + 1], s[off + 2] at 0, 1, 2, evaluated at r:
    s0 (r - 1)(r - 2) / 2 - s1 r (r - 2) + s2 r (r - 1) / 2."""
    var one = ext_one[4]()
    var two = f_add(one, one)
    var half = E(0)
    half[0] = 64
    var r1 = f_sub(r, one)
    var r2 = f_sub(r, two)
    var t0 = ext_mul[4](ext_mul[4](list_e(s, off), ext_mul[4](r1, r2)), half)
    var t1 = ext_mul[4](list_e(s, off + 1), ext_mul[4](r, r2))
    var t2 = ext_mul[4](ext_mul[4](list_e(s, off + 2), ext_mul[4](r, r1)), half)
    return f_add(f_sub(t0, t1), t2)

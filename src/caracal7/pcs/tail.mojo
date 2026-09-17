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

from std.bit import log2_floor
from std.math import ceildiv
from std.gpu import global_idx, thread_idx, block_idx
from max.gpu.host import DeviceContext

from caracal7.core.field import F4, E, f_add, f_sub, f_mul, ext_mul, ext_pow, ext_embed, ext_one, fp_ext_mul, fp_reduce, fp_canonical, E_LEVEL, E_BYTES, EF, to_f32
from caracal7.core.params import Params
from caracal7.core.tables import RsTables, RsDomain
from caracal7.pcs.encode import rs_encode_on, pack_index, split_index
from caracal7.core.backend import BACKEND
from caracal7.core.bytes import Base, Buf, u32, list_e
from caracal7.core.arena import Arena

comptime ROUND_THREADS = 16384     # partial sums of one sumcheck round (1024 left the GPU idle: 42 ms a round at N = 82,944)
comptime ROUND_SUM_BLOCKS = 64     # first pass of the sum over the partials (one block walking all 16,384 was 0.85 ms)
comptime ROUND_ROWS = ROUND_THREADS + ROUND_SUM_BLOCKS   # rows of (3, e) in the partial buffer
comptime DOM_BYTES = 20            # an RsDomain in the arena: g (4), then gamma4^k for k < 4


@always_inline
def e_mul_f4(v: E, w: F4) -> E:
    """E x F4: E is a vector space over F4 with a 4-byte slice per coordinate, so each slice is multiplied by w."""
    var out = E(0)
    comptime for k in range(E_BYTES // 4):
        var s = ext_mul[2](v.slice[4, offset=4 * k](), w)
        comptime for c in range(4):
            out[4 * k + c] = s[c]
    return out


@always_inline
def rbar_at(r: InlineArray[E, 3], a: Int, digits: Int) -> E:
    """prod_{i < digits} (bit i of a ? r_i : 1 - r_i): the fold weight of digit value a."""
    var w = ext_one[E_LEVEL]()
    for i in range(digits):
        w = ext_mul[E_LEVEL](w, r[i] if (a >> i) & 1 else f_sub(ext_one[E_LEVEL](), r[i]))
    return w


@always_inline
def _r3(base: Base, r: Buf[E_BYTES]) -> InlineArray[E, 3]:
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


def k_running0[p: Params](base: Base, w_z: Buf[E_BYTES], gamma: Buf[E_BYTES], P: Int32, dst: Buf[E_BYTES]):
    """The level-2 running query: sum_p gamma_p w_{z_p}."""
    comptime N = p.N()
    var slot = global_idx.x
    if slot >= N:
        return
    var acc = EF(0)
    for pt in range(Int(P)):
        acc += fp_ext_mul[E_LEVEL](to_f32(gamma.load(base, pt)), to_f32(w_z.load(base, slot * Int(P) + pt)))
        if pt % 4 == 3:
            acc = fp_reduce(acc)
    dst.store(base, slot, fp_canonical(acc))


comptime POW_LO = 256              # pt^i = pt^(i & 255) pt^(256 (i >> 8)): the level-1 power table holds both factors


@always_inline
def table_len(exponents: Int) -> Int:
    """Entries per query of a power table for pt^i, i < exponents: the low factors, then the high ones."""
    return POW_LO + ceildiv(exponents, POW_LO)


@always_inline
def power_table_len[p: Params]() -> Int:
    """Entries per query of the level-1 power table, the largest one (i' < K); the tail levels reuse it."""
    return table_len(p.K())


def k_power_table(base: Base, pts: Buf[4], count: Int32, tab: Int32, ptab: Buf[4]):
    """ptab[q, k] = pt_q^k for k < POW_LO, then pt_q^(POW_LO (k - POW_LO)), `tab` entries per query."""
    var gid = global_idx.x
    var TAB = Int(tab)
    if gid >= Int(count) * TAB:
        return
    var k = gid % TAB
    var n = k if k < POW_LO else POW_LO * (k - POW_LO)
    ptab.store(base, gid, ext_pow[2](pts.load(base, gid // TAB), n))


def k_materialize_level1[p: Params](base: Base, running: Buf[E_BYTES], batch: Buf[E_BYTES], ptab: Buf[4],
                                    count: Int32, w_tilde: Buf[E_BYTES]):
    """w~[slot] = batch_0 running[slot] + sum_{q, tau} batch_{1 + 4 (q n_cw + cw) + tau} coord_tau(b_j pt_q^i'),
    (i, j) = pack_index(slot), (i', cw) = split_index(i): the slot's codeword answers to its own four weights."""
    comptime N = p.N()
    comptime TAB = power_table_len[p]()
    var slot = global_idx.x
    if slot >= N:
        return
    var w = fp_reduce(fp_ext_mul[E_LEVEL](to_f32(batch.load(base, 0)), to_f32(running.load(base, slot))))
    var i: Int
    var j: Int
    i, j = pack_index[p](slot)
    var ip: Int
    var cw: Int
    ip, cw = split_index[p](i)
    var bj = F4(0)
    bj[j] = 1
    var lo = ip % POW_LO
    var hi = POW_LO + ip // POW_LO
    for q in range(Int(count)):                     # 4 count products of canonical values, below 16 K each
        var pw = ext_mul[2](ptab.load(base, q * TAB + lo), ptab.load(base, q * TAB + hi))
        var m = to_f32(ext_mul[2](bj, pw))
        comptime for tau in range(4):
            w = to_f32(batch.load(base, 1 + 4 * (q * p.n_cw() + cw) + tau)).fma(EF(m[tau]), w)
        if q % 128 == 127:                          # 512 terms: 512 * 126^2 + 190 < 2^24, no cancellation here
            w = fp_reduce(w)
    w_tilde.store(base, slot, fp_canonical(w))


@always_inline
def tail_split(row: Int, bd: Int, low: Int) -> Tuple[Int, Int]:
    """A tail row (bd binary digits, then the odd digit) -> (row', cw): the codeword is the top bd - low binary
    digits, row' the low digits with the odd digit above them (spec 9.1, the level-1 rule on tail rows)."""
    var b = row & ((1 << bd) - 1)
    var r = row >> bd
    return ((b & ((1 << low) - 1)) | (r << low), b >> low)


@always_inline
def tail_join(rp: Int, cw: Int, bd: Int, low: Int) -> Int:
    """The inverse of tail_split."""
    return ((rp & ((1 << low) - 1)) | (cw << low)) | ((rp >> low) << bd)


def k_tail_pack(base: Base, y: Buf[1], dst: Buf[1], rows: Int32, row_bytes: Int32, n_cw: Int32, bd: Int32, low: Int32):
    """dst[row', cw] = y[row]: the level's rows regrouped by codeword for the RS passes (n_cw > 1)."""
    var gid = Int(global_idx.x)
    var RB = Int(row_bytes)
    if gid >= Int(rows) * RB:
        return
    var row = gid // RB
    var rp: Int
    var cw: Int
    rp, cw = tail_split(row, Int(bd), Int(low))
    dst.store(base, (rp * Int(n_cw) + cw) * RB + gid % RB, y.load(base, gid))


def k_materialize_tail(base: Base, running: Buf[E_BYTES], batch: Buf[E_BYTES], ptab: Buf[4], tab: Int32,
                       count: Int32, rows: Int32, w_tilde: Buf[E_BYTES], n_cw: Int32, bd: Int32, low: Int32):
    """w~[row] = batch_0 running[row] + sum_q batch_{1 + q n_cw + cw} pt_q^row', (row', cw) = tail_split(row),
    the power from the table."""
    var row = Int(global_idx.x)
    if row >= Int(rows):
        return
    var TAB = Int(tab)
    var rp: Int
    var cw: Int
    rp, cw = tail_split(row, Int(bd), Int(low))
    var lo = rp % POW_LO
    var hi = POW_LO + rp // POW_LO
    var w = ext_mul[E_LEVEL](batch.load(base, 0), running.load(base, row))
    for q in range(Int(count)):
        var pw = ext_mul[2](ptab.load(base, q * TAB + lo), ptab.load(base, q * TAB + hi))
        w = f_add(w, e_mul_f4(batch.load(base, 1 + q * Int(n_cw) + cw), pw))
    w_tilde.store(base, row, w)


def k_round_partial(base: Base, w_tilde: Buf[E_BYTES], y: Buf[E_BYTES], length: Int32, d: Int32, r: Buf[E_BYTES], partial: Buf[E_BYTES]):
    """Round d of the partial sumcheck over the three low digits: digits below d are bound to r_0 ..
    r_{d-1}, digit d is the variable b, everything above is summed. Thread t of ROUND_THREADS sums its
    share of the groups (row, digits above d) into partial[t] = (s(0), s(1), s(2)).
    Registers set the speed, not memory or arithmetic (2026-09-17, N = 258,048: the loads alone take
    0.1 ms a round, the kernel 3.7 ms with the multipliers in fp32 registers, 1.6 ms with them as
    bytes converted at use; one evaluation alone runs in 0.3 ms). So the multipliers stay bytes."""
    var t = global_idx.x
    var dd = Int(d)
    var m = 1 << dd
    var rr = _r3(base, r)
    var rb = InlineArray[E, 4](fill=E(0))
    for a in range(m):
        rb[a] = rbar_at(rr, a, dd)
    # fp32 lanes: up to four products of canonical values (below 2.1 M each) per sum, reduced to
    # |x| <= 190 before the products of sums (below 4.7 M), one reduction of each accumulator per group
    var acc0 = EF(0)
    var acc1 = EF(0)
    var acc2 = EF(0)
    var groups = Int(length) // (2 * m)
    for grp in range(t, groups, ROUND_THREADS):
        var n0 = grp * 2 * m
        var y0 = EF(0)
        var y1 = EF(0)
        var w0 = EF(0)
        var w1 = EF(0)
        for a in range(m):
            var ra = to_f32(rb[a])
            y0 += fp_ext_mul[E_LEVEL](ra, to_f32(y.load(base, n0 + a)))
            y1 += fp_ext_mul[E_LEVEL](ra, to_f32(y.load(base, n0 + m + a)))
            w0 += fp_ext_mul[E_LEVEL](ra, to_f32(w_tilde.load(base, n0 + a)))
            w1 += fp_ext_mul[E_LEVEL](ra, to_f32(w_tilde.load(base, n0 + m + a)))
        y0 = fp_reduce(y0)
        y1 = fp_reduce(y1)
        w0 = fp_reduce(w0)
        w1 = fp_reduce(w1)
        acc0 = fp_reduce(acc0 + fp_ext_mul[E_LEVEL](y0, w0))
        acc1 = fp_reduce(acc1 + fp_ext_mul[E_LEVEL](y1, w1))
        acc2 = fp_reduce(acc2 + fp_ext_mul[E_LEVEL](fp_reduce(y1 + y1 - y0), fp_reduce(w1 + w1 - w0)))
    partial.store(base, 3 * t, fp_canonical(acc0))
    partial.store(base, 3 * t + 1, fp_canonical(acc1))
    partial.store(base, 3 * t + 2, fp_canonical(acc2))


def k_round_sum(base: Base, src: Buf[E_BYTES], dst: Buf[E_BYTES], rows: Int32):
    """Block b, 3 e threads (one per evaluation and byte): dst[b] = the sum of src rows b rows .. (b + 1) rows.
    Two launches sum the ROUND_THREADS partials: ROUND_SUM_BLOCKS blocks, then one block over their sums."""
    var i = Int(thread_idx.x)
    if i >= 3 * E_BYTES:
        return
    var b = Int(block_idx.x)
    var acc = SIMD[DType.uint8, 1](0)
    for t in range(b * Int(rows), (b + 1) * Int(rows)):
        acc = f_add(acc, base.unsafe_load[width=1](src.at(3 * t) + i))
    base.unsafe_store(dst.at(3 * b) + i, acc)


def k_fold8(base: Base, src: Buf[E_BYTES], rows: Int32, r: Buf[E_BYTES], dst: Buf[E_BYTES]):
    """dst[row] = sum_{a < 8} rbar[a] src[8 row + a], rbar = (x) (1 - r_i, r_i)."""
    var row = global_idx.x
    if row >= Int(rows):
        return
    var rr = _r3(base, r)
    var acc = EF(0)
    for a in range(8):                              # eight products of canonical values: below 16.3 M
        acc += fp_ext_mul[E_LEVEL](to_f32(rbar_at(rr, a, 3)), to_f32(src.load(base, 8 * row + a)))
    dst.store(base, row, fp_canonical(acc))


# ---- host launchers ----

@always_inline
def _grid(n: Int) -> Int:
    return ceildiv(n, BACKEND.block)


comptime TAIL_F4 = 8 * (E_BYTES // 4)   # F4 symbols per tail row: 8 E values, E_BYTES // 4 F4 coordinates each


def tail_encode(ctx: DeviceContext, arena: Arena,
                y: Int, rows: Int, L0: Int, m: Int, etmp: Int, code: Int, rs: RsTables,
                n_cw: Int = 1, bd: Int = 0, packed: Int = 0) raises:
    """Mat(y) (rows, 8, e) -> code (m L0, n_cw, 8, e): the RS encoder on TAIL_F4 n_cw F4 columns of rows / n_cw
    symbols, no inverse. With a split the rows are first regrouped by codeword into `packed` (rows * 8 e bytes
    of scratch), `bd` the binary digits of a row."""
    var src = y
    if n_cw > 1:
        var low = bd - log2_floor(n_cw)
        var total = rows * 8 * E_BYTES
        ctx.enqueue_function[k_tail_pack](arena.buf, Buf[1](y), Buf[1](packed), Int32(rows), Int32(8 * E_BYTES),
                                          Int32(n_cw), Int32(bd), Int32(low), grid_dim=_grid(total), block_dim=BACKEND.block)
        src = packed
    rs_encode_on(ctx, arena, src, etmp, code, TAIL_F4 * n_cw, rows // n_cw, L0, m, rs)


def points(ctx: DeviceContext, arena: Arena, positions: Int, count: Int, dom: Int, L0: Int, pts: Int) raises:
    ctx.enqueue_function[k_points](arena.buf, Buf[4](positions), Int32(count), Buf[4](dom), Int32(L0), Buf[4](pts),
                                   grid_dim=_grid(count), block_dim=BACKEND.block)


def running0[p: Params](ctx: DeviceContext, arena: Arena, w_z: Int, gamma: Int, P: Int, dst: Int) raises:
    ctx.enqueue_function[k_running0[p]](arena.buf, Buf[E_BYTES](w_z), Buf[E_BYTES](gamma), Int32(P), Buf[E_BYTES](dst),
                                        grid_dim=_grid(p.N()), block_dim=BACKEND.block)


def tail_materialize[p: Params](ctx: DeviceContext, arena: Arena, level1: Bool,
                                running: Int, batch: Int, pts: Int, count: Int, length: Int, w_tilde: Int, ptab: Int,
                                n_cw: Int = 1, bd: Int = 0) raises:
    """`ptab` is (count, power_table_len) F4 of scratch, the level-1 size; every level fills its own table.
    A tail level passes the opened level's codeword count and the binary digits of its rows."""
    var tab = power_table_len[p]() if level1 else table_len(length // n_cw)     # exponents below the codeword's rows
    if tab > power_table_len[p]():
        raise Error("tail_materialize: the power table holds N / 4 exponents")
    ctx.enqueue_function[k_power_table](arena.buf, Buf[4](pts), Int32(count), Int32(tab), Buf[4](ptab),
                                        grid_dim=_grid(count * tab), block_dim=BACKEND.block)
    if level1:
        ctx.enqueue_function[k_materialize_level1[p]](arena.buf, Buf[E_BYTES](running), Buf[E_BYTES](batch), Buf[4](ptab), Int32(count), Buf[E_BYTES](w_tilde),
                                                      grid_dim=_grid(p.N()), block_dim=BACKEND.block)
    else:
        ctx.enqueue_function[k_materialize_tail](arena.buf, Buf[E_BYTES](running), Buf[E_BYTES](batch), Buf[4](ptab), Int32(tab),
                                                 Int32(count), Int32(length), Buf[E_BYTES](w_tilde),
                                                 Int32(n_cw), Int32(bd), Int32(bd - log2_floor(n_cw)),
                                                 grid_dim=_grid(length), block_dim=BACKEND.block)


def tail_round(ctx: DeviceContext, arena: Arena,
               w_tilde: Int, y: Int, length: Int, digit: Int, r: Int, partial: Int, dst: Int) raises:
    """dst (3, e) = the round message of digit `digit` given r_0 .. r_{digit-1} at `r`. `partial` holds
    ROUND_THREADS + ROUND_SUM_BLOCKS rows of (3, e): the threads' sums, then the blocks' sums."""
    comptime rows = ROUND_THREADS // ROUND_SUM_BLOCKS
    var sums = partial + ROUND_THREADS * 3 * E_BYTES
    ctx.enqueue_function[k_round_partial](arena.buf, Buf[E_BYTES](w_tilde), Buf[E_BYTES](y), Int32(length), Int32(digit), Buf[E_BYTES](r), Buf[E_BYTES](partial),
                                          grid_dim=_grid(ROUND_THREADS), block_dim=BACKEND.block)
    ctx.enqueue_function[k_round_sum](arena.buf, Buf[E_BYTES](partial), Buf[E_BYTES](sums), Int32(rows), grid_dim=ROUND_SUM_BLOCKS, block_dim=64)
    ctx.enqueue_function[k_round_sum](arena.buf, Buf[E_BYTES](sums), Buf[E_BYTES](dst), Int32(ROUND_SUM_BLOCKS), grid_dim=1, block_dim=64)


def tail_fold(ctx: DeviceContext, arena: Arena, src: Int, rows: Int, r: Int, dst: Int) raises:
    """dst = Mat(src) r_bar, for the message and for the query."""
    ctx.enqueue_function[k_fold8](arena.buf, Buf[E_BYTES](src), Int32(rows), Buf[E_BYTES](r), Buf[E_BYTES](dst), grid_dim=_grid(rows), block_dim=BACKEND.block)


# ---- host side of the same formulas (verifier, tests) ----


def host_r3(r: Span[UInt8, _]) -> InlineArray[E, 3]:
    var out = InlineArray[E, 3](fill=E(0))
    for i in range(3):
        out[i] = list_e(r, i)
    return out^


def tail_encode_at(y: Span[UInt8, _], rows: Int, pt: F4, cw: Int = 0, n_cw: Int = 1, bd: Int = 0) -> E:
    """Enc(y)(pt) of codeword cw over E: sum_row' y[tail_join(row', cw)] pt^row'."""
    var low = bd - log2_floor(n_cw)
    var acc = E(0)
    var pw = F4(1, 0, 0, 0)
    for rp in range(rows // n_cw):
        acc = f_add(acc, e_mul_f4(list_e(y, tail_join(rp, cw, bd, low)), pw))
        pw = ext_mul[2](pw, pt)
    return acc


def fold8_host(src: Span[UInt8, _], rows: Int, r: Span[UInt8, _]) -> List[UInt8]:
    var rr = host_r3(r)
    var out = List[UInt8](capacity=rows * E_BYTES)
    for row in range(rows):
        var acc = E(0)
        for a in range(8):
            acc = f_add(acc, ext_mul[E_LEVEL](rbar_at(rr, a, 3), list_e(src, 8 * row + a)))
        for t in range(E_BYTES):
            out.append(acc[t])
    return out^


def quadratic_at(s: Span[UInt8, _], off: Int, r: E) -> E:
    """The degree-2 polynomial with values s[off], s[off + 1], s[off + 2] at 0, 1, 2, evaluated at r:
    s0 (r - 1)(r - 2) / 2 - s1 r (r - 2) + s2 r (r - 1) / 2."""
    var one = ext_one[E_LEVEL]()
    var two = f_add(one, one)
    var half = E(0)
    half[0] = 64
    var r1 = f_sub(r, one)
    var r2 = f_sub(r, two)
    var t0 = ext_mul[E_LEVEL](ext_mul[E_LEVEL](list_e(s, off), ext_mul[E_LEVEL](r1, r2)), half)
    var t1 = ext_mul[E_LEVEL](list_e(s, off + 1), ext_mul[E_LEVEL](r, r2))
    var t2 = ext_mul[E_LEVEL](ext_mul[E_LEVEL](list_e(s, off + 2), ext_mul[E_LEVEL](r, r1)), half)
    return f_add(f_sub(t0, t1), t2)

"""Sketched level-1 code, experimental (branch `sketch`): C_H(x) = (x, RS(H x)).

Bolt (Gurkan, Novakovic, Rothblum 2026, section 3): H is a random sparse (rows x K) matrix over F with
column weight c and row weight d, K c = rows d, and nonzero entries uniform in F*; RS encodes the short
sketch u = H x at rate 1/2. The codeword is the K packed message symbols as they are, then the RS
codeword of u: (K + L_s) rows of `columns` F4 symbols, leaf-major like the RS code (encode.mojo). The
coefficients of H are in F (width 1: H acts on each F4 coordinate alone) or in F4 (width 4, a larger
alphabet for the [YHC+11] bound: distance 0.098 against 0.072 at 16/128).

Nothing in the prover or the verifier calls this module. The distance of H is a probabilistic claim
over the seed (Bolt section 6.2, [YHC+11] Theorem 6.2); the opening side is not written.

Buffers (bytes; shapes slowest ... fastest):
    start   (rows + 1)            u32 CSR row starts into `ent`
    ent     (K c)                 u32 column * 128 + the F coefficient (0 at width 4)
    coef    (K c, 4)              F4 coefficients, width 4 only
    u       (rows, column, 4)     F4, the sketch H x
    etmp    (L0, column, 4)       F4, RS intermediate
    code    (K + L_s, column, 4)  F4, rows 0 .. K-1 the message (encode.pack writes them), then RS(u)
"""

from std.math import ceildiv
from std.builtin.sort import sort
from std.gpu import global_idx
from std.memory import bitcast
from max.gpu.host import DeviceContext, HostBuffer

from core.field import F4, f_reduce, f_reduce_signed, f4_mac_wide
from core.params import domain_for
from core.tables import RsDomain, RsTables, build_rs_tables
from core.arena import Arena, Bump, ST_LOAD, ST_END
from core.bytes import Base, Buf, u32
from core.backend import BACKEND
from pcs.encode import rs_encode_on

comptime SKETCH_RATE_INV = 2        # the RS code of the sketch (Bolt 7.3)


struct SketchMatrix(Movable):
    """H on the host, CSR: row a holds the entries start[a] .. start[a + 1] - 1."""
    var K: Int
    var rows: Int
    var start: List[Int]
    var col: List[Int]
    var width: Int                  # coefficient bytes: 1 (F) or 4 (F4)
    var coef: List[UInt8]           # (entry, width)

    def __init__(out self, K: Int, col_weight: Int, row_weight: Int, seed: UInt64, width: Int = 1) raises:
        """The biregular ensemble: K c sockets in a seeded random order, cut into rows of `row_weight` (the
        last row takes the rest), one uniform nonzero coefficient of F or F4 per socket. A column repeated
        in a row keeps both entries; they add."""
        if row_weight % 4 != 0:
            raise Error("row weight not a multiple of 4 (k_sketch loads four entries)")
        if width != 1 and width != 4:
            raise Error("coefficient width is 1 (F) or 4 (F4)")
        self.K = K
        self.width = width
        self.rows = ceildiv(K * col_weight, row_weight)
        var rng = seed
        self.col = List[Int](capacity=K * col_weight)
        for i in range(K):
            for _ in range(col_weight):
                self.col.append(i)
        for i in range(len(self.col) - 1, 0, -1):     # Fisher-Yates
            var j = Int(_splitmix(rng) % UInt64(i + 1))
            var t = self.col[i]
            self.col[i] = self.col[j]
            self.col[j] = t
        for a in range(ceildiv(len(self.col), row_weight)):  # each row by column: the same H; the rows in
            var lo = a * row_weight                             # flight then read x in one moving window
            var hi = min(lo + row_weight, len(self.col))
            sort(Span(self.col)[lo:hi])
        self.coef = List[UInt8](capacity=len(self.col) * width)
        for _ in range(len(self.col)):
            if width == 1:
                self.coef.append(UInt8(1 + _splitmix(rng) % 126))
            else:
                var z = 0
                while z == 0:                               # uniform in F4*: reject zero
                    z = 0
                    for _ in range(4):
                        var v = Int(_splitmix(rng) % 127)
                        self.coef.append(UInt8(v))
                        z += v
                    if z == 0:
                        self.coef.resize(len(self.coef) - 4, UInt8(0))
        self.start = List[Int](capacity=self.rows + 1)
        for a in range(self.rows):
            self.start.append(a * row_weight)
        self.start.append(len(self.col))


@always_inline
def _splitmix(mut s: UInt64) -> UInt64:
    s += 0x9E3779B97F4A7C15
    var z = s
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


struct SketchLayout(TrivialRegisterPassable):
    """Arena offsets of the sketched code of K symbols per column for `columns` columns."""
    var K: Int
    var rows: Int
    var entries: Int
    var width: Int
    var columns: Int
    var L0: Int
    var m: Int
    var start: Int
    var ent: Int
    var coef: Int
    var u: Int
    var etmp: Int
    var code: Int
    var rs: RsTables

    def __init__(out self, mut bump: Bump, h: SketchMatrix, columns: Int, stage: Int = ST_LOAD) raises:
        self.K = h.K
        self.rows = h.rows
        self.entries = len(h.col)
        self.width = h.width
        self.columns = columns
        var dom = domain_for(h.rows, SKETCH_RATE_INV)
        if dom[0] == 0:
            raise Error("no RS domain for the sketch")
        self.m = dom[1]
        self.L0 = dom[0] // dom[1]
        var L = dom[0]
        self.start = bump.alloc((h.rows + 1) * 4)
        self.ent = bump.alloc(self.entries * 4)
        self.coef = bump.alloc(self.entries * 4 if h.width == 4 else 0)
        self.u = bump.alloc(h.rows * columns * 4, stage, stage)
        self.etmp = bump.alloc(self.L0 * columns * 4, stage, stage)
        self.code = bump.alloc((h.K + L) * columns * 4, stage)
        self.rs = RsTables(bump.alloc(0), self.L0, self.m, h.rows)
        _ = bump.alloc(self.rs.bytes)

    def leaves(self) -> Int:
        return self.K + self.m * self.L0

    def row_bytes(self) -> Int:
        return self.columns * 4


def upload_sketch(ctx: DeviceContext, arena: Arena, s: SketchLayout, h: SketchMatrix) raises:
    """H as CSR and the sketch's RS tables. Setup only."""
    var st = ctx.enqueue_create_host_buffer[DType.uint8]((h.rows + 1) * 4)
    var en = ctx.enqueue_create_host_buffer[DType.uint8](s.entries * 4)
    ctx.synchronize()
    for a in range(h.rows + 1):
        _put_u32(st, a * 4, h.start[a])
    for i in range(s.entries):
        _put_u32(en, i * 4, h.col[i] * 128 + (Int(h.coef[i]) if h.width == 1 else 0))
    arena.upload(ctx, s.start, st)
    arena.upload(ctx, s.ent, en)
    if h.width == 4:
        var cf = ctx.enqueue_create_host_buffer[DType.uint8](s.entries * 4)
        ctx.synchronize()
        for i in range(s.entries * 4):
            cf[i] = h.coef[i]
        arena.upload(ctx, s.coef, cf)
    arena.upload(ctx, s.rs.base, build_rs_tables(ctx, s.rs, RsDomain(s.L0, s.m), h.rows))


def _put_u32(h: HostBuffer[DType.uint8], at: Int, v: Int):
    for b in range(4):
        h[at + b] = UInt8((v >> (8 * b)) & 255)


def k_sketch[V: Int, W: Int](base: Base, start: Buf[4], ent: Buf[4], coef: Buf[4], x: Buf[4], u: Buf[4],
                             rows: Int32, columns: Int32):
    """u[a, c] = sum over row a of H of coef * x[col, c]; one thread per (a, V consecutive columns), columns
    fastest: an entry is decoded once per V columns, and a SIMD group reads 32 V 4 contiguous bytes of one
    message row per entry. Rows start at multiples of the row weight (a multiple of 4), so the entries and
    the F4 coefficients load as aligned groups of four. A row of x is 4 columns bytes, so the V-column loads
    are 4-byte aligned. Width 1 reduces every 128 entries (128 * 126^2 + 126 < 2^21, the f_reduce bound),
    width 4 every 28 (under F4_MAC_MAX)."""
    var gid = Int(global_idx.x)
    var cols = Int(columns)
    var groups = ceildiv(cols, V)
    if gid >= Int(rows) * groups:
        return
    var c = (gid % groups) * V
    var a = gid // groups
    var n = min(V, cols - c)
    var lo = u32(base, start.at(a))
    var hi = u32(base, start.at(a + 1))
    var r: SIMD[DType.uint8, 4 * V]
    comptime if W == 1:
        var acc = SIMD[DType.uint32, 4 * V](0)
        for i in range(lo, hi, 4):
            var w = bitcast[DType.uint32, 4](base.unsafe_load[width=16, alignment=16](ent.at(i)))
            comptime for k in range(4):
                if i + k < hi:
                    acc += _load_x[V](base, x.at(Int(w[k] >> 7) * cols + c), n).cast[DType.uint32]() * (w[k] & 127)
            if (i - lo) % 128 == 124:
                acc = f_reduce(acc).cast[DType.uint32]()
        r = f_reduce(acc)
    else:
        var acc = InlineArray[SIMD[DType.int32, 4], V](fill=SIMD[DType.int32, 4](0))   # one per column: Metal
        for i in range(lo, hi, 4):                                                     # has no int32 vector insert
            var w = bitcast[DType.uint32, 4](base.unsafe_load[width=16, alignment=16](ent.at(i)))
            var h = base.unsafe_load[width=16, alignment=16](coef.at(i))
            comptime for k in range(4):
                if i + k < hi:
                    var v = _load_x[V](base, x.at(Int(w[k] >> 7) * cols + c), n)
                    comptime for j in range(V):
                        f4_mac_wide(acc[j], _get4(v, j), _get4(h, k))
            if (i - lo) % 28 == 24:
                comptime for j in range(V):
                    acc[j] = f_reduce_signed(acc[j]).cast[DType.int32]()
        r = SIMD[DType.uint8, 4 * V](0)
        comptime for j in range(V):
            var q = f_reduce_signed(acc[j])
            comptime for l in range(4):
                r[4 * j + l] = q[l]
    if n == V:
        base.unsafe_store[width=4 * V, alignment=4](u.at(a * cols + c), r)
    else:
        for j in range(n):
            u.store(base, a * cols + c + j, _get4(r, j))


@always_inline
def _load_x[V: Int](base: Base, at: Int, n: Int) -> SIMD[DType.uint8, 4 * V]:
    """The n <= V symbols at `at`, zero past n."""
    if n == V:
        return base.unsafe_load[width=4 * V, alignment=4](at)
    var v = SIMD[DType.uint8, 4 * V](0)
    for j in range(n):
        var q = base.unsafe_load[width=4, alignment=4](at + 4 * j)
        for l in range(4):
            v[4 * j + l] = q[l]
    return v


@always_inline
def _get4[w: SIMDLength](v: SIMD[DType.uint8, w], j: Int) -> SIMD[DType.uint8, 4]:
    return SIMD[DType.uint8, 4](v[4 * j], v[4 * j + 1], v[4 * j + 2], v[4 * j + 3])


def sketch[V: Int = 4](ctx: DeviceContext, arena: Arena, s: SketchLayout) raises:
    """u = H x on the message rows of `code`."""
    var grid = ceildiv(s.rows * ceildiv(s.columns, V), BACKEND.block)
    if s.width == 1:
        ctx.enqueue_function[k_sketch[V, 1]](arena.buf, Buf[4](s.start), Buf[4](s.ent), Buf[4](s.coef), Buf[4](s.code),
                                             Buf[4](s.u), Int32(s.rows), Int32(s.columns), grid_dim=grid, block_dim=BACKEND.block)
    else:
        ctx.enqueue_function[k_sketch[V, 4]](arena.buf, Buf[4](s.start), Buf[4](s.ent), Buf[4](s.coef), Buf[4](s.code),
                                             Buf[4](s.u), Int32(s.rows), Int32(s.columns), grid_dim=grid, block_dim=BACKEND.block)


def sketch_code(ctx: DeviceContext, arena: Arena, s: SketchLayout) raises:
    """code rows 0 .. K-1 (the message, already in place) -> the sketch, then its RS code into rows K .. ."""
    sketch(ctx, arena, s)
    rs_encode_on(ctx, arena, s.u, s.etmp, s.code + s.K * s.columns * 4, s.columns, s.rows, s.L0, s.m, s.rs)

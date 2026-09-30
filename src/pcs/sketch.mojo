"""Sketched level-1 code, experimental (branch `sketch`): C_H(x) = (x, RS(H x)).

Bolt (Gurkan, Novakovic, Rothblum 2026, section 3): H is a random sparse (rows x K) matrix over F with
column weight c and row weight d, K c = rows d, and nonzero entries uniform in F*; RS encodes the short
sketch u = H x at rate 1/2. The codeword is the K packed message symbols as they are, then the RS
codeword of u: (K + L_s) rows of `columns` F4 symbols, leaf-major like the RS code (encode.mojo). H has
F coefficients, so it acts on each F4 coordinate alone.

Nothing in the prover or the verifier calls this module. The distance of H is a probabilistic claim
over the seed (Bolt section 6.2, [YHC+11] Theorem 6.2); the opening side is not written.

Buffers (bytes; shapes slowest ... fastest):
    start   (rows + 1)            u32 CSR row starts into `ent`
    ent     (K c)                 u32 column * 128 + coefficient
    u       (rows, column, 4)     F4, the sketch H x
    etmp    (L0, column, 4)       F4, RS intermediate
    code    (K + L_s, column, 4)  F4, rows 0 .. K-1 the message (encode.pack writes them), then RS(u)
"""

from std.math import ceildiv
from std.gpu import global_idx
from max.gpu.host import DeviceContext, HostBuffer

from core.field import F4, f_reduce
from core.params import domain_for
from core.tables import RsDomain, RsTables, build_rs_tables
from core.arena import Arena, Bump, ST_LOAD, ST_END
from core.bytes import Base, Buf, u32
from core.backend import BACKEND
from pcs.encode import rs_encode_on

comptime SKETCH_RATE_INV = 2        # the RS code of the sketch (Bolt 7.3)
comptime MAX_ROW_WEIGHT = 128       # k_sketch: 128 * 126^2 < 2^21, the f_reduce input bound


struct SketchMatrix(Movable):
    """H on the host, CSR: row a holds the entries start[a] .. start[a + 1] - 1."""
    var K: Int
    var rows: Int
    var start: List[Int]
    var col: List[Int]
    var coef: List[UInt8]

    def __init__(out self, K: Int, col_weight: Int, row_weight: Int, seed: UInt64) raises:
        """The biregular ensemble: K c sockets in a seeded random order, cut into rows of `row_weight` (the
        last row takes the rest), one uniform F* coefficient per socket. A column repeated in a row keeps
        both entries; they add."""
        if row_weight > MAX_ROW_WEIGHT:
            raise Error("row weight above MAX_ROW_WEIGHT")
        self.K = K
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
        self.coef = List[UInt8](capacity=len(self.col))
        for _ in range(len(self.col)):
            self.coef.append(UInt8(1 + _splitmix(rng) % 126))
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
    var columns: Int
    var L0: Int
    var m: Int
    var start: Int
    var ent: Int
    var u: Int
    var etmp: Int
    var code: Int
    var rs: RsTables

    def __init__(out self, mut bump: Bump, h: SketchMatrix, columns: Int, stage: Int = ST_LOAD) raises:
        self.K = h.K
        self.rows = h.rows
        self.entries = len(h.col)
        self.columns = columns
        var dom = domain_for(h.rows, SKETCH_RATE_INV)
        if dom[0] == 0:
            raise Error("no RS domain for the sketch")
        self.m = dom[1]
        self.L0 = dom[0] // dom[1]
        var L = dom[0]
        self.start = bump.alloc((h.rows + 1) * 4)
        self.ent = bump.alloc(self.entries * 4)
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
        _put_u32(en, i * 4, h.col[i] * 128 + Int(h.coef[i]))
    arena.upload(ctx, s.start, st)
    arena.upload(ctx, s.ent, en)
    arena.upload(ctx, s.rs.base, build_rs_tables(ctx, s.rs, RsDomain(s.L0, s.m), h.rows))


def _put_u32(h: HostBuffer[DType.uint8], at: Int, v: Int):
    for b in range(4):
        h[at + b] = UInt8((v >> (8 * b)) & 255)


def k_sketch(base: Base, start: Buf[4], ent: Buf[4], x: Buf[4], u: Buf[4], rows: Int32, columns: Int32):
    """u[a, c] = sum over row a of H of coef * x[col, c]; one thread per (a, c), c fastest, so a SIMD group
    reads 32 consecutive columns of one message row per entry."""
    var gid = Int(global_idx.x)
    var cols = Int(columns)
    if gid >= Int(rows) * cols:
        return
    var c = gid % cols
    var a = gid // cols
    var acc = SIMD[DType.uint32, 4](0)
    for i in range(u32(base, start.at(a)), u32(base, start.at(a + 1))):
        var w = u32(base, ent.at(i))
        acc += x.load(base, (w >> 7) * cols + c).cast[DType.uint32]() * UInt32(w & 127)
    u.store(base, a * cols + c, f_reduce(acc))


def sketch_code(ctx: DeviceContext, arena: Arena, s: SketchLayout) raises:
    """code rows 0 .. K-1 (the message, already in place) -> the sketch, then its RS code into rows K .. ."""
    ctx.enqueue_function[k_sketch](arena.buf, Buf[4](s.start), Buf[4](s.ent), Buf[4](s.code), Buf[4](s.u),
                                   Int32(s.rows), Int32(s.columns),
                                   grid_dim=ceildiv(s.rows * s.columns, BACKEND.block), block_dim=BACKEND.block)
    rs_encode_on(ctx, arena, s.u, s.etmp, s.code + s.K * s.columns * 4, s.columns, s.rows, s.L0, s.m, s.rs)

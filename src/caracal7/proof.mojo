"""Proof shape and byte layout (statement-layer section 7, spec 9.5), milestone 1.

Order, every integer little-endian, every field element e bytes:

    header      version u32, params digest 32 B, public inputs (u32 length + bytes)
    W root      H.DIGEST
    Q root      H.DIGEST                       (Z root, Z2, Q3 join in milestone 2)
    openings    alpha_{c,p}: P x (columns_w + columns_q) x e
    per level l = 2 .. ell-1:
                Mat(y_l) root 32 B; multiproof of level l-1 (u32 length + bytes);
                expected symbols v_{l-1}; three sumcheck messages (9 e)
    last        clear vector y_ell (|y_ell| x e); multiproof of level ell-1 (u32 length + bytes)

Multiproofs are length-prefixed because the sibling frontier depends on the sampled positions.
Everything else has a size fixed by `Shape`, so the verifier can check the total length up front.
"""

from std.math import log2

from caracal7.params import Params

comptime VERSION: UInt32 = 1
comptime H4_ORDER = 161280          # largest smooth subgroup of F4*; every code domain divides it
comptime TAIL_FOLD = 8              # 2^tail_digits columns per tail matrix
comptime TAIL_RATE_INV = 16         # rate rule of spec 9.5: L_l is the smallest divisor of 161280 >= 16 n_l


@fieldwise_init
struct TailLevel(TrivialRegisterPassable, Writable):
    var length: Int      # |y_l| in E elements
    var rows: Int        # n_l = length / 8
    var L: Int           # domain size
    var queries: Int     # |S_l|

    def write_to(self, mut w: Some[Writer]):
        w.write("TailLevel(len=", self.length, ", rows=", self.rows, ", L=", self.L, ", q=", self.queries, ")")


def tail_schedule[p: Params]() raises -> List[TailLevel]:
    """Committed tail levels l = 2 .. ell-1, derived from N and e (design section 2). The level after
    the last entry is sent in the clear; with no entries, y_2 itself is the clear vector."""
    var levels = List[TailLevel]()
    var length = p.N()
    while length > p.tail_clear_max:
        var rows = length // TAIL_FOLD
        var L = 0
        for d in range(1, H4_ORDER + 1):
            if H4_ORDER % d == 0 and d >= TAIL_RATE_INV * rows:
                L = d
                break
        if L == 0:
            raise Error("tail level does not fit the F4 domain")
        # queries at the exact rate rows / L, same formula as level 1
        var rate = Float64(rows) / Float64(L)
        var queries = Int((Float64(p.lambda_bits) / _log2(2.0 / (1.0 + rate))).__ceil__())
        levels.append(TailLevel(length=length, rows=rows, L=L, queries=queries))
        length = rows
    return levels^


def _log2(x: Float64) -> Float64:
    return log2(x)


struct Shape(Writable):
    """Everything the proof size depends on besides Params: set by the IR program."""
    var columns_w: Int          # witness tree
    var columns_q: Int          # quotient tree, 3 e coordinate columns
    var points: Int             # P opening points
    var tail: List[TailLevel]
    var clear_length: Int       # |y_ell|

    def __init__[p: Params](out self, columns_w: Int, points: Int) raises:
        self.columns_w = columns_w
        self.columns_q = 3 * p.e
        self.points = points
        self.tail = tail_schedule[p]()
        self.clear_length = p.N() if len(self.tail) == 0 else self.tail[len(self.tail) - 1].rows

    def columns(self) -> Int:
        return self.columns_w + self.columns_q

    def fixed_bytes[p: Params, digest: Int](self, public_bytes: Int) -> Int:
        """Proof length without the length-prefixed multiproofs."""
        var n = 4 + digest + 4 + public_bytes + digest + digest
        n += self.points * self.columns() * p.e
        for i in range(len(self.tail)):
            var prev_queries = p.queries() if i == 0 else self.tail[i - 1].queries
            var v_count = 4 * p.n_cw() * prev_queries if i == 0 else prev_queries
            n += digest + 4 + v_count * p.e + 9 * p.e
        n += self.clear_length * p.e + 4
        return n

    def write_to(self, mut w: Some[Writer]):
        w.write("Shape(columns=", self.columns_w, "+", self.columns_q, ", P=", self.points,
                ", tail levels=", len(self.tail), ", clear=", self.clear_length, ")")


struct ProofWriter:
    var bytes: List[UInt8]

    def __init__(out self):
        self.bytes = List[UInt8]()

    def u32(mut self, v: Int):
        for i in range(4):
            self.bytes.append(UInt8((v >> (8 * i)) & 255))

    def raw(mut self, src: List[UInt8]):
        self.bytes.extend(src.copy())

    def prefixed(mut self, src: List[UInt8]):
        self.u32(len(src))
        self.raw(src)


struct ProofReader:
    var bytes: List[UInt8]
    var pos: Int

    def __init__(out self, var bytes: List[UInt8]):
        self.bytes = bytes^
        self.pos = 0

    def u32(mut self) raises -> Int:
        var v = 0
        for i in range(4):
            v |= Int(self.take(1)[0]) << (8 * i)
        return v

    def take(mut self, n: Int) raises -> List[UInt8]:
        if self.pos + n > len(self.bytes):
            raise Error("proof truncated")
        var out = List[UInt8](capacity=n)
        for i in range(n):
            out.append(self.bytes[self.pos + i])
        self.pos += n
        return out^

    def prefixed(mut self) raises -> List[UInt8]:
        var n = self.u32()
        return self.take(n)

    def done(self) raises:
        if self.pos != len(self.bytes):
            raise Error("trailing bytes in proof")

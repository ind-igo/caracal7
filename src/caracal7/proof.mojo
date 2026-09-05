"""Proof shape and byte layout (statement-layer section 7, spec 9.5), milestone 1.

Order, every integer little-endian, every field element e bytes:

    header      version u32, public inputs (u32 length + bytes); the parameters are bound through
                the transcript prefix (prefix_bytes), not sent
    W root      H.DIGEST
    Q root      H.DIGEST                       (Z root, Z2, Q3 join in milestone 2)
    openings    alpha_{c,p}: P x (columns_w + columns_q) x e
    per level l = 2 .. ell-1:
                Mat(y_l) root 32 B; multiproof(s) of level l-1 (u32 length + bytes each; two at
                level 1, W and Q); three sumcheck messages (9 e)
    last        clear vector y_ell (|y_ell| x e); multiproof(s) of level ell-1

Multiproofs are length-prefixed because the sibling frontier depends on the sampled positions.
Everything else has a size fixed by `Shape`, so the verifier can check the total length up front.
"""

from std.math import log2
from std.memory import memcpy
from max.gpu.host import DeviceContext, HostBuffer

from caracal7.params import Params
from caracal7.arena import Arena
from caracal7.residual import ENTRY, shift_points
from caracal7.hash import Hash

comptime VERSION: UInt32 = 1
comptime H4_ORDER = 161280          # largest smooth subgroup of F4*; every code domain is m cosets of a divisor
comptime TAIL_RATE_INV = 32         # rate rule of spec 9.5: the smallest domain at rate <= 1/32 ...
comptime TAIL_RATE_MIN_INV = 16     # ... or the largest domain (4 x 161280) if that still gives rate <= 1/16


@fieldwise_init
struct TailLevel(TrivialRegisterPassable, Writable):
    var length: Int      # |y_l| in E elements
    var rows: Int        # n_l = length / 2^tail_digits
    var L: Int           # domain size, cosets * L0
    var cosets: Int      # m: 1, 2, or 4
    var queries: Int     # |S_l|

    def write_to(self, mut w: Some[Writer]):
        w.write("TailLevel(len=", self.length, ", rows=", self.rows, ", L=", self.L, "=", self.cosets,
                "x", self.L // self.cosets, ", q=", self.queries, ")")


def tail_schedule[p: Params]() raises -> List[TailLevel]:
    """Committed tail levels l = 2 .. ell-1, derived from N and e (design section 2). The level after
    the last entry is sent in the clear; with no entries, y_2 itself is the clear vector. A level folds
    `tail_digits` binary digits, so the tail stops when fewer remain (the odd digit is never folded).
    Domains with odd part 1 are skipped: the encoder scatters from its last odd-radix stage."""
    var levels = List[TailLevel]()
    var length = p.N()
    var digits = p.a1 + p.a2
    while length > p.tail_clear_max and digits >= p.tail_digits:
        var rows = length >> p.tail_digits
        var L = 0
        var cosets = 0
        for m in [1, 2, 4]:
            for d in range(1, H4_ORDER + 1):
                if H4_ORDER % d == 0 and (d & (d - 1)) != 0 and m * d >= TAIL_RATE_INV * rows and (L == 0 or m * d < L):
                    L = m * d
                    cosets = m
                    break
        if L == 0 and 4 * H4_ORDER >= TAIL_RATE_MIN_INV * rows:
            L = 4 * H4_ORDER
            cosets = 4
        if L == 0:
            raise Error("tail level does not fit the F4 domain")
        # queries at the exact rate rows / L, same formula as level 1
        var rate = Float64(rows) / Float64(L)
        var queries = Int((Float64(p.lambda_bits) / _log2(2.0 / (1.0 + rate))).__ceil__())
        levels.append(TailLevel(length=length, rows=rows, L=L, cosets=cosets, queries=queries))
        length = rows
        digits -= p.tail_digits
    return levels^


def _log2(x: Float64) -> Float64:
    return log2(x)


struct Shape(Writable):
    """Everything the proof size depends on besides Params: set by the IR program."""
    var columns_w: Int          # witness tree
    var columns_q: Int          # quotient tree, 3 e coordinate columns
    var points: Int             # P opening points
    var entries: Int            # family table entries (residual.mojo)
    var tail: List[TailLevel]
    var clear_length: Int       # |y_ell|

    def __init__[p: Params](out self, columns_w: Int, families: List[UInt8]) raises:
        """P and the entry count come from the family table (residual.mojo)."""
        p.check()
        if len(families) % ENTRY != 0:
            raise Error("family table is not whole entries")
        self.columns_w = columns_w
        self.columns_q = 3 * p.e
        self.points = len(shift_points(families)) // 4
        self.entries = len(families) // ENTRY
        self.tail = tail_schedule[p]()
        self.clear_length = p.N() if len(self.tail) == 0 else self.tail[len(self.tail) - 1].rows

    def columns(self) -> Int:
        return self.columns_w + self.columns_q

    def fixed_bytes[p: Params, digest: Int](self, public_bytes: Int) -> Int:
        """Proof length without the multiproof bodies: their u32 prefixes are counted, one per tree
        opened (two at level 1: W and Q). Mirrors ProofWriter's order exactly."""
        var n = 4 + 4 + public_bytes + digest + digest
        n += self.points * self.columns() * p.e
        for i in range(len(self.tail)):
            n += digest + (2 if i == 0 else 1) * 4 + 9 * p.e
        n += self.clear_length * p.e + (2 if len(self.tail) == 0 else 1) * 4
        return n

    def write_to(self, mut w: Some[Writer]):
        w.write("Shape(columns=", self.columns_w, "+", self.columns_q, ", P=", self.points,
                ", tail levels=", len(self.tail), ", clear=", self.clear_length, ")")


def prefix_bytes[p: Params, H: Hash](shape: Shape, public_inputs: List[UInt8], mut families: List[UInt8]) -> List[UInt8]:
    """The transcript prefix of spec 9.4: version, field and grid parameters, domains and rates per
    level, shape, public inputs, and H(family table) as the statement artifact hash of
    statement-layer 6 step 1. Prover and verifier build the same bytes."""
    var w = _U32Writer()
    w.u32(Int(VERSION))
    for v in [p.e, p.a1, p.m1, p.a2, p.m2, p.L0, p.m_cosets, p.leaf_bytes, p.tail_digits, p.tail_clear_max,
              p.lambda_bits, p.queries(), p.n_cw()]:
        w.u32(v)
    w.u32(shape.columns_w)
    w.u32(shape.columns_q)
    w.u32(shape.points)
    w.u32(len(shape.tail))
    for lvl in shape.tail:
        for v in [lvl.length, lvl.rows, lvl.L, lvl.cosets, lvl.queries]:
            w.u32(v)
    w.u32(shape.clear_length)
    w.u32(len(public_inputs))
    w.bytes.extend(public_inputs.copy())
    var digest = List[UInt8](length=H.DIGEST, fill=0)
    H.leaf(rebind[Pointer[UInt8, MutAnyOrigin]](families.unsafe_ptr()), len(families),
           rebind[Pointer[UInt8, MutAnyOrigin]](digest.unsafe_ptr()))
    w.bytes.extend(digest^)
    return w.bytes.copy()


struct _U32Writer:
    var bytes: List[UInt8]

    def __init__(out self):
        self.bytes = List[UInt8]()

    def u32(mut self, v: Int):
        for i in range(4):
            self.bytes.append(UInt8((v >> (8 * i)) & 255))


struct ProofWriter:
    """Collects the proof in order into one host staging pool allocated once per prover (design
    rule 6). Host values are written into the pool directly; device values are staged as async
    copies out of the arena into the pool and assembled after the one synchronize in `finish`, so
    the prover never waits on a read-back mid-stream. A staged multiproof carries its byte count in
    its first u32 (merkle.mojo) and is emitted length-prefixed and trimmed."""
    var ctx: DeviceContext
    var pool: HostBuffer[DType.uint8]
    var pos: Int
    var starts: List[Int]
    var lens: List[Int]
    var multiproof: List[Bool]

    def __init__(out self, ctx: DeviceContext, bytes: Int) raises:
        self.ctx = ctx
        self.pool = ctx.enqueue_create_host_buffer[DType.uint8](bytes)
        ctx.synchronize()
        self.pos = 0
        self.starts = List[Int]()
        self.lens = List[Int]()
        self.multiproof = List[Bool]()

    def reset(mut self):
        self.pos = 0
        self.starts.clear()
        self.lens.clear()
        self.multiproof.clear()

    def _take(mut self, bytes: Int) raises -> Int:
        var start = self.pos
        if start + bytes > len(self.pool):
            raise Error("proof staging pool exhausted")
        self.pos += bytes
        return start

    def scratch(mut self, bytes: Int) raises -> HostBuffer[DType.uint8]:
        """A pool region that is not part of the proof (host bytes to upload)."""
        return self.pool.create_sub_buffer[DType.uint8](self._take(bytes), bytes)

    def raw(mut self, src: List[UInt8]) raises:
        if len(src) == 0:
            return
        var start = self._take(len(src))
        for i in range(len(src)):
            self.pool[start + i] = src[i]
        self.starts.append(start)
        self.lens.append(len(src))
        self.multiproof.append(False)

    def u32(mut self, v: Int) raises:
        var w = _U32Writer()
        w.u32(v)
        self.raw(w.bytes)

    def prefixed(mut self, src: List[UInt8]) raises:
        self.u32(len(src))
        self.raw(src)

    def stage(mut self, arena: Arena, off: Int, bytes: Int, multiproof: Bool = False) raises:
        var start = self._take(bytes)
        arena.download(self.ctx, off, self.pool.create_sub_buffer[DType.uint8](start, bytes))
        self.starts.append(start)
        self.lens.append(bytes)
        self.multiproof.append(multiproof)

    def finish(mut self) raises -> List[UInt8]:
        self.ctx.synchronize()
        var src = self.pool.unsafe_ptr()
        var starts = List[Int]()      # trimmed segments and their u32 prefixes
        var stops = List[Int]()
        var total = 0
        for i in range(len(self.starts)):
            var start = self.starts[i]
            var stop = start + self.lens[i]
            if self.multiproof[i]:
                var n = Int(src[start]) | Int(src[start + 1]) << 8 | Int(src[start + 2]) << 16 | Int(src[start + 3]) << 24
                if n < 4 or n > self.lens[i]:
                    raise Error("multiproof header out of range")
                stop = start + n        # n bytes: the header becomes the u32 prefix, then the body
            starts.append(start)
            stops.append(stop)
            total += stop - start
        var out = List[UInt8](unsafe_uninit_length=total)
        var dst = out.unsafe_ptr()
        var at = 0
        for i in range(len(starts)):
            if self.multiproof[i]:
                var n = stops[i] - starts[i]        # the body follows its own header; emit n - 4 then the body
                var w = _U32Writer()
                w.u32(n - 4)
                for j in range(4):
                    dst[at + j] = w.bytes[j]
                at += 4
                memcpy(dest=dst + at, src=src + starts[i] + 4, count=n - 4)
                at += n - 4
            else:
                memcpy(dest=dst + at, src=src + starts[i], count=stops[i] - starts[i])
                at += stops[i] - starts[i]
        return out^


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

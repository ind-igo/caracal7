"""Proof shape and byte layout (statement-layer section 7, spec 9.5), milestone 1.

Order, every integer little-endian, every field element e bytes:

    header      version u32, public inputs (u32 length + bytes); the parameters are bound through
                the transcript prefix (prefix_bytes), not sent
    W root      H.DIGEST
    Z root      H.DIGEST; Z2 per accumulator (h2 e)
    Q root      H.DIGEST; Q3 (2 h2 e) when there are accumulators
    openings    alpha_{c,p}: P x (columns_w + columns_z + columns_q) x e
    per level l = 2 .. ell-1:
                Mat(y_l) root 32 B; multiproof(s) of level l-1 (u32 length + bytes each; three at
                level 1: W, Z, Q); three sumcheck messages (9 e)
    last        clear vector y_ell (|y_ell| x e); multiproof(s) of level ell-1

Multiproofs are length-prefixed because the sibling frontier depends on the sampled positions.
Everything else has a size fixed by `Shape`, so the verifier can check the total length up front.
"""

from std.math import log2
from std.memory import unsafe_memcpy
from max.gpu.host import DeviceContext, HostBuffer

from caracal7.core.params import Params
from caracal7.core.arena import Arena
from caracal7.relations import ENTRY, ACC, KIND_LOOKUP, shift_points
from caracal7.core.hash import Hash
from caracal7.core.bytes import append_u32, host_base

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
    var columns_z: Int          # accumulator tree, e coordinate columns per accumulator
    var columns_q: Int          # quotient tree, 3 e coordinate columns
    var points: Int             # P opening points
    var entries: Int            # family table entries (residual.mojo)
    var accs: List[UInt8]       # accumulator descriptors (accumulate.mojo), part of the artifact
    var tables: List[List[UInt8]]   # lookup tables, (K, w) bytes each, part of the artifact (milestone-3-lookup.md)
    var tail: List[TailLevel]
    var clear_length: Int       # |y_ell|

    def __init__[p: Params](out self, columns_w: Int, families: List[UInt8], accs: List[UInt8] = List[UInt8](),
                            tables: List[List[UInt8]] = List[List[UInt8]]()) raises:
        """P and the entry count come from the family table (residual.mojo). A lookup descriptor names its
        table by index; the table's row width is the record width.
        TODO(memory): a KIND_MEMORY descriptor (spec 6.4) has no table and its own column roles; validate here."""
        p.check()
        if len(families) % ENTRY != 0 or len(accs) % ACC != 0:
            raise Error("family or accumulator table is not whole entries")
        self.columns_w = columns_w
        self.columns_z = p.e * (len(accs) // ACC)
        self.columns_q = 3 * p.e
        self.points = len(shift_points(families)) // 4
        self.entries = len(families) // ENTRY
        self.accs = accs.copy()
        self.tables = tables.copy()
        for k in range(len(accs) // ACC):
            if (Int(accs[k * ACC]) | Int(accs[k * ACC + 1]) << 8) != columns_w + k * p.e:
                raise Error("accumulator z_col must be columns_w + k e in registration order (the Z tree packs Z_k at that block)")
            for j in range(16):                        # record columns are witness columns (the factor kernel reads the W trace)
                var at = k * ACC + 6 + 2 * j
                var w = Int(accs[k * ACC + 2]) | Int(accs[k * ACC + 3]) << 8 if j < 8 else Int(accs[k * ACC + 4]) | Int(accs[k * ACC + 5]) << 8
                if (j % 8) < w and (Int(accs[at]) | Int(accs[at + 1]) << 8) >= columns_w:
                    raise Error("accumulator record columns must be witness columns")
            if Int(accs[k * ACC + 38]) == KIND_LOOKUP:
                var w = Int(accs[k * ACC + 2]) | Int(accs[k * ACC + 3]) << 8
                var t = Int(accs[k * ACC + 39])
                if t >= len(tables) or len(tables[t]) == 0 or len(tables[t]) % w != 0 or w != (Int(accs[k * ACC + 4]) | Int(accs[k * ACC + 5]) << 8):
                    raise Error("lookup descriptor needs a table of its record width")
        self.tail = tail_schedule[p]()
        self.clear_length = p.N() if len(self.tail) == 0 else self.tail[len(self.tail) - 1].rows

    def accumulators(self) -> Int:
        return len(self.accs) // ACC

    def lookups(self) -> Int:
        var n = 0
        for k in range(self.accumulators()):
            n += 1 if Int(self.accs[k * ACC + 38]) == KIND_LOOKUP else 0
        return n

    def table_rows(self, k: Int) -> Int:
        """K of the table lookup descriptor k reads."""
        return len(self.tables[Int(self.accs[k * ACC + 39])]) // (Int(self.accs[k * ACC + 2]) | Int(self.accs[k * ACC + 3]) << 8)

    def max_table_rows(self) -> Int:
        var m = 0
        for k in range(self.accumulators()):
            if Int(self.accs[k * ACC + 38]) == KIND_LOOKUP:
                m = max(m, self.table_rows(k))
        return m

    def trees(self) -> Int:
        """Trees opened at level 1: W and Q, plus Z when there are accumulators."""
        return 3 if self.columns_z > 0 else 2

    def columns(self) -> Int:
        return self.columns_w + self.columns_z + self.columns_q

    def fixed_bytes[p: Params, digest: Int](self, public_bytes: Int) -> Int:
        """Proof length without the multiproof bodies: their u32 prefixes are counted, one per tree
        opened (three at level 1: W, Z, Q). Mirrors ProofWriter's order exactly."""
        var n = 4 + 4 + public_bytes + 2 * digest
        if self.accumulators() > 0:                    # Z root, Z2, Q3 exist only with accumulators
            n += digest + self.accumulators() * p.h2() * p.e + 2 * p.h2() * p.e
        n += self.points * self.columns() * p.e
        for i in range(len(self.tail)):
            n += digest + (self.trees() if i == 0 else 1) * 4 + 9 * p.e
        n += self.clear_length * p.e + (self.trees() if len(self.tail) == 0 else 1) * 4
        return n

    def write_to(self, mut w: Some[Writer]):
        w.write("Shape(columns=", self.columns_w, "+", self.columns_z, "+", self.columns_q, ", P=", self.points,
                ", tail levels=", len(self.tail), ", clear=", self.clear_length, ")")


def prefix_bytes[p: Params, H: Hash](shape: Shape, public_inputs: Span[UInt8, _], mut families: List[UInt8]) -> List[UInt8]:
    """The transcript prefix of spec 9.4: version, field and grid parameters, domains and rates per
    level, shape, public inputs, and H(family table) as the statement artifact hash of
    statement-layer 6 step 1. Prover and verifier build the same bytes."""
    var bytes = List[UInt8]()
    append_u32(bytes, Int(VERSION))
    for v in [p.e, p.a1, p.m1, p.a2, p.m2, p.L0, p.m_cosets, p.leaf_bytes, p.tail_digits, p.tail_clear_max,
              p.lambda_bits, p.queries(), p.n_cw()]:
        append_u32(bytes, v)
    append_u32(bytes, shape.columns_w)
    append_u32(bytes, shape.columns_z)
    append_u32(bytes, shape.columns_q)
    append_u32(bytes, shape.points)
    append_u32(bytes, len(shape.tail))
    for lvl in shape.tail:
        for v in [lvl.length, lvl.rows, lvl.L, lvl.cosets, lvl.queries]:
            append_u32(bytes, v)
    append_u32(bytes, shape.clear_length)
    append_u32(bytes, len(public_inputs))
    bytes.extend(public_inputs.copy())
    append_u32(bytes, len(shape.accs))
    bytes.extend(shape.accs.copy())
    append_u32(bytes, len(shape.tables))
    for t in shape.tables:
        append_u32(bytes, len(t))
        bytes.extend(t.copy())
    var digest = List[UInt8](length=H.DIGEST, fill=0)
    H.leaf(host_base(families), len(families), host_base(digest))
    bytes.extend(digest^)
    return bytes.copy()


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

    def raw(mut self, src: Span[UInt8, _]) raises:
        if len(src) == 0:
            return
        var start = self._take(len(src))
        for i in range(len(src)):
            self.pool[start + i] = src[i]
        self.starts.append(start)
        self.lens.append(len(src))
        self.multiproof.append(False)

    def u32(mut self, v: Int) raises:
        var bytes = List[UInt8]()
        append_u32(bytes, v)
        self.raw(bytes)

    def prefixed(mut self, src: Span[UInt8, _]) raises:
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
                var n = Int(src[unsafe_offset=start]) | Int(src[unsafe_offset=start + 1]) << 8 | Int(src[unsafe_offset=start + 2]) << 16 | Int(src[unsafe_offset=start + 3]) << 24
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
                var hdr = List[UInt8]()
                append_u32(hdr, n - 4)
                for j in range(4):
                    dst[unsafe_offset=at + j] = hdr[j]
                at += 4
                unsafe_memcpy(dest=dst.unsafe_offset(at), src=src.unsafe_offset(starts[i] + 4), count=n - 4)
                at += n - 4
            else:
                unsafe_memcpy(dest=dst.unsafe_offset(at), src=src.unsafe_offset(starts[i]), count=stops[i] - starts[i])
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

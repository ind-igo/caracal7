"""The sorted copy of a lookup instance (spec 6.3, docs/milestone-3-lookup.md): the record columns
f permuted into table order, written into the trace columns s before encode.

The frontend hands the prover an advice index, one u32 per row, the table position of the record
at that row. It is prover-only: never committed, never hashed. A wrong index gives a sorted copy
whose product identity fails against C_T; it cannot make a false proof pass. Equal records are
identical bytes, so the order inside a bin does not matter and a counting sort is enough.

Buffers (bytes), per lookup instance, over the descriptor layout of accumulate.mojo (num columns
are f, den columns are s, both of width w_num):
    idx      (row, u32)      advice index, uploaded by the frontend
    bins     (K + 1, u32)    histogram, then exclusive prefix sums; bins[K] = N
    cursor   (K, u32)        the scatter's running write position per bin

TODO(memory): spec 6.4 sorts on (addr, ts) keys, where equal keys are not identical records, so a
stable multi-pass radix sort replaces the histogram-scan-scatter triple here. It lands with the
memory descriptor kind; the client profile has no memory (configuration.md 3).

ponytail: one thread scans the K bins; a two-level scan when K grows past a few thousand.
"""

from std.math import ceildiv
from std.gpu import global_idx
from std.atomic import Atomic
from max.gpu.host import DeviceContext

from core.params import Params
from core.backend import BACKEND
from core.bytes import Base, Buf, u16, u32, put_u32
from core.arena import Arena


@always_inline
def _counter(base: Base, b: Buf[4], i: Int) -> MutPointer[Int32, MutAnyOrigin]:
    return b.ptr(base, i).unsafe_bitcast[Int32]()


def k_histogram(base: Base, idx: Buf[4], bins: Buf[4], n: Int32, k: Int32):
    """One thread per row. An index outside the table is dropped, not written: the frontend's advice is
    untrusted input, and a dropped record leaves a sorted copy the verifier rejects."""
    var i = global_idx.x
    if i < Int(n):
        var j = u32(base, idx.at(i))
        if j < Int(k):
            _ = Atomic.fetch_add(_counter(base, bins, j), Int32(1))


def k_scan(base: Base, bins: Buf[4], cursor: Buf[4], k: Int32):
    """One thread: counts to exclusive prefix sums, the cursor starting at each bin's first slot."""
    var total = 0
    for b in range(Int(k)):
        var c = u32(base, bins.at(b))
        put_u32(base, bins.at(b), total)
        put_u32(base, cursor.at(b), total)
        total += c
    put_u32(base, bins.at(Int(k)), total)


def k_scatter[p: Params](base: Base, trace: Buf[1], acc: Buf[1], idx: Buf[4], cursor: Buf[4], k: Int32):
    """One thread per row: copy the record at row i to the next slot of its bin, in the s columns."""
    comptime N = p.N()
    var i = global_idx.x
    if i >= N:
        return
    var j = u32(base, idx.at(i))
    if j >= Int(k):
        return
    var slot = Int(Atomic.fetch_add(_counter(base, cursor, j), Int32(1)))
    var w = u16(base, acc.at(2))
    for j in range(w):
        trace.store(base, u16(base, acc.at(22 + 2 * j)) * N + slot, trace.load(base, u16(base, acc.at(6 + 2 * j)) * N + i))


def counting_sort[p: Params](ctx: DeviceContext, arena: Arena, trace: Int, acc: Int, idx: Int, bins: Int, cursor: Int, k: Int) raises:
    """One lookup instance: its descriptor at `acc`, table size `k`; fills the s columns of the trace."""
    comptime N = p.N()
    comptime B = BACKEND.block
    ctx.enqueue_memset(arena.buf.create_sub_buffer[DType.uint8](bins, 4 * (k + 1)), 0)
    ctx.enqueue_function[k_histogram](arena.buf, Buf[4](idx), Buf[4](bins), Int32(N), Int32(k), grid_dim=ceildiv(N, B), block_dim=B)
    ctx.enqueue_function[k_scan](arena.buf, Buf[4](bins), Buf[4](cursor), Int32(k), grid_dim=1, block_dim=1)
    ctx.enqueue_function[k_scatter[p]](arena.buf, Buf[1](trace), Buf[1](acc), Buf[4](idx), Buf[4](cursor), Int32(k),
                                       grid_dim=ceildiv(N, B), block_dim=B)

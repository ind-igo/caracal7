"""The counting sort on a width-2 lookup instance, reference profile: the s columns equal the host
sort of the records by advice index, the bins are the exclusive counts with one bin left empty."""

from std.testing import assert_equal, TestSuite
from max.gpu.host import DeviceContext, HostBuffer

from core.params import Params, CLIENT
from core.arena import Arena, Bump
from core.bytes import append_u32, set_u16
from relations.accumulate import ACC
from relations.sort import counting_sort

comptime p = CLIENT.grid(72, 32)
comptime N = p.N()
comptime K = 37
comptime EMPTY = 5      # no record has this index; its bin must stay empty
comptime COLS = 4       # f = (c0, c1), s = (c2, c3)


def _host(ctx: DeviceContext, l: List[UInt8]) raises -> HostBuffer[DType.uint8]:
    var h = ctx.enqueue_create_host_buffer[DType.uint8](len(l))
    ctx.synchronize()
    for i in range(len(l)):
        h[i] = l[i]
    return h^


def _down(ctx: DeviceContext, arena: Arena, off: Int, n: Int) raises -> List[UInt8]:
    var h = ctx.enqueue_create_host_buffer[DType.uint8](n)
    arena.download(ctx, off, h)
    ctx.synchronize()
    var l = List[UInt8](capacity=n)
    for i in range(n):
        l.append(h[i])
    return l^


def _descriptor() -> List[UInt8]:
    var d = List[UInt8](length=ACC, fill=0)
    set_u16(d, 2, 2)
    set_u16(d, 4, 2)
    set_u16(d, 6, 0)
    set_u16(d, 8, 1)
    set_u16(d, 22, 2)
    set_u16(d, 24, 3)
    return d^


def test_counting_sort() raises:
    var ctx = DeviceContext()
    # table entry j is the pair (j, 2 j + 1); record i holds the entry at its advice index
    var idx = List[Int](capacity=N)
    var s = 7
    for _ in range(N):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        var j = (s >> 8) % K
        idx.append(j if j != EMPTY else EMPTY + 1)
    var trace = List[UInt8](length=COLS * N, fill=0)
    var idx_bytes = List[UInt8]()
    for i in range(N):
        trace[i] = UInt8(idx[i])
        trace[N + i] = UInt8(2 * idx[i] + 1)
        append_u32(idx_bytes, idx[i])

    var bump = Bump()
    var o_trace = bump.alloc(COLS * N)
    var o_acc = bump.alloc(ACC)
    var o_idx = bump.alloc(4 * N)
    var o_bins = bump.alloc(4 * (K + 1))
    var o_cursor = bump.alloc(4 * K)
    var arena = Arena(ctx, bump.used)
    arena.upload(ctx, o_trace, _host(ctx, trace))
    arena.upload(ctx, o_acc, _host(ctx, _descriptor()))
    arena.upload(ctx, o_idx, _host(ctx, idx_bytes))
    counting_sort[p](ctx, arena, o_trace, o_acc, o_idx, o_bins, o_cursor, K)
    var out = _down(ctx, arena, o_trace, COLS * N)
    var bins = _down(ctx, arena, o_bins, 4 * (K + 1))

    # host: counts, exclusive sums, then the records in bin order
    var count = List[Int](length=K, fill=0)
    for i in range(N):
        count[idx[i]] += 1
    var start = 0
    for b in range(K):
        assert_equal(Int(bins[4 * b]) | Int(bins[4 * b + 1]) << 8 | Int(bins[4 * b + 2]) << 16, start)
        for r in range(count[b]):
            assert_equal(Int(out[2 * N + start + r]), b)
            assert_equal(Int(out[3 * N + start + r]), 2 * b + 1)
        start += count[b]
    assert_equal(Int(bins[4 * K]) | Int(bins[4 * K + 1]) << 8 | Int(bins[4 * K + 2]) << 16, N)
    assert_equal(count[EMPTY], 0)
    for i in range(N):      # f untouched
        assert_equal(Int(out[i]), idx[i])
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks (decisions.md 2026-09-16)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

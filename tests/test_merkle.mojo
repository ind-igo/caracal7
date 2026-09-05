"""Merkle tree with uneven levels (630 leaves) against a host reference; multiproof round trip."""

from std.testing import assert_equal, assert_true, assert_raises, TestSuite
from max.gpu.host import DeviceContext

from caracal7.params import REFERENCE
from caracal7.bytes import Base
from caracal7.hash import Blake3
from caracal7.arena import Arena, Bump
from caracal7.merkle import merkle, query_gather, root_offset, tree_nodes, multiproof_region, check_multiproof

comptime p = REFERENCE
comptime LEAVES = 630
comptime ROW = 12
comptime QUERIES = 9


def _ptr(mut l: List[UInt8]) -> Base:
    return rebind[Base](l.unsafe_ptr())


def _host_root(mut code: List[UInt8]) -> List[UInt8]:
    var level = List[UInt8](length=LEAVES * 32, fill=0)
    for i in range(LEAVES):
        Blake3.leaf(_ptr(code).unsafe_offset(i * ROW), ROW, _ptr(level).unsafe_offset(i * 32))
    var n = LEAVES
    while n > 1:
        var m = (n + 1) // 2
        var next = List[UInt8](length=m * 32, fill=0)
        for j in range(m):
            if 2 * j + 1 < n:
                Blake3.node(_ptr(level).unsafe_offset(2 * j * 32), _ptr(level).unsafe_offset((2 * j + 1) * 32),
                            _ptr(next).unsafe_offset(j * 32))
            else:
                for b in range(32):
                    next[j * 32 + b] = level[2 * j * 32 + b]
        level = next^
        n = m
    return level^


def test_tree_and_multiproof() raises:
    var ctx = DeviceContext()
    var bump = Bump()
    var code_off = bump.alloc(LEAVES * ROW)
    var tree = bump.alloc(tree_nodes(LEAVES) * 32)
    var pos_off = bump.alloc(QUERIES * 4)
    var stage = bump.alloc(multiproof_region[Blake3](ROW, LEAVES, QUERIES))
    var arena = Arena(ctx, bump.used)

    var ch = ctx.enqueue_create_host_buffer[DType.uint8](LEAVES * ROW)
    var ph = ctx.enqueue_create_host_buffer[DType.uint8](QUERIES * 4)
    ctx.synchronize()
    var code = List[UInt8](capacity=LEAVES * ROW)
    for i in range(LEAVES * ROW):
        ch[i] = UInt8((i * 37 + 11) % 127)
        code.append(ch[i])
    var positions: List[Int] = [0, 629, 5, 4, 300, 5, 17, 628, 314]     # duplicates, both edges, odd-level nodes
    for q in range(QUERIES):
        for b in range(4):
            ph[4 * q + b] = UInt8((positions[q] >> (8 * b)) & 255)
    arena.upload(ctx, code_off, ch)
    arena.upload(ctx, pos_off, ph)

    merkle[p, Blake3](ctx, arena.base(), code_off, ROW, LEAVES, tree)
    var bound = query_gather[p, Blake3](ctx, arena.base(), code_off, ROW, LEAVES, tree, pos_off, QUERIES, stage)

    var rh = ctx.enqueue_create_host_buffer[DType.uint8](32)
    var mh = ctx.enqueue_create_host_buffer[DType.uint8](bound)
    arena.download(ctx, root_offset[Blake3](tree, LEAVES), rh)
    arena.download(ctx, stage, mh)
    ctx.synchronize()
    var root = List[UInt8](capacity=32)
    for i in range(32):
        root.append(rh[i])
    assert_equal(root, _host_root(code))

    var n = Int(mh[0]) | Int(mh[1]) << 8 | Int(mh[2]) << 16 | Int(mh[3]) << 24
    assert_true(n <= bound)
    var proof = List[UInt8](capacity=n - 4)
    for i in range(4, n):
        proof.append(mh[i])
    var rows = check_multiproof[Blake3](root, LEAVES, ROW, positions, proof)
    var distinct = [0, 4, 5, 17, 300, 314, 628, 629]
    assert_equal(len(rows), len(distinct) * ROW)
    for i in range(len(distinct)):
        for b in range(ROW):
            assert_equal(rows[i * ROW + b], code[distinct[i] * ROW + b])

    proof[len(proof) - 1] ^= 1
    with assert_raises(contains="root mismatch"):
        _ = check_multiproof[Blake3](root, LEAVES, ROW, positions, proof)
    proof[len(proof) - 1] ^= 1
    var short = proof.copy()
    _ = short.pop()
    with assert_raises(contains="truncated"):
        _ = check_multiproof[Blake3](root, LEAVES, ROW, positions, short)
    var long = proof.copy()
    long.append(0)
    with assert_raises(contains="trailing"):
        _ = check_multiproof[Blake3](root, LEAVES, ROW, positions, long)
    proof[3] ^= 1
    with assert_raises(contains="root mismatch"):
        _ = check_multiproof[Blake3](root, LEAVES, ROW, positions, proof)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

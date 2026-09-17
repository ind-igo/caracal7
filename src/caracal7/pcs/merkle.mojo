"""Merkle tree over leaf-major codeword rows (design section 4): one launch per level, `H.leaf` on
rows, `H.node` above. Levels have ceil(n/2) nodes; an odd last node is copied up unchanged (the
verifier knows every level size, so this is unambiguous). Tree layout (node, H.DIGEST), level 0 first.

Multiproof for sampled positions: [u32 bytes][rows of the sorted distinct positions][sibling frontier],
the frontier walked level by level, ascending, each sibling once. `check_multiproof` is the host side.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import row_major, stack_allocation
from max.gpu.host import DeviceContext

from caracal7.core.params import Params
from caracal7.core.hash import Hash
from caracal7.core.backend import BACKEND
from caracal7.core.bytes import Base, Buf, u32, put_u32, host_base, check_field_bytes
from caracal7.core.arena import Arena
from std.gpu import global_idx

comptime MAX_QUERIES = 1024        # k_frontier holds four Int32 lists of this length in threadgroup memory (16 KB)


def level_sizes(leaves: Int) -> List[Int]:
    var sizes = List[Int]()
    var n = leaves
    sizes.append(n)
    while n > 1:
        n = (n + 1) // 2
        sizes.append(n)
    return sizes^


def tree_nodes(leaves: Int) -> Int:
    var total = 0
    for n in level_sizes(leaves):
        total += n
    return total


def root_offset[H: Hash](tree: Int, leaves: Int) -> Int:
    return tree + (tree_nodes(leaves) - 1) * H.DIGEST


def k_leaves[H: Hash](base: Base, code: Buf[1], row_bytes: Int32, leaves: Int32, tree: Buf[H.DIGEST]):
    var i = Int(global_idx.x)
    if i < Int(leaves):
        H.leaf(code.ptr(base, i * Int(row_bytes)), Int(row_bytes), tree.ptr(base, i))


def k_level[H: Hash](base: Base, src: Buf[H.DIGEST], n: Int32, dst: Buf[H.DIGEST]):
    var j = Int(global_idx.x)
    if j < (Int(n) + 1) // 2:
        if 2 * j + 1 < Int(n):
            H.node(src.ptr(base, 2 * j), src.ptr(base, 2 * j + 1), dst.ptr(base, j))
        else:
            dst.store(base, j, src.load(base, 2 * j))


def merkle[p: Params, H: Hash](ctx: DeviceContext, arena: Arena,
                               code: Int, row_bytes: Int, leaves: Int, tree: Int) raises:
    """Hash `leaves` rows of `row_bytes` at `code` into `tree`."""
    ctx.enqueue_function[k_leaves[H]](arena.buf, Buf[1](code), Int32(row_bytes), Int32(leaves), Buf[H.DIGEST](tree),
                                      grid_dim=(leaves + BACKEND.block - 1) // BACKEND.block, block_dim=BACKEND.block)
    var level = tree
    var n = leaves
    while n > 1:
        var next = level + n * H.DIGEST
        var m = (n + 1) // 2
        ctx.enqueue_function[k_level[H]](arena.buf, Buf[H.DIGEST](level), Int32(n), Buf[H.DIGEST](next),
                                         grid_dim=(m + BACKEND.block - 1) // BACKEND.block, block_dim=BACKEND.block)
        level = next
        n = m


def k_frontier[H: Hash](base: Base, leaves: Int32, positions: Buf[4],
                        count: Int32, row_bytes: Int32, dst: Buf[1], order: Buf[4], sibs: Buf[4]):
    """One block of index arithmetic, no digest traffic. Sorts the positions, writes the distinct list
    to `order` (u32 m, then m u32), the sibling frontier's node indexes to `sibs` (u32 count, then the
    tree node of each sibling in emission order; k_sibs copies the digests), and the total byte count
    at dst[0:4]. Thread i owns list entry i (and i + block, ...): the sort is a rank sort over the
    first occurrences, each level flags its entries (1: second of a sibling pair, dropped; 2: emits
    its sibling) and places them by short scans. One thread doing the same took 0.8 ms a call, a
    chain of dependent instructions with nothing to hide their latency (2026-09-17); this block takes
    0.38 ms. ponytail: the O(m) scans per level are the rest, a tree prefix sum would go under 0.1 ms."""
    comptime assert 4 * MAX_QUERIES * 4 <= BACKEND.threadgroup_bytes, "the frontier's four lists fit threadgroup memory"
    comptime B = BACKEND.block
    var tid = Int(thread_idx.x)
    var cnt = Int(count)
    var pos = stack_allocation[DType.int32, address_space=AddressSpace.SHARED](row_major[MAX_QUERIES]())
    var known = stack_allocation[DType.int32, address_space=AddressSpace.SHARED](row_major[MAX_QUERIES]())
    var next = stack_allocation[DType.int32, address_space=AddressSpace.SHARED](row_major[MAX_QUERIES]())
    var flag = stack_allocation[DType.int32, address_space=AddressSpace.SHARED](row_major[MAX_QUERIES]())
    for i in range(tid, cnt, B):
        pos[i] = Int32(u32(base, positions.at(i)))
    barrier()
    for i in range(tid, cnt, B):                    # first occurrence of its value
        var f = Int32(1)
        for j in range(i):
            if pos[j] == pos[i]:
                f = 0
        flag[i] = f
    barrier()
    var m = 0
    for j in range(cnt):
        if flag[j] == 1:
            m += 1
    for i in range(tid, cnt, B):                    # rank among the first occurrences
        if flag[i] == 1:
            var r = 0
            for j in range(cnt):
                if flag[j] == 1 and pos[j] < pos[i]:
                    r += 1
            known[r] = pos[i]
            put_u32(base, order.at(1 + r), Int(pos[i]))
    if tid == 0:
        put_u32(base, order.at(0), m)
    barrier()

    var w = m
    var ns = 0
    var node = 0                                    # the level's first node in the tree
    var n = Int(leaves)
    while n > 1:
        for i in range(tid, m, B):                  # known is ascending and distinct: a pair is (even k, k + 1)
            var k = Int(known[i])
            var s = k ^ 1
            var f = Int32(0)
            if i > 0 and Int(known[i - 1]) == s:
                f = 1
            elif not (i + 1 < m and Int(known[i + 1]) == s) and s < n:
                f = 2
            flag[i] = f
        barrier()
        for i in range(tid, m, B):
            var f = flag[i]
            if f != 1:
                var drop = 0
                var emit = 0
                for j in range(i):
                    if flag[j] == 1:
                        drop += 1
                    elif flag[j] == 2:
                        emit += 1
                next[i - drop] = Int32(Int(known[i]) >> 1)
                if f == 2:
                    put_u32(base, sibs.at(1 + ns + emit), node + (Int(known[i]) ^ 1))
        var drops = 0
        var emits = 0
        for j in range(m):
            if flag[j] == 1:
                drops += 1
            elif flag[j] == 2:
                emits += 1
        barrier()
        m -= drops
        ns += emits
        for i in range(tid, m, B):
            known[i] = next[i]
        node += n
        n = (n + 1) // 2
        barrier()
    if tid == 0:
        put_u32(base, sibs.at(0), ns)
        put_u32(base, dst.at(0), 4 + w * Int(row_bytes) + ns * H.DIGEST)


def k_sibs[H: Hash](base: Base, tree: Buf[H.DIGEST], row_bytes: Int32, order: Buf[4], sibs: Buf[4], dst: Buf[1]):
    """Block per frontier sibling, thread per byte: copy its digest after the rows."""
    var r = Int(block_idx.x)
    if r >= u32(base, sibs.at(0)):
        return
    var src = tree.at(u32(base, sibs.at(1 + r)))
    var out = dst.at(4 + u32(base, order.at(0)) * Int(row_bytes) + r * H.DIGEST)
    Buf[1](out).store(base, Int(thread_idx.x), Buf[1](src).load(base, Int(thread_idx.x)))


def k_rows(base: Base, code: Buf[1], row_bytes: Int32, order: Buf[4], dst: Buf[1]):
    """Thread per (row, byte): copy the distinct opened rows in ascending order."""
    var r = Int(block_idx.x)
    var m = u32(base, order.at(0))
    if r >= m:
        return
    var pos = u32(base, order.at(1 + r))
    var src = code.at(pos * Int(row_bytes))
    var out = dst.at(4 + r * Int(row_bytes))
    var b = Int(thread_idx.x)
    while b < Int(row_bytes):
        Buf[1](out).store(base, b, Buf[1](src).load(base, b))
        b += Int(block_dim.x)


def multiproof_bound[H: Hash](row_bytes: Int, leaves: Int, count: Int) -> Int:
    """Upper bound on the multiproof bytes; the arena region must also hold the order scratch after it."""
    return 4 + count * row_bytes + count * (len(level_sizes(leaves)) - 1) * H.DIGEST


def multiproof_region[H: Hash](row_bytes: Int, leaves: Int, count: Int) -> Int:
    """The bound, the order scratch (u32 m, m u32), then the sibling index scratch (u32 count, one u32 per
    frontier node: at most count per level)."""
    return multiproof_bound[H](row_bytes, leaves, count) + 4 + 4 * count + 4 + 4 * count * (len(level_sizes(leaves)) - 1)


def query_gather[p: Params, H: Hash](ctx: DeviceContext, arena: Arena,
                                     code: Int, row_bytes: Int, leaves: Int, tree: Int,
                                     positions: Int, count: Int, dst: Int) raises -> Int:
    """Multiproof of the rows at `positions` into `dst`; returns the byte bound to read back
    (the exact length is the u32 at dst)."""
    if count > MAX_QUERIES:
        raise Error("too many queries for the frontier kernel")
    var bound = multiproof_bound[H](row_bytes, leaves, count)
    var order = dst + bound
    var sibs = order + 4 + 4 * count
    var levels = len(level_sizes(leaves)) - 1
    ctx.enqueue_function[k_frontier[H]](arena.buf, Int32(leaves), Buf[4](positions), Int32(count),
                                        Int32(row_bytes), Buf[1](dst), Buf[4](order), Buf[4](sibs), grid_dim=1, block_dim=BACKEND.block)
    ctx.enqueue_function[k_rows](arena.buf, Buf[1](code), Int32(row_bytes), Buf[4](order), Buf[1](dst),
                                 grid_dim=count, block_dim=BACKEND.block)
    ctx.enqueue_function[k_sibs[H]](arena.buf, Buf[H.DIGEST](tree), Int32(row_bytes), Buf[4](order), Buf[4](sibs), Buf[1](dst),
                                    grid_dim=count * levels, block_dim=H.DIGEST)
    return bound


def distinct_sorted(positions: List[Int]) -> List[Int]:
    """The opened rows' order: ascending, each position once."""
    var known = List[Int]()
    for v in positions:
        var i = 0
        while i < len(known) and known[i] < v:
            i += 1
        if i == len(known) or known[i] != v:
            known.insert(i, v)
    return known^


def check_multiproof[H: Hash](root: Span[UInt8, _], leaves: Int, row_bytes: Int, positions: List[Int],
                              mut proof: List[UInt8]) raises -> List[UInt8]:
    """Host side. Recomputes the root from the multiproof (without its u32 header); raises on mismatch.
    Returns canonical F127 row coordinates in ascending distinct position order.
    Only rows are field bytes; roots and sibling digests may contain any byte."""
    var known = distinct_sorted(positions)
    var m = len(known)
    if len(proof) < m * row_bytes:
        raise Error("multiproof truncated")
    var rows = List[UInt8](capacity=m * row_bytes)
    for i in range(m * row_bytes):
        rows.append(proof[i])
    var digests = List[UInt8](length=m * H.DIGEST, fill=0)
    for i in range(m):
        H.leaf(host_base(rows[i * row_bytes:]), row_bytes, host_base(digests[i * H.DIGEST:]))
    var at = m * row_bytes
    var n = leaves
    while n > 1:
        var next_known = List[Int]()
        var next_digests = List[UInt8]()
        var i = 0
        while i < m:
            var k = known[i]
            var s = k ^ 1
            var out = List[UInt8](length=H.DIGEST, fill=0)
            var mine = host_base(digests[i * H.DIGEST:])
            if i + 1 < m and known[i + 1] == s:
                H.node(mine, host_base(digests[(i + 1) * H.DIGEST:]), host_base(out))
                i += 2
            elif s < n:
                if at + H.DIGEST > len(proof):
                    raise Error("multiproof truncated")
                var sib = host_base(proof[at:])
                at += H.DIGEST
                if k & 1 == 0:
                    H.node(mine, sib, host_base(out))
                else:
                    H.node(sib, mine, host_base(out))
                i += 1
            else:
                for b in range(H.DIGEST):
                    out[b] = digests[i * H.DIGEST + b]
                i += 1
            next_known.append(k >> 1)
            next_digests.extend(out^)
        known = next_known^
        digests = next_digests^
        m = len(known)
        n = (n + 1) // 2
    if at != len(proof):
        raise Error("multiproof has trailing bytes")
    if Span(digests) != root:
        raise Error("multiproof root mismatch")
    check_field_bytes(rows)
    return rows^

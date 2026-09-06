"""Merkle tree over leaf-major codeword rows (design section 4): one launch per level, `H.leaf` on
rows, `H.node` above. Levels have ceil(n/2) nodes; an odd last node is copied up unchanged (the
verifier knows every level size, so this is unambiguous). Tree layout (node, H.DIGEST), level 0 first.

Multiproof for sampled positions: [u32 bytes][rows of the sorted distinct positions][sibling frontier],
the frontier walked level by level, ascending, each sibling once. `check_multiproof` is the host side.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext

from caracal7.core.params import Params
from caracal7.core.hash import Hash
from caracal7.core.backend import BACKEND
from caracal7.core.bytes import Base, Buf, u32, put_u32, host_base
from caracal7.core.arena import Arena
from std.gpu import global_idx

comptime MAX_QUERIES = 1024        # ponytail: frontier walk keeps the position list in registers


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


def k_frontier[H: Hash](base: Base, tree: Buf[H.DIGEST], leaves: Int32, positions: Buf[4],
                        count: Int32, row_bytes: Int32, dst: Buf[1], order: Buf[4]):
    """One thread. Sorts the positions, writes the distinct list to `order` (u32 m, then m u32),
    the sibling frontier after the rows, and the total byte count at dst[0:4]."""
    var known = InlineArray[Int32, MAX_QUERIES](fill=0)
    var next = InlineArray[Int32, MAX_QUERIES](fill=0)
    var m = 0
    for q in range(Int(count)):
        var v = Int32(u32(base, positions.at(q)))
        var i = m
        while i > 0 and known[i - 1] > v:
            known[i] = known[i - 1]
            i -= 1
        known[i] = v
        m += 1
    var w = 0
    for i in range(m):
        if i == 0 or known[i] != known[i - 1]:
            known[w] = known[i]
            w += 1
    m = w
    put_u32(base, order.at(0), m)
    for i in range(m):
        put_u32(base, order.at(1 + i), Int(known[i]))

    var out = dst.at(4 + m * Int(row_bytes))
    var level = tree.at(0)
    var n = Int(leaves)
    while n > 1:
        var nm = 0
        var i = 0
        while i < m:
            var k = Int(known[i])
            var s = k ^ 1
            if i + 1 < m and Int(known[i + 1]) == s:
                i += 2
            else:
                if s < n:
                    Buf[H.DIGEST](out).store(base, 0, Buf[H.DIGEST](level).load(base, s))
                    out += H.DIGEST
                i += 1
            next[nm] = Int32(k >> 1)
            nm += 1
        for j in range(nm):
            known[j] = next[j]
        m = nm
        level += n * H.DIGEST
        n = (n + 1) // 2
    put_u32(base, dst.at(0), out - dst.at(0))


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
    return multiproof_bound[H](row_bytes, leaves, count) + 4 + 4 * count


def query_gather[p: Params, H: Hash](ctx: DeviceContext, arena: Arena,
                                     code: Int, row_bytes: Int, leaves: Int, tree: Int,
                                     positions: Int, count: Int, dst: Int) raises -> Int:
    """Multiproof of the rows at `positions` into `dst`; returns the byte bound to read back
    (the exact length is the u32 at dst)."""
    if count > MAX_QUERIES:
        raise Error("too many queries for the frontier kernel")
    var bound = multiproof_bound[H](row_bytes, leaves, count)
    var order = dst + bound
    ctx.enqueue_function[k_frontier[H]](arena.buf, Buf[H.DIGEST](tree), Int32(leaves), Buf[4](positions), Int32(count),
                                        Int32(row_bytes), Buf[1](dst), Buf[4](order), grid_dim=1, block_dim=1)
    ctx.enqueue_function[k_rows](arena.buf, Buf[1](code), Int32(row_bytes), Buf[4](order), Buf[1](dst),
                                 grid_dim=count, block_dim=BACKEND.block)
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
    Returns the opened rows in ascending distinct position order."""
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
    return rows^

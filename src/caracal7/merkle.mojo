"""Merkle tree over leaf-major codeword rows (design section 4): one launch per level, `H.leaf` on
rows, `H.node` above. Levels have ceil(n/2) nodes; an odd last node is copied up unchanged (the
verifier knows every level size, so this is unambiguous). Tree layout (node, H.DIGEST), level 0 first.

Multiproof for sampled positions: [u32 bytes][rows of the sorted distinct positions][sibling frontier],
the frontier walked level by level, ascending, each sibling once. `check_multiproof` is the host side.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext

from caracal7.params import Params
from caracal7.hash import Hash

comptime BLOCK = 256
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


def _gid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def k_leaves[H: Hash](base: Pointer[UInt8, MutAnyOrigin], code: Int64, row_bytes: Int32, leaves: Int32, tree: Int64):
    var i = _gid()
    if i < Int(leaves):
        H.leaf(base.unsafe_offset(Int(code) + i * Int(row_bytes)), Int(row_bytes),
               base.unsafe_offset(Int(tree) + i * H.DIGEST))


def k_level[H: Hash](base: Pointer[UInt8, MutAnyOrigin], src: Int64, n: Int32, dst: Int64):
    var j = _gid()
    if j < (Int(n) + 1) // 2:
        var left = base.unsafe_offset(Int(src) + 2 * j * H.DIGEST)
        var dst_ptr = base.unsafe_offset(Int(dst) + j * H.DIGEST)
        if 2 * j + 1 < Int(n):
            H.node(left, left.unsafe_offset(H.DIGEST), dst_ptr)
        else:
            comptime for b in range(H.DIGEST):
                dst_ptr[unsafe_offset=b] = left[unsafe_offset=b]


def merkle[p: Params, H: Hash](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin],
                               code: Int, row_bytes: Int, leaves: Int, tree: Int) raises:
    """Hash `leaves` rows of `row_bytes` at `code` into `tree`."""
    ctx.enqueue_function[k_leaves[H]](base, Int64(code), Int32(row_bytes), Int32(leaves), Int64(tree),
                                      grid_dim=(leaves + BLOCK - 1) // BLOCK, block_dim=BLOCK)
    var level = tree
    var n = leaves
    while n > 1:
        var next = level + n * H.DIGEST
        var m = (n + 1) // 2
        ctx.enqueue_function[k_level[H]](base, Int64(level), Int32(n), Int64(next),
                                         grid_dim=(m + BLOCK - 1) // BLOCK, block_dim=BLOCK)
        level = next
        n = m


def k_frontier[H: Hash](base: Pointer[UInt8, MutAnyOrigin], tree: Int64, leaves: Int32, positions: Int64,
                        count: Int32, row_bytes: Int32, dst: Int64, order: Int64):
    """One thread. Sorts the positions, writes the distinct list to `order` (u32 m, then m u32),
    the sibling frontier after the rows, and the total byte count at dst[0:4]."""
    var known = InlineArray[Int32, MAX_QUERIES](fill=0)
    var next = InlineArray[Int32, MAX_QUERIES](fill=0)
    var m = 0
    for q in range(Int(count)):
        var v = Int32(0)
        comptime for b in range(4):
            v |= Int32(base[unsafe_offset=Int(positions) + 4 * q + b]) << Int32(8 * b)
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
    var ord_ptr = base.unsafe_offset(Int(order))
    _put_u32(ord_ptr, 0, m)
    for i in range(m):
        _put_u32(ord_ptr, 4 + 4 * i, Int(known[i]))

    var out = Int(dst) + 4 + m * Int(row_bytes)
    var level = Int(tree)
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
                    var src = base.unsafe_offset(level + s * H.DIGEST)
                    comptime for b in range(H.DIGEST):
                        base[unsafe_offset=out + b] = src[unsafe_offset=b]
                    out += H.DIGEST
                i += 1
            next[nm] = Int32(k >> 1)
            nm += 1
        for j in range(nm):
            known[j] = next[j]
        m = nm
        level += n * H.DIGEST
        n = (n + 1) // 2
    _put_u32(base, Int(dst), out - Int(dst))


def _put_u32(ptr: Pointer[UInt8, MutAnyOrigin], off: Int, v: Int):
    comptime for b in range(4):
        ptr[unsafe_offset=off + b] = UInt8((v >> (8 * b)) & 255)


def k_rows(base: Pointer[UInt8, MutAnyOrigin], code: Int64, row_bytes: Int32, order: Int64, dst: Int64):
    """Thread per (row, byte): copy the distinct opened rows in ascending order."""
    var r = Int(block_idx.x)
    var m = 0
    comptime for b in range(4):
        m |= Int(base[unsafe_offset=Int(order) + b]) << (8 * b)
    if r >= m:
        return
    var pos = 0
    comptime for b in range(4):
        pos |= Int(base[unsafe_offset=Int(order) + 4 + 4 * r + b]) << (8 * b)
    var src = Int(code) + pos * Int(row_bytes)
    var out = Int(dst) + 4 + r * Int(row_bytes)
    var b = Int(thread_idx.x)
    while b < Int(row_bytes):
        base[unsafe_offset=out + b] = base[unsafe_offset=src + b]
        b += Int(block_dim.x)


def multiproof_bound[H: Hash](row_bytes: Int, leaves: Int, count: Int) -> Int:
    """Upper bound on the multiproof bytes; the arena region must also hold the order scratch after it."""
    return 4 + count * row_bytes + count * (len(level_sizes(leaves)) - 1) * H.DIGEST


def multiproof_region[H: Hash](row_bytes: Int, leaves: Int, count: Int) -> Int:
    return multiproof_bound[H](row_bytes, leaves, count) + 4 + 4 * count


def query_gather[p: Params, H: Hash](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin],
                                     code: Int, row_bytes: Int, leaves: Int, tree: Int,
                                     positions: Int, count: Int, dst: Int) raises -> Int:
    """Multiproof of the rows at `positions` into `dst`; returns the byte bound to read back
    (the exact length is the u32 at dst)."""
    if count > MAX_QUERIES:
        raise Error("too many queries for the frontier kernel")
    var bound = multiproof_bound[H](row_bytes, leaves, count)
    var order = dst + bound
    ctx.enqueue_function[k_frontier[H]](base, Int64(tree), Int32(leaves), Int64(positions), Int32(count),
                                        Int32(row_bytes), Int64(dst), Int64(order), grid_dim=1, block_dim=1)
    ctx.enqueue_function[k_rows](base, Int64(code), Int32(row_bytes), Int64(order), Int64(dst),
                                 grid_dim=count, block_dim=BLOCK)
    return bound


def _ptr(mut l: List[UInt8]) -> Pointer[UInt8, MutAnyOrigin]:
    return rebind[Pointer[UInt8, MutAnyOrigin]](l.unsafe_ptr())


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


def check_multiproof[H: Hash](root: List[UInt8], leaves: Int, row_bytes: Int, positions: List[Int],
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
        H.leaf(_ptr(rows).unsafe_offset(i * row_bytes), row_bytes, _ptr(digests).unsafe_offset(i * H.DIGEST))
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
            var mine = _ptr(digests).unsafe_offset(i * H.DIGEST)
            if i + 1 < m and known[i + 1] == s:
                H.node(mine, mine.unsafe_offset(H.DIGEST), _ptr(out))
                i += 2
            elif s < n:
                if at + H.DIGEST > len(proof):
                    raise Error("multiproof truncated")
                var sib = _ptr(proof).unsafe_offset(at)
                at += H.DIGEST
                if k & 1 == 0:
                    H.node(mine, sib, _ptr(out))
                else:
                    H.node(sib, mine, _ptr(out))
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
    if digests != root:
        raise Error("multiproof root mismatch")
    return rows^

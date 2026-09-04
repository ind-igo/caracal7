"""Merkle tree over leaf-major codeword rows (design section 4): one launch per level, `H.leaf` on
rows, `H.node` above. Tree layout (node, H.DIGEST), leaves first; root is the last node.
Multiproof: the unique sibling frontier of the sampled positions, each sibling emitted once."""

from max.gpu.host import DeviceContext

from caracal7.params import Params
from caracal7.hash import Hash


def root_offset[H: Hash](tree: Int, leaves: Int) -> Int:
    return tree + (2 * leaves - 2) * H.DIGEST


def merkle[p: Params, H: Hash](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin],
                               code: Int, row_bytes: Int, leaves: Int, tree: Int) raises:
    """Hash `leaves` rows of `row_bytes` at `code` into `tree`."""
    raise Error("not implemented: merkle")


def query_gather[p: Params, H: Hash](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin],
                                     code: Int, row_bytes: Int, leaves: Int, tree: Int,
                                     positions: Int, count: Int, dst: Int) raises -> Int:
    """Opened rows at `positions` plus the sibling frontier, written to `dst`; returns the byte count."""
    raise Error("not implemented: query_gather")

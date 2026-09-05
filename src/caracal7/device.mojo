"""TEMPORARY shim during the bytes.mojo migration; deleted when the last module imports bytes instead.
byte layouts (one buffer type, docs/design.md rule 2)."""

from std.gpu import thread_idx, block_idx, block_dim

from caracal7.field import E


@always_inline
def gid() -> Int:
    return Int(block_idx.x * block_dim.x + thread_idx.x)


@always_inline
def load_e(base: Pointer[UInt8, MutAnyOrigin], off: Int) -> E:
    return base.unsafe_load[width=16](off)


@always_inline
def load_u16(base: Pointer[UInt8, MutAnyOrigin], at: Int) -> Int:
    return Int(base[unsafe_offset=at]) | Int(base[unsafe_offset=at + 1]) << 8

"""The byte vocabulary (docs/design.md rule 2): the arena base, width-typed regions of it, and the
host readers of the same layouts. Every load and store of a committed layout goes through here; no
kernel module defines its own."""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder

from caracal7.field import E

comptime Base = MutPointer[UInt8, MutAnyOrigin]   # the arena base every kernel takes


struct Buf[W: Int](TrivialRegisterPassable, DevicePassable):
    """A region of the arena holding W-byte elements, element i at off + i W. One Int64 in a
    register; the width is the type, so an F2 buffer cannot be passed where an E buffer is read."""
    var off: Int64

    comptime device_type: AnyType = Self

    def __init__(out self, off: Int):
        self.off = Int64(off)

    def _to_device_type(self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "Buf"

    @always_inline
    def at(self, i: Int) -> Int:
        """Byte offset of element i."""
        return Int(self.off) + i * Self.W

    @always_inline
    def offset(self, i: Int) -> Self:
        """The region starting at element i."""
        return Self(self.at(i))

    @always_inline
    def load(self, base: Base, i: Int) -> SIMD[DType.uint8, Self.W]:
        return base.unsafe_load[width=Self.W](self.at(i))

    @always_inline
    def store(self, base: Base, i: Int, v: SIMD[DType.uint8, Self.W]):
        base.unsafe_store[width=Self.W](self.at(i), v)


# ---- device reads of the little-endian integer fields of descriptors and headers ----

@always_inline
def u16(base: Base, off: Int) -> Int:
    return Int(base[unsafe_offset=off]) | Int(base[unsafe_offset=off + 1]) << 8


@always_inline
def u32(base: Base, off: Int) -> Int:
    var v = base.unsafe_load[width=4](off)
    return Int(v[0]) | Int(v[1]) << 8 | Int(v[2]) << 16 | Int(v[3]) << 24


@always_inline
def put_u32(base: Base, off: Int, v: Int):
    comptime for b in range(4):
        base[unsafe_offset=off + b] = UInt8((v >> (8 * b)) & 255)


# ---- host side, the same layouts in a List ----

def get_u16(l: List[UInt8], at: Int) -> Int:
    return Int(l[at]) | Int(l[at + 1]) << 8


def set_u16(mut l: List[UInt8], at: Int, v: Int):
    l[at] = UInt8(v & 255)
    l[at + 1] = UInt8(v >> 8)


def append_u32(mut l: List[UInt8], v: Int):
    for i in range(4):
        l.append(UInt8((v >> (8 * i)) & 255))


def list_e(l: List[UInt8], i: Int) -> E:
    """Element i of a (.., e) List."""
    var v = E(0)
    for t in range(16):
        v[t] = l[i * 16 + t]
    return v

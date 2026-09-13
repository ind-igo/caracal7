"""The byte vocabulary (docs/design.md rule 2): the arena base, width-typed regions of it, and the
host readers of the same layouts. Every load and store of a committed layout goes through here; no
kernel module defines its own."""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder

from caracal7.core.field import E, E_BYTES, E_WIDTH

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
    def ptr(self, base: Base, i: Int) -> Base:
        """Pointer to element i, for the Hash methods."""
        return base.unsafe_offset(self.at(i))

    # An E value is E_BYTES in memory and E_WIDTH lanes in a register (field.mojo): the access is 16 + 4.
    comptime split: Bool = Self.W == E_BYTES and E_BYTES != E_WIDTH
    comptime R: Int = E_WIDTH if Self.split else Self.W

    @always_inline
    def load(self, base: Base, i: Int) -> SIMD[DType.uint8, Self.R]:
        comptime if Self.split:
            var lo = base.unsafe_load[width=16](self.at(i))
            var hi = base.unsafe_load[width=4](self.at(i) + 16)
            var r = lo.join(SIMD[DType.uint8, 16](0))       # lane stores: Metal has no llvm.vector.insert
            comptime for l in range(4):
                r[16 + l] = hi[l]
            return rebind[SIMD[DType.uint8, Self.R]](r)
        else:
            return rebind[SIMD[DType.uint8, Self.R]](base.unsafe_load[width=Self.W](self.at(i)))

    @always_inline
    def store(self, base: Base, i: Int, v: SIMD[DType.uint8, Self.R]):
        comptime if Self.split:
            var x = rebind[SIMD[DType.uint8, E_WIDTH]](v)
            base.unsafe_store[width=16](self.at(i), x.slice[16]())
            base.unsafe_store[width=4](self.at(i) + 16, x.slice[4, offset=16]())
        else:
            base.unsafe_store[width=Self.W](self.at(i), rebind[SIMD[DType.uint8, Self.W]](v))


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


@always_inline
def u64(base: Base, off: Int) -> Int:
    var v = base.unsafe_load[width=8](off)
    var c = 0
    comptime for i in range(8):
        c |= Int(v[i]) << (8 * i)
    return c


@always_inline
def put_u64(base: Base, off: Int, v: Int):
    comptime for b in range(8):
        base[unsafe_offset=off + b] = UInt8((v >> (8 * b)) & 255)


# ---- host side, the same layouts in a byte Span (a List converts, slices are free) ----

def host_base(s: Span[UInt8, _]) -> Base:
    """The one place host bytes become a Base, for the Hash methods that run on both sides."""
    return rebind[Base](s.unsafe_ptr())


def get_u16(l: Span[UInt8, _], at: Int) -> Int:
    return Int(l[at]) | Int(l[at + 1]) << 8


def set_u16(mut l: List[UInt8], at: Int, v: Int):
    l[at] = UInt8(v & 255)
    l[at + 1] = UInt8(v >> 8)


def append_u32(mut l: List[UInt8], v: Int):
    for i in range(4):
        l.append(UInt8((v >> (8 * i)) & 255))


def check_field_bytes(bytes: Span[UInt8, _]) raises:
    """Validate F127 coordinates once at a host trust boundary, before field arithmetic."""
    for b in bytes:
        if b >= 127:
            raise Error("noncanonical field byte (expected < 127)")


def list_e(l: Span[UInt8, _], i: Int) -> E:
    """Element i of a (.., e) byte span; coordinates must already be canonical (< 127)."""
    var v = E(0)
    for t in range(E_BYTES):
        v[t] = l[i * E_BYTES + t]
    return v

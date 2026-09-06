"""Fiat-Shamir transcript (docs/design.md section 6, spec 9.4), device resident with a host mirror.

State (STATE_BYTES in the arena): H.DIGEST bytes of hash state, a u64 squeeze counter, H.DIGEST bytes
of squeeze scratch. `absorb` runs H.absorb under a domain separator and resets the counter; a squeeze
streams H.squeeze(state, counter++) blocks through rejection sampling: F elements are bytes below 127,
positions are u32 below `below`. The same `sample` runs in the device kernel and in `HostTranscript`.
"""

from max.gpu.host import DeviceContext

from caracal7.core.params import Params
from caracal7.core.arena import Bump
from caracal7.core.hash import Hash
from caracal7.core.bytes import Base, Buf, u32, put_u32, u64, put_u64, host_base
from caracal7.core.arena import Arena

# Domain separators, one per line of spec 9.4, in transcript order.
comptime DS_PREFIX: UInt8 = 0       # protocol version, tower constants, grid, domains and rates, public inputs
comptime DS_TREE_W: UInt8 = 1       # -> beta_1, delta, gamma (stage 1)
comptime DS_TREE_Z: UInt8 = 2       # with Z2 -> alpha
comptime DS_TREE_Q: UInt8 = 3       # with Q3 -> z
comptime DS_OPENINGS: UInt8 = 4     # alpha_{c,p} -> beta (E^columns), gamma (E^P)
comptime DS_TAIL_ROOT: UInt8 = 5    # Mat(y_l) root -> S_{l-1}; the batching scalars follow the multiproof(s)
comptime DS_TAIL_ROUND: UInt8 = 6   # sumcheck message s_i -> r_i
comptime DS_CLEAR: UInt8 = 7        # y_ell in the clear -> S_{ell-1}

comptime STATE_BYTES = 128          # [0, DIGEST) state, [64, 72) counter, [72, 72 + DIGEST) scratch
comptime _COUNTER = 64
comptime _SCRATCH = 72


struct TranscriptLayout(TrivialRegisterPassable):
    """Arena offset of the hash state. Challenges are squeezed into caller-owned regions, one per
    challenge of the schedule, so nothing is overwritten before the kernel that reads it runs."""
    var state: Int

    def __init__(out self, mut bump: Bump):
        self.state = bump.alloc(STATE_BYTES)


def _get_counter(state: Base) -> Int:
    return u64(state, _COUNTER)


def _set_counter(state: Base, c: Int):
    put_u64(state, _COUNTER, c)


def absorb_into[H: Hash](state: Base, ds: UInt8, src: Base, bytes: Int):
    H.absorb(state, ds, src, bytes)
    _set_counter(state, 0)


def sample[H: Hash](state: Base, dst: Base, count: Int, below: Int):
    """below == 0: `count` F elements, one byte each. Else `count` u32 positions below `below`."""
    var scratch = state.unsafe_offset(_SCRATCH)
    var counter = _get_counter(state)
    var pos = H.DIGEST
    var produced = 0
    var step = 1 if below == 0 else 4
    var limit = UInt64(0) if below == 0 else UInt64(1 << 32) - (UInt64(1 << 32) % UInt64(below))
    while produced < count:
        if pos + step > H.DIGEST:
            H.squeeze(state, counter, scratch)
            counter += 1
            pos = 0
        if below == 0:
            var b = scratch[unsafe_offset=pos] & 127
            pos += 1
            if b != 127:
                dst[unsafe_offset=produced] = b
                produced += 1
        else:
            var u = UInt64(u32(scratch, pos))
            pos += 4
            if u < limit:
                var v = u % UInt64(below)
                put_u32(dst, 4 * produced, Int(v))
                produced += 1
    _set_counter(state, counter)


def k_reset(base: Base, state: Buf[1]):
    comptime for i in range(STATE_BYTES):
        state.store(base, i, 0)


def k_absorb[H: Hash](base: Base, state: Buf[1], ds: UInt8, src: Buf[1], bytes: Int32):
    absorb_into[H](state.ptr(base, 0), ds, src.ptr(base, 0), Int(bytes))


def k_sample[H: Hash](base: Base, state: Buf[1], dst: Buf[1], count: Int32, below: Int32):
    sample[H](state.ptr(base, 0), dst.ptr(base, 0), Int(count), Int(below))


def reset(ctx: DeviceContext, arena: Arena, t: TranscriptLayout) raises:
    """Zero the state: every proof starts from the same transcript as the verifier."""
    ctx.enqueue_function[k_reset](arena.buf, Buf[1](t.state), grid_dim=1, block_dim=1)


def absorb[p: Params, H: Hash](ctx: DeviceContext, arena: Arena, t: TranscriptLayout,
                               ds: UInt8, src: Int, bytes: Int) raises:
    """Hash `bytes` at arena offset `src` into the state under separator `ds`. One thread: serial by definition."""
    ctx.enqueue_function[k_absorb[H]](arena.buf, Buf[1](t.state), ds, Buf[1](src), Int32(bytes), grid_dim=1, block_dim=1)


def squeeze_elements[p: Params, H: Hash](ctx: DeviceContext, arena: Arena, t: TranscriptLayout,
                                         dst: Int, count: Int) raises:
    """Write `count` E elements (e bytes each) to arena offset `dst`."""
    ctx.enqueue_function[k_sample[H]](arena.buf, Buf[1](t.state), Buf[1](dst), Int32(count * p.e), Int32(0),
                                      grid_dim=1, block_dim=1)


def squeeze_positions[p: Params, H: Hash](ctx: DeviceContext, arena: Arena, t: TranscriptLayout,
                                          dst: Int, count: Int, below: Int) raises:
    """Write `count` uniform positions below `below` (u32 each) to arena offset `dst`."""
    ctx.enqueue_function[k_sample[H]](arena.buf, Buf[1](t.state), Buf[1](dst), Int32(count), Int32(below),
                                      grid_dim=1, block_dim=1)


struct HostTranscript[p: Params, H: Hash]:
    """Host mirror for the verifier: same state, same separators, same sampling code."""
    var state: List[UInt8]

    def __init__(out self):
        self.state = List[UInt8](length=STATE_BYTES, fill=0)

    def absorb(mut self, ds: UInt8, msg: Span[UInt8, _]):
        absorb_into[Self.H](host_base(self.state), ds, host_base(msg), len(msg))

    def elements(mut self, count: Int) -> List[UInt8]:
        var out = List[UInt8](length=count * Self.p.e, fill=0)
        sample[Self.H](host_base(self.state), host_base(out), count * Self.p.e, 0)
        return out^

    def positions(mut self, count: Int, below: Int) -> List[Int]:
        var raw = List[UInt8](length=4 * count, fill=0)
        sample[Self.H](host_base(self.state), host_base(raw), count, below)
        var out = List[Int](capacity=count)
        for i in range(count):
            out.append(Int(raw[4 * i]) | Int(raw[4 * i + 1]) << 8 | Int(raw[4 * i + 2]) << 16 | Int(raw[4 * i + 3]) << 24)
        return out^

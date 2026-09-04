"""Fiat-Shamir transcript (docs/design.md section 6, spec 9.4), device resident with a host mirror.

State (STATE_BYTES in the arena): H.DIGEST bytes of hash state, a u64 squeeze counter, H.DIGEST bytes
of squeeze scratch. `absorb` runs H.absorb under a domain separator and resets the counter; a squeeze
streams H.squeeze(state, counter++) blocks through rejection sampling: F elements are bytes below 127,
positions are u32 below `below`. The same `sample` runs in the device kernel and in `HostTranscript`.
"""

from max.gpu.host import DeviceContext

from caracal7.params import Params
from caracal7.arena import Bump
from caracal7.hash import Hash

# Domain separators, one per line of spec 9.4, in transcript order.
comptime DS_PREFIX: UInt8 = 0       # protocol version, tower constants, grid, domains and rates, public inputs
comptime DS_TREE_W: UInt8 = 1       # -> beta_1, delta, gamma (stage 1)
comptime DS_TREE_Z: UInt8 = 2       # with Z2 -> alpha
comptime DS_TREE_Q: UInt8 = 3       # with Q3 -> z
comptime DS_OPENINGS: UInt8 = 4     # alpha_{c,p} -> beta (E^columns), gamma (E^P)
comptime DS_TAIL_ROOT: UInt8 = 5    # Mat(y_l) root -> S_{l-1}
comptime DS_TAIL_V: UInt8 = 6       # expected symbols v -> batching scalars
comptime DS_TAIL_ROUND: UInt8 = 7   # sumcheck message s_i -> r_i
comptime DS_CLEAR: UInt8 = 8        # y_ell in the clear -> S_{ell-1}

comptime STATE_BYTES = 128          # [0, DIGEST) state, [64, 72) counter, [72, 72 + DIGEST) scratch
comptime _COUNTER = 64
comptime _SCRATCH = 72


struct TranscriptLayout(TrivialRegisterPassable):
    """Arena offset of the hash state. Challenges are squeezed into caller-owned regions, one per
    challenge of the schedule, so nothing is overwritten before the kernel that reads it runs."""
    var state: Int

    def __init__(out self, mut bump: Bump):
        self.state = bump.alloc(STATE_BYTES)


def _get_counter(state: Pointer[UInt8, MutAnyOrigin]) -> Int:
    var c = 0
    comptime for i in range(8):
        c |= Int(state[unsafe_offset=_COUNTER + i]) << (8 * i)
    return c


def _set_counter(state: Pointer[UInt8, MutAnyOrigin], c: Int):
    comptime for i in range(8):
        state[unsafe_offset=_COUNTER + i] = UInt8((c >> (8 * i)) & 255)


def absorb_into[H: Hash](state: Pointer[UInt8, MutAnyOrigin], ds: UInt8, src: Pointer[UInt8, MutAnyOrigin], bytes: Int):
    H.absorb(state, ds, src, bytes)
    _set_counter(state, 0)


def sample[H: Hash](state: Pointer[UInt8, MutAnyOrigin], dst: Pointer[UInt8, MutAnyOrigin], count: Int, below: Int):
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
            var u = UInt64(0)
            comptime for i in range(4):
                u |= UInt64(scratch[unsafe_offset=pos + i]) << UInt64(8 * i)
            pos += 4
            if u < limit:
                var v = u % UInt64(below)
                comptime for i in range(4):
                    dst[unsafe_offset=4 * produced + i] = UInt8((v >> UInt64(8 * i)) & 255)
                produced += 1
    _set_counter(state, counter)


def k_reset(base: Pointer[UInt8, MutAnyOrigin], state: Int64):
    comptime for i in range(STATE_BYTES):
        base[unsafe_offset=Int(state) + i] = 0


def k_absorb[H: Hash](base: Pointer[UInt8, MutAnyOrigin], state: Int64, ds: UInt8, src: Int64, bytes: Int32):
    absorb_into[H](base.unsafe_offset(Int(state)), ds, base.unsafe_offset(Int(src)), Int(bytes))


def k_sample[H: Hash](base: Pointer[UInt8, MutAnyOrigin], state: Int64, dst: Int64, count: Int32, below: Int32):
    sample[H](base.unsafe_offset(Int(state)), base.unsafe_offset(Int(dst)), Int(count), Int(below))


def reset(ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin], t: TranscriptLayout) raises:
    """Zero the state: every proof starts from the same transcript as the verifier."""
    ctx.enqueue_function[k_reset](base, Int64(t.state), grid_dim=1, block_dim=1)


def absorb[p: Params, H: Hash](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin], t: TranscriptLayout,
                               ds: UInt8, src: Int, bytes: Int) raises:
    """Hash `bytes` at arena offset `src` into the state under separator `ds`. One thread: serial by definition."""
    ctx.enqueue_function[k_absorb[H]](base, Int64(t.state), ds, Int64(src), Int32(bytes), grid_dim=1, block_dim=1)


def squeeze_elements[p: Params, H: Hash](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin], t: TranscriptLayout,
                                         dst: Int, count: Int) raises:
    """Write `count` E elements (e bytes each) to arena offset `dst`."""
    ctx.enqueue_function[k_sample[H]](base, Int64(t.state), Int64(dst), Int32(count * p.e), Int32(0),
                                      grid_dim=1, block_dim=1)


def squeeze_positions[p: Params, H: Hash](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin], t: TranscriptLayout,
                                          dst: Int, count: Int, below: Int) raises:
    """Write `count` uniform positions below `below` (u32 each) to arena offset `dst`."""
    ctx.enqueue_function[k_sample[H]](base, Int64(t.state), Int64(dst), Int32(count), Int32(below),
                                      grid_dim=1, block_dim=1)


def _ptr(mut l: List[UInt8]) -> Pointer[UInt8, MutAnyOrigin]:
    return rebind[Pointer[UInt8, MutAnyOrigin]](l.unsafe_ptr())


struct HostTranscript[p: Params, H: Hash]:
    """Host mirror for the verifier: same state, same separators, same sampling code."""
    var state: List[UInt8]

    def __init__(out self):
        self.state = List[UInt8](length=STATE_BYTES, fill=0)

    def absorb(mut self, ds: UInt8, mut msg: List[UInt8]):
        absorb_into[Self.H](_ptr(self.state), ds, _ptr(msg), len(msg))

    def elements(mut self, count: Int) -> List[UInt8]:
        var out = List[UInt8](length=count * Self.p.e, fill=0)
        sample[Self.H](_ptr(self.state), _ptr(out), count * Self.p.e, 0)
        return out^

    def positions(mut self, count: Int, below: Int) -> List[Int]:
        var raw = List[UInt8](length=4 * count, fill=0)
        sample[Self.H](_ptr(self.state), _ptr(raw), count, below)
        var out = List[Int](capacity=count)
        for i in range(count):
            out.append(Int(raw[4 * i]) | Int(raw[4 * i + 1]) << 8 | Int(raw[4 * i + 2]) << 16 | Int(raw[4 * i + 3]) << 24)
        return out^

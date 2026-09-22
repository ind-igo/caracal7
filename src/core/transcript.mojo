"""Fiat-Shamir transcript (docs/design.md section 6, spec 9.4), device resident with a host mirror.

State (STATE_BYTES in the arena): H.DIGEST bytes of hash state, a u64 squeeze counter, H.DIGEST bytes
of squeeze scratch. `absorb` runs H.absorb under a domain separator and resets the counter; a squeeze
streams H.squeeze(state, counter++) blocks through rejection sampling: F elements are bytes below 127,
positions are u32 below `below`. The same `sample` runs in the device kernel and in `HostTranscript`.
"""

from std.math import ceildiv
from std.gpu import thread_idx, global_idx
from std.atomic import Atomic
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from layout import row_major, stack_allocation

from core.params import Params
from core.arena import Bump
from core.hash import Hash
from core.bytes import Base, Buf, u32, put_u32, u64, put_u64, host_base
from core.arena import Arena
from core.backend import BACKEND

# Domain separators, one per line of spec 9.4, in transcript order.
comptime DS_PREFIX: UInt8 = 0       # protocol version, tower constants, grid, domains and rates, public inputs
comptime DS_TREE_W: UInt8 = 1       # -> beta_1, delta, gamma (stage 1)
comptime DS_TREE_Z: UInt8 = 2       # with Z2 -> alpha
comptime DS_TREE_Q: UInt8 = 3       # with Q3 -> z
comptime DS_OPENINGS: UInt8 = 4     # alpha_{c,p} -> beta (E^columns), gamma (E^P)
comptime DS_TAIL_ROOT: UInt8 = 5    # Mat(y_l) root -> S_{l-1}; the batching scalars follow the multiproof(s)
comptime DS_TAIL_ROUND: UInt8 = 6   # sumcheck message s_i -> r_i
comptime DS_CLEAR: UInt8 = 7        # y_ell in the clear -> S_{ell-1}
comptime DS_GRIND: UInt8 = 8        # the nonce of a query seed -> the grind word (block 0), then S from block 1 on

comptime STATE_BYTES = 128          # [0, DIGEST) state, [64, 72) counter, [72, 72 + DIGEST) scratch
comptime _COUNTER = 64
comptime _SCRATCH = 72
comptime MAX_CHUNKS = 1024          # ponytail: one block per absorb, 1 MiB; larger messages fall back to the serial thread
comptime _CV_BYTES = 32             # per-chunk value slot, >= H.DIGEST
comptime GRIND_THREADS = BACKEND.grind_threads   # nonce search: threads, each walking nonces t, t + GRIND_THREADS, ...
comptime GRIND_POLL = BACKEND.grind_poll         # iterations between polls of `found`
comptime _NO_NONCE: UInt32 = 0xFFFFFFFF


struct TranscriptLayout(TrivialRegisterPassable):
    """Arena offset of the hash state. Challenges are squeezed into caller-owned regions, one per
    challenge of the schedule, so nothing is overwritten before the kernel that reads it runs."""
    var state: Int
    var cvs: Int      # chunk values of one multi-chunk absorb

    def __init__(out self, mut bump: Bump):
        self.state = bump.alloc(STATE_BYTES)
        self.cvs = bump.alloc(MAX_CHUNKS * _CV_BYTES)


def _get_counter(state: Base) -> Int:
    return u64(state, _COUNTER)


def _set_counter(state: Base, c: Int):
    put_u64(state, _COUNTER, c)


def absorb_into[H: Hash](state: Base, ds: UInt8, src: Base, bytes: Int):
    H.absorb(state, ds, src, bytes)
    _set_counter(state, 0)


def grind_word[H: Hash](state: Base) -> UInt32:
    """The proof-of-work word after a nonce is absorbed: the first u32 of squeeze block 0; the counter moves
    to 1 so the positions sampled next come from block 1 on."""
    var scratch = state.unsafe_offset(_SCRATCH)
    H.squeeze(state, 0, scratch)
    _set_counter(state, 1)
    return UInt32(u32(scratch, 0))


@always_inline
def grind_ok(word: UInt32, bits: Int) -> Bool:
    """The top `bits` bits of the word are zero."""
    return bits == 0 or (word >> UInt32(32 - bits)) == 0


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


def k_absorb_tree[H: Hash](base: Base, state: Buf[1], ds: UInt8, src: Buf[1], bytes: Int32, cvs: Buf[1], n: Int32):
    """One block, one thread per chunk; thread 0 merges after the barrier. Same state as `k_absorb`."""
    var k = Int(thread_idx.x)
    var st = state.ptr(base, 0)
    if k < Int(n):
        H.chunk(st, ds, src.ptr(base, 0), Int(bytes), k, cvs.ptr(base, H.DIGEST * k))
    barrier()
    if k == 0:
        H.merge(st, Int(n), cvs.ptr(base, 0))
        _set_counter(st, 0)


def k_sample[H: Hash](base: Base, state: Buf[1], dst: Buf[1], count: Int32, below: Int32):
    sample[H](state.ptr(base, 0), dst.ptr(base, 0), Int(count), Int(below))


def k_grind_init(base: Base, found: Buf[4]):
    put_u32(base, found.at(0), Int(_NO_NONCE))


def k_grind[H: Hash](base: Base, state: Buf[1], bits: Int32, found: Buf[4]):
    """Thread t tries the nonces t, t + GRIND_THREADS, ... with `H.grind_probe` (two compressions from the
    state's key words, nothing written) until a nonce has been found; the smallest passing nonce wins
    (Atomic.min), so the proof is deterministic. Every GRIND_POLL iterations one thread per block reads
    `found` atomically (a plain load is hoisted out of the loop; 32768 atomics per poll were a third of
    the search) and the block leaves together once its nonces pass the one found, so every thread of a
    block meets every barrier. There is no try limit. Probes use the low 32 nonce bits, so a
    fixed state has only 2^32 distinct trials; wraparound repeats them. The usual 2^bits
    fresh-trial expectation does not guarantee termination for this finite search space.
    A limit below the thread count left blocks partial at barriers, and an exhausted
    search staged an invalid nonce. See docs/security-assurance.md for the open work model."""
    var t = Int(global_idx.x)
    var src = state.ptr(base, 0)
    var flag = stack_allocation[DType.uint32, address_space=AddressSpace.SHARED](row_major[1]())
    var k = t
    while True:
        if (k // GRIND_THREADS) % GRIND_POLL == 0:
            if thread_idx.x == 0:
                flag[0] = Atomic.fetch_add(found.ptr(base, 0).unsafe_bitcast[UInt32](), UInt32(0))
            barrier()
            var done = UInt32(k - Int(thread_idx.x)) > flag[0]   # block-uniform: every nonce below the one found is tried
            barrier()
            if done:
                return
        if grind_ok(H.grind_probe(src, DS_GRIND, UInt32(k)), Int(bits)):
            Atomic.min(found.ptr(base, 0).unsafe_bitcast[UInt32](), UInt32(k))
        k += GRIND_THREADS


def k_grind_commit[H: Hash](base: Base, state: Buf[1], found: Buf[4], nonce: Buf[1]):
    """Write the nonce found (8 bytes, little-endian u64) where the proof stages it, absorb it, and take
    the grind word so the positions come from block 1 on."""
    var n = u32(base, found.at(0))
    put_u32(base, nonce.at(0), n)
    put_u32(base, nonce.at(0) + 4, 0)
    absorb_into[H](state.ptr(base, 0), DS_GRIND, nonce.ptr(base, 0), 8)
    _ = grind_word[H](state.ptr(base, 0))


def reset(ctx: DeviceContext, arena: Arena, t: TranscriptLayout) raises:
    """Zero the state: every proof starts from the same transcript as the verifier."""
    ctx.enqueue_function[k_reset](arena.buf, Buf[1](t.state), grid_dim=1, block_dim=1)


def absorb[p: Params, H: Hash](ctx: DeviceContext, arena: Arena, t: TranscriptLayout,
                               ds: UInt8, src: Int, bytes: Int) raises:
    """Hash `bytes` at arena offset `src` into the state under separator `ds`: one thread per chunk of
    the message `ds || src`, or one thread when there is one chunk."""
    comptime assert H.DIGEST <= _CV_BYTES
    var n = ceildiv(bytes + 1, H.CHUNK)
    if n <= 1 or n > MAX_CHUNKS:      # one chunk, or past the one-block ceiling: the serial thread
        ctx.enqueue_function[k_absorb[H]](arena.buf, Buf[1](t.state), ds, Buf[1](src), Int32(bytes), grid_dim=1, block_dim=1)
        return
    ctx.enqueue_function[k_absorb_tree[H]](arena.buf, Buf[1](t.state), ds, Buf[1](src), Int32(bytes),
                                           Buf[1](t.cvs), Int32(n), grid_dim=1, block_dim=n)


def squeeze_elements[p: Params, H: Hash](ctx: DeviceContext, arena: Arena, t: TranscriptLayout,
                                         dst: Int, count: Int) raises:
    """Write `count` E elements (e bytes each) to arena offset `dst`."""
    ctx.enqueue_function[k_sample[H]](arena.buf, Buf[1](t.state), Buf[1](dst), Int32(count * p.e), Int32(0),
                                      grid_dim=1, block_dim=1)


def grind[p: Params, H: Hash](ctx: DeviceContext, arena: Arena, t: TranscriptLayout, found: Int, nonce: Int) raises:
    """Find and absorb the nonce of the next query seed (p.grind_bits bits of proof of work), the nonce
    written to arena offset `nonce` for the proof."""
    ctx.enqueue_function[k_grind_init](arena.buf, Buf[4](found), grid_dim=1, block_dim=1)
    ctx.enqueue_function[k_grind[H]](arena.buf, Buf[1](t.state), Int32(p.grind_bits), Buf[4](found),
                                     grid_dim=GRIND_THREADS // 256, block_dim=256)
    ctx.enqueue_function[k_grind_commit[H]](arena.buf, Buf[1](t.state), Buf[4](found), Buf[1](nonce), grid_dim=1, block_dim=1)


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

    def grind(mut self, nonce: Span[UInt8, _], bits: Int) raises:
        """Absorb a query seed's nonce and check its proof of work."""
        if len(nonce) != 8:
            raise Error("grinding nonce is 8 bytes")
        self.absorb(DS_GRIND, nonce)
        if not grind_ok(grind_word[Self.H](host_base(self.state)), bits):
            raise Error("grinding check fails")

    def positions(mut self, count: Int, below: Int) -> List[Int]:
        var raw = List[UInt8](length=4 * count, fill=0)
        sample[Self.H](host_base(self.state), host_base(raw), count, below)
        var out = List[Int](capacity=count)
        for i in range(count):
            out.append(Int(raw[4 * i]) | Int(raw[4 * i + 1]) << 8 | Int(raw[4 * i + 2]) << 16 | Int(raw[4 * i + 3]) << 24)
        return out^

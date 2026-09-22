"""Device transcript equals the host mirror; samples are in range."""

from std.testing import assert_equal, assert_true, TestSuite
from max.gpu.host import DeviceContext

from core.params import CLIENT
from core.hash import Blake3
from core.arena import Arena, Bump
from core.transcript import TranscriptLayout, HostTranscript, reset, absorb, squeeze_elements, squeeze_positions, DS_PREFIX, DS_TREE_W, DS_GRIND, STATE_BYTES, grind_word, grind_ok
from core.bytes import host_base

comptime p = CLIENT.grid(72, 32)
comptime MSG = 85_120     # the Keccak openings: 84 chunks, one block of the tree absorb
comptime ELEMS = 500
comptime POS = 64
comptime BELOW = 161280


def _download(ctx: DeviceContext, arena: Arena, off: Int, n: Int) raises -> List[UInt8]:
    var h = ctx.enqueue_create_host_buffer[DType.uint8](n)
    arena.download(ctx, off, h)
    ctx.synchronize()
    var l = List[UInt8](capacity=n)
    for i in range(n):
        l.append(h[i])
    return l^


def test_device_matches_host() raises:
    var ctx = DeviceContext()
    var bump = Bump()
    var t = TranscriptLayout(bump)
    var chal = bump.alloc(max(ELEMS * p.e, POS * 4))
    var msg_off = bump.alloc(MSG)
    var arena = Arena(ctx, bump.used)
    var mh = ctx.enqueue_create_host_buffer[DType.uint8](MSG)
    ctx.synchronize()
    var msg = List[UInt8](capacity=MSG)
    for i in range(MSG):
        mh[i] = UInt8((i * 31 + 7) % 251)
        msg.append(mh[i])
    arena.upload(ctx, msg_off, mh)

    var host = HostTranscript[p, Blake3]()
    reset(ctx, arena, t)
    absorb[p, Blake3](ctx, arena, t, DS_PREFIX, msg_off, MSG)
    host.absorb(DS_PREFIX, msg)
    squeeze_elements[p, Blake3](ctx, arena, t, chal, ELEMS)
    var want_e = host.elements(ELEMS)
    var got_e = _download(ctx, arena, chal, ELEMS * p.e)
    assert_equal(got_e, want_e)
    for b in got_e:
        assert_true(b < 127)

    absorb[p, Blake3](ctx, arena, t, DS_TREE_W, chal, 32)   # absorb the first 32 challenge bytes
    var first = List[UInt8](capacity=32)
    for i in range(32):
        first.append(got_e[i])
    host.absorb(DS_TREE_W, first)
    squeeze_positions[p, Blake3](ctx, arena, t, chal, POS, BELOW)
    var want_p = host.positions(POS, BELOW)
    var got = _download(ctx, arena, chal, POS * 4)
    var distinct = 0
    for i in range(POS):
        var v = Int(got[4 * i]) | Int(got[4 * i + 1]) << 8 | Int(got[4 * i + 2]) << 16 | Int(got[4 * i + 3]) << 24
        assert_equal(v, want_p[i])
        assert_true(v < BELOW)
        if i > 0 and v != want_p[i - 1]:
            distinct += 1
    assert_true(distinct > POS // 2)

    # a second proof starts from the same state: reset, absorb, squeeze again equals the first run
    reset(ctx, arena, t)
    absorb[p, Blake3](ctx, arena, t, DS_PREFIX, msg_off, MSG)
    squeeze_elements[p, Blake3](ctx, arena, t, chal, ELEMS)
    assert_equal(_download(ctx, arena, chal, ELEMS * p.e), want_e)
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks


def test_state_depends_on_separator() raises:
    var a = HostTranscript[p, Blake3]()
    var b = HostTranscript[p, Blake3]()
    var msg = List[UInt8](length=10, fill=3)
    a.absorb(DS_PREFIX, msg)
    b.absorb(DS_TREE_W, msg)
    assert_true(a.elements(4) != b.elements(4))
    # a second squeeze continues the stream instead of repeating
    var c = HostTranscript[p, Blake3]()
    c.absorb(DS_PREFIX, msg)
    var x = c.elements(4)
    var y = c.elements(4)
    assert_true(x != y)


def test_one_chunk_absorb_is_its_chunk_value() raises:
    """The search kernel absorbs the 8-byte nonce through Hash.chunk (no merge stack per thread): the same state as absorb."""
    var state = List[UInt8](length=STATE_BYTES, fill=0)
    for i in range(32):
        state[i] = UInt8((i * 29 + 3) % 251)
    var nonce = List[UInt8](length=8, fill=0)
    nonce[0] = 7
    nonce[3] = 250
    var a = state.copy()
    Blake3.absorb(host_base(a), DS_GRIND, host_base(nonce), 8)
    var b = state.copy()
    Blake3.chunk(host_base(b), DS_GRIND, host_base(nonce), 8, 0, host_base(b))
    for i in range(32):
        assert_equal(a[i], b[i])
    var probe = Blake3.grind_probe(host_base(state), DS_GRIND, UInt32(7) | UInt32(250) << 24)
    assert_equal(probe, grind_word[Blake3](host_base(a)))
    # the grind word of a fresh nonce is uniform-ish: both zero and nonzero top bits occur over a few nonces
    var zero = 0
    for n in range(64):
        var c = state.copy()
        nonce[0] = UInt8(n)
        Blake3.absorb(host_base(c), DS_GRIND, host_base(nonce), 8)
        if grind_ok(grind_word[Blake3](host_base(c)), 1):
            zero += 1
    assert_true(zero > 8 and zero < 56, "grind word top bit is not balanced")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

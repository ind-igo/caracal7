"""The SHA-256 row group: three groups on one 144-row grid, messages 2 and 3 embedding the first digest (inside
a block, and across a block boundary), the digest fingerprint wired to both embedded ones; the second digest
pinned by a public factor (the oracle)."""

from std.testing import assert_equal, assert_true, TestSuite
from max.gpu.host import DeviceContext

from core.params import CLIENT
from core.hash import Blake3
from relations.statement import Statement, Compiled
from workloads.sha256 import sha256, WORDS
from workloads.sha256g import ShaGroup, Zeta, ZETA, sha256_group, sha256_group_trace, sha256_group_public, BASE, SLOT, STREAM
from prover import Prover, load_trace, load_public
from verifier import verify

comptime p = CLIENT.grid(144, 96)
comptime N = p.N()
comptime h1 = p.h1()
comptime h2 = p.h2()
comptime CHAINS_1 = 17          # one block and an idle chain
comptime CHAINS_2 = 17          # one block
comptime CHAINS_3 = 33          # two blocks
comptime EMBED = 12             # byte offset of digest 1 in message 2: s0 = 96, chains 0, 1, 2
comptime EMBED_3 = 44           # in message 3: s0 = 96 too, chains 2, 3 and 16 (across the block boundary)
comptime GROUPS = 3


def _message1() -> List[UInt8]:
    var m = List[UInt8](capacity=50)
    for i in range(50):
        m.append(UInt8((i * 37 + 11) & 0xFF))
    return m^


def _embedding(digest: List[UInt8], at: Int, length: Int) -> List[UInt8]:
    """A message of `length` bytes with `digest` at byte `at`."""
    var m = List[UInt8](capacity=length)
    for i in range(length):
        m.append(UInt8((200 + i * 5) & 0xFF))
    for i in range(32):
        m[at + i] = digest[i]
    return m^


def _statement(m1: List[UInt8], m2: List[UInt8]) raises -> Tuple[Statement, ShaGroup, ShaGroup, ShaGroup]:
    var st = Statement()
    var zeta = Zeta()
    var g1 = sha256_group(st, "h1", CHAINS_1, h2, zeta, 0)
    var g2 = sha256_group(st, "h2", CHAINS_2, h2, zeta, 0, embeds=[EMBED])
    var g3 = sha256_group(st, "h3", CHAINS_3, h2, zeta, 0, embeds=[EMBED_3])
    var dg = g1.digest.copy()
    dg.extend(g2.digest.copy())
    dg.extend(g3.digest.copy())
    st.horner("dg", dg, scale=ZETA)
    var em = g2.embedded.copy()
    em.extend(g3.embedded.copy())
    st.horner("em", em, scale=ZETA)
    var sdg = st.slot("dg")
    var sem = st.slot("em")
    st.wire(sdg, g1.base + g1.digest_chain(m1), sem, g2.base + g2.embed_chain())
    st.wire(sdg, g1.base + g1.digest_chain(m1), sem, g3.base + g3.embed_chain())
    st.public_factor("d2", "dg", sdg, g2.base + g2.digest_chain(m2))
    return (st^, g1^, g2^, g3^)


def _factor(digest: List[UInt8]) -> List[UInt8]:
    """The factor bytes: h1 per dg ingest entry (g1's eight, then g2's), the digest words on slot 3."""
    var v = List[UInt8](length=GROUPS * WORDS * h1, fill=0)
    for i in range(WORDS):
        for z in range(SLOT):
            v[(WORDS + i) * h1 + BASE + 3 * SLOT + z] = (digest[4 * i + z // 8] >> UInt8(7 - z % 8)) & 1
    return v^


def _verdict(proof: List[UInt8], c: Compiled, public: List[UInt8]) raises -> String:
    var fam = c.families.copy()
    try:
        _ = verify[p, Blake3](proof.copy(), c.shape, List[UInt8](), fam, public)
    except e:
        return String(e)
    return String("accepted")


def test_two_groups_wire_the_digest() raises:
    var ctx = DeviceContext()
    var m1 = _message1()
    var d1 = sha256(m1)
    var m2 = _embedding(d1, EMBED, 47)
    var m3 = _embedding(d1, EMBED_3, 80)
    var d2 = sha256(m2)
    var parts = _statement(m1, m2)
    assert_equal(parts[3].embed_chain(), 16)
    var c = parts[0].compile[p]()
    var L = c.layout.copy()
    assert_true(L.columns_w() < 100)                     # the groups pool: 43 + 21 + 4 columns, 11 exclusive
    var trace = List[UInt8](length=L.columns_w() * N, fill=0)
    sha256_group_trace[p](L, parts[1], m1, trace)
    sha256_group_trace[p](L, parts[2], m2, trace)
    sha256_group_trace[p](L, parts[3], m3, trace)
    var cols = sha256_group_public[p](L, parts[1], m1)
    cols.extend(sha256_group_public[p](L, parts[2], m2))
    cols.extend(sha256_group_public[p](L, parts[3], m3))
    var public = cols.copy()
    public.extend(_factor(d2))
    var prover = Prover[p, Blake3](ctx, _statement(m1, m2)[0].compile[p]().take_shape(), c.families.copy())
    load_trace[p, Blake3](ctx, prover, trace)
    load_public[p, Blake3](ctx, prover, cols)
    var proof = prover.prove(ctx, List[UInt8]())
    assert_equal(_verdict(proof, c, public), "accepted")
    # the factor names another digest
    var wrong = public.copy()
    var off = len(public) - GROUPS * WORDS * h1 + WORDS * h1 + BASE + 3 * SLOT + 5
    wrong[off] = 1 - wrong[off]
    assert_equal(_verdict(proof, c, wrong), "wiring grand product is not the public factor")
    # message 2 embeds a different digest: an honest trace of that message fails the wire
    var d1x = d1.copy()
    d1x[3] ^= 0x10
    var m2x = _embedding(d1x, EMBED, 47)
    var tx = List[UInt8](length=L.columns_w() * N, fill=0)
    sha256_group_trace[p](L, parts[1], m1, tx)
    sha256_group_trace[p](L, parts[2], m2x, tx)
    sha256_group_trace[p](L, parts[3], m3, tx)
    var cx = sha256_group_public[p](L, parts[1], m1)
    cx.extend(sha256_group_public[p](L, parts[2], m2x))
    cx.extend(sha256_group_public[p](L, parts[3], m3))
    var px = cx.copy()
    px.extend(_factor(sha256(m2x)))
    load_trace[p, Blake3](ctx, prover, tx)
    load_public[p, Blake3](ctx, prover, cx)
    var proofx = prover.prove(ctx, List[UInt8]())
    assert_equal(_verdict(proofx, c, px), "wiring grand product is not the public factor")
    # a message bit of group 1 flipped in the trace alone breaks the round
    var bad = trace.copy()
    bad[L.col("h1.w") * N + BASE + 7] ^= 1
    load_trace[p, Blake3](ctx, prover, bad)
    load_public[p, Blake3](ctx, prover, cols)
    var proof2 = prover.prove(ctx, List[UInt8]())
    assert_equal(_verdict(proof2, c, public), "residual identity fails at z")
    # the padding is public: a padding bit flipped in the trace fails too
    var bad2 = trace.copy()
    bad2[L.col("h2.w") * N + (parts[2].base + 3) * h1 + BASE + 3 * SLOT + 31] ^= 1   # the length's low bit (block word 15)
    load_trace[p, Blake3](ctx, prover, bad2)
    var proof3 = prover.prove(ctx, List[UInt8]())
    assert_equal(_verdict(proof3, c, public), "residual identity fails at z")
    _ = ctx


def test_group_near_the_grid_end_is_refused() raises:
    """Masks wrap mod h2: a group whose backward reads wrap into itself would over-constrain the honest
    trace (Codex), so the builder refuses it."""
    var st = Statement()
    var zeta = Zeta()
    var err = String("")
    try:
        _ = sha256_group(st, "h1", h2 - 14, h2, zeta, 0)
    except e:
        err = String(e)
    assert_equal(err, "a SHA-256 group ends at least 15 chains before the grid's end: h1")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

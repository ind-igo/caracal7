"""A passport SOD verified in one proof: three SHA-256 groups and the RSA verify on one grid.

    DG1 --sha--> digest 1, embedded in the eContent (the LDS security object) at byte embed 1
    eContent --sha--> digest 2, embedded in the signed attributes at byte embed 2
    signed attributes --sha--> digest 3 = limb 0 of m, where s^e = m (mod n) for the DSC key n

The three messages are witness; their lengths and the two offsets are pinned public inputs (they fix the
statement: the groups' chains and the embedding masks). Each digest is a Horner fingerprint at zeta wired
to its consumer: digests 1 and 2 to the embedding of the next group (`sha256g`), digest 3 to the cz
accumulator of the RSA group's last modmul (`rsa.m_chain`), so the RSA message limb 0 is not a public
factor: the verifier is given s, n and the upper limbs of m (the PKCS#1 v1.5 padding and DigestInfo) only.

Chains: the SHA groups take the grid's first chains (16 per 64-byte block plus one each), the RSA chains
follow (`rsa.chain_count`). Public inputs: limbs, muls, the three lengths and two offsets (u16 each), then
s, n, and m without its low limb, little endian."""

from core.params import Params
from relations.statement import Statement, Layout, Term
from workloads.bigint import Big
from workloads.rsa import rsa_build, rsa_trace, rsa_public_data, m_chain, LIMB
from workloads.sha256g import ShaGroup, Zeta, ZETA, STREAM, sha256_group, sha256_group_trace, sha256_group_public
from workloads.sha256 import BLOCK, ROUNDS
from workload import Workload

comptime GROUPS = 3
comptime HEAD = 2 + 2 * GROUPS + 2 * (GROUPS - 1)   # the pinned head: limbs, muls, lengths, offsets


def sha_chains(length: Int) -> Int:
    """The group of a message of `length` bytes: 16 chains per block and an idle one."""
    return ROUNDS // 4 * ((length + 9 + BLOCK - 1) // BLOCK) + 1


def _u16(mut v: List[UInt8], x: Int):
    v.append(UInt8(x & 255))
    v.append(UInt8(x >> 8))


def _get(v: List[UInt8], i: Int) -> Int:
    return Int(v[i]) | (Int(v[i + 1]) << 8)


def sod_head(limbs: Int, muls: Int, lengths: List[Int], embeds: List[Int]) raises -> List[UInt8]:
    if len(lengths) != GROUPS or len(embeds) != GROUPS - 1:
        raise Error("an SOD has three messages and two embedding offsets")
    var v: List[UInt8] = [UInt8(limbs), UInt8(muls)]
    for l in lengths:
        _u16(v, l)
    for e in embeds:
        _u16(v, e)
    return v^


def _group(i: Int, lengths: List[Int], embeds: List[Int], base: Int) -> ShaGroup:
    """Group i's placement (the ingest terms empty: the trace and public data do not read them)."""
    return ShaGroup("h" + String(i), base, sha_chains(lengths[i]), embeds[i - 1] if i > 0 else -1, List[Term](), List[Term]())


def sod_statement(h2: Int, limbs: Int, muls: Int, lengths: List[Int], embeds: List[Int]) raises -> Statement:
    var head = sod_head(limbs, muls, lengths, embeds)
    for i in range(GROUPS - 1):
        if embeds[i] + 32 > lengths[i + 1]:
            raise Error("an embedded digest lies inside its message")
    var st = Statement()
    var zeta = Zeta()
    var gs = List[ShaGroup]()
    var fp = List[Term]()
    for i in range(GROUPS):
        var shift = 128 - 8 * embeds[i] % STREAM if i < GROUPS - 1 else 0
        var g = sha256_group(st, "h" + String(i), sha_chains(lengths[i]), h2, zeta, shift, embed=embeds[i - 1] if i > 0 else -1)
        fp.extend(g.digest.copy())
        fp.extend(g.embedded.copy())
        gs.append(g^)
    # one accumulator for every fingerprint: a group's digest sits on its last live chain (15 mod 16), an
    # embedded digest on a message chain (below 4 mod 16), so no chain ingests two; one slot, not two
    st.horner("fp", fp, scale=ZETA)
    var base = st.chains_declared()
    var cz = rsa_build(st, limbs, muls, base, m_wired=True)
    st.pin(head)
    var sfp = st.slot("fp")
    for i in range(GROUPS):
        var message = List[UInt8](length=lengths[i], fill=0)      # the digest chain depends on the length only
        var from_chain = gs[i].base + gs[i].digest_chain(message)
        if i < GROUPS - 1:
            st.wire(sfp, from_chain, sfp, gs[i + 1].base + gs[i + 1].embed_chain())
        else:
            st.wire(sfp, from_chain, cz, base + m_chain(limbs, muls, 0))
    return st^


struct SOD(Workload, Copyable, Movable):
    """`sod_statement` with the prover's messages (empty on the verifier's side)."""
    var limbs: Int
    var muls: Int
    var s: Big
    var n: Big
    var m: Big                  # the full m on the prover's side; the verifier's low limb is ignored
    var messages: List[List[UInt8]]
    var lengths: List[Int]
    var embeds: List[Int]

    def __init__(out self, limbs: Int, muls: Int, s: Big, n: Big, m: Big, var lengths: List[Int], var embeds: List[Int],
                 var messages: List[List[UInt8]] = List[List[UInt8]]()):
        self.limbs = limbs
        self.muls = muls
        self.s = s.copy()
        self.n = n.copy()
        self.m = m.copy()
        self.lengths = lengths^
        self.embeds = embeds^
        self.messages = messages^

    def statement[p: Params](self) raises -> Statement:
        return sod_statement(p.h2(), self.limbs, self.muls, self.lengths, self.embeds)

    def trace[p: Params](self, layout: Layout) raises -> List[UInt8]:
        if len(self.messages) != GROUPS:
            raise Error("the prover's SOD has three messages")
        var base = 0
        for l in self.lengths:
            base += sha_chains(l)
        var trace = rsa_trace[p](layout, self.limbs, self.muls, self.s, self.n, self.m, base)
        var at = 0
        for i in range(GROUPS):
            if len(self.messages[i]) != self.lengths[i]:
                raise Error("message " + String(i) + " has the declared length")
            sha256_group_trace[p](layout, _group(i, self.lengths, self.embeds, at), self.messages[i], trace)
            at += sha_chains(self.lengths[i])
        return trace^

    def public_inputs[p: Params](self) raises -> List[UInt8]:
        var lb = LIMB // 8 * self.limbs
        var v = sod_head(self.limbs, self.muls, self.lengths, self.embeds)
        v.extend(self.s.bytes(lb))
        v.extend(self.n.bytes(lb))
        v.extend(self.m.shr(LIMB).bytes(lb - LIMB // 8))
        return v^

    @staticmethod
    def public_data[p: Params](layout: Layout, public_inputs: List[UInt8]) raises -> List[UInt8]:
        if len(public_inputs) < HEAD:
            raise Error("public inputs start with limbs, muls, three lengths and two offsets")
        var limbs = Int(public_inputs[0])
        var lb = LIMB // 8 * limbs
        if len(public_inputs) != HEAD + 3 * lb - LIMB // 8:
            raise Error("public inputs are the head, then s, n and m without its low limb")
        var lengths = List[Int]()
        var embeds = List[Int]()
        for i in range(GROUPS):
            lengths.append(_get(public_inputs, 2 + 2 * i))
        for i in range(GROUPS - 1):
            embeds.append(_get(public_inputs, 2 + 2 * GROUPS + 2 * i))
        for i in range(GROUPS):                                         # before any index: the derivation runs before the pin check
            if layout.group_chains["h" + String(i)] != sha_chains(lengths[i]) or (i > 0 and embeds[i - 1] + 32 > lengths[i]):
                raise Error("the public inputs' lengths and offsets are not the statement's")
        var out = List[UInt8]()
        var at = 0
        for i in range(GROUPS):
            var message = List[UInt8](length=lengths[i], fill=0)       # the length and the offsets are public, the bytes are not
            out.extend(sha256_group_public[p](layout, _group(i, lengths, embeds, at), message))
            at += sha_chains(lengths[i])
        var rsa: List[UInt8] = [public_inputs[0], public_inputs[1]]
        for i in range(2 * lb):
            rsa.append(public_inputs[HEAD + i])
        rsa.extend(List[UInt8](length=LIMB // 8, fill=0))
        for i in range(lb - LIMB // 8):
            rsa.append(public_inputs[HEAD + 2 * lb + i])
        out.extend(rsa_public_data[p](layout, rsa, at, m_wired=True))
        return out^

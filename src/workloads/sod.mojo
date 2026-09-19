"""A passport SOD verified in one proof: three SHA-256 groups, the commitment to the DSC key and the RSA verify
on one grid.

    DG1 --sha--> digest 1, embedded in the eContent (the LDS security object) at byte embed 1
    eContent --sha--> digest 2, embedded in the signed attributes at byte embed 2
    signed attributes --sha--> digest 3 = limb 0 of m, where s^e = m (mod n) for the DSC key n
    n || r --sha--> the commitment digest, public: the DSC proof (`dsc.mojo`) opens the same commitment

The three messages, the signature s, the DSC modulus n and the 32 random bytes r are witness; the lengths and
the two offsets are pinned public inputs (they fix the statement: the groups' chains and the embedding masks).
Each digest is a Horner fingerprint at zeta wired to its consumer: digests 1 and 2 to the embedding of the next
group (`sha256g`), digest 3 to the cz accumulator of the RSA group's last modmul (`rsa.m_chain`); the limbs
of n are the commitment message's 32-byte windows, wired to the rb slot of every QN product that multiplies
them (`rsa.n_chains`); the occurrences of s are wired to each other. The verifier is given the upper limbs of
m (the PKCS#1 v1.5 padding and DigestInfo) and the commitment digest only.

Chains: the SHA groups take the grid's first chains (16 per 64-byte block plus one each), the commitment group
follows, then the RSA chains (`rsa.chain_count`). Public inputs: limbs, muls, the three lengths and two
offsets (u16 each), then m without its low limb (little endian) and the commitment digest."""

from core.params import Params
from relations.statement import Statement, Layout, Term
from workloads.bigint import Big
from workloads.rsa import rsa_build, rsa_trace, rsa_public_data, m_chain, n_chains, LIMB, SLOT_RB, SLOT_CZ
from workloads.sha256 import sha256
from workloads.sha256g import ShaGroup, Zeta, ZETA, sha256_group, sha256_group_trace, sha256_group_public, sha256_digest_factor
from workloads.sha256 import BLOCK, ROUNDS, WORDS
from workload import Workload

comptime GROUPS = 3
comptime HEAD = 2 + 2 * GROUPS + 2 * (GROUPS - 1)   # the pinned head: limbs, muls, lengths, offsets
comptime RANDOM = 32                                 # bytes of r in the commitment message n || r
comptime WINDOW = 32                                 # bytes per embedded window (one limb)


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


def commitment_message(n: Big, limbs: Int, r: List[UInt8]) raises -> List[UInt8]:
    """The commitment group's message: n big endian (limbs x 32 bytes), then r."""
    if len(r) != RANDOM:
        raise Error("the commitment takes " + String(RANDOM) + " random bytes")
    var v = n.bytes(LIMB // 8 * limbs)
    v.reverse()
    v.extend(r.copy())
    return v^


def commitment_windows(limbs: Int) -> List[Int]:
    """Window k of the commitment message holds limb limbs - 1 - k of n."""
    var v = List[Int]()
    for k in range(limbs):
        v.append(WINDOW * k)
    return v^


def commitment_length(limbs: Int) -> Int:
    return LIMB // 8 * limbs + RANDOM


def wire_windows(mut st: Statement, slot_fp: Int, cm: ShaGroup, slot_rb: Int, base: Int, limbs: Int, muls: Int) raises:
    """Wire the commitment group's windows to the rb slot of every QN product of the RSA at `base`."""
    for k in range(limbs):
        for x in n_chains(limbs, muls, limbs - 1 - k):
            st.wire(slot_fp, cm.base + cm.embed_chain(k), slot_rb, base + x)


def _group(i: Int, lengths: List[Int], embeds: List[Int], base: Int) -> ShaGroup:
    """Group i's placement (the ingest terms empty: the trace and public data do not read them)."""
    var windows = List[Int]()
    if i > 0:
        windows.append(embeds[i - 1])
    return ShaGroup("h" + String(i), base, sha_chains(lengths[i]), windows^, List[Term](), List[Term]())


def _cm_group(limbs: Int, base: Int) -> ShaGroup:
    return ShaGroup("cm", base, sha_chains(commitment_length(limbs)), commitment_windows(limbs), List[Term](), List[Term]())


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
        var windows = List[Int]()
        if i > 0:
            windows.append(embeds[i - 1])
        var g = sha256_group(st, "h" + String(i), sha_chains(lengths[i]), h2, zeta, 0, embeds=windows)
        fp.extend(g.digest.copy())
        fp.extend(g.embedded.copy())
        gs.append(g^)
    # the commitment: its digest is a public factor, its windows the limbs of n
    var cm = sha256_group(st, "cm", sha_chains(commitment_length(limbs)), h2, zeta, 0, embeds=commitment_windows(limbs))
    fp.extend(cm.digest.copy())
    fp.extend(cm.embedded.copy())
    # one accumulator for every fingerprint: a group's digest sits on its last live chain (15 mod 16), an
    # embedded window on a message chain (below 4 mod 16), so no chain ingests two; one slot, not two
    if len(fp) != _cm_entry()[1]:
        raise Error("the commitment digest's factor index is not the accumulator's term count")
    st.horner("fp", fp, scale=ZETA)
    var base = st.chains_declared()
    var slots = rsa_build(st, limbs, muls, base, m_wired=True, s_wired=True, n_wired=True)
    st.pin(head)
    var sfp = st.slot("fp")
    for i in range(GROUPS):
        var message = List[UInt8](length=lengths[i], fill=0)      # the digest chain depends on the length only
        var from_chain = gs[i].base + gs[i].digest_chain(message)
        if i < GROUPS - 1:
            st.wire(sfp, from_chain, sfp, gs[i + 1].base + gs[i + 1].embed_chain())
        else:
            st.wire(sfp, from_chain, slots[SLOT_CZ], base + m_chain(limbs, muls, 0))
    wire_windows(st, sfp, cm, slots[SLOT_RB], base, limbs, muls)
    st.public_factor("cmd", "fp", sfp, cm.base + cm.digest_chain(List[UInt8](length=commitment_length(limbs), fill=0)))
    return st^


def _cm_entry() -> Tuple[Int, Int]:
    """(first entry, entries) of the commitment digest's terms on the fp accumulator: after the hash groups'
    digests and embeddings; the commitment's own embeddings follow."""
    var entry = WORDS * GROUPS + 5 * (GROUPS - 1)
    return (entry, entry + WORDS + 5)


struct SOD(Workload, Copyable, Movable):
    """`sod_statement` with the prover's messages, s, n and r (empty or zero on the verifier's side)."""
    var limbs: Int
    var muls: Int
    var s: Big
    var n: Big
    var m: Big                  # the full m on the prover's side; the verifier's low limb is ignored
    var r: List[UInt8]
    var committed: Big          # the modulus in the commitment: n, or (a dishonest prover) another
    var messages: List[List[UInt8]]
    var lengths: List[Int]
    var embeds: List[Int]

    def __init__(out self, limbs: Int, muls: Int, s: Big, n: Big, m: Big, var lengths: List[Int], var embeds: List[Int],
                 var messages: List[List[UInt8]] = List[List[UInt8]](), var r: List[UInt8] = List[UInt8](), committed: Big = Big()):
        self.limbs = limbs
        self.muls = muls
        self.s = s.copy()
        self.n = n.copy()
        self.m = m.copy()
        self.r = r^
        self.committed = committed.copy() if not committed.is_zero() else n.copy()
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
        base += sha_chains(commitment_length(self.limbs))
        var trace = rsa_trace[p](layout, self.limbs, self.muls, self.s, self.n, self.m, base)
        var at = 0
        for i in range(GROUPS):
            if len(self.messages[i]) != self.lengths[i]:
                raise Error("message " + String(i) + " has the declared length")
            sha256_group_trace[p](layout, _group(i, self.lengths, self.embeds, at), self.messages[i], trace)
            at += sha_chains(self.lengths[i])
        sha256_group_trace[p](layout, _cm_group(self.limbs, at), commitment_message(self.committed, self.limbs, self.r), trace)
        return trace^

    def public_inputs[p: Params](self) raises -> List[UInt8]:
        var lb = LIMB // 8 * self.limbs
        var v = sod_head(self.limbs, self.muls, self.lengths, self.embeds)
        v.extend(self.m.shr(LIMB).bytes(lb - LIMB // 8))
        v.extend(sha256(commitment_message(self.committed, self.limbs, self.r)))
        return v^

    @staticmethod
    def public_data[p: Params](layout: Layout, public_inputs: List[UInt8]) raises -> List[UInt8]:
        if len(public_inputs) < HEAD:
            raise Error("public inputs start with limbs, muls, three lengths and two offsets")
        var limbs = Int(public_inputs[0])
        var lb = LIMB // 8 * limbs
        if len(public_inputs) != HEAD + lb:
            raise Error("public inputs are the head, then m without its low limb and the commitment digest")
        var lengths = List[Int]()
        var embeds = List[Int]()
        for i in range(GROUPS):
            lengths.append(_get(public_inputs, 2 + 2 * i))
        for i in range(GROUPS - 1):
            embeds.append(_get(public_inputs, 2 + 2 * GROUPS + 2 * i))
        for i in range(GROUPS):                                         # before any index: the derivation runs before the pin check
            if layout.group_chains["h" + String(i)] != sha_chains(lengths[i]) or (i > 0 and embeds[i - 1] + 32 > lengths[i]):
                raise Error("the public inputs' lengths and offsets are not the statement's")
        if layout.group_chains["cm"] != sha_chains(commitment_length(limbs)):
            raise Error("the public inputs' limbs are not the statement's")
        var out = List[UInt8]()
        var at = 0
        for i in range(GROUPS):
            var message = List[UInt8](length=lengths[i], fill=0)       # the length and the offsets are public, the bytes are not
            out.extend(sha256_group_public[p](layout, _group(i, lengths, embeds, at), message))
            at += sha_chains(lengths[i])
        out.extend(sha256_group_public[p](layout, _cm_group(limbs, at), List[UInt8](length=commitment_length(limbs), fill=0)))
        at += sha_chains(commitment_length(limbs))
        var rsa: List[UInt8] = [public_inputs[0], public_inputs[1]]
        rsa.extend(List[UInt8](length=2 * lb + LIMB // 8, fill=0))     # s, n witness; m's low limb wired
        for i in range(lb - LIMB // 8):
            rsa.append(public_inputs[HEAD + i])
        out.extend(rsa_public_data[p](layout, rsa, at, m_wired=True, s_wired=True, n_wired=True))
        var digest = List[UInt8]()
        for i in range(LIMB // 8):
            digest.append(public_inputs[HEAD + lb - LIMB // 8 + i])
        var e = _cm_entry()
        out.extend(sha256_digest_factor(e[0], e[1], digest))
        return out^

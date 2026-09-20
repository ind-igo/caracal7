"""A passport in one proof: the SOD (`sod.mojo`) and the DSC certificate check (`dsc.mojo`) on one grid, the
DSC key wired from the certificate body to the SOD's RSA verify, no commitment.

    DG1 --sha--> digest 1, embedded in the eContent at byte embed 1
    eContent --sha--> digest 2, embedded in the signed attributes at byte embed 2
    signed attributes --sha--> digest 3 = limb 0 of m1, where s1^e = m1 (mod n_dsc)
    digest 2 || scope --sha--> the nullifier, public
    DG1 bytes [mrz, mrz + 32) --> the disclosed window, public
    TBS --sha--> digest 4 = limb 0 of m2, where s2^e = m2 (mod n_csca), n_csca public
    TBS bytes [offset, offset + 32 limbs) = n_dsc, window by window the rb slot of the SOD verify's q n products

The messages, both signatures, the body and n_dsc are witness; the lengths and offsets are pinned. The two
RSA verifies share one column set (`rsa_columns`) on disjoint chains (`rsa_instance`), so one pinned
(limbs, muls) covers both: the DSC key and the CSCA key have one size and one public exponent (the two-proof
form pins them apart; ICAO keys are 65537 and 2048 bits on both, the registry checks the CSCA's). Against the two
proofs of `sod.mojo` and `dsc.mojo` this drops the two commitment groups (81 chains each) and the random
bytes r: the link is a wire, not a digest the verifier compares.

Chains: the three SOD groups, the nullifier group, the body group, then the two RSA verifies. Public inputs:
limbs, muls, the three lengths, two offsets, the window offset, the body length and the key offset (u16
each), then m1 without its low limb, n_csca, m2 without its low limb (little endian), the window, the
scope and the nullifier."""

from core.params import Params
from relations.statement import Statement, Layout, Term
from workloads.bigint import Big
from workloads.rsa import rsa_columns, rsa_instance, rsa_trace, rsa_trace_into, rsa_public_columns, rsa_factor_data, m_chain, chain_count, LIMB, SLOT_RB, SLOT_CZ
from workloads.sha256 import sha256, WORDS
from workloads.sha256g import ShaGroup, Zeta, ZETA, sha256_group, sha256_group_trace, sha256_group_public, sha256_digest_factor, sha256_window_factor
from workloads.sod import GROUPS, TERMS, WINDOW, NULL_LENGTH, sha_chains, sod_head, nullifier_message, wire_windows, _group, _nu_group, _get
from workloads.dsc import tbs_windows, _tb_group, dsc_head
from workload import Workload

comptime HEAD = 2 + 2 * GROUPS + 2 * (GROUPS - 1) + 2 + 4     # the SOD head, then the body length and the key offset
comptime FACTORS = 3                                          # the public factors after the RSA limbs: window, scope, nullifier


def passport_head(limbs: Int, muls: Int, lengths: List[Int], embeds: List[Int], mrz: Int, length: Int, offset: Int) raises -> List[UInt8]:
    var v = sod_head(limbs, muls, lengths, embeds, mrz)
    var d = dsc_head(limbs, muls, length, offset)
    for i in range(2, len(d)):
        v.append(d[i])
    return v^


def _bases(lengths: List[Int], length: Int) -> Tuple[Int, Int, Int]:
    """(nullifier group, body group, first RSA) chain bases; the second RSA follows the first."""
    var nu = 0
    for l in lengths:
        nu += sha_chains(l)
    var tb = nu + sha_chains(NULL_LENGTH)
    return (nu, tb, tb + sha_chains(length))


def passport_chains(lengths: List[Int], length: Int, limbs: Int, muls: Int) raises -> Int:
    return _bases(lengths, length)[2] + 2 * chain_count(limbs, muls)


def passport_statement(h2: Int, limbs: Int, muls: Int, lengths: List[Int], embeds: List[Int], mrz: Int, length: Int, offset: Int) raises -> Statement:
    var head = passport_head(limbs, muls, lengths, embeds, mrz, length, offset)
    for i in range(GROUPS - 1):
        if embeds[i] + WINDOW > lengths[i + 1]:
            raise Error("an embedded digest lies inside its message")
    var st = Statement()
    var zeta = Zeta()
    var gs = List[ShaGroup]()
    var fp = List[Term]()
    for i in range(GROUPS):
        var windows: List[Int] = [mrz] if i == 0 else [embeds[i - 1]]
        var g = sha256_group(st, "h" + String(i), sha_chains(lengths[i]), h2, zeta, 0, embeds=windows)
        fp.extend(g.digest.copy())
        fp.extend(g.embedded.copy())
        gs.append(g^)
    var nu = sha256_group(st, "nu", sha_chains(NULL_LENGTH), h2, zeta, 0, embeds=[0, WINDOW])
    fp.extend(nu.digest.copy())
    fp.extend(nu.embedded.copy())
    var tb = sha256_group(st, "tb", sha_chains(length), h2, zeta, 0, embeds=tbs_windows(limbs, offset))
    fp.extend(tb.digest.copy())
    fp.extend(tb.embedded.copy())
    if len(fp) != (GROUPS + 2) * TERMS:
        raise Error("the factors' entry indices are not the accumulator's term count")
    st.horner("fp", fp, scale=ZETA)
    var bases = _bases(lengths, length)
    var base1 = bases[2]
    var base2 = base1 + chain_count(limbs, muls)
    var slots = rsa_columns(st, limbs, muls)
    rsa_instance(st, slots, limbs, muls, base1, m_wired=True, s_wired=True, n_wired=True, prefix="f")
    rsa_instance(st, slots, limbs, muls, base2, m_wired=True, s_wired=True, prefix="g")
    st.pin(head)
    var sfp = st.slot("fp")
    for i in range(GROUPS):
        var message = List[UInt8](length=lengths[i], fill=0)
        var from_chain = gs[i].base + gs[i].digest_chain(message)
        if i < GROUPS - 1:
            st.wire(sfp, from_chain, sfp, gs[i + 1].base + gs[i + 1].embed_chain())
        else:
            st.wire(sfp, from_chain, slots[SLOT_CZ], base1 + m_chain(limbs, muls, 0))
    st.wire(sfp, gs[1].base + gs[1].digest_chain(List[UInt8](length=lengths[1], fill=0)), sfp, nu.base + nu.embed_chain(0))
    # the DSC key: the body's windows feed the SOD verify's q n products, and the body's digest is m2's low limb
    wire_windows(st, sfp, tb, slots[SLOT_RB], base1, limbs, muls)
    st.wire(sfp, tb.base + tb.digest_chain(List[UInt8](length=length, fill=0)), slots[SLOT_CZ], base2 + m_chain(limbs, muls, 0))
    st.public_factor("mrz", "fp", sfp, gs[0].base + gs[0].embed_chain())
    st.public_factor("scp", "fp", sfp, nu.base + nu.embed_chain(1))
    st.public_factor("nul", "fp", sfp, nu.base + nu.digest_chain(List[UInt8](length=NULL_LENGTH, fill=0)))
    return st^


struct Passport(Workload, Copyable, Movable):
    """`passport_statement` with the prover's messages, body, signatures and n_dsc (empty or zero on the
    verifier's side); the verifier's m1 and m2 carry the padding limbs only, n_csca is public."""
    var limbs: Int
    var muls: Int
    var s1: Big
    var n_dsc: Big
    var m1: Big
    var s2: Big
    var n_csca: Big
    var m2: Big
    var lengths: List[Int]
    var embeds: List[Int]
    var mrz: Int
    var length: Int
    var offset: Int
    var scope: List[UInt8]
    var messages: List[List[UInt8]]
    var tbs: List[UInt8]

    def __init__(out self, limbs: Int, muls: Int, s1: Big, n_dsc: Big, m1: Big, s2: Big, n_csca: Big, m2: Big,
                 var lengths: List[Int], var embeds: List[Int], mrz: Int, length: Int, offset: Int, var scope: List[UInt8],
                 var messages: List[List[UInt8]] = List[List[UInt8]](), var tbs: List[UInt8] = List[UInt8]()):
        self.limbs = limbs
        self.muls = muls
        self.s1 = s1.copy()
        self.n_dsc = n_dsc.copy()
        self.m1 = m1.copy()
        self.s2 = s2.copy()
        self.n_csca = n_csca.copy()
        self.m2 = m2.copy()
        self.lengths = lengths^
        self.embeds = embeds^
        self.mrz = mrz
        self.length = length
        self.offset = offset
        self.scope = scope^
        self.messages = messages^
        self.tbs = tbs^

    def statement[p: Params](self) raises -> Statement:
        return passport_statement(p.h2(), self.limbs, self.muls, self.lengths, self.embeds, self.mrz, self.length, self.offset)

    def trace[p: Params](self, layout: Layout) raises -> List[UInt8]:
        if len(self.messages) != GROUPS or len(self.tbs) != self.length:
            raise Error("the prover's passport has three messages and the body of the declared length")
        var bases = _bases(self.lengths, self.length)
        var trace = rsa_trace[p](layout, self.limbs, self.muls, self.s1, self.n_dsc, self.m1, bases[2])
        rsa_trace_into[p](layout, self.limbs, self.muls, self.s2, self.n_csca, self.m2, bases[2] + chain_count(self.limbs, self.muls), trace)
        var at = 0
        for i in range(GROUPS):
            if len(self.messages[i]) != self.lengths[i]:
                raise Error("message " + String(i) + " has the declared length")
            sha256_group_trace[p](layout, _group(i, self.lengths, self.embeds, self.mrz, at), self.messages[i], trace)
            at += sha_chains(self.lengths[i])
        sha256_group_trace[p](layout, _nu_group(bases[0]), nullifier_message(self.messages[1], self.scope), trace)
        var tb = _tb_group(self.limbs, self.length, self.offset)
        tb.base = bases[1]
        sha256_group_trace[p](layout, tb, self.tbs, trace)
        return trace^

    def window(self) raises -> List[UInt8]:
        """The disclosed bytes of the prover's DG1."""
        if len(self.messages) != GROUPS or len(self.messages[0]) != self.lengths[0]:
            raise Error("the prover's passport has three messages of the declared lengths")
        var v = List[UInt8]()
        for i in range(WINDOW):
            v.append(self.messages[0][self.mrz + i])
        return v^

    def public_inputs[p: Params](self) raises -> List[UInt8]:
        var lb = LIMB // 8 * self.limbs
        var v = passport_head(self.limbs, self.muls, self.lengths, self.embeds, self.mrz, self.length, self.offset)
        v.extend(self.m1.shr(LIMB).bytes(lb - LIMB // 8))
        v.extend(self.n_csca.bytes(lb))
        v.extend(self.m2.shr(LIMB).bytes(lb - LIMB // 8))
        v.extend(self.window())
        v.extend(self.scope.copy())
        v.extend(sha256(nullifier_message(self.messages[1], self.scope)))
        return v^

    @staticmethod
    def public_data[p: Params](layout: Layout, public_inputs: List[UInt8]) raises -> List[UInt8]:
        if len(public_inputs) < HEAD:
            raise Error("public inputs start with limbs, muls, three lengths, two offsets, the window offset, the body length and the key offset")
        var limbs = Int(public_inputs[0])
        var muls = Int(public_inputs[1])
        var lb = LIMB // 8 * limbs
        if len(public_inputs) != HEAD + 3 * lb - 2 * (LIMB // 8) + FACTORS * WINDOW:
            raise Error("public inputs are the head, then m1 without its low limb, n_csca, m2 without its low limb, the window, the scope and the nullifier")
        var lengths = List[Int]()
        var embeds = List[Int]()
        for i in range(GROUPS):
            lengths.append(_get(public_inputs, 2 + 2 * i))
        for i in range(GROUPS - 1):
            embeds.append(_get(public_inputs, 2 + 2 * GROUPS + 2 * i))
        var mrz = _get(public_inputs, HEAD - 6)
        var length = _get(public_inputs, HEAD - 4)
        var offset = _get(public_inputs, HEAD - 2)
        for i in range(GROUPS):                                         # before any index: the derivation runs before the pin check
            if layout.group_chains["h" + String(i)] != sha_chains(lengths[i]) or (i > 0 and embeds[i - 1] + WINDOW > lengths[i]):
                raise Error("the public inputs' lengths and offsets are not the statement's")
        if mrz + WINDOW > lengths[0]:
            raise Error("the public inputs' lengths and offsets are not the statement's")
        if layout.group_chains["tb"] != sha_chains(length) or offset + lb > length:
            raise Error("the public inputs' body length and key offset are not the statement's")
        var bases = _bases(lengths, length)
        var out = List[UInt8]()
        var at = 0
        for i in range(GROUPS):
            var message = List[UInt8](length=lengths[i], fill=0)       # the length and the offsets are public, the bytes are not
            out.extend(sha256_group_public[p](layout, _group(i, lengths, embeds, mrz, at), message))
            at += sha_chains(lengths[i])
        out.extend(sha256_group_public[p](layout, _nu_group(bases[0]), List[UInt8](length=NULL_LENGTH, fill=0)))
        var tb = _tb_group(limbs, length, offset)
        tb.base = bases[1]
        out.extend(sha256_group_public[p](layout, tb, List[UInt8](length=length, fill=0)))
        var base2 = bases[2] + chain_count(limbs, muls)
        out.extend(rsa_public_columns[p](limbs, muls, [bases[2], base2]))
        # the SOD verify: s1, n_dsc witness, m1's low limb wired; the DSC check: s2 witness, n_csca public
        var rsa1: List[UInt8] = [public_inputs[0], public_inputs[1]]
        rsa1.extend(List[UInt8](length=2 * lb + LIMB // 8, fill=0))
        for i in range(lb - LIMB // 8):
            rsa1.append(public_inputs[HEAD + i])
        out.extend(rsa_factor_data(rsa1, m_wired=True, s_wired=True, n_wired=True))
        var rsa2: List[UInt8] = [public_inputs[0], public_inputs[1]]
        rsa2.extend(List[UInt8](length=lb, fill=0))
        for i in range(lb):
            rsa2.append(public_inputs[HEAD + lb - LIMB // 8 + i])
        rsa2.extend(List[UInt8](length=LIMB // 8, fill=0))
        for i in range(lb - LIMB // 8):
            rsa2.append(public_inputs[HEAD + 2 * lb - LIMB // 8 + i])
        out.extend(rsa_factor_data(rsa2, m_wired=True, s_wired=True))
        # the factors: fp's terms are TERMS per group, h0 h1 h2 nu tb, a group's digest first then its windows
        var tail = HEAD + 3 * lb - 2 * (LIMB // 8)
        var values = List[List[UInt8]]()
        for k in range(FACTORS):
            values.append(List[UInt8]())
            for i in range(WINDOW):
                values[k].append(public_inputs[tail + WINDOW * k + i])
        var entries = (GROUPS + 2) * TERMS
        out.extend(sha256_window_factor(WORDS, entries, _group(0, lengths, embeds, mrz, 0).segments(), values[0]))
        out.extend(sha256_window_factor(GROUPS * TERMS + WORDS, entries, _nu_group(0).segments(1), values[1]))
        out.extend(sha256_digest_factor(GROUPS * TERMS, entries, values[2]))
        return out^

"""The DSC certificate check in one proof: the CSCA's RSA signature over the certificate body (the TBS), the
DSC key inside that body, and the commitment to the DSC key the SOD proof (`sod.mojo`) opens.

    TBS --sha--> digest = limb 0 of m, where s^e = m (mod n_csca)
    TBS bytes [offset, offset + 32 limbs) = n_dsc, window by window equal to the commitment message's windows
    n_dsc || r --sha--> the commitment digest, public

The TBS, the signature s, the DSC modulus n_dsc and the random bytes r are witness; the TBS length and the
offset of n_dsc in it are pinned public inputs (they fix the statement: the group's chains and the window
masks). The verifier is given n_csca, the upper limbs of m (the PKCS#1 v1.5 padding and DigestInfo) and the
commitment digest. Both groups' windows fingerprint to the plain value of the limb (`sha256g`) and wire to
each other.

Chains: the TBS group, the commitment group, then the RSA chains. Public inputs: limbs, muls, the TBS length
and the offset (u16 each), then n_csca and m without its low limb (little endian) and the commitment digest."""

from core.params import Params
from relations.statement import Statement, Layout, Term
from workloads.bigint import Big
from workloads.rsa import rsa_build, rsa_trace, rsa_public_data, m_chain, LIMB, SLOT_CZ
from workloads.sha256 import sha256, WORDS
from workloads.sha256g import ShaGroup, Zeta, ZETA, sha256_group, sha256_group_trace, sha256_group_public, sha256_digest_factor
from workloads.sod import sha_chains, commitment_message, commitment_windows, commitment_length, WINDOW, _u16, _get
from workload import Workload

comptime HEAD = 6                   # the pinned head: limbs, muls, TBS length, offset of n_dsc


def dsc_head(limbs: Int, muls: Int, length: Int, offset: Int) raises -> List[UInt8]:
    if offset < 0 or offset + LIMB // 8 * limbs > length:
        raise Error("the DSC key lies inside the certificate body")
    var v: List[UInt8] = [UInt8(limbs), UInt8(muls)]
    _u16(v, length)
    _u16(v, offset)
    return v^


def tbs_windows(limbs: Int, offset: Int) -> List[Int]:
    """Window k of the TBS holds limb limbs - 1 - k of n_dsc, like the commitment's."""
    var v = List[Int]()
    for k in range(limbs):
        v.append(offset + WINDOW * k)
    return v^


def _tb_group(limbs: Int, length: Int, offset: Int) -> ShaGroup:
    return ShaGroup("tb", 0, sha_chains(length), tbs_windows(limbs, offset), List[Term](), List[Term]())


def _cm_group(limbs: Int, base: Int) -> ShaGroup:
    return ShaGroup("cm", base, sha_chains(commitment_length(limbs)), commitment_windows(limbs), List[Term](), List[Term]())


def dsc_statement(h2: Int, limbs: Int, muls: Int, length: Int, offset: Int) raises -> Statement:
    var head = dsc_head(limbs, muls, length, offset)
    var st = Statement()
    var zeta = Zeta()
    var tb = sha256_group(st, "tb", sha_chains(length), h2, zeta, 0, embeds=tbs_windows(limbs, offset))
    var cm = sha256_group(st, "cm", sha_chains(commitment_length(limbs)), h2, zeta, 0, embeds=commitment_windows(limbs))
    var fp = tb.digest.copy()
    fp.extend(tb.embedded.copy())
    fp.extend(cm.digest.copy())
    fp.extend(cm.embedded.copy())
    if len(fp) != _cm_entry()[1]:
        raise Error("the commitment digest's factor index is not the accumulator's term count")
    st.horner("fp", fp, scale=ZETA)
    var base = st.chains_declared()
    var slots = rsa_build(st, limbs, muls, base, m_wired=True, s_wired=True)
    st.pin(head)
    var sfp = st.slot("fp")
    st.wire(sfp, tb.base + tb.digest_chain(List[UInt8](length=length, fill=0)), slots[SLOT_CZ], base + m_chain(limbs, muls, 0))
    for k in range(limbs):
        st.wire(sfp, tb.base + tb.embed_chain(k), sfp, cm.base + cm.embed_chain(k))
    st.public_factor("cmd", "fp", sfp, cm.base + cm.digest_chain(List[UInt8](length=commitment_length(limbs), fill=0)))
    return st^


def _cm_entry() -> Tuple[Int, Int]:
    """(first entry, entries) of the commitment digest's terms on fp: after the TBS group's digest and embeddings."""
    return (WORDS + 5, 2 * (WORDS + 5))


struct DSC(Workload, Copyable, Movable):
    """`dsc_statement` with the prover's TBS, s, n_dsc and r (empty or zero on the verifier's side)."""
    var limbs: Int
    var muls: Int
    var s: Big
    var n_csca: Big
    var m: Big                  # the full m on the prover's side; the verifier's low limb is ignored
    var n_dsc: Big
    var r: List[UInt8]
    var tbs: List[UInt8]
    var length: Int
    var offset: Int

    def __init__(out self, limbs: Int, muls: Int, s: Big, n_csca: Big, m: Big, length: Int, offset: Int,
                 var tbs: List[UInt8] = List[UInt8](), n_dsc: Big = Big(), var r: List[UInt8] = List[UInt8]()):
        self.limbs = limbs
        self.muls = muls
        self.s = s.copy()
        self.n_csca = n_csca.copy()
        self.m = m.copy()
        self.n_dsc = n_dsc.copy()
        self.r = r^
        self.tbs = tbs^
        self.length = length
        self.offset = offset

    def statement[p: Params](self) raises -> Statement:
        return dsc_statement(p.h2(), self.limbs, self.muls, self.length, self.offset)

    def trace[p: Params](self, layout: Layout) raises -> List[UInt8]:
        if len(self.tbs) != self.length:
            raise Error("the certificate body has the declared length")
        var cm_base = sha_chains(self.length)
        var base = cm_base + sha_chains(commitment_length(self.limbs))
        var trace = rsa_trace[p](layout, self.limbs, self.muls, self.s, self.n_csca, self.m, base)
        sha256_group_trace[p](layout, _tb_group(self.limbs, self.length, self.offset), self.tbs, trace)
        sha256_group_trace[p](layout, _cm_group(self.limbs, cm_base), commitment_message(self.n_dsc, self.limbs, self.r), trace)
        return trace^

    def public_inputs[p: Params](self) raises -> List[UInt8]:
        var lb = LIMB // 8 * self.limbs
        var v = dsc_head(self.limbs, self.muls, self.length, self.offset)
        v.extend(self.n_csca.bytes(lb))
        v.extend(self.m.shr(LIMB).bytes(lb - LIMB // 8))
        v.extend(sha256(commitment_message(self.n_dsc, self.limbs, self.r)))
        return v^

    @staticmethod
    def public_data[p: Params](layout: Layout, public_inputs: List[UInt8]) raises -> List[UInt8]:
        if len(public_inputs) < HEAD:
            raise Error("public inputs start with limbs, muls, the TBS length and the offset")
        var limbs = Int(public_inputs[0])
        var lb = LIMB // 8 * limbs
        if len(public_inputs) != HEAD + 2 * lb:
            raise Error("public inputs are the head, then n_csca, m without its low limb and the commitment digest")
        var length = _get(public_inputs, 2)
        var offset = _get(public_inputs, 4)
        if layout.group_chains["cm"] != sha_chains(commitment_length(limbs)):
            raise Error("the public inputs' limbs are not the statement's")
        if layout.group_chains["tb"] != sha_chains(length) or offset + lb > length:
            raise Error("the public inputs' length and offset are not the statement's")
        var out = sha256_group_public[p](layout, _tb_group(limbs, length, offset), List[UInt8](length=length, fill=0))
        var at = sha_chains(length)
        out.extend(sha256_group_public[p](layout, _cm_group(limbs, at), List[UInt8](length=commitment_length(limbs), fill=0)))
        at += sha_chains(commitment_length(limbs))
        var rsa: List[UInt8] = [public_inputs[0], public_inputs[1]]
        rsa.extend(List[UInt8](length=lb, fill=0))                     # s witness
        for i in range(lb):
            rsa.append(public_inputs[HEAD + i])
        rsa.extend(List[UInt8](length=LIMB // 8, fill=0))              # m's low limb wired
        for i in range(lb - LIMB // 8):
            rsa.append(public_inputs[HEAD + lb + i])
        out.extend(rsa_public_data[p](layout, rsa, at, m_wired=True, s_wired=True))
        var digest = List[UInt8]()
        for i in range(LIMB // 8):
            digest.append(public_inputs[HEAD + 2 * lb - LIMB // 8 + i])
        var e = _cm_entry()
        out.extend(sha256_digest_factor(e[0], e[1], digest))
        return out^

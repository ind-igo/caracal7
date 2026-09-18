"""Proof shape and byte layout (statement-layer section 7, spec 9.5), milestone 1.

Order, every integer little-endian, every field element e bytes:

    header      version u32, public inputs (u32 length + bytes); the parameters are bound through
                the transcript prefix (prefix_bytes), not sent
    W root      H.DIGEST
    Z root      H.DIGEST; Z2 per accumulator (h2 e)
    Q root      H.DIGEST; Q3 (2 h2 e) when there are accumulators
    openings    alpha_{c,p}: P x (columns_w + columns_z + columns_q) x e
    per level l = 2 .. ell-1:
                Mat(y_l) root 32 B; multiproof(s) of level l-1 (u32 length + bytes each; three at
                level 1: W, Z, Q); three sumcheck messages (9 e)
    last        clear vector y_ell (|y_ell| x e); multiproof(s) of level ell-1

Multiproofs are length-prefixed because the sibling frontier depends on the sampled positions.
Everything else has a size fixed by `Shape`, so the verifier can check the total length up front.
Field coordinates must be canonical bytes below 127, including authenticated codeword rows.
Roots, sibling hashes, length words, and raw public inputs are opaque bytes.
"""

from std.math import log2
from std.memory import unsafe_memcpy
from max.gpu.host import DeviceContext, HostBuffer

from core.params import Params, domain_for, query_count
from core.arena import Arena
from relations import ENTRY, NONE, NO_BASIS, ACC, ACC_W_MAX, END, WIRE, PUBF, GRP, KIND_LOOKUP, KIND_HORNER, HORNER_TRANSITIONS, group_offsets, acc_z_col, acc_start, acc_kind, acc_table, acc_family, PUB, RES, ZERO, POINT, CHAL, CHAL_ADD, CHAL_MUL, CHAL_ONE, SAMPLED, FIX_ONE, FIX_E, entry, shift_points, required_points, standard_chals, chal_count, point_index, value_bytes
from core.hash import Hash
from core.tables import F2_ORDER, Domains, f2_primitive
from core.field import ext_mul, ext_pow
from core.bytes import append_u32, get_u16, host_base, check_field_bytes

comptime VERSION: UInt32 = 1


@fieldwise_init
struct TailLevel(TrivialRegisterPassable, Writable):
    var length: Int      # |y_l| in E elements
    var rows: Int        # n_l = length / 2^tail_digits
    var L: Int           # domain size, cosets * L0
    var cosets: Int      # m: 1, 2, or 4
    var queries: Int     # |S_l|
    var codewords: Int   # n_cw of the level: rows / codewords symbols per codeword, split on the top binary digits of the row

    def write_to(self, mut w: Some[Writer]):
        w.write("TailLevel(len=", self.length, ", rows=", self.rows, ", n_cw=", self.codewords, ", L=", self.L, "=", self.cosets,
                "x", self.L // self.cosets, ", q=", self.queries, ")")


def tail_schedule[p: Params]() raises -> List[TailLevel]:
    """Committed tail levels l = 2 .. ell-1, derived from N and e (design section 2). The level after
    the last entry is sent in the clear; with no entries, y_2 itself is the clear vector. A level folds
    `tail_digits` binary digits, so the tail stops when fewer remain (the odd digit is never folded).
    Domains with odd part 1 are skipped: the encoder scatters from its last odd-radix stage."""
    var levels = List[TailLevel]()
    var length = p.N()
    var digits = p.a1 + p.a2
    while length > p.tail_clear_max and digits >= p.tail_digits:
        var rows = length >> p.tail_digits
        # the codeword split: the smallest power of two whose rows / cw symbols a domain holds (spec 9.1)
        var cw = 1
        var dom = domain_for(rows, p.tail_rate_inv)
        while dom[1] == 0 and cw < (1 << (digits - p.tail_digits)):
            cw *= 2
            dom = domain_for(rows // cw, p.tail_rate_inv)
        var L = dom[0]
        var cosets = dom[1]
        if L == 0:
            raise Error("tail level does not fit the F4 domain")
        # queries at the exact rate (rows / cw) / L, same formula as level 1
        var rate = Float64(rows // cw) / Float64(L)
        var queries = query_count(p.lambda_bits - p.grind_bits, rate, p.regime, p.eta_inv)
        levels.append(TailLevel(length=length, rows=rows, L=L, cosets=cosets, queries=queries, codewords=cw))
        length = rows
        digits -= p.tail_digits
    return levels^


def _log2(x: Float64) -> Float64:
    return log2(x)


struct Shape(Writable):
    """Everything the proof size depends on besides Params: set by the IR program."""
    var columns_w: Int          # witness tree
    var columns_z: Int          # accumulator tree, e coordinate columns per accumulator
    var columns_q: Int          # quotient tree, 3 e coordinate columns
    var points: Int             # P opening points
    var point_list: List[UInt8] # the P points (POINT bytes each), part of the artifact; index 0 is z
    var chals: List[UInt8]      # challenge derivation table (CHAL bytes per row), part of the artifact
    var entries: Int            # family table entries (residual.mojo)
    var accs: List[UInt8]       # accumulator descriptors (accumulate.mojo), part of the artifact
    var tables: List[List[UInt8]]   # lookup tables, (K, w) bytes each, part of the artifact (milestone-3-lookup.md)
    var columns_p: Int          # public columns: on the LDE buffer after W and Z, never committed (docs/public-columns.md)
    var publics: List[UInt8]    # m per public column (PUB bytes each), part of the artifact
    var restrictions: List[UInt8]   # (column, coordinate, coefficient count) per restriction (RES bytes each), part of the artifact
    var ends: List[UInt8]       # chain-end terms on the small grid (END bytes each, smallgrid.mojo), part of the artifact
    var wires: List[UInt8]      # wiring products (WIRE bytes each, accumulate.mojo), part of the artifact
    var sigma: List[UInt8]      # the wiring permutation: F2 per slot and chain, slot-major, part of the artifact
    var pubf: List[UInt8]       # public factors (PUBF bytes each): virtual slots whose value the verifier fingerprints from the public data
    var zeros: List[UInt8]      # zero rows (ZERO bytes each): a column's opening at (1, z2) or (e1, z2) is zero, part of the artifact
    var pinned: List[UInt8]     # the public inputs must start with these bytes (a circuit description the statement was compiled from), part of the artifact
    var groups: List[UInt8]     # group public columns (ir.group_record): the verifier checks each one's public data against its group's mask, part of the artifact
    var tail: List[TailLevel]
    var clear_length: Int       # |y_ell|

    def __init__[p: Params](out self, columns_w: Int, families: List[UInt8], accs: List[UInt8] = List[UInt8](),
                            tables: List[List[UInt8]] = List[List[UInt8]](), publics: List[UInt8] = List[UInt8](),
                            restrictions: List[UInt8] = List[UInt8](), points: List[UInt8] = List[UInt8](),
                            chals: List[UInt8] = standard_chals(), ends: List[UInt8] = List[UInt8](), wires: List[UInt8] = List[UInt8](),
                            sigma: List[UInt8] = List[UInt8](), pubf: List[UInt8] = List[UInt8](), zeros: List[UInt8] = List[UInt8](),
                            pinned: List[UInt8] = List[UInt8](), groups: List[UInt8] = List[UInt8]()) raises:
        """The entry count comes from the family table (residual.mojo). `points` is the opening list (an empty
        list means the default, ir.shift_points); it must hold every point of ir.required_points. `chals` is the
        challenge derivation table. A lookup descriptor names its table by index; the table's row width is the
        record width.
        TODO(memory): a KIND_MEMORY descriptor (spec 6.4) has no table and its own column roles; validate here."""
        p.check()
        if len(families) % ENTRY != 0 or len(accs) % ACC != 0 or len(publics) % PUB != 0 or len(restrictions) % RES != 0 or len(points) % POINT != 0 or len(chals) % CHAL != 0 or len(ends) % END != 0 or len(wires) % WIRE != 0 or len(pubf) % PUBF != 0 or len(zeros) % ZERO != 0:
            raise Error("family, accumulator, public, restriction, point, challenge, chain-end, wiring, public factor, zero-row or group table is not whole entries")
        self.columns_w = columns_w
        self.columns_z = p.e * (len(accs) // ACC)
        self.columns_q = 3 * p.e
        self.columns_p = len(publics) // PUB
        self.point_list = points.copy() if len(points) > 0 else shift_points(families, restrictions, len(accs) > 0, zeros)
        self.points = len(self.point_list) // POINT
        self.chals = chals.copy()
        self.entries = len(families) // ENTRY
        self.accs = accs.copy()
        self.tables = tables.copy()
        self.publics = publics.copy()
        self.restrictions = restrictions.copy()
        self.ends = ends.copy()
        self.wires = wires.copy()
        self.sigma = sigma.copy()
        self.pubf = pubf.copy()
        self.zeros = zeros.copy()
        self.pinned = pinned.copy()
        self.groups = groups.copy()
        var opened = self.columns_w + self.columns_z
        if get_u16(self.point_list, 0) != 0 or get_u16(self.point_list, 2) != 0:
            raise Error("opening point 0 must be z")
        for i in range(self.points):
            var dj1 = get_u16(self.point_list, i * POINT)
            var dj2 = get_u16(self.point_list, i * POINT + 2)
            if (dj1 >= 2 * p.h1() and dj1 != FIX_ONE and dj1 != FIX_E) or (dj2 >= 2 * p.h2() and dj2 != FIX_ONE and dj2 != FIX_E):
                raise Error("opening point is outside the grid")
            if point_index(self.point_list, dj1, dj2) != i:
                raise Error("opening points repeat")
        var need = required_points(families, restrictions, len(accs) > 0, zeros)
        for i in range(len(need) // POINT):
            if point_index(self.point_list, get_u16(need, i * POINT), get_u16(need, i * POINT + 2)) < 0:
                raise Error("opening points must include every read shift, restriction line, and accumulator boundary point")
        if chal_count(chals) > CHAL_ONE - 1:
            raise Error("challenge derivation table holds at most 251 rows (elements are u8 indices, 255 is the constant)")
        for i in range(len(chals) // CHAL):
            var op = Int(chals[i * CHAL])
            var a = Int(chals[i * CHAL + 1])
            var b = Int(chals[i * CHAL + 2])
            if (op != CHAL_ADD and op != CHAL_MUL) or (a != CHAL_ONE and a >= SAMPLED + i) or (b != CHAL_ONE and b >= SAMPLED + i):
                raise Error("challenge derivation row must add or multiply earlier elements")
        if len(accs) > 0:                              # the factor kernels read 1 + beta and (1 + beta) delta at 3 and 4
            var std = standard_chals()
            if len(chals) < len(std) or chals[: len(std)] != Span(std):
                raise Error("accumulators need the derivation table to start with 1 + beta and (1 + beta) delta")
        for i in range(self.columns_p):
            var m = Int(publics[i * PUB]) | Int(publics[i * PUB + 1]) << 8
            if m == 0 or p.h2() % m != 0:
                raise Error("public column period divides h2: need m >= 1, h2 % m == 0")
        for i in range(len(restrictions) // RES):
            var col = Int(restrictions[i * RES]) | Int(restrictions[i * RES + 1]) << 8
            var coord = Int(restrictions[i * RES + 2]) | Int(restrictions[i * RES + 3]) << 8
            var count = Int(restrictions[i * RES + 4]) | Int(restrictions[i * RES + 5]) << 8
            if col >= opened or (coord != FIX_ONE and coord != FIX_E) or count == 0 or count > p.h1():
                raise Error("restriction needs an opened column, a fixed axis-2 coordinate, and a coefficient count in [1, h1]")
        for i in range(len(zeros) // ZERO):
            var coord = Int(zeros[i * ZERO + 2]) | Int(zeros[i * ZERO + 3]) << 8
            if (Int(zeros[i * ZERO]) | Int(zeros[i * ZERO + 1]) << 8) >= opened or (coord != FIX_ONE and coord != FIX_E):
                raise Error("zero row needs an opened column and a fixed axis-1 coordinate")
        for k in range(self.entries):
            var en = entry(families, k)
            var pub_a = en.col_a >= opened
            var pub_b = en.col_b != NONE and en.col_b >= opened
            if en.col_a >= opened + self.columns_p or (en.col_b != NONE and en.col_b >= opened + self.columns_p):
                raise Error("family entry reads a column past the public columns")
            if en.dj1_a >= 2 * p.h1() or en.dj2_a >= 2 * p.h2() or (en.col_b != NONE and (en.dj1_b >= 2 * p.h1() or en.dj2_b >= 2 * p.h2())):
                raise Error("family entry shift is outside the residual grid")
            if en.chal > chal_count(chals):
                raise Error("family entry names a challenge element past the derivation table")
            if pub_a and pub_b:
                raise Error("a quadratic entry may read at most one public column")
            if en.mult > 2 or (en.mult == 2 and en.col_b != NONE) or en.coef >= 127 or (en.basis != NO_BASIS and en.basis >= p.e) or (en.basis2 != NO_BASIS and en.basis2 >= p.e):
                raise Error("family entry gate, coefficient, or basis out of range")
        for k in range(len(accs) // ACC):
            if (Int(accs[k * ACC]) | Int(accs[k * ACC + 1]) << 8) != columns_w + k * p.e:
                raise Error("accumulator z_col must be columns_w + k e in registration order (the Z tree packs Z_k at that block)")
            if acc_kind(accs, k) == KIND_HORNER:   # the ingest range is what the kernel reads: linear W reads inside the chain
                var first = Int(accs[k * ACC + 2]) | Int(accs[k * ACC + 3]) << 8
                var count = Int(accs[k * ACC + 4]) | Int(accs[k * ACC + 5]) << 8
                if count == 0 or first + count > self.entries or acc_start(accs, k) > 1 or Int(accs[k * ACC + 7]) > chal_count(chals):
                    raise Error("horner descriptor: ingest range inside the family table, start in {0, 1}, scale a stage-1 element")
                if first < HORNER_TRANSITIONS:
                    raise Error("horner descriptor: the transition entries precede the ingest range")
                for t in range(HORNER_TRANSITIONS):        # what merge_tables drops; the residual kernel reads only the first pair's weights and applies them to every coordinate, so all 2 e must match Families.horner exactly
                    var en = entry(families, first - HORNER_TRANSITIONS + t)
                    var nxt = t % 2 == 0                   # even: R(omega1 x1) coef 1; odd: -scale R with the scale element
                    if (en.col_a != acc_z_col(accs, k) + t // 2 or en.col_b != NONE or en.mult != 1 or en.basis != t // 2 or en.basis2 != NO_BASIS
                            or en.family != acc_family(accs, k) or en.dj1_a != (2 if nxt else 0) or en.dj2_a != 0
                            or en.coef != (1 if nxt else 126) or en.chal != (0 if nxt else Int(accs[k * ACC + 7]))):
                        raise Error("horner descriptor: the entries before the ingest range are not its transition entries")
                for i in range(first, first + count):
                    var en = entry(families, i)
                    if en.col_a >= columns_w or en.mult != 1 or en.dj2_a != 0 or en.dj1_a % 2 != 0 or en.basis != NO_BASIS or en.basis2 != NO_BASIS or en.family != acc_family(accs, k):
                        raise Error("horner ingest entries are gated reads of witness columns in the accumulator's family")
                    if en.col_b != NONE and (en.col_b < opened or en.col_b >= opened + self.columns_p or en.dj2_b != 0 or en.dj1_b % 2 != 0):
                        raise Error("a horner ingest entry's selector is a public column read on the same chain")
                continue
            var w_num = Int(accs[k * ACC + 2]) | Int(accs[k * ACC + 3]) << 8
            var w_den = Int(accs[k * ACC + 4]) | Int(accs[k * ACC + 5]) << 8
            if w_num == 0 or w_num > ACC_W_MAX or w_den == 0 or w_den > ACC_W_MAX:
                raise Error("accumulator record width")
            for j in range(16):                        # record columns are witness columns (the factor kernel reads the W trace)
                var at = k * ACC + 6 + 2 * j
                if (j % 8) < (w_num if j < 8 else w_den) and (Int(accs[at]) | Int(accs[at + 1]) << 8) >= columns_w:
                    raise Error("accumulator record columns must be witness columns")
            if acc_kind(accs, k) == KIND_LOOKUP:
                var t = acc_table(accs, k)
                if w_num != w_den or t >= len(tables) or len(tables[t]) == 0 or len(tables[t]) % w_num != 0:
                    raise Error("lookup descriptor needs a table of its record width")
                for b in tables[t]:                    # the verifier compares raw bytes for the break rule; [0,127] must not pass as [0,0]
                    if Int(b) >= 127:
                        raise Error("lookup table bytes must be canonical field elements (< 127)")
                for i in range(w_num):                 # the sort reads f and writes s in one pass
                    for j in range(w_num):
                        if accs[k * ACC + 6 + 2 * i] == accs[k * ACC + 22 + 2 * j] and accs[k * ACC + 7 + 2 * i] == accs[k * ACC + 23 + 2 * j]:
                            raise Error("lookup record and sorted columns must be distinct")
        for i in range(len(ends) // END):
            var ca = get_u16(ends, i * END)
            var cb = get_u16(ends, i * END + 2)
            var ok_a = ca >= columns_w and ca < opened and (ca - columns_w) % p.e == 0
            var ok_b = cb == NONE or (cb >= columns_w and cb < opened and (cb - columns_w) % p.e == 0)
            if not ok_a or not ok_b or Int(ends[i * END + 6]) >= 127 or Int(ends[i * END + 7]) > chal_count(chals) or Int(ends[i * END + 8]) > 1:
                raise Error("chain-end term reads Z blocks by their first column, with a canonical coefficient, a stage-1 element, and a gate flag")
            for k in range(len(accs) // ACC):              # R2 sums every term with alpha^family: a (W) pair and a chain-end family never share one
                if acc_kind(accs, k) != KIND_HORNER and acc_family(accs, k) == get_u16(ends, i * END + 4):
                    raise Error("chain-end family index collides with a grand-product accumulator's")
        var products = 0
        for k in range(len(accs) // ACC):
            if acc_kind(accs, k) == KIND_HORNER:
                continue
            products += 1
            for j in range(k):                             # one alpha power per (W) pair
                if acc_kind(accs, j) != KIND_HORNER and acc_family(accs, j) == acc_family(accs, k):
                    raise Error("grand-product accumulators must carry distinct family indices")
        var slots = 0
        for g in range(len(wires) // WIRE):
            var fam = get_u16(wires, g * WIRE + 4)
            for s in range(2):
                var c = get_u16(wires, g * WIRE + 2 * s)
                if c == NONE and (s == 0 or g + 1 < len(wires) // WIRE):    # slot index 2 g + s indexes sigma: only the last product is short
                    raise Error("only the last wiring product may have one slot")
                if c != NONE and (c < columns_w or c >= opened or (c - columns_w) % p.e != 0):
                    raise Error("wiring slots are Z blocks by their first column")
                slots += 0 if c == NONE else 1
            for k in range(len(accs) // ACC):
                if acc_kind(accs, k) != KIND_HORNER and acc_family(accs, k) == fam:
                    raise Error("wiring family index collides with a grand-product accumulator's")
            for i in range(len(ends) // END):
                if get_u16(ends, i * END + 4) == fam:
                    raise Error("wiring family index collides with a chain-end family's")
            for h in range(g):
                if get_u16(wires, h * WIRE + 4) == fam:
                    raise Error("wiring products must carry distinct family indices")
        if slots * p.h2() > F2_ORDER:
            raise Error("wiring slots exceed the cosets of H2 in F2*")
        if len(sigma) != 2 * slots * p.h2():
            raise Error("sigma holds one F2 element per wiring slot and chain")
        for b in sigma:
            if Int(b) >= 127:
                raise Error("sigma bytes must be canonical field elements (< 127)")
        for i in range(len(pubf) // PUBF):
            var k = get_u16(pubf, i * PUBF)
            if len(wires) == 0 or k >= len(accs) // ACC or acc_kind(accs, k) != KIND_HORNER:
                raise Error("public factor names a horner accumulator of a wired statement")
            for t in range(2, 6):
                if Int(pubf[i * PUBF + t]) >= 127:
                    raise Error("public factor id and sigma must be canonical field elements (< 127)")
            if get_u16(pubf, i * PUBF + 6) >= p.h2():
                raise Error("public factor chain is below h2")
        for off in group_offsets(groups):
            var base = get_u16(groups, off)
            var chains = get_u16(groups, off + 2)
            if chains < 1 or base + chains > p.h2():
                raise Error("group chains lie below h2")
            var c = get_u16(groups, off + 4)
            if c >= self.columns_p or get_u16(publics, c * PUB) != 1 or groups[off + 6] > 1:
                raise Error("group public column is a public column with m = 1")
            var n = Int(groups[off + 7])
            if n < 1 or get_u16(groups, off + GRP) != 0:   # sorted shifts starting at 0: the mask lies inside the group
                raise Error("group shifts start at 0")
            for i in range(n):
                if get_u16(groups, off + GRP + 2 * i) >= p.h2():
                    raise Error("group shifts lie below h2")
        if slots > 0:                                  # sigma is a permutation of the ids: the slots' cosets kappa^s H2 and the public factors' own
            var ids = Dict[Int, Int]()
            var kappa = f2_primitive()
            var omega2 = Domains.__init__[p]().omega2
            for s in range(slots):
                var x = ext_pow[1](kappa, s)
                for _ in range(p.h2()):
                    ids[Int(x[0]) | Int(x[1]) << 8] = 1
                    x = ext_mul[1](x, omega2)
            for i in range(len(pubf) // PUBF):
                var key = Int(pubf[i * PUBF + 2]) | Int(pubf[i * PUBF + 3]) << 8
                if key in ids:
                    raise Error("public factor id collides with another id")
                ids[key] = 1
            for i in range(slots * p.h2() + len(pubf) // PUBF):
                var at = i * 2 if i < slots * p.h2() else (i - slots * p.h2()) * PUBF + 4
                var key = Int(sigma[at]) | Int(sigma[at + 1]) << 8 if i < slots * p.h2() else Int(pubf[at]) | Int(pubf[at + 1]) << 8
                if key not in ids or ids[key] == 0:
                    raise Error("sigma must map every slot to a distinct id")
                ids[key] = 0
        products += len(wires) // WIRE
        if len(accs) > 0 and products == 0 and len(ends) == 0:
            raise Error("horner accumulators need a chain-end family or a wiring product (nothing else writes Q3)")
        self.tail = tail_schedule[p]()
        self.clear_length = p.N() if len(self.tail) == 0 else self.tail[len(self.tail) - 1].rows

    def accumulators(self) -> Int:
        return len(self.accs) // ACC

    def products(self) -> Int:
        """Z2 lines in the clear: the grand-product accumulators (KIND_PERM, KIND_LOOKUP), then the wiring products."""
        var n = self.wiring_products()
        for k in range(self.accumulators()):
            n += 0 if acc_kind(self.accs, k) == KIND_HORNER else 1
        return n

    def wiring_products(self) -> Int:
        return len(self.wires) // WIRE

    def product_of(self, k: Int) -> Int:
        """The product index of accumulator k, its Z2, n_end and d_end line; -1 for a Horner accumulator."""
        if acc_kind(self.accs, k) == KIND_HORNER:
            return -1
        var pi = 0
        for j in range(k):
            pi += 0 if acc_kind(self.accs, j) == KIND_HORNER else 1
        return pi

    def wiring_product(self, g: Int) -> Int:
        """The product index of wiring product g: the wiring products' lines follow the accumulators'."""
        return self.products() - self.wiring_products() + g

    def factor_bytes[p: Params](self, i: Int) -> Int:
        """Public data of public factor i: h1 bytes per ingest entry of its accumulator (ir.horner_chain_end)."""
        return p.h1() * get_u16(self.accs, get_u16(self.pubf, i * PUBF) * ACC + 4)

    def family_of(self, k: Int) -> Int:
        """The family index of accumulator k: its alpha power on the small grid."""
        return acc_family(self.accs, k)

    def chal_count(self) -> Int:
        """Stage-1 elements: the sampled ones and one per derivation row."""
        return chal_count(self.chals)

    def lookups(self) -> Int:
        var n = 0
        for k in range(self.accumulators()):
            n += 1 if acc_kind(self.accs, k) == KIND_LOOKUP else 0
        return n

    def table_rows(self, k: Int) -> Int:
        """K of the table lookup descriptor k reads."""
        return len(self.tables[acc_table(self.accs, k)]) // (Int(self.accs[k * ACC + 2]) | Int(self.accs[k * ACC + 3]) << 8)

    def max_table_rows(self) -> Int:
        var m = 0
        for k in range(self.accumulators()):
            if acc_kind(self.accs, k) == KIND_LOOKUP:
                m = max(m, self.table_rows(k))
        return m

    def public_bytes[p: Params](self) -> Int:
        """Host bytes of the public data both sides derive: the column periods, the restriction polynomials, then
        the public factors' ingest columns."""
        var n = value_bytes(self.publics, p.h1(), p.h2())
        for i in range(len(self.restrictions) // RES):
            n += (Int(self.restrictions[i * RES + 4]) | Int(self.restrictions[i * RES + 5]) << 8) * 2
        for i in range(len(self.pubf) // PUBF):
            n += self.factor_bytes[p](i)
        return n

    def trees(self) -> Int:
        """Trees opened at level 1: W and Q, plus Z when there are accumulators."""
        return 3 if self.columns_z > 0 else 2

    def columns(self) -> Int:
        return self.columns_w + self.columns_z + self.columns_q

    def fixed_bytes[p: Params, digest: Int](self, public_bytes: Int) -> Int:
        """Proof length without the multiproof bodies: their u32 prefixes are counted, one per tree
        opened (three at level 1: W, Z, Q). Mirrors ProofWriter's order exactly."""
        var n = 4 + 4 + public_bytes + 2 * digest
        if self.accumulators() > 0:                    # Z root, Z2 (products only), Q3 exist only with accumulators
            n += digest + self.products() * p.h2() * p.e + 2 * p.h2() * p.e
        n += self.points * self.columns() * p.e
        for i in range(len(self.tail)):
            n += digest + (self.trees() if i == 0 else 1) * 4 + 9 * p.e
        n += self.clear_length * p.e + (self.trees() if len(self.tail) == 0 else 1) * 4
        if p.grind_bits > 0:
            n += 8 * (len(self.tail) + 1)          # one nonce before every opened level's multiproof(s)
        return n

    def write_to(self, mut w: Some[Writer]):
        w.write("Shape(columns=", self.columns_w, "+", self.columns_z, "+", self.columns_q, ", public=", self.columns_p, ", P=", self.points,
                ", tail levels=", len(self.tail), ", clear=", self.clear_length, ")")


def prefix_bytes[p: Params, H: Hash](shape: Shape, public_inputs: Span[UInt8, _], mut families: List[UInt8]) -> List[UInt8]:
    """The transcript prefix of spec 9.4: version, field and grid parameters, domains and rates per
    level, shape, public inputs, H(family table) as the statement artifact hash of
    statement-layer 6 step 1, and H(lookup tables). Prover and verifier build the same bytes."""
    var bytes = List[UInt8]()
    append_u32(bytes, Int(VERSION))
    for v in [p.e, p.a1, p.m1, p.a2, p.m2, p.L0, p.m_cosets, p.leaf_bytes, p.tail_digits, p.tail_clear_max,
              p.lambda_bits, p.grind_bits, p.regime, p.eta_inv, p.queries(), p.n_cw()]:
        append_u32(bytes, v)
    append_u32(bytes, shape.columns_w)
    append_u32(bytes, shape.columns_z)
    append_u32(bytes, shape.columns_q)
    append_u32(bytes, len(shape.point_list))
    bytes.extend(shape.point_list.copy())
    append_u32(bytes, len(shape.chals))
    bytes.extend(shape.chals.copy())
    append_u32(bytes, len(shape.tail))
    for lvl in shape.tail:
        for v in [lvl.length, lvl.rows, lvl.L, lvl.cosets, lvl.queries, lvl.codewords]:
            append_u32(bytes, v)
    append_u32(bytes, shape.clear_length)
    append_u32(bytes, len(public_inputs))
    bytes.extend(public_inputs.copy())
    append_u32(bytes, len(shape.accs))
    bytes.extend(shape.accs.copy())
    append_u32(bytes, len(shape.publics))
    bytes.extend(shape.publics.copy())
    append_u32(bytes, len(shape.restrictions))
    bytes.extend(shape.restrictions.copy())
    append_u32(bytes, len(shape.ends))
    bytes.extend(shape.ends.copy())
    append_u32(bytes, len(shape.wires))
    bytes.extend(shape.wires.copy())
    append_u32(bytes, len(shape.sigma))
    bytes.extend(shape.sigma.copy())
    append_u32(bytes, len(shape.pubf))
    bytes.extend(shape.pubf.copy())
    append_u32(bytes, len(shape.zeros))
    bytes.extend(shape.zeros.copy())
    append_u32(bytes, len(shape.pinned))
    bytes.extend(shape.pinned.copy())
    append_u32(bytes, len(shape.groups))
    bytes.extend(shape.groups.copy())
    var digest = List[UInt8](length=H.DIGEST, fill=0)
    H.leaf(host_base(families), len(families), host_base(digest))
    bytes.extend(digest.copy())
    var tabs = List[UInt8]()                           # the tables by digest: a table can exceed the prefix region
    append_u32(tabs, len(shape.tables))
    for t in shape.tables:
        append_u32(tabs, len(t))
        tabs.extend(t.copy())
    H.leaf(host_base(tabs), len(tabs), host_base(digest))
    bytes.extend(digest^)
    return bytes.copy()


struct ProofWriter:
    """Collects the proof in order into one host staging pool allocated once per prover (design
    rule 6). Host values are written into the pool directly; device values are staged as async
    copies out of the arena into the pool and assembled after the one synchronize in `finish`, so
    the prover never waits on a read-back mid-stream. A staged multiproof carries its byte count in
    its first u32 (merkle.mojo) and is emitted length-prefixed and trimmed."""
    var ctx: DeviceContext
    var pool: HostBuffer[DType.uint8]
    var pos: Int
    var starts: List[Int]
    var lens: List[Int]
    var multiproof: List[Bool]

    def __init__(out self, ctx: DeviceContext, bytes: Int) raises:
        self.ctx = ctx
        self.pool = ctx.enqueue_create_host_buffer[DType.uint8](bytes)
        ctx.synchronize()
        self.pos = 0
        self.starts = List[Int]()
        self.lens = List[Int]()
        self.multiproof = List[Bool]()

    def reset(mut self):
        self.pos = 0
        self.starts.clear()
        self.lens.clear()
        self.multiproof.clear()

    def _take(mut self, bytes: Int) raises -> Int:
        var start = self.pos
        if start + bytes > len(self.pool):
            raise Error("proof staging pool exhausted")
        self.pos += bytes
        return start

    def scratch(mut self, bytes: Int) raises -> HostBuffer[DType.uint8]:
        """A pool region that is not part of the proof (host bytes to upload)."""
        return self.pool.create_sub_buffer[DType.uint8](self._take(bytes), bytes)

    def raw(mut self, src: Span[UInt8, _]) raises:
        if len(src) == 0:
            return
        var start = self._take(len(src))
        for i in range(len(src)):
            self.pool[start + i] = src[i]
        self.starts.append(start)
        self.lens.append(len(src))
        self.multiproof.append(False)

    def u32(mut self, v: Int) raises:
        var bytes = List[UInt8]()
        append_u32(bytes, v)
        self.raw(bytes)

    def prefixed(mut self, src: Span[UInt8, _]) raises:
        self.u32(len(src))
        self.raw(src)

    def stage(mut self, arena: Arena, off: Int, bytes: Int, multiproof: Bool = False) raises:
        var start = self._take(bytes)
        arena.download(self.ctx, off, self.pool.create_sub_buffer[DType.uint8](start, bytes))
        self.starts.append(start)
        self.lens.append(bytes)
        self.multiproof.append(multiproof)

    def finish(mut self) raises -> List[UInt8]:
        self.ctx.synchronize()
        var src = self.pool.unsafe_ptr()
        var starts = List[Int]()      # trimmed segments and their u32 prefixes
        var stops = List[Int]()
        var total = 0
        for i in range(len(self.starts)):
            var start = self.starts[i]
            var stop = start + self.lens[i]
            if self.multiproof[i]:
                var n = Int(src[unsafe_offset=start]) | Int(src[unsafe_offset=start + 1]) << 8 | Int(src[unsafe_offset=start + 2]) << 16 | Int(src[unsafe_offset=start + 3]) << 24
                if n < 4 or n > self.lens[i]:
                    raise Error("multiproof header out of range")
                stop = start + n        # n bytes: the header becomes the u32 prefix, then the body
            starts.append(start)
            stops.append(stop)
            total += stop - start
        var out = List[UInt8](unsafe_uninit_length=total)
        var dst = out.unsafe_ptr()
        var at = 0
        for i in range(len(starts)):
            if self.multiproof[i]:
                var n = stops[i] - starts[i]        # the body follows its own header; emit n - 4 then the body
                var hdr = List[UInt8]()
                append_u32(hdr, n - 4)
                for j in range(4):
                    dst[unsafe_offset=at + j] = hdr[j]
                at += 4
                unsafe_memcpy(dest=dst.unsafe_offset(at), src=src.unsafe_offset(starts[i] + 4), count=n - 4)
                at += n - 4
            else:
                unsafe_memcpy(dest=dst.unsafe_offset(at), src=src.unsafe_offset(starts[i]), count=stops[i] - starts[i])
                at += stops[i] - starts[i]
        return out^


struct ProofReader:
    var bytes: List[UInt8]
    var pos: Int

    def __init__(out self, var bytes: List[UInt8]):
        self.bytes = bytes^
        self.pos = 0

    def u32(mut self) raises -> Int:
        if self.pos + 4 > len(self.bytes):
            raise Error("proof truncated")
        var v = 0
        for i in range(4):
            v |= Int(self.bytes[self.pos + i]) << (8 * i)
        self.pos += 4
        return v

    def take(mut self, n: Int) raises -> List[UInt8]:
        """Opaque bytes; use field_bytes for coordinates that will enter field arithmetic."""
        if self.pos + n > len(self.bytes):
            raise Error("proof truncated")
        var out = List[UInt8](capacity=n)
        for i in range(n):
            out.append(self.bytes[self.pos + i])
        self.pos += n
        return out^

    def field_bytes(mut self, n: Int) raises -> List[UInt8]:
        """Read n bytes of canonical field coordinates; hashes and raw inputs use take/prefixed."""
        var bytes = self.take(n)
        check_field_bytes(bytes)
        return bytes^

    def prefixed(mut self) raises -> List[UInt8]:
        var n = self.u32()
        return self.take(n)

    def done(self) raises:
        if self.pos != len(self.bytes):
            raise Error("trailing bytes in proof")

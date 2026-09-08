"""The statement builder (docs/statement-builder.md): columns, reads, families, accumulators, public
columns, and restrictions by name. `compile` resolves the names, runs the statement-layer checks, emits
the family bytes through `Families`, derives the opening list, and returns the Shape with a `Layout`
(name to column, group to columns, the descriptors and tables the trace helpers read). Names, groups, and
kinds are prover-side only: the artifact is the bytes Shape already hashes.

Kinds decide the certificate the builder emits: a BIT column gets its Booleanity family, a LIMB6 column a
lookup into the [64] table through a sorted column the builder allocates (`<name>.sorted`), a BYTE column
nothing. An E column is a Z block and only `acc` creates one. Padding is a function of the kinds too: an
idle row of a lookup record column holds a table row, any other column zero, and Z blocks and sorted
columns are the prover's (`pad_trace`).
ponytail: one lookup per LIMB6 column (16 Z columns each); a shared range lookup when ECDSA's limb count
is measured. Groups are labels for `pad_trace`; a per-group live-row count is the frontend's.
"""

from caracal7.core.field import F2, f_add, f_pow, ext_mul, ext_pow
from caracal7.core.params import Params
from caracal7.core.tables import Domains
from caracal7.core.bytes import set_u16, get_u16, append_u32
from caracal7.relations.ir import Families, standard_chals, shift_points, chal_count, wire_record, public_factor_record, CHAL_ADD, CHAL_MUL, CHAL_ONE, FIX_ONE, FIX_E, PUB, RES, ACC, ACC_W_MAX, KIND_PERM, KIND_LOOKUP, KIND_HORNER
from caracal7.core.field import F2, ext_mul, ext_pow
from caracal7.core.tables import Domains, f2_primitive, F2_ORDER
from caracal7.proof import Shape

comptime BIT = 0
comptime LIMB6 = 1
comptime BYTE = 2       # any F value
comptime GATE_NONE = 0
comptime GATE_1 = 1     # (X1 - e1): the family holds on every row but the last of each chain
comptime GATE_2 = 2     # (X2 - e2): every chain but the last
comptime LIMB_ROWS = 64


@fieldwise_init
struct Read(Copyable, Movable):
    """A column at (omega1^k1 x1, omega2^k2 x2): k1 cyclic on the chain, k2 = 1 the next chain."""
    var col: String
    var k1: Int
    var k2: Int


struct Term(Copyable, Movable):
    """coef * element(chal) * b_basis * b_basis2 * a * b. `chal` is a stage-1 element index (-1 none; 0 beta,
    1 delta, 2 gamma, 3 = 1 + beta, 4 = (1 + beta) delta, then `Statement.derived` rows); basis, basis2 are
    coordinates of E (-1 none)."""
    var coef: Int
    var a: Read
    var b: Optional[Read]
    var chal: Int
    var basis: Int
    var basis2: Int

    def __init__(out self, coef: Int, a: Read, b: Optional[Read] = None, chal: Int = -1, basis: Int = -1, basis2: Int = -1):
        self.coef = coef
        self.a = a.copy()
        self.b = b.copy()
        self.chal = chal
        self.basis = basis
        self.basis2 = basis2


@fieldwise_init
struct _Family(Copyable, Movable):
    var name: String
    var terms: List[Term]
    var gate: Int


@fieldwise_init
struct _Acc(Copyable, Movable):
    var name: String
    var kind: Int
    var num: List[String]       # KIND_HORNER: unused
    var den: List[String]
    var table: Int
    var start: Int              # KIND_HORNER: R(1, X2); scale the element index (-1: 1); ingest the weighted linear reads
    var scale: Int
    var ingest: List[Term]


@fieldwise_init
struct _End(Copyable, Movable):
    var name: String
    var terms: List[Term]       # reads name Z blocks; coef chal a [b]
    var gated: Bool


struct Layout(Copyable, Movable):
    """Prover-side names: W columns in index order with their kind and group, plus the descriptors, tables,
    public specs, and restriction records the trace helpers read. Not part of the artifact."""
    var names: List[String]
    var index: Dict[String, Int]
    var kinds: List[Int]
    var groups: List[String]
    var accs: List[UInt8]
    var tables: List[List[UInt8]]
    var publics: List[UInt8]
    var restrictions: List[UInt8]
    var slots: List[Tuple[Int, Int, Int]]   # per W column: (table id, record position, width) of its lookup record;
                                            # (-1, 0, 0) none, (-2, 0, 0) a sorted column, (-3, 0, 0) a record of two lookups

    def __init__(out self, names: List[String], index: Dict[String, Int], kinds: List[Int], groups: List[String],
                 accs: List[UInt8], tables: List[List[UInt8]], publics: List[UInt8], restrictions: List[UInt8]):
        self.names = names.copy()
        self.index = index.copy()
        self.kinds = kinds.copy()
        self.groups = groups.copy()
        self.accs = accs.copy()
        self.tables = tables.copy()
        self.publics = publics.copy()
        self.restrictions = restrictions.copy()
        self.slots = List[Tuple[Int, Int, Int]](length=len(names), fill=(-1, 0, 0))
        for k in range(len(accs) // ACC):
            if Int(accs[k * ACC + 38]) != KIND_LOOKUP:
                continue
            var width = get_u16(accs, k * ACC + 2)
            for j in range(width):
                var c = get_u16(accs, k * ACC + 6 + 2 * j)
                if self.slots[c][0] == -1:
                    self.slots[c] = (Int(accs[k * ACC + 39]), j, width)
                else:
                    self.slots[c] = (-3, 0, 0)
                self.slots[get_u16(accs, k * ACC + 22 + 2 * j)] = (-2, 0, 0)

    def col(self, name: String) raises -> Int:
        if name in self.index:
            return self.index[name]
        raise Error("unknown column " + name)

    def columns_w(self) -> Int:
        return len(self.names)


struct Compiled(Movable):
    var shape: Shape
    var families: List[UInt8]
    var layout: Layout

    def __init__(out self, var shape: Shape, var families: List[UInt8], var layout: Layout):
        self.shape = shape^
        self.families = families^
        self.layout = layout^

    def take_shape(deinit self) -> Shape:
        """Consume the compiled statement for its Shape (Shape is move-only; the Prover owns one)."""
        return self.shape^


struct Statement(Movable):
    var cols: List[String]
    var col_index: Dict[String, Int]
    var kinds: List[Int]
    var col_group: List[String]
    var fams: List[_Family]
    var accs: List[_Acc]
    var ends: List[_End]
    var order: List[Tuple[Int, Int]]    # (0 family i | 1 accumulator i | 2 chain-end family i) in call order: the position is the family index
    var pubs: List[UInt8]               # PUB records
    var pub_names: List[String]
    var res: List[UInt8]                # RES records, column resolved at compile
    var res_names: List[String]
    var tables: List[List[UInt8]]
    var widths: List[Int]
    var chals: List[UInt8]
    var slots: List[Int]                # wiring slots: accumulator index each (accumulate.mojo, "Wiring")
    var edges: List[Tuple[Int, Int, Int, Int]]   # (slot, chain, slot, chain) equalities
    var factors: List[Tuple[String, Int, Int, Int]]   # public factors: (name, accumulator, slot, chain)

    def __init__(out self):
        self.cols = List[String]()
        self.col_index = Dict[String, Int]()
        self.kinds = List[Int]()
        self.col_group = List[String]()
        self.fams = List[_Family]()
        self.accs = List[_Acc]()
        self.ends = List[_End]()
        self.order = List[Tuple[Int, Int]]()
        self.pubs = List[UInt8]()
        self.pub_names = List[String]()
        self.res = List[UInt8]()
        self.res_names = List[String]()
        self.tables = List[List[UInt8]]()
        self.widths = List[Int]()
        self.chals = standard_chals()
        self.slots = List[Int]()
        self.edges = List[Tuple[Int, Int, Int, Int]]()
        self.factors = List[Tuple[String, Int, Int, Int]]()

    def _fresh(self, name: String) raises:
        if name in self.col_index:
            raise Error("name in use: " + name)
        for a in self.accs:
            if a.name == name:
                raise Error("name in use: " + name)
        for n in self.pub_names:
            if n == name:
                raise Error("name in use: " + name)
        for f in self.factors:
            if f[0] == name:
                raise Error("name in use: " + name)

    def col(mut self, name: String, kind: Int = BYTE, group: String = "") raises:
        """A witness column; the index is the declaration order."""
        self._fresh(name)
        if kind < BIT or kind > BYTE:
            raise Error("column kind is BIT, LIMB6, or BYTE")
        self.col_index[name] = len(self.cols)
        self.cols.append(name)
        self.kinds.append(kind)
        self.col_group.append(group)

    def table(mut self, rows: List[UInt8], width: Int) raises -> Int:
        """Register a lookup table (canonical bytes, `width` per row); returns its id. Rows are distinct (the
        advice picks a record's row by value) and two consecutive rows differ (ir.lookup_constant)."""
        if width < 1 or len(rows) == 0 or len(rows) % width != 0:
            raise Error("lookup table is whole rows")
        var k = len(rows) // width
        if k < 2:
            raise Error("lookup table needs two distinct entries")
        var seen = Dict[Int, Int]()
        for i in range(k):
            var key = _row_key(rows, i * width, width)
            if key < 0:
                raise Error("lookup table bytes must be canonical field elements (< 127)")
            if key in seen:
                raise Error("lookup table rows must be distinct")
            seen[key] = i
        self.tables.append(rows.copy())
        self.widths.append(width)
        return len(self.tables) - 1

    def acc(mut self, name: String, kind: Int, num: List[String], den: List[String], table: Int = -1) raises:
        """A Z block: KIND_PERM (num against den) or KIND_LOOKUP (records num against `table`, den the sorted
        columns the prover fills). Its family index is its position among family and acc calls."""
        self._fresh(name)
        if kind == KIND_LOOKUP and (table < 0 or table >= len(self.tables)):
            raise Error("lookup accumulator needs a registered table")
        if kind == KIND_LOOKUP and len(num) != self.widths[table]:
            raise Error("lookup record width differs from its table's")
        self.order.append((1, len(self.accs)))
        self.accs.append(_Acc(name, kind, num.copy(), den.copy(), table, 0, -1, List[Term]()))

    def horner(mut self, name: String, ingest: List[Term], scale: Int = -1, start: Int = 0) raises:
        """The second Z kind (polynomial-mulmod 5): R(1, X2) = start and R(omega1 x1, x2) = scale R + sum of the
        ingest terms, each coef chal read of a W column inside the chain (k2 = 0, linear, no basis). `scale` is a
        stage-1 element index (-1: 1). Chain ends meet in `chain_end` families."""
        self._fresh(name)
        if len(ingest) == 0 or (start != 0 and start != 1) or scale < -1:
            raise Error("horner accumulator needs ingest terms, start in {0, 1}, and a scale element: " + name)
        for t in ingest:
            if t.b or t.a.k2 != 0 or t.basis >= 0 or t.basis2 >= 0:
                raise Error("horner ingest terms are linear same-chain reads without basis factors: " + name)
        self.order.append((1, len(self.accs)))
        self.accs.append(_Acc(name, KIND_HORNER, List[String](), List[String](), -1, start, scale, ingest.copy()))

    def chain_end(mut self, name: String, terms: List[Term], gated: Bool = False) raises:
        """A family on the small grid (spec 7.4): sum of coef chal A(e1, X2) [B(e1, X2)] = 0 on H2, the reads
        naming accumulators; `gated` multiplies by (X2 - e2) so the last chain is exempt."""
        if len(terms) == 0:
            raise Error("chain-end family needs terms: " + name)
        for t in terms:
            if t.a.k1 != 0 or t.a.k2 != 0 or t.basis >= 0 or t.basis2 >= 0 or (t.b and (t.b.value().k1 != 0 or t.b.value().k2 != 0)):
                raise Error("chain-end terms read accumulators at (e1, X2) with no shift or basis: " + name)
        self.order.append((2, len(self.ends)))
        self.ends.append(_End(name, terms.copy(), gated))

    def slot(mut self, acc: String) raises -> Int:
        """Register accumulator `acc`'s chain-end values as a wiring slot (polynomial-mulmod "Wiring"); returns
        the slot index. Slots pair up into products in registration order."""
        self.slots.append(self._acc_index(acc))
        return len(self.slots) - 1

    def wire(mut self, slot_a: Int, chain_a: Int, slot_b: Int, chain_b: Int) raises:
        """Slot `slot_a` on chain `chain_a` equals slot `slot_b` on chain `chain_b` (a copy constraint edge)."""
        if slot_a < 0 or slot_a >= len(self.slots) or slot_b < 0 or slot_b >= len(self.slots) or chain_a < 0 or chain_b < 0:
            raise Error("wire names registered slots and chain indices")
        self.edges.append((slot_a, chain_a, slot_b, chain_b))

    def public_factor(mut self, name: String, acc: String, slot: Int, chain: Int) raises:
        """A public value `name` equal to `slot` on `chain`: the verifier fingerprints its public data (h1 bytes per
        ingest entry of Horner accumulator `acc`, after the restriction lines) and closes the wiring product on it."""
        self._fresh(name)
        if slot < 0 or slot >= len(self.slots) or chain < 0:
            raise Error("public factor names a registered slot and a chain index")
        var k = self._acc_index(acc)
        if self.accs[k].kind != KIND_HORNER:
            raise Error("public factor fingerprints by a horner accumulator: " + name)
        self.factors.append((name, k, slot, chain))

    def pub(mut self, name: String, m: Int = 1) raises:
        """A public column (docs/public-columns.md): a polynomial in (X1, X2^m), a column periodic along axis 2
        with period h2 / m; its public data is one period of values. Indexed after W and Z."""
        self._fresh(name)
        if m < 1 or m > 65535:
            raise Error("public column needs m >= 1, a u16")
        var b = List[UInt8](length=PUB, fill=0)
        set_u16(b, 0, m)
        self.pubs.extend(b^)
        self.pub_names.append(name)

    def restrict(mut self, name: String, coord: Int, count: Int = 0) raises:
        """Restrict W column `name` on the chain X2 = 1 (FIX_ONE) or e2 (FIX_E) to a public line of degree < count
        (0: the full line, h1 coefficients)."""
        if (coord != FIX_ONE and coord != FIX_E) or count < 0 or count > 65535:
            raise Error("restriction coordinate is FIX_ONE or FIX_E, count a u16")
        var r = List[UInt8](length=RES, fill=0)
        set_u16(r, 2, coord)
        set_u16(r, 4, count)
        self.res.extend(r^)
        self.res_names.append(name)

    def read(self, name: String, k1: Int = 0, k2: Int = 0) -> Read:
        return Read(name, k1, k2)

    def derived(mut self, op: Int, a: Int, b: Int) raises -> Int:
        """Append a challenge derivation row (CHAL_ADD or CHAL_MUL of two earlier elements, CHAL_ONE the
        constant 1); returns the new element's index for `Term.chal`."""
        var n = chal_count(self.chals)
        if (op != CHAL_ADD and op != CHAL_MUL) or (a != CHAL_ONE and (a < 0 or a >= n)) or (b != CHAL_ONE and (b < 0 or b >= n)):
            raise Error("challenge derivation row must add or multiply earlier elements")
        self.chals.extend([UInt8(op), UInt8(a), UInt8(b)])
        return n

    def family(mut self, name: String, terms: List[Term], gate: Int = GATE_NONE) raises:
        """sum of terms = 0 on the rows the gate admits. One entry per term, sharing the family index."""
        if gate < GATE_NONE or gate > GATE_2 or len(terms) == 0:
            raise Error("family needs terms and a gate in {GATE_NONE, GATE_1, GATE_2}")
        for t in terms:
            if t.basis < -1 or t.basis >= 16 or t.basis2 < -1 or t.basis2 >= 16:
                raise Error("basis is a coordinate of E: -1 or [0, 16)")
        self.order.append((0, len(self.fams)))
        self.fams.append(_Family(name, terms.copy(), gate))

    def _acc_index(self, name: String) raises -> Int:
        for i in range(len(self.accs)):
            if self.accs[i].name == name:
                return i
        raise Error("unknown accumulator " + name)

    def _wcol(self, index: Dict[String, Int], name: String) raises -> Int:
        if name in index:
            return index[name]
        raise Error("unknown witness column " + name)

    def _resolve[p: Params](self, index: Dict[String, Int], r: Read, pub_at: Int, mut touched: List[Bool], mut read: List[Bool]) raises -> Int:
        """A read to a column index: W by name, else a public column past W and Z."""
        if r.k1 < 0 or r.k1 >= p.h1() or r.k2 < 0 or r.k2 > 1:
            raise Error("read shift: k1 in [0, h1), k2 in {0, 1}")
        if r.col in index:
            var i = index[r.col]
            touched[i] = True
            read[i] = True
            return i
        for i in range(len(self.pub_names)):
            if self.pub_names[i] == r.col:
                return pub_at + i
        raise Error("unknown column " + r.col)

    def compile[p: Params](self) raises -> Compiled:
        """Resolve, check, emit. Raises on the first failed check; the Shape constructor runs the byte-level
        ones (degree bound, public bounds, table canonicality, point list)."""
        if len(self.order) + len(self.cols) > 65535:
            raise Error("family index is a u16")
        var names = self.cols.copy()
        var index = self.col_index.copy()
        var kinds = self.kinds.copy()
        var groups = self.col_group.copy()
        var tables = self.tables.copy()
        var limbs = List[Int]()
        var limb_table = -1
        for i in range(len(self.cols)):
            if kinds[i] == LIMB6:
                if limb_table < 0:
                    var t = List[UInt8](capacity=LIMB_ROWS)
                    for j in range(LIMB_ROWS):
                        t.append(UInt8(j))
                    limb_table = len(tables)
                    tables.append(t^)
                limbs.append(i)
                self._fresh(self.cols[i] + ".sorted")
                index[self.cols[i] + ".sorted"] = len(names)
                names.append(self.cols[i] + ".sorted")
                kinds.append(BYTE)
                groups.append(self.col_group[i])
        var w = len(names)
        var pub_at = w + (len(self.accs) + len(limbs)) * p.e
        var touched = List[Bool](length=w, fill=False)
        var read = List[Bool](length=w, fill=False)       # by a family
        var acc_uses = List[Int](length=w, fill=0)        # as a record or sorted column
        var sorted_of = List[Int](length=w, fill=-1)      # the lookup whose sorted copy the column is
        var f = Families()
        for k in range(len(self.order)):
            var it = self.order[k]
            if it[0] == 0:
                var fam = self.fams[it[1]].copy()
                for t in fam.terms:
                    var quadratic = Bool(t.b)
                    var ca = self._resolve[p](index, t.a, pub_at, touched, read)
                    var cb = -1
                    var k1b = 0
                    var k2b = 0
                    if quadratic:
                        cb = self._resolve[p](index, t.b.value(), pub_at, touched, read)
                        k1b = t.b.value().k1
                        k2b = t.b.value().k2
                    if (t.a.k2 == 1 or k2b == 1) and (fam.gate != GATE_2 or quadratic):
                        raise Error("a next-chain read (k2 = 1) needs a linear family with the axis-2 gate: " + fam.name)
                    f.add(k, ((t.coef % 127) + 127) % 127, ca, k1_a=t.a.k1, k2_a=t.a.k2, col_b=cb, k1_b=k1b, k2_b=k2b,
                          mult=fam.gate, chal=0 if t.chal < 0 else t.chal + 1, basis=t.basis, basis2=t.basis2)
            elif it[0] == 2:
                var en = self.ends[it[1]].copy()
                for t in en.terms:
                    var cb = -1
                    if t.b:
                        cb = w + self._acc_index(t.b.value().col) * p.e
                    f.chain_end(k, t.coef, w + self._acc_index(t.a.col) * p.e, cb, t.chal, en.gated)
            elif self.accs[it[1]].kind == KIND_HORNER:
                var a = self.accs[it[1]].copy()
                var ingest = List[Tuple[Int, Int, Int, Int]]()
                for t in a.ingest:
                    if t.a.k1 < 0 or t.a.k1 >= p.h1():
                        raise Error("read shift: k1 in [0, h1), k2 in {0, 1}")
                    var c = self._wcol(index, t.a.col)
                    touched[c] = True
                    read[c] = True
                    ingest.append((c, t.a.k1, t.coef, t.chal))
                f.horner(k, w + it[1] * p.e, a.start, a.scale, ingest)
            else:
                var a = self.accs[it[1]].copy()
                var num = List[Int]()
                var den = List[Int]()
                for n in a.num:
                    num.append(self._wcol(index, n))
                    touched[num[len(num) - 1]] = True
                    acc_uses[num[len(num) - 1]] += 1
                for n in a.den:
                    den.append(self._wcol(index, n))
                    touched[den[len(den) - 1]] = True
                    acc_uses[den[len(den) - 1]] += 1
                var z_col = w + it[1] * p.e
                if a.kind == KIND_PERM:
                    f.accumulator(k, z_col, num, den)
                elif a.kind == KIND_LOOKUP:
                    if len(tables[a.table]) // self.widths[a.table] > p.N():
                        raise Error("lookup table has more rows than the grid: " + a.name)
                    for s in den:
                        sorted_of[s] = k
                    f.lookup(k, z_col, num, den, a.table)
                else:
                    raise Error("accumulator kind is KIND_PERM or KIND_LOOKUP")
        var fam_index = len(self.order)
        for i in range(w):
            if kinds[i] == BIT:                          # Booleanity: c^2 - c
                f.add(fam_index, 1, i, col_b=i)
                f.add(fam_index, 126, i)
                touched[i] = True
                fam_index += 1
        for j in range(len(limbs)):                      # range: the limb is in [64]
            var s = len(self.cols) + j
            f.lookup(fam_index, w + (len(self.accs) + j) * p.e, [limbs[j]], [s], limb_table)
            touched[limbs[j]] = True
            touched[s] = True
            acc_uses[limbs[j]] += 1
            acc_uses[s] += 1
            sorted_of[s] = fam_index
            fam_index += 1
        var pubs = self.pubs.copy()
        for i in range(len(self.pub_names)):
            if p.h2() % get_u16(pubs, i * PUB) != 0:
                raise Error("public column period: m divides h2: " + self.pub_names[i])
        var res = self.res.copy()
        for i in range(len(self.res_names)):
            var c = self._wcol(index, self.res_names[i])
            set_u16(res, i * RES, c)
            if get_u16(res, i * RES + 4) == 0:
                set_u16(res, i * RES + 4, p.h1())
            touched[c] = True
            read[c] = True
        for i in range(w):
            if not touched[i]:
                raise Error("column is read by nothing and constrained by nothing: " + names[i])
            if sorted_of[i] >= 0 and (read[i] or acc_uses[i] != 1):   # the sort overwrites it: one lookup's, read by nothing
                raise Error("a sorted column belongs to one lookup and is read by nothing else: " + names[i])
        var wires = List[UInt8]()
        var sigma = List[UInt8]()
        var pubf = List[UInt8]()
        if len(self.slots) > 0:
            comptime h2 = p.h2()
            var ns = len(self.slots)
            var nf = len(self.factors)
            if (ns + (nf + h2 - 1) // h2) * h2 > F2_ORDER:
                raise Error("wiring slots and public factors exceed the cosets of H2 in F2*")
            for i in range(ns):
                if i % 2 == 0:
                    wires.extend(wire_record(fam_index, w + self.slots[i] * p.e, -1 if i + 1 == ns else w + self.slots[i + 1] * p.e))
                    fam_index += 1
            var kappa = f2_primitive()
            var omega2 = Domains.__init__[p]().omega2
            var ids = List[F2]()                      # node s h2 + j is slot s on chain j; node ns h2 + i the factor i
            for s in range(ns + (nf + h2 - 1) // h2):
                var k = ext_pow[1](kappa, s)
                for j in range(h2):
                    ids.append(ext_mul[1](k, ext_pow[1](omega2, j)))
            var parent = List[Int](length=len(ids), fill=0)
            for n in range(len(ids)):
                parent[n] = n
            var pairs = List[Tuple[Int, Int]]()
            for ed in self.edges:
                if ed[1] >= h2 or ed[3] >= h2:
                    raise Error("wire chain index is outside the grid")
                pairs.append((ed[0] * h2 + ed[1], ed[2] * h2 + ed[3]))
            for i in range(nf):
                if self.factors[i][3] >= h2:
                    raise Error("public factor chain index is outside the grid: " + self.factors[i][0])
                pairs.append((ns * h2 + i, self.factors[i][2] * h2 + self.factors[i][3]))
            for pr in pairs:
                var a = pr[0]
                var b = pr[1]
                while parent[a] != a:
                    a = parent[a]
                while parent[b] != b:
                    b = parent[b]
                parent[a] = b
            var members = Dict[Int, List[Int]]()      # sigma maps every node to the next of its cycle
            for n in range(len(ids)):
                var r = n
                while parent[r] != r:
                    r = parent[r]
                if r not in members:
                    members[r] = List[Int]()
                members[r].append(n)
            var succ = List[Int](length=len(ids), fill=0)
            for e in members.items():
                for i in range(len(e.value)):
                    succ[e.value[i]] = e.value[(i + 1) % len(e.value)]
            for n in range(ns * h2):
                sigma.extend([ids[succ[n]][0], ids[succ[n]][1]])
            for i in range(nf):
                pubf.extend(public_factor_record(self.factors[i][1], ids[ns * h2 + i], ids[succ[ns * h2 + i]]))
        var points = shift_points(f.bytes, res, len(f.accs) > 0)
        var shape = Shape.__init__[p](w, f.bytes, f.accs, tables, pubs, res, points, self.chals, f.ends, wires, sigma, pubf)
        var layout = Layout(names, index, kinds, groups, f.accs, tables, pubs, res)
        return Compiled(shape^, f.bytes.copy(), layout^)


# ---- trace helpers ----

def _row_key(bytes: List[UInt8], off: Int, width: Int) -> Int:
    """A record of at most ACC_W_MAX canonical bytes packed 7 bits each; -1 if a byte is not canonical."""
    var key = 0
    for j in range(width):
        if Int(bytes[off + j]) >= 127:
            return -1
        key = key * 128 + Int(bytes[off + j])
    return key


def pad_trace[p: Params](layout: Layout, mut trace: List[UInt8], group: String, live_rows: Int) raises:
    """Fill rows [live_rows, N) of the group's columns with the neutral value: table row i mod K for a lookup
    record column (its record position picks the byte), zero otherwise. Z blocks and sorted columns are the
    prover's. Table filler covers the table when the idle rows are at least K (the dummy rule). Neutrality of
    the frontend's own families under this filler (statement-layer decision 1) is the frontend's to arrange:
    a family tying a lookup record column to a zero-padded column does not hold on the idle rows."""
    comptime N = p.N()
    if len(trace) != layout.columns_w() * N or live_rows < 0 or live_rows > N:
        raise Error("trace has the wrong size or live_rows is outside [0, N]")
    if live_rows % p.h1() != 0:
        raise Error("live_rows is whole chains (a multiple of h1): cyclic reads wrap inside a chain")
    var seen = False
    for c in range(layout.columns_w()):
        if layout.groups[c] != group:
            continue
        seen = True
        var slot = layout.slots[c]
        if slot[0] == -2:
            continue
        if slot[0] == -3:
            raise Error("column is a record of two lookups; no single table row pads it: " + layout.names[c])
        for i in range(live_rows, N):
            if slot[0] < 0:
                trace[c * N + i] = 0
            else:
                var rows = len(layout.tables[slot[0]]) // slot[2]
                trace[c * N + i] = layout.tables[slot[0]][(i % rows) * slot[2] + slot[1]]
    if not seen:
        raise Error("unknown group " + group)


def advice[p: Params](layout: Layout, trace: List[UInt8]) raises -> List[UInt8]:
    """The advice index list the sort reads (sort.mojo): per lookup descriptor, per row, the table row the
    record equals. Every table row must occur (the dummy rule: the boundary constant is the whole table's),
    else the first missing row is named here instead of failing at the verifier."""
    comptime N = p.N()
    if len(trace) != layout.columns_w() * N:
        raise Error("trace has the wrong size")
    var out = List[UInt8]()
    var rec = List[UInt8](length=ACC_W_MAX, fill=0)
    for k in range(len(layout.accs) // ACC):
        if Int(layout.accs[k * ACC + 38]) != KIND_LOOKUP:
            continue
        var width = get_u16(layout.accs, k * ACC + 2)
        var t = layout.tables[Int(layout.accs[k * ACC + 39])].copy()
        var rows = len(t) // width
        var at = Dict[Int, Int]()
        for r in range(rows):
            at[_row_key(t, r * width, width)] = r
        var used = List[Bool](length=rows, fill=False)
        for i in range(N):
            var found = -1
            for j in range(width):
                rec[j] = trace[get_u16(layout.accs, k * ACC + 6 + 2 * j) * N + i]
            var key = _row_key(rec, 0, width)
            if key in at:
                found = at[key]
            if found < 0:
                raise Error("lookup record at row " + String(i) + " is not in its table")
            used[found] = True
            append_u32(out, found)
        for r in range(rows):
            if not used[r]:
                raise Error("lookup table row " + String(r) + " occurs in no record (pad with the table)")
    return out^


def chain_values[p: Params](layout: Layout, trace: List[UInt8], i: Int) raises -> List[UInt8]:
    """The h1 values of restriction i's column on its chain, read from the trace: the prover's public inputs
    for that restriction (the verifier gets them, not the trace)."""
    comptime h1 = p.h1()
    if len(trace) != layout.columns_w() * p.N():
        raise Error("trace has the wrong size")
    var c = get_u16(layout.restrictions, i * RES)
    var chain = p.h2() - 1 if get_u16(layout.restrictions, i * RES + 2) == FIX_E else 0
    var line = List[UInt8](capacity=h1)
    for x1 in range(h1):
        line.append(trace[c * p.N() + chain * h1 + x1])
    return line^


def restriction_line[p: Params](layout: Layout, i: Int, vals: List[UInt8]) raises -> List[UInt8]:
    """The `count` F2 coefficients of restriction i from its h1 chain values: the interpolant, rejected if a
    coefficient past `count` is nonzero. Both sides call it (`chain_values` on the prover's)."""
    comptime h1 = p.h1()
    if len(vals) != h1:
        raise Error("restriction line takes h1 values")
    var count = get_u16(layout.restrictions, i * RES + 4)
    var d = Domains.__init__[p]()
    var coeffs = interpolate_line(vals, d.omega1, h1)
    for t in range(count * 2, h1 * 2):
        if coeffs[t] != 0:
            raise Error("restriction line has degree at least the coefficient count")
    coeffs.resize(count * 2, 0)
    return coeffs^


def interpolate_line(vals: List[UInt8], omega: F2, h: Int) raises -> List[UInt8]:
    """h F values on <omega> -> h F2 monomial coefficients (2 bytes each): c_k = h^-1 sum_x v(x) omega^(-k x)."""
    var inv_h = f_pow(SIMD[DType.uint8, 1](UInt8(h % 127)), 125)
    var out = List[UInt8](capacity=h * 2)
    for k in range(h):
        var acc = F2(0)
        for x in range(h):
            acc = f_add(acc, ext_mul[1](F2(vals[x], 0), ext_pow[1](omega, (h - (k * x) % h) % h)))
        var c = ext_mul[1](acc, F2(inv_h[0], 0))
        out.append(c[0])
        out.append(c[1])
    return out^

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
from caracal7.relations.ir import Families, standard_chals, shift_points, chal_count, CHAL_ADD, CHAL_MUL, CHAL_ONE, FIX_ONE, FIX_E, PUB, RES, ACC, KIND_PERM, KIND_LOOKUP
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
    var num: List[String]
    var den: List[String]
    var table: Int


struct Layout(Copyable, Movable):
    """Prover-side names: W columns in index order with their kind and group, plus the descriptors, tables,
    public specs, and restriction records the trace helpers read. Not part of the artifact."""
    var names: List[String]
    var kinds: List[Int]
    var groups: List[String]
    var accs: List[UInt8]
    var tables: List[List[UInt8]]
    var publics: List[UInt8]
    var restrictions: List[UInt8]

    def __init__(out self, names: List[String], kinds: List[Int], groups: List[String], accs: List[UInt8],
                 tables: List[List[UInt8]], publics: List[UInt8], restrictions: List[UInt8]):
        self.names = names.copy()
        self.kinds = kinds.copy()
        self.groups = groups.copy()
        self.accs = accs.copy()
        self.tables = tables.copy()
        self.publics = publics.copy()
        self.restrictions = restrictions.copy()

    def col(self, name: String) raises -> Int:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        raise Error("unknown column " + name)

    def columns_w(self) -> Int:
        return len(self.names)

    def _record_slot(self, c: Int) -> Tuple[Int, Int, Int]:
        """(table id, record position, record width) if W column c is a lookup record column, else (-1, 0, 0);
        (-2, 0, 0) if it is a sorted column (the prover's)."""
        for k in range(len(self.accs) // ACC):
            if Int(self.accs[k * ACC + 38]) != KIND_LOOKUP:
                continue
            var width = get_u16(self.accs, k * ACC + 2)
            for j in range(width):
                if get_u16(self.accs, k * ACC + 6 + 2 * j) == c:
                    return (Int(self.accs[k * ACC + 39]), j, width)
                if get_u16(self.accs, k * ACC + 22 + 2 * j) == c:
                    return (-2, 0, 0)
        return (-1, 0, 0)


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
    var kinds: List[Int]
    var col_group: List[String]
    var fams: List[_Family]
    var accs: List[_Acc]
    var order: List[Tuple[Int, Int]]    # (0 family i | 1 accumulator i) in call order: the position is the family index
    var pubs: List[UInt8]               # PUB records
    var pub_names: List[String]
    var res: List[UInt8]                # RES records, column resolved at compile
    var res_names: List[String]
    var tables: List[List[UInt8]]
    var chals: List[UInt8]

    def __init__(out self):
        self.cols = List[String]()
        self.kinds = List[Int]()
        self.col_group = List[String]()
        self.fams = List[_Family]()
        self.accs = List[_Acc]()
        self.order = List[Tuple[Int, Int]]()
        self.pubs = List[UInt8]()
        self.pub_names = List[String]()
        self.res = List[UInt8]()
        self.res_names = List[String]()
        self.tables = List[List[UInt8]]()
        self.chals = standard_chals()

    def _fresh(self, name: String) raises:
        for n in self.cols:
            if n == name:
                raise Error("name in use: " + name)
        for a in self.accs:
            if a.name == name:
                raise Error("name in use: " + name)
        for n in self.pub_names:
            if n == name:
                raise Error("name in use: " + name)

    def col(mut self, name: String, kind: Int = BYTE, group: String = "") raises:
        """A witness column; the index is the declaration order."""
        self._fresh(name)
        if kind < BIT or kind > BYTE:
            raise Error("column kind is BIT, LIMB6, or BYTE")
        self.cols.append(name)
        self.kinds.append(kind)
        self.col_group.append(group)

    def table(mut self, rows: List[UInt8], width: Int) raises -> Int:
        """Register a lookup table (canonical bytes, `width` per row); returns its id. Rows are distinct (the
        advice picks a record's row by value) and two consecutive rows differ (ir.lookup_constant)."""
        if width < 1 or len(rows) == 0 or len(rows) % width != 0:
            raise Error("lookup table is whole rows")
        var k = len(rows) // width
        var has_break = False
        for i in range(k):
            for j in range(i + 1, k):
                if rows[i * width : (i + 1) * width] == rows[j * width : (j + 1) * width]:
                    raise Error("lookup table rows must be distinct")
            if i + 1 < k:
                has_break = True
        if not has_break:
            raise Error("lookup table needs two distinct entries")
        self.tables.append(rows.copy())
        return len(self.tables) - 1

    def acc(mut self, name: String, kind: Int, num: List[String], den: List[String], table: Int = -1) raises:
        """A Z block: KIND_PERM (num against den) or KIND_LOOKUP (records num against `table`, den the sorted
        columns the prover fills). Its family index is its position among family and acc calls."""
        self._fresh(name)
        if kind == KIND_LOOKUP and (table < 0 or table >= len(self.tables)):
            raise Error("lookup accumulator needs a registered table")
        self.order.append((1, len(self.accs)))
        self.accs.append(_Acc(name, kind, num.copy(), den.copy(), table))

    def pub(mut self, name: String, m: Int, d2: Int = 0) raises:
        """A public column (docs/public-columns.md): block (d2, h1), row j at X2^(m j); d2 = 0 means h2 / m, the
        full block of a column periodic along axis 2 with period h2 / m. Indexed after W and Z."""
        self._fresh(name)
        if m < 1 or d2 < 0 or m > 65535 or d2 > 65535:
            raise Error("public column needs m >= 1 and u16 dimensions")
        var b = List[UInt8](length=PUB, fill=0)
        set_u16(b, 0, m)
        set_u16(b, 2, d2)
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

    def _wcol(self, names: List[String], name: String) raises -> Int:
        for i in range(len(names)):
            if names[i] == name:
                return i
        raise Error("unknown witness column " + name)

    def _resolve[p: Params](self, names: List[String], r: Read, pub_at: Int, mut touched: List[Bool]) raises -> Int:
        """A read to a column index: W by name, else a public column past W and Z."""
        if r.k1 < 0 or r.k1 >= p.h1() or r.k2 < 0 or r.k2 > 1:
            raise Error("read shift: k1 in [0, h1), k2 in {0, 1}")
        for i in range(len(names)):
            if names[i] == r.col:
                touched[i] = True
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
                names.append(self.cols[i] + ".sorted")
                kinds.append(BYTE)
                groups.append(self.col_group[i])
        var w = len(names)
        var pub_at = w + (len(self.accs) + len(limbs)) * p.e
        var touched = List[Bool](length=w, fill=False)
        var f = Families()
        for k in range(len(self.order)):
            var it = self.order[k]
            if it[0] == 0:
                var fam = self.fams[it[1]].copy()
                for t in fam.terms:
                    var quadratic = Bool(t.b)
                    var ca = self._resolve[p](names, t.a, pub_at, touched)
                    var cb = -1
                    var k1b = 0
                    var k2b = 0
                    if quadratic:
                        cb = self._resolve[p](names, t.b.value(), pub_at, touched)
                        k1b = t.b.value().k1
                        k2b = t.b.value().k2
                    if (t.a.k2 == 1 or k2b == 1) and (fam.gate != GATE_2 or quadratic):
                        raise Error("a next-chain read (k2 = 1) needs a linear family with the axis-2 gate: " + fam.name)
                    f.add(k, ((t.coef % 127) + 127) % 127, ca, k1_a=t.a.k1, k2_a=t.a.k2, col_b=cb, k1_b=k1b, k2_b=k2b,
                          mult=fam.gate, chal=0 if t.chal < 0 else t.chal + 1, basis=t.basis, basis2=t.basis2)
            else:
                var a = self.accs[it[1]].copy()
                var num = List[Int]()
                var den = List[Int]()
                for n in a.num:
                    num.append(self._wcol(names, n))
                    touched[num[len(num) - 1]] = True
                for n in a.den:
                    den.append(self._wcol(names, n))
                    touched[den[len(den) - 1]] = True
                var z_col = w + it[1] * p.e
                if a.kind == KIND_PERM:
                    f.accumulator(k, z_col, num, den)
                elif a.kind == KIND_LOOKUP:
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
            fam_index += 1
        var pubs = self.pubs.copy()
        for i in range(len(self.pub_names)):
            if get_u16(pubs, i * PUB + 2) == 0:
                set_u16(pubs, i * PUB + 2, p.h2() // get_u16(pubs, i * PUB))
        var res = self.res.copy()
        for i in range(len(self.res_names)):
            var c = self._wcol(names, self.res_names[i])
            set_u16(res, i * RES, c)
            if get_u16(res, i * RES + 4) == 0:
                set_u16(res, i * RES + 4, p.h1())
            touched[c] = True
        for i in range(w):
            if not touched[i]:
                raise Error("column is read by nothing and constrained by nothing: " + names[i])
        var points = shift_points(f.bytes, res, len(f.accs) > 0)
        var shape = Shape.__init__[p](w, f.bytes, f.accs, tables, pubs, res, points, self.chals)
        var layout = Layout(names, kinds, groups, f.accs, tables, pubs, res)
        return Compiled(shape^, f.bytes.copy(), layout^)


# ---- trace helpers ----

def pad_trace[p: Params](layout: Layout, mut trace: List[UInt8], group: String, live_rows: Int) raises:
    """Fill rows [live_rows, N) of the group's columns with the neutral value: table row i mod K for a lookup
    record column (its record position picks the byte), zero otherwise. Z blocks and sorted columns are the
    prover's. Table filler covers the table when the idle rows are at least K (the dummy rule). Neutrality of
    the frontend's own families under this filler (statement-layer decision 1) is the frontend's to arrange:
    a family tying a lookup record column to a zero-padded column does not hold on the idle rows."""
    comptime N = p.N()
    if len(trace) != layout.columns_w() * N or live_rows < 0 or live_rows > N:
        raise Error("trace has the wrong size or live_rows is outside [0, N]")
    var seen = False
    for c in range(layout.columns_w()):
        if layout.groups[c] != group:
            continue
        seen = True
        var slot = layout._record_slot(c)
        if slot[0] == -2:
            continue
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
    record equals. ponytail: linear scan of the table per row; index the table when N K w matters."""
    comptime N = p.N()
    if len(trace) != layout.columns_w() * N:
        raise Error("trace has the wrong size")
    var out = List[UInt8]()
    for k in range(len(layout.accs) // ACC):
        if Int(layout.accs[k * ACC + 38]) != KIND_LOOKUP:
            continue
        var width = get_u16(layout.accs, k * ACC + 2)
        var t = layout.tables[Int(layout.accs[k * ACC + 39])].copy()
        var rows = len(t) // width
        for i in range(N):
            var found = -1
            for r in range(rows):
                var same = True
                for j in range(width):
                    if trace[get_u16(layout.accs, k * ACC + 6 + 2 * j) * N + i] != t[r * width + j]:
                        same = False
                        break
                if same:
                    found = r
                    break
            if found < 0:
                raise Error("lookup record at row " + String(i) + " is not in its table")
            append_u32(out, found)
    return out^


def public_block[p: Params](vals: List[UInt8], m: Int, d2: Int) raises -> List[UInt8]:
    """The (d2, h1, 2) block of a public column from its N values (x2, x1) on H: the interpolant's rows
    k2 = m j. The other rows must be zero (the column is periodic along axis 2), else the block is rejected."""
    comptime h1 = p.h1()
    comptime h2 = p.h2()
    var full = interpolate_grid[p](vals)
    var block = List[UInt8](capacity=d2 * h1 * 2)
    for k2 in range(h2):
        var kept = k2 % m == 0 and k2 // m < d2
        for t in range(h1 * 2):
            if kept:
                block.append(full[(k2 * h1) * 2 + t])
            elif full[(k2 * h1) * 2 + t] != 0:
                raise Error("public column is not periodic with period h2 / m")
    return block^


def restriction_line[p: Params](layout: Layout, trace: List[UInt8], i: Int) raises -> List[UInt8]:
    """The `count` F2 coefficients of restriction i from the trace: the interpolant of the column on its chain,
    rejected if a coefficient past `count` is nonzero."""
    comptime h1 = p.h1()
    var c = get_u16(layout.restrictions, i * RES)
    var chain = p.h2() - 1 if get_u16(layout.restrictions, i * RES + 2) == FIX_E else 0
    var count = get_u16(layout.restrictions, i * RES + 4)
    var line = List[UInt8](capacity=h1)
    for x1 in range(h1):
        line.append(trace[c * p.N() + chain * h1 + x1])
    var d = Domains.__init__[p]()
    var coeffs = interpolate_line(line, d.omega1, h1)
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


def interpolate_grid[p: Params](vals: List[UInt8]) raises -> List[UInt8]:
    """N F values (x2, x1) on H -> (k2, k1, 2) F2 coefficients, axis 1 then axis 2 (host, O(N (h1 + h2)))."""
    comptime h1 = p.h1()
    comptime h2 = p.h2()
    var d = Domains.__init__[p]()
    var inv_h2 = f_pow(SIMD[DType.uint8, 1](UInt8(h2 % 127)), 125)
    var ctmp = List[UInt8](length=p.N() * 2, fill=0)       # (x2, k1, 2)
    for x2 in range(h2):
        var line = List[UInt8](capacity=h1)
        for x1 in range(h1):
            line.append(vals[x2 * h1 + x1])
        var c = interpolate_line(line, d.omega1, h1)
        for t in range(h1 * 2):
            ctmp[x2 * h1 * 2 + t] = c[t]
    var out = List[UInt8](length=p.N() * 2, fill=0)         # (k2, k1, 2)
    for k1 in range(h1):
        for k2 in range(h2):
            var acc = F2(0)
            for x2 in range(h2):
                acc = f_add(acc, ext_mul[1](F2(ctmp[(x2 * h1 + k1) * 2], ctmp[(x2 * h1 + k1) * 2 + 1]),
                                            ext_pow[1](d.omega2, (h2 - (k2 * x2) % h2) % h2)))
            var c = ext_mul[1](acc, F2(inv_h2[0], 0))
            out[(k2 * h1 + k1) * 2] = c[0]
            out[(k2 * h1 + k1) * 2 + 1] = c[1]
    return out^

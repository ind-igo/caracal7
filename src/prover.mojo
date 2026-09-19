"""Host orchestration (docs/design.md section 6): plan the arena once, then enqueue every stage of
spec section 10 in order on one stream. One synchronize at the end reads the proof bytes.

`prove` reads as the protocol: the transcript steps are visible there in order, except the three that
belong to a repeated unit and live in it (`_commit` absorbs a root, `_tail_level` absorbs a level's root
and rounds and squeezes its scalars, `_open_previous` squeezes the positions). Each stage between them is a helper that takes its own layout group (accumulators, small
grid, residual grid, openings, one tail level) and computes its own offsets. The layout groups are
plain structs of arena offsets built from one Bump; the stage modules own the groups whose internal
packing they read (AccLayout in accumulate.mojo, SmallGridLayout in smallgrid.mojo).
"""

from std.time import perf_counter_ns
from std.math import ceildiv
from max.gpu.host import DeviceContext, HostBuffer

from std.memory import unsafe_memcpy
from core.params import Params
from core.field import F2, ext_pow
from core.arena import Arena, Bump, ST_LOAD, ST_SORT, ST_W, ST_ACC, ST_Z, ST_SG, ST_LDE, ST_RES, ST_QUO, ST_Q, ST_OPEN, ST_FOLD, ST_RUN0, ST_TAIL, ST_END
from core.tables import F2_ORDER, Domains, TableLayout, RsDomain, RsTables, build_tables, build_rs_tables, f2_primitive
from pcs.encode import EncLayout, encode, idft2
from core.transcript import TranscriptLayout, reset, absorb, squeeze_elements, squeeze_positions, grind
from core.transcript import DS_PREFIX, DS_TREE_W, DS_TREE_Z, DS_TREE_Q, DS_OPENINGS, DS_TAIL_ROOT, DS_TAIL_ROUND, DS_CLEAR
from proof import Shape, ProofWriter, TailLevel, VERSION, prefix_bytes
from core.hash import Hash
from pcs import merkle, query_gather, root_offset, tree_nodes, multiproof_region, build_queries, open, open_splits, fold, table_len, factor_len, TAIL_F4
from pcs import DOM_BYTES, ROUND_ROWS, domain_bytes, tail_encode, points, running0, tail_materialize, tail_round, tail_fold, power_table_len
from relations import ENTRY, POINT, ACC, END, WIRE, CHAL, KIND_LOOKUP, KIND_HORNER, acc_kind, tile_values
from relations.statement import Compiled
from relations import AccLayout, accumulate, horner, wiring, derive_chals, counting_sort, merge_tables
from relations import lde, residual, quotient, quotient_elems, k_values_to_trace
from relations import SmallGridLayout, small_grid_accumulator, small_grid_wiring, small_grid_end, small_grid_values
from core.bytes import Buf, get_u16
from core.backend import BACKEND

comptime PREFIX_MAX = 1 << 16       # arena bytes for the transcript prefix (public inputs included)


# ---- layout groups: arena offsets by the stage that uses them ----

struct CommitLayout(TrivialRegisterPassable):
    """One committed tree (W, Z or Q): the encoder's buffers from trace to code, and the Merkle nodes."""
    var enc: EncLayout
    var tree: Int           # (node, 32), level 0 first, tree_nodes(L0) nodes
    var row: Int            # bytes per codeword row: 4 n_cw per column

    def __init__[p: Params, H: Hash](out self, mut bump: Bump, columns: Int, trace_from: Int, trace_to: Int, stage: Int, coeff_to: Int, coeff_from: Int = -1):
        self.enc = EncLayout.__init__[p](bump, columns, trace_from, trace_to, stage, coeff_to, coeff_from)
        self.tree = bump.alloc(tree_nodes(p.L()) * H.DIGEST, stage)
        self.row = 4 * p.n_cw() * columns


struct SortLayout(TrivialRegisterPassable):
    """The lookup sorts (sort.mojo): advice indices per lookup descriptor, bins and the cursor of the largest table."""
    var idx: Int            # (lookup, row, u32)
    var bins: Int           # (max K + 1, u32)
    var cursor: Int         # (max K, u32)

    def __init__[p: Params](out self, mut bump: Bump, lookups: Int, max_rows: Int):
        self.idx = bump.alloc(lookups * p.N() * 4)           # loaded once, must survive repeated proves
        self.bins = bump.alloc((max_rows + 1) * 4, ST_SORT, ST_SORT)
        self.cursor = bump.alloc(max_rows * 4, ST_SORT, ST_SORT)


struct LdeLayout(TrivialRegisterPassable):
    """The residual grid: the public columns' values and coefficients, the LDE of every column
    (witness, accumulator, then public), the residual and the quotient's interpolation scratch."""
    var pub_vals: Int       # (public column, x2, x1)       public column values on H (load_public)
    var pub_coeff: Int      # (public column, k2, k1, 2)    their coefficients (idft2 in load_public), the LDE input
    var ltmp: Int           # (column, G2, G1, 2)           LDE after axis 1, then the axis-2 scratch
    var lde: Int            # (column, G2, G1, 2)
    var residual: Int       # (G2, G1, e)
    var quotient: Int       # quotient_elems x e            Q1, Q2 interpolation scratch (residual.mojo)
    var block: Int          # bytes of one column's LDE

    def __init__[p: Params](out self, mut bump: Bump, shape: Shape):
        comptime N = p.N()
        self.block = 4 * N * 2
        self.pub_vals = bump.alloc(shape.columns_p * N)   # loaded once too: the selected ingest terms read it in every prove's Z stage
        self.pub_coeff = bump.alloc(shape.columns_p * N * 2)   # loaded once, must survive repeated proves
        self.ltmp = bump.alloc(max(shape.columns_w, max(shape.columns_z, shape.columns_p)) * self.block, ST_LOAD, ST_LDE)   # load_public's scratch too
        self.lde = bump.alloc((shape.columns_w + shape.columns_z + shape.columns_p) * self.block, ST_LDE, ST_RES)
        self.residual = bump.alloc(4 * N * p.e, ST_RES, ST_QUO)
        self.quotient = bump.alloc(quotient_elems[p]() * p.e, ST_QUO, ST_QUO)

    def lde_at(self, column: Int) -> Int:
        return self.lde + column * self.block


struct OpenLayout(TrivialRegisterPassable):
    """The openings at the P points (open.mojo) and the fold to the level-2 message."""
    var w_tab: Int          # (P, table_len, e)     per-point powers and Lagrange factors, then (P, factor_len, e)
    var w_z: Int            # (slot, P, e)          evaluation queries
    var openings: Int       # (P, column, e)
    var open_partial: Int   # (splits, column, P, e) split-K partials of `open`
    var fold_y: Int         # (slot, e)             y = sum beta_c stored(c), the level-2 message
    var running0: Int       # (slot, e)             sum_p gamma_p w_{z_p}, the level-2 running query

    def __init__[p: Params](out self, mut bump: Bump, shape: Shape):
        comptime N = p.N()
        var widest = max(shape.columns_w, max(shape.columns_z, shape.columns_q))
        self.w_tab = bump.alloc(shape.points * (table_len[p]() + factor_len[p]()) * p.e, ST_OPEN, ST_OPEN)
        self.w_z = bump.alloc(shape.points * N * p.e, ST_OPEN, ST_RUN0)
        self.openings = bump.alloc(shape.points * shape.columns() * p.e, ST_OPEN)
        self.open_partial = bump.alloc(shape.points * open_splits[p]() * widest * p.e, ST_OPEN, ST_OPEN)
        self.fold_y = bump.alloc(N * p.e, ST_FOLD, ST_TAIL)             # tail level 0 folds it (or it is the clear vector)
        self.running0 = bump.alloc(N * p.e, ST_RUN0, ST_TAIL)


struct ChalLayout(TrivialRegisterPassable):
    """The challenges, one region each so nothing is overwritten before its consumer runs."""
    var stage1: Int         # (chal_count, e)       beta, delta, gamma, then the derivation table's rows
    var table: Int          # (rows, CHAL)          the derivation table
    var wchal: Int          # (2, e)                beta_w, gamma_w of the wiring copy constraint
    var alpha: Int          # (1, e)
    var z: Int              # (2, e)
    var beta_gamma: Int     # (columns + P, e)      beta per column, gamma per point
    var batch: Int          # (max v_count + 1, e)  tail batching scalars
    var r: Int              # (3, e)                sumcheck round challenges of the current level

    def __init__[p: Params](out self, mut bump: Bump, shape: Shape, max_v: Int):
        self.stage1 = bump.alloc(shape.chal_count() * p.e)
        self.table = bump.alloc(len(shape.chals))
        self.wchal = bump.alloc(2 * p.e)
        self.alpha = bump.alloc(p.e)
        self.z = bump.alloc(2 * p.e)
        self.beta_gamma = bump.alloc((shape.columns() + shape.points) * p.e)
        self.batch = bump.alloc((max_v + 1) * p.e)
        self.r = bump.alloc(3 * p.e)


struct QueryLayout(TrivialRegisterPassable):
    """Opening one committed level: the sampled positions, their leaf points and power tables, the
    sumcheck partials, the level-1 domain, and the staging region of one multiproof."""
    var positions: Int      # (max queries, u32)    S of the level being opened
    var pts: Int            # (max queries, 4)      leaf points of the opened positions
    var ptab: Int           # (queries, table, 4)   level-1 powers of the leaf points
    var partial: Int        # (ROUND_ROWS, 3, e) sumcheck partial sums
    var dom1: Int           # DOM_BYTES             the level-1 domain
    var stage: Int          # gathered rows and siblings of one multiproof, staged to the host in stream order
    var found: Int          # u32                   the nonce found
    var nonce: Int          # 8 bytes               the nonce as the proof stages it

    def __init__[p: Params](out self, mut bump: Bump, max_queries: Int, stage_bytes: Int):
        self.positions = bump.alloc(max_queries * 4)
        self.pts = bump.alloc(max_queries * 4)
        self.ptab = bump.alloc(max_queries * power_table_len[p]() * 4)
        self.partial = bump.alloc(ROUND_ROWS * 3 * p.e)
        self.dom1 = bump.alloc(DOM_BYTES)
        self.stage = bump.alloc(stage_bytes)
        self.found = bump.alloc(4)
        self.nonce = bump.alloc(8)


struct TailLayout(TrivialRegisterPassable):
    """Arena offsets of one committed tail level (spec 9.3), all E-valued."""
    var y: Int          # (rows, e)          the folded message y_{l+1} = Mat(y_l) r_bar
    var running: Int    # (rows, e)          the folded query, the next level's running claim
    var etmp: Int       # (L0, 32, 4)        encoder scratch
    var code: Int       # (s, 8, e)          leaf-major codeword rows
    var tree: Int       # (node, 32)
    var w_tilde: Int    # (slot, e)          batched query on y_l
    var rounds: Int     # (3, 3, e)          sumcheck messages
    var rs: RsTables    # this level's RS domain tables
    var dom: Int        # DOM_BYTES          this level's domain (tail.domain_bytes)

    def __init__[p: Params, H: Hash](out self, mut bump: Bump, lvl: TailLevel, i: Int):
        var L0 = lvl.L // lvl.cosets
        var st = ST_TAIL + i                                # this level's stage; the next level (or the last open) reads its code
        self.y = bump.alloc(lvl.rows * p.e, st)
        self.running = bump.alloc(lvl.rows * p.e, st)
        self.etmp = bump.alloc(L0 * TAIL_F4 * lvl.codewords * 4, st, st)
        self.code = bump.alloc(lvl.L * 8 * p.e * lvl.codewords, st, st + 1)
        self.tree = bump.alloc(tree_nodes(lvl.L) * H.DIGEST, st, st + 1)
        self.w_tilde = bump.alloc(lvl.length * p.e, st, st)  # also the encoder's packed scratch of a split level (free until materialize)
        self.rounds = bump.alloc(9 * p.e, st)
        self.rs = RsTables(bump.alloc(0, ST_LOAD, st), L0, lvl.cosets, lvl.rows // lvl.codewords)   # uploaded at setup
        _ = bump.alloc(self.rs.bytes, ST_LOAD, st)
        self.dom = bump.alloc(DOM_BYTES)                    # uploaded at setup, read by the next level


struct ProverLayout:
    """Every buffer of design section 3 as an arena offset, grouped by stage. Built once per (Params, Shape)."""
    var tables: TableLayout
    var transcript: TranscriptLayout
    var w: CommitLayout             # witness tree
    var z: CommitLayout             # accumulator tree (the Z stage writes its trace)
    var q: CommitLayout             # quotient tree (the quotient writes its trace)
    var families: Int               # (entry, ENTRY)        the family table, kappa folded in on device
    var families_g: Int             # (entry, ENTRY)        the residual's merged table (residual.merge_tables)
    var merge: Int                  # (entry, 2) u16 (start, count), then (entry) u16 indices into `families`
    var accs: Int                   # (accumulator, ACC)    accumulator descriptors
    var shifts: Int                 # (P, POINT)            opening points as (dj1, dj2) on G
    var wires: Int                  # (wiring product, WIRE)
    var sigma: Int                  # (slot, x2, 2)         the wiring permutation
    var acc: AccLayout
    var sort: SortLayout
    var sg: SmallGridLayout
    var lde: LdeLayout
    var open: OpenLayout
    var chal: ChalLayout
    var query: QueryLayout
    var prefix: Int                 # transcript prefix bytes (PREFIX_MAX)
    var tail: List[TailLayout]
    var bytes: Int

    def __init__[p: Params, H: Hash](out self, shape: Shape, keep: Bool = False) raises:
        """Two passes over one construction: the first records every region's stage lifetime, `Bump.plan`
        packs regions whose lifetimes never meet onto the same bytes, the second hands out the packed
        offsets. `keep` skips the packing so every region survives the proof (tests read scratch after it)."""
        var bump = Bump()
        self = Self.__init__[p, H](bump, shape)
        if not keep:
            bump.plan()
            self = Self.__init__[p, H](bump, shape)

    def __init__[p: Params, H: Hash](out self, mut bump: Bump, shape: Shape) raises:
        """One construction pass; every alloc's stage lifetime is recorded (or replayed) by `bump`."""
        self.tables = TableLayout.__init__[p](bump.alloc(0))
        _ = bump.alloc(self.tables.bytes)
        self.transcript = TranscriptLayout(bump)
        # W's trace is loaded once and must survive repeated proves; the LDE reads W's and Z's coefficients; Q's trace is the quotient
        self.w = CommitLayout.__init__[p, H](bump, shape.columns_w, ST_LOAD, ST_END, ST_W, ST_LDE)
        self.z = CommitLayout.__init__[p, H](bump, shape.columns_z, ST_ACC, ST_Z, ST_Z, ST_LDE)
        self.q = CommitLayout.__init__[p, H](bump, shape.columns_q, ST_QUO, ST_Q, ST_Q, ST_Q, coeff_from=ST_QUO)
        self.families = bump.alloc(shape.entries * ENTRY)
        self.families_g = bump.alloc(shape.entries * ENTRY)
        self.merge = bump.alloc(shape.entries * 6)
        self.accs = bump.alloc(len(shape.accs))
        self.shifts = bump.alloc(shape.points * POINT)
        self.wires = bump.alloc(len(shape.wires))
        self.sigma = bump.alloc(len(shape.sigma))
        self.acc = AccLayout.__init__[p](bump, shape.accumulators(), shape.products(), shape.wiring_products())
        self.sort = SortLayout.__init__[p](bump, shape.lookups(), shape.max_table_rows())
        self.sg = SmallGridLayout.__init__[p](bump, 6 * shape.wiring_products())
        self.lde = LdeLayout.__init__[p](bump, shape)
        self.open = OpenLayout.__init__[p](bump, shape)
        var stage = max(multiproof_region[H](4 * p.n_cw() * shape.columns_w, p.L(), p.queries()),
                        max(multiproof_region[H](4 * p.n_cw() * shape.columns_z, p.L(), p.queries()),
                            multiproof_region[H](4 * p.n_cw() * shape.columns_q, p.L(), p.queries())))
        var max_queries = p.queries()
        var max_v = 4 * p.n_cw() * p.queries()
        for lvl in shape.tail:
            stage = max(stage, multiproof_region[H](8 * p.e * lvl.codewords, lvl.L, lvl.queries))
            max_queries = max(max_queries, lvl.queries)
            max_v = max(max_v, lvl.queries)
        self.chal = ChalLayout.__init__[p](bump, shape, max_v)
        self.query = QueryLayout.__init__[p](bump, max_queries, stage)
        self.prefix = bump.alloc(PREFIX_MAX)
        self.tail = List[TailLayout]()
        for i in range(len(shape.tail)):
            self.tail.append(TailLayout.__init__[p, H](bump, shape.tail[i], i))
        self.bytes = bump.used


struct Prover[p: Params, H: Hash]:
    var shape: Shape
    var families: List[UInt8]       # the entry table as built (residual.Families), kappa bytes zero
    var entries_g: Int              # rows of layout.families_g
    var layout: ProverLayout
    var arena: Arena
    var domains: Domains
    var kappa: F2                   # a primitive element of F2*: wiring slot s has the ids kappa^s H2 (accumulate.mojo)
    var profile: Bool               # prove(profile=True): synchronize after every stage and record its time
    var t0: Int                     # the last profile mark
    var profile_names: List[String]     # filled by prove(profile=True): stage label and ms, in order
    var profile_ms: List[Int]
    var proof: ProofWriter          # host staging pool, sized once from the shape
    var trace_host: HostBuffer[DType.uint8]   # trace staging for load_trace, allocated once

    def __init__(out self, ctx: DeviceContext, var c: Compiled, keep: Bool = False) raises:
        """From a compiled statement: its shape and family table travel together."""
        var families = c.families.copy()
        self = Self(ctx, c^.take_shape(), families^, keep)

    def __init__(out self, ctx: DeviceContext, var shape: Shape, var families: List[UInt8], keep: Bool = False) raises:
        """`keep` builds the arena with no region sharing (ProverLayout.keep)."""
        if len(families) != shape.entries * ENTRY:
            raise Error("family table does not match shape.entries")
        if shape.accumulators() > 0 and 2 * Self.p.h2() == F2_ORDER:
            raise Error("the small grid needs a coset of G2 inside F2*: accumulators need h2 < 8064")
        self.shape = shape^
        self.families = families^
        self.layout = ProverLayout.__init__[Self.p, Self.H](self.shape, keep)
        self.arena = Arena(ctx, self.layout.bytes)
        self.domains = Domains.__init__[Self.p]()
        self.kappa = f2_primitive()
        self.profile = False
        self.t0 = 0
        self.profile_names = List[String]()
        self.profile_ms = List[Int]()
        self.proof = ProofWriter(ctx, proof_pool_bytes[Self.p, Self.H](self.shape))
        self.trace_host = ctx.enqueue_create_host_buffer[DType.uint8](self.shape.columns_w * Self.p.N())
        ref L = self.layout
        ref S = self.shape
        self.arena.upload(ctx, L.tables.base, build_tables[Self.p](ctx, L.tables, self.domains))
        var merged = merge_tables(self.families, S.accs, S.entries)
        self.entries_g = merged[2]
        _upload(ctx, self.arena, L.families, self.families)
        _upload(ctx, self.arena, L.families_g, merged[0])
        _upload(ctx, self.arena, L.merge, merged[1])
        _upload(ctx, self.arena, L.shifts, S.point_list)
        if len(S.accs) > 0:
            _upload(ctx, self.arena, L.accs, S.accs)
        if len(S.chals) > 0:
            _upload(ctx, self.arena, L.chal.table, S.chals)
        if len(S.wires) > 0:
            _upload(ctx, self.arena, L.wires, S.wires)
            _upload(ctx, self.arena, L.sigma, S.sigma)
        _upload(ctx, self.arena, L.query.dom1, domain_bytes(self.domains.level1))
        for i in range(len(S.tail)):
            var lvl = S.tail[i]
            var dom = RsDomain(lvl.L // lvl.cosets, lvl.cosets)
            self.arena.upload(ctx, L.tail[i].rs.base, build_rs_tables(ctx, L.tail[i].rs, dom, lvl.rows // lvl.codewords))
            _upload(ctx, self.arena, L.tail[i].dom, domain_bytes(dom))

    def _mark(mut self, ctx: DeviceContext, name: String) raises:
        """With profile on: synchronize and record the time since the last mark. Off: nothing."""
        if self.profile:
            ctx.synchronize()
            var now = perf_counter_ns()
            self.profile_names.append(name)
            self.profile_ms.append((now - self.t0) // 1000000)
            self.t0 = now

    def prove(mut self, ctx: DeviceContext, public_inputs: Span[UInt8, _], profile: Bool = False) raises -> List[UInt8]:
        """Spec section 10 in order. The trace must already be in the arena at layout.w.enc.trace.
        Every call is an enqueue; proof values are staged as async copies and read after the one
        synchronize in `proof.finish`. `profile` inserts a synchronize after every stage and records
        the stage times in profile_names / profile_ms (a measurement mode, never the production path)."""
        self.profile = profile
        self.t0 = perf_counter_ns()
        self.profile_names = List[String]()
        self.profile_ms = List[Int]()
        comptime N = Self.p.N()
        comptime e = Self.p.e
        comptime h2 = Self.p.h2()
        ref L = self.layout
        var T = L.transcript
        ref S = self.shape
        self.proof.reset()

        # header and transcript prefix (spec 9.4, statement-layer 6 step 1)
        self.proof.u32(Int(VERSION))
        self.proof.prefixed(public_inputs)
        self._prefix(ctx, public_inputs)
        self._mark(ctx, "prefix")

        # 2. the sorted copies (spec 6.3): the lookup descriptors' s columns are witness columns, filled before W is committed
        if self._sort(ctx):
            self._mark(ctx, "sort")

        # 3. commit W -> stage-1 challenges
        self._commit(ctx, L.w, DS_TREE_W, "W")
        squeeze_elements[Self.p, Self.H](ctx, self.arena, T, L.chal.stage1, 3)             # beta, delta, gamma
        derive_chals(ctx, self.arena, L.chal.stage1, L.chal.table, len(S.chals) // CHAL)
        if S.wiring_products() > 0:
            squeeze_elements[Self.p, Self.H](ctx, self.arena, T, L.chal.wchal, 2)          # beta_w, gamma_w
        self._mark(ctx, "transcript W")

        # 4-7. the Z stage and commit Z with Z2 -> alpha
        self._accumulators(ctx)
        if S.columns_z > 0:
            self._mark(ctx, "accumulate")
            self._commit(ctx, L.z, DS_TREE_Z, "Z")
            if S.products() > 0:
                absorb[Self.p, Self.H](ctx, self.arena, T, DS_TREE_Z, L.acc.z2, S.products() * h2 * e)
                self.proof.stage(self.arena, L.acc.z2, S.products() * h2 * e)
        squeeze_elements[Self.p, Self.H](ctx, self.arena, T, L.chal.alpha, 1)
        self._mark(ctx, "transcript Z")

        # 7.4: the small grid, Q3 in the clear (sent with the Q root)
        self._small_grid(ctx)
        if S.accumulators() > 0:
            self._mark(ctx, "small grid")

        # 8-10. residual grid, quotient, commit Q -> z
        self._quotient(ctx)
        self._commit(ctx, L.q, DS_TREE_Q, "Q")
        if S.accumulators() > 0:
            absorb[Self.p, Self.H](ctx, self.arena, T, DS_TREE_Q, L.sg.q3, 2 * h2 * e)
            self.proof.stage(self.arena, L.sg.q3, 2 * h2 * e)
        squeeze_elements[Self.p, Self.H](ctx, self.arena, T, L.chal.z, 2)                  # z = (z1, z2)
        self._mark(ctx, "transcript Q")

        # 11. openings at the P points -> beta per column, gamma per point
        self._openings(ctx)
        absorb[Self.p, Self.H](ctx, self.arena, T, DS_OPENINGS, L.open.openings, S.points * S.columns() * e)
        self.proof.stage(self.arena, L.open.openings, S.points * S.columns() * e)
        squeeze_elements[Self.p, Self.H](ctx, self.arena, T, L.chal.beta_gamma, S.columns() + S.points)
        self._mark(ctx, "transcript openings")

        # 12. fold to the level-2 message
        self._fold(ctx)
        self._mark(ctx, "fold")

        # 13. tail: each committed level opens the previous one
        var y = L.open.fold_y
        var y_len = N
        var running = L.open.running0
        if len(S.tail) > 0:
            running0[Self.p](ctx, self.arena, L.open.w_z, L.chal.beta_gamma + S.columns() * e, S.points, L.open.running0)
        for i in range(len(S.tail)):
            self._tail_level(ctx, i, T, y, y_len, running)
            y = L.tail[i].y
            y_len = S.tail[i].rows
            running = L.tail[i].running

        # last: the clear vector, then open the last committed level
        absorb[Self.p, Self.H](ctx, self.arena, T, DS_CLEAR, y, y_len * e)
        self.proof.stage(self.arena, y, y_len * e)
        self._mark(ctx, "transcript clear")
        self._open_previous(ctx, len(S.tail), T)
        self._mark(ctx, "open last")
        var out = self.proof.finish()
        self._mark(ctx, "finish")
        return out^

    # ---- stages, in protocol order ----

    def _prefix(mut self, ctx: DeviceContext, public_inputs: Span[UInt8, _]) raises:
        """Build the transcript prefix on the host, upload it, and start the transcript from it."""
        var prefix = prefix_bytes[Self.p, Self.H](self.shape, public_inputs, self.families)
        if len(prefix) > PREFIX_MAX:
            raise Error("public inputs too large for the prefix region")
        var prefix_host = self.proof.scratch(len(prefix))
        for i in range(len(prefix)):
            prefix_host[i] = prefix[i]
        self.arena.upload(ctx, self.layout.prefix, prefix_host)
        reset(ctx, self.arena, self.layout.transcript)
        absorb[Self.p, Self.H](ctx, self.arena, self.layout.transcript, DS_PREFIX, self.layout.prefix, len(prefix))

    def _sort(mut self, ctx: DeviceContext) raises -> Bool:
        """Fill the sorted-copy columns of every lookup descriptor. Returns whether there was one."""
        ref L = self.layout
        ref S = self.shape
        var li = 0
        for k in range(S.accumulators()):
            if acc_kind(S.accs, k) == KIND_LOOKUP:
                counting_sort[Self.p](ctx, self.arena, L.w.enc.trace, L.accs + k * ACC, L.sort.idx + li * Self.p.N() * 4,
                                      L.sort.bins, L.sort.cursor, S.table_rows(k))
                li += 1
        return li > 0

    def _commit(mut self, ctx: DeviceContext, C: CommitLayout, ds: UInt8, name: String) raises:
        """Encode the tree's trace, hash the codeword rows, absorb the root and stage it for the proof."""
        encode[Self.p](ctx, self.arena, C.enc, self.layout.tables)
        self._mark(ctx, "encode " + name)
        merkle[Self.p, Self.H](ctx, self.arena, C.enc.code, C.row, Self.p.L(), C.tree)
        self._mark(ctx, "merkle " + name)
        var root = root_offset[Self.H](C.tree, Self.p.L())
        absorb[Self.p, Self.H](ctx, self.arena, self.layout.transcript, ds, root, Self.H.DIGEST)
        self.proof.stage(self.arena, root, Self.H.DIGEST)

    def _accumulators(mut self, ctx: DeviceContext) raises:
        """Z per accumulator (Horner scans and grand products with their chain-end lines), the wiring
        products, then the Z values as the Z tree's trace. The Z2, n_end, d_end lines are per product
        (Shape.product_of, Shape.wiring_product)."""
        ref L = self.layout
        ref S = self.shape
        var A = L.acc
        var horners = 0
        for k in range(S.accumulators()):
            if acc_kind(S.accs, k) == KIND_HORNER:
                horners += 1
            else:
                accumulate[Self.p](ctx, self.arena, L.w.enc.trace, L.accs + k * ACC, L.chal.stage1, A, k, S.product_of(k))
        if horners > 0:
            horner[Self.p](ctx, self.arena, L.w.enc.trace, L.families, L.accs, L.chal.stage1, A, S.accumulators(), L.lde.pub_vals, S.columns_w + S.columns_z)
        if S.wiring_products() > 0:
            wiring[Self.p](ctx, self.arena, A, S.wiring_products(), S.wiring_product(0), L.wires, L.sigma, S.columns_w, L.chal.wchal,
                           self.kappa, self.domains.omega2)
        if S.columns_z > 0:
            ctx.enqueue_function[k_values_to_trace[Self.p]](self.arena.buf, Buf[1](A.zval), Buf[1](L.z.enc.trace), Int32(S.accumulators()),
                                                            grid_dim=ceildiv(S.columns_z * Self.p.N(), BACKEND.block), block_dim=BACKEND.block)

    def _small_grid(mut self, ctx: DeviceContext) raises:
        """R2 on the coset as the sum of every grand-product term and chain-end term, then Q3 on G2."""
        ref L = self.layout
        ref S = self.shape
        var A = L.acc
        var e2 = ext_pow[1](self.domains.omega2, Self.p.h2() - 1)
        for k in range(S.accumulators()):                  # the first term written starts R2; the rest add to it
            var pi = S.product_of(k)
            if pi >= 0:
                small_grid_accumulator[Self.p](ctx, self.arena, L.tables, A, k, pi, L.sg, L.chal.alpha, S.family_of(k), e2, pi == 0)
        if S.wiring_products() > 0:
            small_grid_wiring[Self.p](ctx, self.arena, L.tables, A, S.wiring_products(), S.wiring_product(0), L.wires, L.sg, L.chal.alpha, e2,
                                      S.wiring_product(0) == 0)
        for i in range(len(S.ends) // END):
            small_grid_end[Self.p](ctx, self.arena, L.tables, A.zval, S.columns_w, S.ends, i, L.sg, L.chal.alpha, L.chal.stage1, e2, S.products() == 0 and i == 0)
        if S.accumulators() > 0:
            small_grid_values[Self.p](ctx, self.arena, L.tables, L.sg)

    def _quotient(mut self, ctx: DeviceContext) raises:
        """The LDE of every column onto the residual grid, the residual, and the quotient as the Q tree's coefficients."""
        ref L = self.layout
        ref S = self.shape
        lde[Self.p](ctx, self.arena, L.w.enc.coeff, S.columns_w, L.tables, L.lde.ltmp, L.lde.lde_at(0))
        if S.columns_z > 0:
            lde[Self.p](ctx, self.arena, L.z.enc.coeff, S.columns_z, L.tables, L.lde.ltmp, L.lde.lde_at(S.columns_w))
        if S.columns_p > 0:
            lde[Self.p](ctx, self.arena, L.lde.pub_coeff, S.columns_p, L.tables, L.lde.ltmp, L.lde.lde_at(S.columns_w + S.columns_z))
        self._mark(ctx, "lde")
        residual[Self.p](ctx, self.arena, L.lde.lde, L.families, S.entries, L.tables, L.chal.alpha, L.chal.stage1, L.lde.residual,
                         L.families_g, self.entries_g, L.accs, len(S.accs) // ACC, L.merge)
        self._mark(ctx, "residual")
        quotient[Self.p](ctx, self.arena, L.lde.residual, L.tables, L.lde.quotient, L.q.enc.coeff)
        self._mark(ctx, "quotient")

    def _openings(mut self, ctx: DeviceContext) raises:
        """The evaluation queries at the P points, then every committed column's opening at each."""
        ref L = self.layout
        ref S = self.shape
        comptime e = Self.p.e
        build_queries[Self.p](ctx, self.arena, L.chal.z, L.shifts, S.points, L.tables, self.domains, L.open.w_tab, L.open.w_z)
        self._mark(ctx, "build_queries")
        for t in [(L.w, 0), (L.z, S.columns_w), (L.q, S.columns_w + S.columns_z)]:
            if t[0].enc.columns == 0:
                continue                                    # no Z tree without accumulators
            open[Self.p](ctx, self.arena, L.open.w_z, S.points, t[0].enc.stored, t[0].enc.columns, L.open.open_partial,
                         L.open.openings + t[1] * e, S.columns())
        self._mark(ctx, "open")

    def _fold(mut self, ctx: DeviceContext) raises:
        """The level-2 message y = sum_c beta_c stored(c) over the three trees."""
        ref L = self.layout
        ref S = self.shape
        comptime e = Self.p.e
        fold[Self.p, False](ctx, self.arena, L.chal.beta_gamma, L.w.enc.stored, S.columns_w, L.open.fold_y)
        if S.columns_z > 0:
            fold[Self.p, True](ctx, self.arena, L.chal.beta_gamma + S.columns_w * e, L.z.enc.stored, S.columns_z, L.open.fold_y)
        fold[Self.p, True](ctx, self.arena, L.chal.beta_gamma + (S.columns_w + S.columns_z) * e, L.q.enc.stored, S.columns_q, L.open.fold_y)

    def _tail_level(mut self, ctx: DeviceContext, i: Int, T: TranscriptLayout, y: Int, y_len: Int, running: Int) raises:
        """Tail level i (spec 9.3): commit Enc(y), open the level before it, sample the batching
        scalars, run the three sumcheck rounds, and fold y and the running query into this level's
        buffers. The transcript steps interleave with the kernels, so the level's protocol lives here."""
        ref L = self.layout
        ref S = self.shape
        comptime e = Self.p.e
        var lvl = S.tail[i]
        var tl = L.tail[i]
        var tag = String(i)
        comptime D = Self.p.a1 + Self.p.a2
        tail_encode(ctx, self.arena, y, lvl.rows, lvl.L // lvl.cosets, lvl.cosets, tl.etmp, tl.code, tl.rs,
                    lvl.codewords, D - Self.p.tail_digits * (i + 1), tl.w_tilde)
        self._mark(ctx, "tail encode " + tag)
        merkle[Self.p, Self.H](ctx, self.arena, tl.code, 8 * e * lvl.codewords, lvl.L, tl.tree)
        self._mark(ctx, "tail merkle " + tag)
        absorb[Self.p, Self.H](ctx, self.arena, T, DS_TAIL_ROOT, root_offset[Self.H](tl.tree, lvl.L), Self.H.DIGEST)
        self.proof.stage(self.arena, root_offset[Self.H](tl.tree, lvl.L), Self.H.DIGEST)
        self._open_previous(ctx, i, T)
        self._mark(ctx, "open previous " + tag)
        var count = self._prev_queries(i)
        var prev_dom = L.query.dom1 if i == 0 else L.tail[i - 1].dom
        var prev_L0 = Self.p.L0 if i == 0 else S.tail[i - 1].L // S.tail[i - 1].cosets
        points(ctx, self.arena, L.query.positions, count, prev_dom, prev_L0, L.query.pts)
        # the expected symbols v are the verifier's to compute from the opened rows (spec 9.3); nothing is sent
        squeeze_elements[Self.p, Self.H](ctx, self.arena, T, L.chal.batch, self._v_count(i) + 1)   # batching scalars
        self._mark(ctx, "transcript batch " + tag)
        tail_materialize[Self.p](ctx, self.arena, i == 0, running, L.chal.batch, L.query.pts, count, y_len, tl.w_tilde, L.query.ptab,
                                 1 if i == 0 else S.tail[i - 1].codewords, D - Self.p.tail_digits * i)
        self._mark(ctx, "materialize " + tag)
        for d in range(3):
            tail_round(ctx, self.arena, tl.w_tilde, y, y_len, d, L.chal.r, L.query.partial, tl.rounds + d * 3 * e)
            absorb[Self.p, Self.H](ctx, self.arena, T, DS_TAIL_ROUND, tl.rounds + d * 3 * e, 3 * e)
            squeeze_elements[Self.p, Self.H](ctx, self.arena, T, L.chal.r + d * e, 1)          # r_d
        self.proof.stage(self.arena, tl.rounds, 9 * e)
        self._mark(ctx, "rounds " + tag)
        tail_fold(ctx, self.arena, y, lvl.rows, L.chal.r, tl.y)
        tail_fold(ctx, self.arena, tl.w_tilde, lvl.rows, L.chal.r, tl.running)
        self._mark(ctx, "fold " + tag)

    def _prev_queries(self, i: Int) -> Int:
        return Self.p.queries() if i == 0 else self.shape.tail[i - 1].queries

    def _v_count(self, i: Int) -> Int:
        return 4 * Self.p.n_cw() * Self.p.queries() if i == 0 else self.shape.tail[i - 1].codewords * self.shape.tail[i - 1].queries

    def _open_previous(mut self, ctx: DeviceContext, i: Int, T: TranscriptLayout) raises:
        """Sample S on the level before tail level i (level 1 when i == 0), gather its multiproof(s),
        and stage them. The stage region is reused: the copy out is enqueued before the next gather."""
        ref L = self.layout
        ref S = self.shape
        if Self.p.grind_bits > 0:
            grind[Self.p, Self.H](ctx, self.arena, T, L.query.found, L.query.nonce)
            self.proof.stage(self.arena, L.query.nonce, 8)
        if i == 0:
            squeeze_positions[Self.p, Self.H](ctx, self.arena, T, L.query.positions, Self.p.queries(), Self.p.L())
            for C in [L.w, L.z, L.q]:
                if C.enc.columns == 0:
                    continue                                    # no Z tree without accumulators
                var bound = query_gather[Self.p, Self.H](ctx, self.arena, C.enc.code, C.row, Self.p.L(),
                                                         C.tree, L.query.positions, Self.p.queries(), L.query.stage)
                self.proof.stage(self.arena, L.query.stage, bound, multiproof=True)
        else:
            var lvl = S.tail[i - 1]
            squeeze_positions[Self.p, Self.H](ctx, self.arena, T, L.query.positions, lvl.queries, lvl.L)
            var bound = query_gather[Self.p, Self.H](ctx, self.arena, L.tail[i - 1].code, 8 * Self.p.e * lvl.codewords, lvl.L, L.tail[i - 1].tree,
                                                     L.query.positions, lvl.queries, L.query.stage)
            self.proof.stage(self.arena, L.query.stage, bound, multiproof=True)


def proof_pool_bytes[p: Params, H: Hash](shape: Shape) -> Int:
    """Host staging for one proof: the fixed bytes at the largest prefix, every multiproof region
    bound, and the transcript prefix upload."""
    var n = shape.fixed_bytes[p, H.DIGEST](PREFIX_MAX) + PREFIX_MAX
    n += multiproof_region[H](4 * p.n_cw() * shape.columns_w, p.L(), p.queries())
    if shape.columns_z > 0:
        n += multiproof_region[H](4 * p.n_cw() * shape.columns_z, p.L(), p.queries())
    n += multiproof_region[H](4 * p.n_cw() * shape.columns_q, p.L(), p.queries())
    for lvl in shape.tail:
        n += multiproof_region[H](8 * p.e * lvl.codewords, lvl.L, lvl.queries)
    return n


def _upload(ctx: DeviceContext, arena: Arena, off: Int, l: Span[UInt8, _]) raises:
    var h = ctx.enqueue_create_host_buffer[DType.uint8](len(l))
    ctx.synchronize()
    unsafe_memcpy(dest=h.unsafe_ptr(), src=l.unsafe_ptr(), count=len(l))
    arena.upload(ctx, off, h)


def load_trace[p: Params, H: Hash](ctx: DeviceContext, mut prover: Prover[p, H], trace: Span[UInt8, _]) raises:
    """Copy a host trace (columns_w x N bytes, values below 127) into the arena through the staging buffer."""
    var n = prover.shape.columns_w * p.N()
    if len(trace) != n:
        raise Error("trace has the wrong size")
    ctx.synchronize()
    unsafe_memcpy(dest=prover.trace_host.unsafe_ptr(), src=trace.unsafe_ptr(), count=n)
    prover.arena.upload(ctx, prover.layout.w.enc.trace, prover.trace_host)


def load_advice[p: Params, H: Hash](ctx: DeviceContext, mut prover: Prover[p, H], advice: Span[UInt8, _]) raises:
    """Upload the advice indices: one u32 per row per lookup descriptor, in descriptor order (sort.mojo)."""
    if len(advice) != prover.shape.lookups() * p.N() * 4:
        raise Error("advice has the wrong size")
    _upload(ctx, prover.arena, prover.layout.sort.idx, advice)


def load_public[p: Params, H: Hash](ctx: DeviceContext, mut prover: Prover[p, H], values: Span[UInt8, _]) raises:
    """Upload the public columns: the public data's columns in order, each a dense period or a term list
    (docs/public-columns.md; bytes past the columns are ignored), tiled to H and transformed to the coefficients
    the LDE reads with the trace's `idft2` (ltmp as scratch)."""
    ref L = prover.layout
    _upload(ctx, prover.arena, L.lde.pub_vals, tile_values(prover.shape.publics, values, p.h1(), p.h2()))
    if prover.shape.columns_p > 0:
        idft2[p](ctx, prover.arena, L.lde.pub_vals, L.lde.ltmp, L.lde.pub_coeff, prover.shape.columns_p, L.tables)

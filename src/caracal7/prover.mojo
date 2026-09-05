"""Host orchestration (docs/design.md section 6): plan the arena once, then enqueue every stage of
spec section 10 in order on one stream. One synchronize at the end reads the proof bytes.

Milestone 1: W tree and Q tree on synthetic columns, residual and quotient on synthetic families,
openings at P points, the tail, and the clear vector. No Z tree (milestone 2), no frontend.
"""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from caracal7.params import Params
from caracal7.arena import Arena, Bump
from caracal7.tables import Domains, TableLayout, RsDomain, RsTables, build_tables, build_rs_tables
from caracal7.encode import EncLayout, encode
from caracal7.transcript import TranscriptLayout, reset, absorb, squeeze_elements, squeeze_positions
from caracal7.transcript import DS_PREFIX, DS_TREE_W, DS_TREE_Q, DS_OPENINGS, DS_TAIL_ROOT, DS_TAIL_V, DS_TAIL_ROUND, DS_CLEAR
from caracal7.proof import Shape, ProofWriter, TailLevel, VERSION, prefix_bytes
from caracal7.hash import Hash
from caracal7.merkle import merkle, query_gather, root_offset, tree_nodes, multiproof_region
from caracal7.residual import lde, residual, quotient, quotient_elems, shift_points, ENTRY, POINT
from caracal7.open import build_queries, open, open_splits, fold
from caracal7.tail import DOM_BYTES, ROUND_THREADS, domain_bytes, tail_encode, points, running0, expected_level1, expected_tail
from caracal7.tail import tail_materialize, tail_round, tail_fold

comptime PREFIX_MAX = 1 << 16       # arena bytes for the transcript prefix (public inputs included)

struct TailLayout(TrivialRegisterPassable):
    """Arena offsets of one committed tail level (spec 9.3), all E-valued."""
    var y: Int          # (rows, e)          the folded message y_{l+1} = Mat(y_l) r_bar
    var running: Int    # (rows, e)          the folded query, the next level's running claim
    var etmp: Int       # (L0, 32, 4)        encoder scratch
    var code: Int       # (s, 8, e)          leaf-major codeword rows
    var tree: Int       # (node, 32)
    var v: Int          # expected symbols for the previous level
    var w_tilde: Int    # (slot, e)          batched query on y_l
    var rounds: Int     # (3, 3, e)          sumcheck messages
    var rs: RsTables    # this level's RS domain tables
    var dom: Int        # DOM_BYTES          this level's domain (tail.domain_bytes)

    def __init__[p: Params, H: Hash](out self, mut bump: Bump, lvl: TailLevel, v_count: Int):
        var L0 = lvl.L // lvl.cosets
        self.y = bump.alloc(lvl.rows * p.e)
        self.running = bump.alloc(lvl.rows * p.e)
        self.etmp = bump.alloc(L0 * 32 * 4)
        self.code = bump.alloc(lvl.L * 8 * p.e)
        self.tree = bump.alloc(tree_nodes(lvl.L) * H.DIGEST)
        self.v = bump.alloc(v_count * p.e)
        self.w_tilde = bump.alloc(lvl.length * p.e)
        self.rounds = bump.alloc(9 * p.e)
        self.rs = RsTables(bump.alloc(0), L0, lvl.cosets, lvl.rows)
        _ = bump.alloc(self.rs.bytes)
        self.dom = bump.alloc(DOM_BYTES)


struct ProverLayout:
    """Every buffer of design section 3 as an arena offset. Built once per (Params, Shape)."""
    var tables: TableLayout
    var transcript: TranscriptLayout
    var enc_w: EncLayout            # witness tree: trace .. code
    var enc_q: EncLayout            # quotient tree: trace .. code (quotient writes the trace)
    var tree_w: Int                 # (node, 32), level 0 first, tree_nodes(L0) nodes
    var tree_q: Int
    var families: Int               # (entry, ENTRY)        the family table, kappa folded in on device
    var shifts: Int                 # (P, POINT)            opening points as (dj1, dj2) on G
    var ltmp: Int                   # (column, k2, G1, 2)   LDE after axis 1
    var lde: Int                    # (column, G2, G1, 2)   witness columns on the residual grid
    var residual: Int               # (G2, G1, e)
    var quotient: Int               # quotient_elems x e    Q1, Q2 interpolation scratch (residual.mojo)
    var w_z: Int                    # (P, slot, e)          evaluation queries
    var openings: Int               # (P, column, e)
    var open_partial: Int           # (P, splits, column, e) split-K partials of `open`
    var fold_y: Int                 # (slot, e)             y = sum beta_c stored(c), the level-2 message
    var running0: Int               # (slot, e)             sum_p gamma_p w_{z_p}, the level-2 running query
    var dom1: Int                   # DOM_BYTES             the level-1 domain
    var pts: Int                    # (max queries, 4)      leaf points of the opened positions
    var partial: Int                # (ROUND_THREADS, 3, e) sumcheck partial sums
    var proof_stage: Int            # gathered rows and siblings of one multiproof, staged to the host in stream order
    var prefix: Int                 # transcript prefix bytes (PREFIX_MAX)
    # challenges, one region each so nothing is overwritten before its consumer runs
    var stage1: Int                 # (4, e)                beta_1, delta, gamma, alpha
    var z: Int                      # (2, e)
    var beta_gamma: Int             # (columns + P, e)      beta per column, gamma per point
    var positions: Int              # (max queries, u32)    S of the level being opened
    var batch: Int                  # (max v_count + 1, e)  tail batching scalars
    var r: Int                      # (3, e)                sumcheck round challenges of the current level
    var tail: List[TailLayout]
    var bytes: Int

    def __init__[p: Params, H: Hash](out self, shape: Shape) raises:
        comptime N = p.N()
        comptime G = 4 * N
        var bump = Bump()
        self.tables = TableLayout.__init__[p](bump.alloc(0))
        _ = bump.alloc(self.tables.bytes)
        self.transcript = TranscriptLayout(bump)
        self.enc_w = EncLayout.__init__[p](bump, shape.columns_w)
        self.enc_q = EncLayout.__init__[p](bump, shape.columns_q)
        self.tree_w = bump.alloc(tree_nodes(p.L()) * H.DIGEST)
        self.tree_q = bump.alloc(tree_nodes(p.L()) * H.DIGEST)
        self.families = bump.alloc(shape.entries * ENTRY)
        self.shifts = bump.alloc(shape.points * POINT)
        self.ltmp = bump.alloc(shape.columns_w * p.h2() * 2 * p.h1() * 2)
        self.lde = bump.alloc(shape.columns_w * G * 2)
        self.residual = bump.alloc(G * p.e)
        self.quotient = bump.alloc(quotient_elems[p]() * p.e)
        self.w_z = bump.alloc(shape.points * N * p.e)
        self.openings = bump.alloc(shape.points * shape.columns() * p.e)
        self.open_partial = bump.alloc(shape.points * open_splits[p]() * max(shape.columns_w, shape.columns_q) * p.e)
        self.fold_y = bump.alloc(N * p.e)
        self.running0 = bump.alloc(N * p.e)
        self.dom1 = bump.alloc(DOM_BYTES)
        var stage = max(multiproof_region[H](4 * p.n_cw() * shape.columns_w, p.L(), p.queries()),
                        multiproof_region[H](4 * p.n_cw() * shape.columns_q, p.L(), p.queries()))
        var max_queries = p.queries()
        var max_v = 4 * p.n_cw() * p.queries()
        for lvl in shape.tail:
            stage = max(stage, multiproof_region[H](8 * p.e, lvl.L, lvl.queries))
            max_queries = max(max_queries, lvl.queries)
            max_v = max(max_v, lvl.queries)
        self.proof_stage = bump.alloc(stage)
        self.prefix = bump.alloc(PREFIX_MAX)
        self.stage1 = bump.alloc(4 * p.e)
        self.z = bump.alloc(2 * p.e)
        self.beta_gamma = bump.alloc((shape.columns() + shape.points) * p.e)
        self.positions = bump.alloc(max_queries * 4)
        self.pts = bump.alloc(max_queries * 4)
        self.partial = bump.alloc(ROUND_THREADS * 3 * p.e)
        self.batch = bump.alloc((max_v + 1) * p.e)
        self.r = bump.alloc(3 * p.e)
        self.tail = List[TailLayout]()
        for i in range(len(shape.tail)):
            var prev_q = p.queries() if i == 0 else shape.tail[i - 1].queries
            var v_count = 4 * p.n_cw() * prev_q if i == 0 else prev_q
            self.tail.append(TailLayout.__init__[p, H](bump, shape.tail[i], v_count))
        self.bytes = bump.used


struct Prover[p: Params, H: Hash]:
    var shape: Shape
    var families: List[UInt8]       # the entry table as built (residual.Families), kappa bytes zero
    var layout: ProverLayout
    var arena: Arena
    var domains: Domains
    var profile_names: List[String]     # filled by prove(profile=True): stage label and ms, in order
    var profile_ms: List[Int]

    def __init__(out self, ctx: DeviceContext, var shape: Shape, var families: List[UInt8]) raises:
        if len(families) != shape.entries * ENTRY:
            raise Error("family table does not match shape.entries")
        self.shape = shape^
        self.families = families^
        self.layout = ProverLayout.__init__[Self.p, Self.H](self.shape)
        self.arena = Arena(ctx, self.layout.bytes)
        self.domains = Domains.__init__[Self.p]()
        self.profile_names = List[String]()
        self.profile_ms = List[Int]()
        self.arena.upload(ctx, self.layout.tables.base, build_tables[Self.p](ctx, self.layout.tables, self.domains))
        var pts = shift_points(self.families)
        var fh = ctx.enqueue_create_host_buffer[DType.uint8](len(self.families))
        var ph = ctx.enqueue_create_host_buffer[DType.uint8](len(pts))
        ctx.synchronize()
        for i in range(len(self.families)):
            fh[i] = self.families[i]
        for i in range(len(pts)):
            ph[i] = pts[i]
        self.arena.upload(ctx, self.layout.families, fh)
        self.arena.upload(ctx, self.layout.shifts, ph)
        _upload(ctx, self.arena, self.layout.dom1, domain_bytes(self.domains.level1))
        for i in range(len(self.shape.tail)):
            var lvl = self.shape.tail[i]
            var dom = RsDomain(lvl.L // lvl.cosets, lvl.cosets)
            self.arena.upload(ctx, self.layout.tail[i].rs.base, build_rs_tables(ctx, self.layout.tail[i].rs, dom, lvl.rows))
            _upload(ctx, self.arena, self.layout.tail[i].dom, domain_bytes(dom))

    def _mark(mut self, ctx: DeviceContext, profile: Bool, name: String, mut t0: Int) raises:
        """With profile on: synchronize and record the time since the last mark. Off: nothing."""
        if profile:
            ctx.synchronize()
            var now = perf_counter_ns()
            self.profile_names.append(name)
            self.profile_ms.append((now - t0) // 1000000)
            t0 = now

    def prove(mut self, ctx: DeviceContext, public_inputs: List[UInt8], profile: Bool = False) raises -> List[UInt8]:
        """Spec section 10 in order. The trace must already be in the arena at layout.enc_w.trace.
        Every call is an enqueue; proof values are staged as async copies and read after the one
        synchronize in `proof.finish`. `profile` inserts a synchronize after every stage and records
        the stage times in profile_names / profile_ms (a measurement mode, never the production path)."""
        self.profile_names = List[String]()
        self.profile_ms = List[Int]()
        var t0 = perf_counter_ns()
        comptime N = Self.p.N()
        comptime e = Self.p.e
        var base = self.arena.base()
        ref L = self.layout
        var T = L.transcript
        ref S = self.shape
        var row_w = 4 * Self.p.n_cw() * S.columns_w
        var row_q = 4 * Self.p.n_cw() * S.columns_q
        var proof = ProofWriter(ctx)

        # header and transcript prefix (spec 9.4, statement-layer 6 step 1)
        proof.u32(Int(VERSION))
        proof.prefixed(public_inputs)
        var prefix = prefix_bytes[Self.p, Self.H](S, public_inputs, self.families)
        if len(prefix) > PREFIX_MAX:
            raise Error("public inputs too large for the prefix region")
        var prefix_host = ctx.enqueue_create_host_buffer[DType.uint8](len(prefix))
        ctx.synchronize()
        for i in range(len(prefix)):
            prefix_host[i] = prefix[i]
        self.arena.upload(ctx, L.prefix, prefix_host)
        reset(ctx, base, T)
        absorb[Self.p, Self.H](ctx, base, T, DS_PREFIX, L.prefix, len(prefix))
        self._mark(ctx, profile, "prefix", t0)

        # 3. commit W. Milestone 1 has no Z tree, so alpha is squeezed here as the fourth stage-1 challenge.
        encode[Self.p](ctx, base, L.enc_w, L.tables)
        self._mark(ctx, profile, "encode W", t0)
        merkle[Self.p, Self.H](ctx, base, L.enc_w.code, row_w, Self.p.L(), L.tree_w)
        self._mark(ctx, profile, "merkle W", t0)
        absorb[Self.p, Self.H](ctx, base, T, DS_TREE_W, root_offset[Self.H](L.tree_w, Self.p.L()), Self.H.DIGEST)
        proof.stage(self.arena, root_offset[Self.H](L.tree_w, Self.p.L()), Self.H.DIGEST)
        squeeze_elements[Self.p, Self.H](ctx, base, T, L.stage1, 4)             # beta_1, delta, gamma, alpha
        self._mark(ctx, profile, "transcript W", t0)

        # 8-10. residual grid, quotient, commit Q
        lde[Self.p](ctx, base, L.enc_w.coeff, S.columns_w, L.tables, L.ltmp, L.lde)
        self._mark(ctx, profile, "lde", t0)
        residual[Self.p](ctx, base, L.lde, L.families, S.entries, L.tables, L.stage1 + 3 * e, L.residual)
        self._mark(ctx, profile, "residual", t0)
        quotient[Self.p](ctx, base, L.residual, L.tables, L.quotient, L.enc_q.trace)
        self._mark(ctx, profile, "quotient", t0)
        encode[Self.p](ctx, base, L.enc_q, L.tables)
        self._mark(ctx, profile, "encode Q", t0)
        merkle[Self.p, Self.H](ctx, base, L.enc_q.code, row_q, Self.p.L(), L.tree_q)
        self._mark(ctx, profile, "merkle Q", t0)
        absorb[Self.p, Self.H](ctx, base, T, DS_TREE_Q, root_offset[Self.H](L.tree_q, Self.p.L()), Self.H.DIGEST)
        proof.stage(self.arena, root_offset[Self.H](L.tree_q, Self.p.L()), Self.H.DIGEST)
        squeeze_elements[Self.p, Self.H](ctx, base, T, L.z, 2)                  # z = (z1, z2)
        self._mark(ctx, profile, "transcript Q", t0)

        # 11. openings at the P points
        build_queries[Self.p](ctx, base, L.z, L.shifts, S.points, L.tables, self.domains, L.w_z)
        self._mark(ctx, profile, "build_queries", t0)
        open[Self.p](ctx, base, L.w_z, S.points, L.enc_w.stored, S.columns_w, L.open_partial, L.openings, S.columns())
        open[Self.p](ctx, base, L.w_z, S.points, L.enc_q.stored, S.columns_q, L.open_partial, L.openings + S.columns_w * e, S.columns())
        self._mark(ctx, profile, "open", t0)
        absorb[Self.p, Self.H](ctx, base, T, DS_OPENINGS, L.openings, S.points * S.columns() * e)
        proof.stage(self.arena, L.openings, S.points * S.columns() * e)
        squeeze_elements[Self.p, Self.H](ctx, base, T, L.beta_gamma, S.columns() + S.points)
        self._mark(ctx, profile, "transcript openings", t0)

        # 12. fold to the level-2 message
        fold[Self.p](ctx, base, L.beta_gamma, L.enc_w.stored, S.columns_w, L.enc_q.stored, S.columns_q, L.fold_y)
        self._mark(ctx, profile, "fold", t0)

        # 13. tail: each committed level opens the previous one
        var y = L.fold_y
        var y_len = N
        var running = L.running0
        if len(S.tail) > 0:
            running0[Self.p](ctx, base, L.w_z, L.beta_gamma + S.columns() * e, S.points, L.running0)
        for i in range(len(S.tail)):
            var lvl = S.tail[i]
            var tl = L.tail[i]
            tail_encode(ctx, base, y, lvl.rows, lvl.L // lvl.cosets, lvl.cosets, tl.etmp, tl.code, tl.rs)
            self._mark(ctx, profile, "tail encode " + String(i), t0)
            merkle[Self.p, Self.H](ctx, base, tl.code, 8 * e, lvl.L, tl.tree)
            self._mark(ctx, profile, "tail merkle " + String(i), t0)
            absorb[Self.p, Self.H](ctx, base, T, DS_TAIL_ROOT, root_offset[Self.H](tl.tree, lvl.L), Self.H.DIGEST)
            proof.stage(self.arena, root_offset[Self.H](tl.tree, lvl.L), Self.H.DIGEST)
            self._open_previous(ctx, i, T, proof)
            self._mark(ctx, profile, "open previous " + String(i), t0)
            var count = self._prev_queries(i)
            var prev_dom = L.dom1 if i == 0 else L.tail[i - 1].dom
            var prev_L0 = Self.p.L0 if i == 0 else S.tail[i - 1].L // S.tail[i - 1].cosets
            points(ctx, base, L.positions, count, prev_dom, prev_L0, L.pts)
            if i == 0:
                expected_level1(ctx, base, L.positions, count, L.enc_w.code, S.columns_w, L.enc_q.code, S.columns_q, L.beta_gamma, tl.v)
            else:
                expected_tail(ctx, base, L.positions, count, L.tail[i - 1].code, L.r, tl.v)
            self._mark(ctx, profile, "expected symbols " + String(i), t0)
            absorb[Self.p, Self.H](ctx, base, T, DS_TAIL_V, tl.v, self._v_count(i) * e)
            proof.stage(self.arena, tl.v, self._v_count(i) * e)
            squeeze_elements[Self.p, Self.H](ctx, base, T, L.batch, self._v_count(i) + 1)   # batching scalars
            self._mark(ctx, profile, "transcript v " + String(i), t0)
            tail_materialize[Self.p](ctx, base, i == 0, running, L.batch, L.pts, count, y_len, tl.w_tilde)
            self._mark(ctx, profile, "materialize " + String(i), t0)
            for d in range(3):
                tail_round(ctx, base, tl.w_tilde, y, y_len, d, L.r, L.partial, tl.rounds + d * 3 * e)
                absorb[Self.p, Self.H](ctx, base, T, DS_TAIL_ROUND, tl.rounds + d * 3 * e, 3 * e)
                squeeze_elements[Self.p, Self.H](ctx, base, T, L.r + d * e, 1)          # r_d
            proof.stage(self.arena, tl.rounds, 9 * e)
            self._mark(ctx, profile, "rounds " + String(i), t0)
            tail_fold(ctx, base, y, lvl.rows, L.r, tl.y)
            tail_fold(ctx, base, tl.w_tilde, lvl.rows, L.r, tl.running)
            self._mark(ctx, profile, "fold " + String(i), t0)
            y = tl.y
            y_len = lvl.rows
            running = tl.running

        # last: the clear vector, then open the last committed level
        absorb[Self.p, Self.H](ctx, base, T, DS_CLEAR, y, y_len * e)
        proof.stage(self.arena, y, y_len * e)
        self._mark(ctx, profile, "transcript clear", t0)
        self._open_previous(ctx, len(S.tail), T, proof)
        self._mark(ctx, profile, "open last", t0)
        var out = proof.finish()
        self._mark(ctx, profile, "finish", t0)
        return out^

    def _prev_queries(self, i: Int) -> Int:
        return Self.p.queries() if i == 0 else self.shape.tail[i - 1].queries

    def _v_count(self, i: Int) -> Int:
        return 4 * Self.p.n_cw() * Self.p.queries() if i == 0 else self.shape.tail[i - 1].queries

    def _open_previous(self, ctx: DeviceContext, i: Int, T: TranscriptLayout, mut proof: ProofWriter) raises:
        """Sample S on the level before tail level i (level 1 when i == 0), gather its multiproof(s),
        and stage them. The stage region is reused: the copy out is enqueued before the next gather."""
        var base = self.arena.base()
        ref L = self.layout
        ref S = self.shape
        if i == 0:
            squeeze_positions[Self.p, Self.H](ctx, base, T, L.positions, Self.p.queries(), Self.p.L())
            var bound = query_gather[Self.p, Self.H](ctx, base, L.enc_w.code, 4 * Self.p.n_cw() * S.columns_w, Self.p.L(),
                                                     L.tree_w, L.positions, Self.p.queries(), L.proof_stage)
            proof.stage(self.arena, L.proof_stage, bound, multiproof=True)
            bound = query_gather[Self.p, Self.H](ctx, base, L.enc_q.code, 4 * Self.p.n_cw() * S.columns_q, Self.p.L(),
                                                 L.tree_q, L.positions, Self.p.queries(), L.proof_stage)
            proof.stage(self.arena, L.proof_stage, bound, multiproof=True)
        else:
            var lvl = S.tail[i - 1]
            squeeze_positions[Self.p, Self.H](ctx, base, T, L.positions, lvl.queries, lvl.L)
            var bound = query_gather[Self.p, Self.H](ctx, base, L.tail[i - 1].code, 8 * Self.p.e, lvl.L, L.tail[i - 1].tree,
                                                     L.positions, lvl.queries, L.proof_stage)
            proof.stage(self.arena, L.proof_stage, bound, multiproof=True)


def _upload(ctx: DeviceContext, arena: Arena, off: Int, l: List[UInt8]) raises:
    var h = ctx.enqueue_create_host_buffer[DType.uint8](len(l))
    ctx.synchronize()
    for i in range(len(l)):
        h[i] = l[i]
    arena.upload(ctx, off, h)


def load_trace[p: Params, H: Hash](ctx: DeviceContext, prover: Prover[p, H], trace: List[UInt8]) raises:
    """Copy a host trace (columns_w x N bytes, values below 127) into the arena."""
    var n = prover.shape.columns_w * p.N()
    if len(trace) != n:
        raise Error("trace has the wrong size")
    var h = ctx.enqueue_create_host_buffer[DType.uint8](n)
    ctx.synchronize()
    for i in range(n):
        h[i] = trace[i]
    prover.arena.upload(ctx, prover.layout.enc_w.trace, h)

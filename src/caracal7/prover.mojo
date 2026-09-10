"""Host orchestration (docs/design.md section 6): plan the arena once, then enqueue every stage of
spec section 10 in order on one stream. One synchronize at the end reads the proof bytes.

Milestone 1: W tree and Q tree on synthetic columns, residual and quotient on synthetic families,
openings at P points, the tail, and the clear vector. No Z tree (milestone 2), no frontend.
"""

from std.time import perf_counter_ns
from std.math import ceildiv
from max.gpu.host import DeviceContext, HostBuffer

from std.memory import unsafe_memcpy
from caracal7.core.params import Params
from caracal7.core.field import F2, ext_pow
from caracal7.core.arena import Arena, Bump
from caracal7.core.tables import F2_ORDER, Domains, TableLayout, RsDomain, RsTables, build_tables, build_rs_tables, f2_primitive
from caracal7.pcs.encode import EncLayout, encode, idft2
from caracal7.core.transcript import TranscriptLayout, reset, absorb, squeeze_elements, squeeze_positions
from caracal7.core.transcript import DS_PREFIX, DS_TREE_W, DS_TREE_Z, DS_TREE_Q, DS_OPENINGS, DS_TAIL_ROOT, DS_TAIL_ROUND, DS_CLEAR
from caracal7.proof import Shape, ProofWriter, TailLevel, VERSION, prefix_bytes
from caracal7.core.hash import Hash
from caracal7.pcs import merkle, query_gather, root_offset, tree_nodes, multiproof_region, build_queries, open, open_splits, fold, table_len
from caracal7.pcs import DOM_BYTES, ROUND_THREADS, domain_bytes, tail_encode, points, running0, tail_materialize, tail_round, tail_fold, power_table_len
from caracal7.relations import ENTRY, POINT, ACC, END, WIRE, CHAL, KIND_LOOKUP, KIND_HORNER, value_bytes, tile_values, accumulate, horner, wiring, derive_chals, counting_sort, lde, residual, quotient, quotient_elems, k_values_to_trace, small_grid_product, small_grid_end, small_grid_values, SG_TOTAL
from caracal7.core.bytes import Buf, get_u16
from caracal7.core.backend import BACKEND

comptime PREFIX_MAX = 1 << 16       # arena bytes for the transcript prefix (public inputs included)

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

    def __init__[p: Params, H: Hash](out self, mut bump: Bump, lvl: TailLevel):
        var L0 = lvl.L // lvl.cosets
        self.y = bump.alloc(lvl.rows * p.e)
        self.running = bump.alloc(lvl.rows * p.e)
        self.etmp = bump.alloc(L0 * 32 * 4)
        self.code = bump.alloc(lvl.L * 8 * p.e)
        self.tree = bump.alloc(tree_nodes(lvl.L) * H.DIGEST)
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
    var enc_z: EncLayout            # accumulator tree: trace .. code (the Z stage writes the trace)
    var enc_q: EncLayout            # quotient tree: trace .. code (quotient writes the trace)
    var tree_w: Int                 # (node, 32), level 0 first, tree_nodes(L0) nodes
    var tree_z: Int
    var tree_q: Int
    var families: Int               # (entry, ENTRY)        the family table, kappa folded in on device
    var families_g: Int             # (entry, ENTRY)        one entry per distinct (reads, gate) of the non-Horner-basis entries: the residual's table
    var merge: Int                  # (entry, 2) u16 (start, count), then (entry) u16 indices into `families`: the entries each families_g row sums
    var accs: Int                   # (accumulator, ACC)    accumulator descriptors
    var shifts: Int                 # (P, POINT)            opening points as (dj1, dj2) on G
    var num: Int                    # (row, e)              the Z stage: N, D, 1/D, Z per accumulator in turn
    var den: Int
    var zscratch: Int
    var zval: Int                   # (accumulator, row, e) Z values, the Z tree's trace before the coordinate split
    var chain_prod: Int             # (x2, e)
    var z2: Int                     # (product accumulator, x2, e) + e  Z2 in the clear; one trailing 1 (accumulate.k_z2)
    var n_end: Int                  # (accumulator, x2, e)  N(e1, x2), D(e1, x2)
    var d_end: Int
    var idx: Int                    # (lookup, row, u32)    advice indices, one list per lookup descriptor in order (sort.mojo)
    var bins: Int                   # (max K + 1, u32)
    var cursor: Int                 # (max K, u32)
    var wires: Int                  # (wiring product, WIRE)
    var sigma: Int                  # (slot, x2, 2)         the wiring permutation
    var wlines: Int                 # (wiring product, 4, x2, e)  the factor lines n0, n1, d0, d1 (accumulate.k_wire_factors)
    var sg: Int                     # small-grid scratch, SG_TOTAL h2 e bytes (smallgrid.mojo)
    var q3: Int                     # (2 h2, e)             Q3 on G2 in the clear
    var pub_vals: Int               # (public column, x2, x1)      public column values on H (load_public)
    var pub_coeff: Int              # (public column, k2, k1, 2)   their coefficients (idft2 in load_public), the LDE input
    var ltmp: Int                   # (column, G2, G1, 2)   LDE after axis 1, then the axis-2 scratch
    var lde: Int                    # (column, G2, G1, 2)   witness, accumulator, then public columns on the residual grid
    var residual: Int               # (G2, G1, e)
    var quotient: Int               # quotient_elems x e    Q1, Q2 interpolation scratch (residual.mojo)
    var w_tab: Int                  # (P, table_len, e)     per-point powers and Lagrange factors (open.mojo)
    var w_z: Int                    # (slot, P, e)          evaluation queries
    var openings: Int               # (P, column, e)
    var open_partial: Int           # (splits, column, P, e) split-K partials of `open`
    var fold_y: Int                 # (slot, e)             y = sum beta_c stored(c), the level-2 message
    var running0: Int               # (slot, e)             sum_p gamma_p w_{z_p}, the level-2 running query
    var dom1: Int                   # DOM_BYTES             the level-1 domain
    var pts: Int                    # (max queries, 4)      leaf points of the opened positions
    var ptab: Int                   # (queries, table, 4)   level-1 powers of the leaf points
    var partial: Int                # (ROUND_THREADS, 3, e) sumcheck partial sums
    var proof_stage: Int            # gathered rows and siblings of one multiproof, staged to the host in stream order
    var prefix: Int                 # transcript prefix bytes (PREFIX_MAX)
    # challenges, one region each so nothing is overwritten before its consumer runs
    var stage1: Int                 # (chal_count, e)       beta, delta, gamma, then the derivation table's rows
    var chal_table: Int             # (rows, CHAL)          the derivation table
    var wchal: Int                  # (2, e)                beta_w, gamma_w of the wiring copy constraint
    var alpha: Int                  # (1, e)
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
        self.enc_z = EncLayout.__init__[p](bump, shape.columns_z)
        self.enc_q = EncLayout.__init__[p](bump, shape.columns_q)
        self.tree_w = bump.alloc(tree_nodes(p.L()) * H.DIGEST)
        self.tree_z = bump.alloc(tree_nodes(p.L()) * H.DIGEST)
        self.tree_q = bump.alloc(tree_nodes(p.L()) * H.DIGEST)
        self.families = bump.alloc(shape.entries * ENTRY)
        self.families_g = bump.alloc(shape.entries * ENTRY)
        self.merge = bump.alloc(shape.entries * 6)
        self.accs = bump.alloc(len(shape.accs))
        self.shifts = bump.alloc(shape.points * POINT)
        self.num = bump.alloc(N * p.e)
        self.den = bump.alloc(N * p.e)
        self.zscratch = bump.alloc(N * p.e)
        self.zval = bump.alloc(shape.accumulators() * N * p.e)
        self.chain_prod = bump.alloc(p.h2() * p.e)
        self.z2 = bump.alloc((shape.products() * p.h2() + 1) * p.e)
        self.n_end = bump.alloc(shape.accumulators() * p.h2() * p.e)
        self.d_end = bump.alloc(shape.accumulators() * p.h2() * p.e)
        self.idx = bump.alloc(shape.lookups() * N * 4)
        self.bins = bump.alloc((shape.max_table_rows() + 1) * 4)
        self.cursor = bump.alloc(shape.max_table_rows() * 4)
        self.wires = bump.alloc(len(shape.wires))
        self.sigma = bump.alloc(len(shape.sigma))
        self.wlines = bump.alloc(shape.wiring_products() * 4 * p.h2() * p.e)
        self.sg = bump.alloc(SG_TOTAL * p.h2() * p.e)
        self.q3 = bump.alloc(2 * p.h2() * p.e)
        self.pub_vals = bump.alloc(shape.columns_p * N)
        self.pub_coeff = bump.alloc(shape.columns_p * N * 2)
        self.ltmp = bump.alloc(max(shape.columns_w, max(shape.columns_z, shape.columns_p)) * 2 * p.h2() * 2 * p.h1() * 2)
        self.lde = bump.alloc((shape.columns_w + shape.columns_z + shape.columns_p) * G * 2)
        self.residual = bump.alloc(G * p.e)
        self.quotient = bump.alloc(quotient_elems[p]() * p.e)
        self.w_tab = bump.alloc(shape.points * table_len[p]() * p.e)
        self.w_z = bump.alloc(shape.points * N * p.e)
        self.openings = bump.alloc(shape.points * shape.columns() * p.e)
        self.open_partial = bump.alloc(shape.points * open_splits[p]() * max(shape.columns_w, max(shape.columns_z, shape.columns_q)) * p.e)
        self.fold_y = bump.alloc(N * p.e)
        self.running0 = bump.alloc(N * p.e)
        self.dom1 = bump.alloc(DOM_BYTES)
        var stage = max(multiproof_region[H](4 * p.n_cw() * shape.columns_w, p.L(), p.queries()),
                        max(multiproof_region[H](4 * p.n_cw() * shape.columns_z, p.L(), p.queries()),
                            multiproof_region[H](4 * p.n_cw() * shape.columns_q, p.L(), p.queries())))
        var max_queries = p.queries()
        var max_v = 4 * p.n_cw() * p.queries()
        for lvl in shape.tail:
            stage = max(stage, multiproof_region[H](8 * p.e, lvl.L, lvl.queries))
            max_queries = max(max_queries, lvl.queries)
            max_v = max(max_v, lvl.queries)
        self.proof_stage = bump.alloc(stage)
        self.prefix = bump.alloc(PREFIX_MAX)
        self.stage1 = bump.alloc(shape.chal_count() * p.e)
        self.chal_table = bump.alloc(len(shape.chals))
        self.wchal = bump.alloc(2 * p.e)
        self.alpha = bump.alloc(p.e)
        self.z = bump.alloc(2 * p.e)
        self.beta_gamma = bump.alloc((shape.columns() + shape.points) * p.e)
        self.positions = bump.alloc(max_queries * 4)
        self.pts = bump.alloc(max_queries * 4)
        self.ptab = bump.alloc(max_queries * power_table_len[p]() * 4)
        self.partial = bump.alloc(ROUND_THREADS * 3 * p.e)
        self.batch = bump.alloc((max_v + 1) * p.e)
        self.r = bump.alloc(3 * p.e)
        self.tail = List[TailLayout]()
        for i in range(len(shape.tail)):
            self.tail.append(TailLayout.__init__[p, H](bump, shape.tail[i]))
        self.bytes = bump.used


struct Prover[p: Params, H: Hash]:
    var shape: Shape
    var families: List[UInt8]       # the entry table as built (residual.Families), kappa bytes zero
    var entries_g: Int              # entries of layout.families_g
    var layout: ProverLayout
    var arena: Arena
    var domains: Domains
    var kappa: F2                   # a primitive element of F2*: wiring slot s has the ids kappa^s H2 (accumulate.mojo)
    var profile_names: List[String]     # filled by prove(profile=True): stage label and ms, in order
    var profile_ms: List[Int]
    var proof: ProofWriter          # host staging pool, sized once from the shape
    var trace_host: HostBuffer[DType.uint8]   # trace staging for load_trace, allocated once

    def __init__(out self, ctx: DeviceContext, var shape: Shape, var families: List[UInt8]) raises:
        if len(families) != shape.entries * ENTRY:
            raise Error("family table does not match shape.entries")
        if shape.accumulators() > 0 and 2 * Self.p.h2() == F2_ORDER:
            raise Error("the small grid needs a coset of G2 inside F2*: accumulators need h2 < 8064")
        self.shape = shape^
        self.families = families^
        self.entries_g = 0
        self.layout = ProverLayout.__init__[Self.p, Self.H](self.shape)
        self.arena = Arena(ctx, self.layout.bytes)
        self.domains = Domains.__init__[Self.p]()
        self.kappa = f2_primitive()
        self.profile_names = List[String]()
        self.profile_ms = List[Int]()
        self.proof = ProofWriter(ctx, proof_pool_bytes[Self.p, Self.H](self.shape))
        self.trace_host = ctx.enqueue_create_host_buffer[DType.uint8](self.shape.columns_w * Self.p.N())
        self.arena.upload(ctx, self.layout.tables.base, build_tables[Self.p](ctx, self.layout.tables, self.domains))
        var pts = self.shape.point_list.copy()
        var fh = ctx.enqueue_create_host_buffer[DType.uint8](len(self.families))
        var ph = ctx.enqueue_create_host_buffer[DType.uint8](len(pts))
        ctx.synchronize()
        for i in range(len(self.families)):
            fh[i] = self.families[i]
        for i in range(len(pts)):
            ph[i] = pts[i]
        self.arena.upload(ctx, self.layout.families, fh)
        var keep = List[Bool](length=self.shape.entries, fill=True)
        for k in range(len(self.shape.accs) // ACC):
            if Int(self.shape.accs[k * ACC + 38]) == KIND_HORNER:
                var first = get_u16(self.shape.accs, k * ACC + 2)
                for i in range(first - 32, first):
                    keep[i] = False
        var fg = ctx.enqueue_create_host_buffer[DType.uint8](len(self.families))
        var mg = ctx.enqueue_create_host_buffer[DType.uint8](self.shape.entries * 6)
        ctx.synchronize()
        # one families_g row per distinct (reads, gate) descriptor; `merge` lists the entries whose
        # kappas it sums (they share X(point), so sum kappa_i X = (sum kappa_i) X)
        if self.shape.entries > 65535:
            raise Error("Prover: the merge table indexes entries as u16")
        var row_of = Dict[String, Int]()
        var members = List[List[Int]]()
        for i in range(self.shape.entries):
            if not keep[i]:
                continue
            var key = String("")
            for j in range(16, 29):
                key += String(Int(self.families[i * ENTRY + j])) + ","
            var u = row_of.get(key, -1)
            if u < 0:
                u = self.entries_g
                row_of[key] = u
                members.append(List[Int]())
                for j in range(ENTRY):
                    fg[u * ENTRY + j] = self.families[i * ENTRY + j]
                self.entries_g += 1
            members[u].append(i)
        for i in range(self.entries_g * ENTRY, len(fg)):
            fg[i] = 0
        for i in range(self.shape.entries * 6):
            mg[i] = 0
        var idx_off = self.shape.entries * 4
        var at = 0
        for u in range(self.entries_g):
            _put_u16(mg, u * 4, at)
            _put_u16(mg, u * 4 + 2, len(members[u]))
            for i in members[u]:
                _put_u16(mg, idx_off + 2 * at, i)
                at += 1
        self.arena.upload(ctx, self.layout.families_g, fg)
        self.arena.upload(ctx, self.layout.merge, mg)
        self.arena.upload(ctx, self.layout.shifts, ph)
        if len(self.shape.accs) > 0:
            _upload(ctx, self.arena, self.layout.accs, self.shape.accs)
        if len(self.shape.chals) > 0:
            _upload(ctx, self.arena, self.layout.chal_table, self.shape.chals)
        if len(self.shape.wires) > 0:
            _upload(ctx, self.arena, self.layout.wires, self.shape.wires)
            _upload(ctx, self.arena, self.layout.sigma, self.shape.sigma)
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

    def prove(mut self, ctx: DeviceContext, public_inputs: Span[UInt8, _], profile: Bool = False) raises -> List[UInt8]:
        """Spec section 10 in order. The trace must already be in the arena at layout.enc_w.trace.
        Every call is an enqueue; proof values are staged as async copies and read after the one
        synchronize in `proof.finish`. `profile` inserts a synchronize after every stage and records
        the stage times in profile_names / profile_ms (a measurement mode, never the production path)."""
        self.profile_names = List[String]()
        self.profile_ms = List[Int]()
        var t0 = perf_counter_ns()
        comptime N = Self.p.N()
        comptime e = Self.p.e
        ref L = self.layout
        var T = L.transcript
        ref S = self.shape
        var row_w = 4 * Self.p.n_cw() * S.columns_w
        var row_z = 4 * Self.p.n_cw() * S.columns_z
        var row_q = 4 * Self.p.n_cw() * S.columns_q
        self.proof.reset()

        # header and transcript prefix (spec 9.4, statement-layer 6 step 1)
        self.proof.u32(Int(VERSION))
        self.proof.prefixed(public_inputs)
        var prefix = prefix_bytes[Self.p, Self.H](S, public_inputs, self.families)
        if len(prefix) > PREFIX_MAX:
            raise Error("public inputs too large for the prefix region")
        var prefix_host = self.proof.scratch(len(prefix))
        for i in range(len(prefix)):
            prefix_host[i] = prefix[i]
        self.arena.upload(ctx, L.prefix, prefix_host)
        reset(ctx, self.arena, T)
        absorb[Self.p, Self.H](ctx, self.arena, T, DS_PREFIX, L.prefix, len(prefix))
        self._mark(ctx, profile, "prefix", t0)

        # 2. the sorted copies (spec 6.3): the lookup descriptors' s columns are witness columns, filled before W is committed
        var li = 0
        for k in range(S.accumulators()):
            if Int(S.accs[k * ACC + 38]) == KIND_LOOKUP:
                counting_sort[Self.p](ctx, self.arena, L.enc_w.trace, L.accs + k * ACC, L.idx + li * N * 4, L.bins, L.cursor, S.table_rows(k))
                li += 1
        if li > 0:
            self._mark(ctx, profile, "sort", t0)

        # 3. commit W -> stage-1 challenges
        encode[Self.p](ctx, self.arena, L.enc_w, L.tables)
        self._mark(ctx, profile, "encode W", t0)
        merkle[Self.p, Self.H](ctx, self.arena, L.enc_w.code, row_w, Self.p.L(), L.tree_w)
        self._mark(ctx, profile, "merkle W", t0)
        absorb[Self.p, Self.H](ctx, self.arena, T, DS_TREE_W, root_offset[Self.H](L.tree_w, Self.p.L()), Self.H.DIGEST)
        self.proof.stage(self.arena, root_offset[Self.H](L.tree_w, Self.p.L()), Self.H.DIGEST)
        squeeze_elements[Self.p, Self.H](ctx, self.arena, T, L.stage1, 3)             # beta, delta, gamma
        derive_chals(ctx, self.arena, L.stage1, L.chal_table, len(S.chals) // CHAL)
        if S.wiring_products() > 0:
            squeeze_elements[Self.p, Self.H](ctx, self.arena, T, L.wchal, 2)          # beta_w, gamma_w
        self._mark(ctx, profile, "transcript W", t0)

        # 4-7. the Z stage and commit Z with Z2 -> alpha
        var pi = 0                                       # product index: Z2, n_end, d_end are per grand product
        for k in range(S.accumulators()):
            if Int(S.accs[k * ACC + 38]) == KIND_HORNER:
                horner[Self.p](ctx, self.arena, L.enc_w.trace, L.families, L.accs + k * ACC, L.stage1, L.num, L.zval + k * N * e)
                continue
            accumulate[Self.p](ctx, self.arena, L.enc_w.trace, L.accs + k * ACC, L.stage1, L.num, L.den, L.zscratch,
                               L.zval + k * N * e, L.chain_prod, L.z2 + pi * Self.p.h2() * e, L.n_end + pi * Self.p.h2() * e, L.d_end + pi * Self.p.h2() * e)
            pi += 1
        comptime h2 = Self.p.h2()
        for g in range(S.wiring_products()):
            wiring[Self.p](ctx, self.arena, L.zval, L.wires + g * WIRE, L.sigma, 2 * g, S.columns_w, L.wchal,
                           ext_pow[1](self.kappa, 2 * g), ext_pow[1](self.kappa, 2 * g + 1), self.domains.omega2,
                           L.chain_prod, L.wlines + g * 4 * h2 * e, L.z2 + pi * h2 * e)
            pi += 1
        if S.columns_z > 0:
            ctx.enqueue_function[k_values_to_trace[Self.p]](self.arena.buf, Buf[1](L.zval), Buf[1](L.enc_z.trace), Int32(S.accumulators()),
                                                            grid_dim=ceildiv(S.columns_z * N, BACKEND.block), block_dim=BACKEND.block)
            self._mark(ctx, profile, "accumulate", t0)
            encode[Self.p](ctx, self.arena, L.enc_z, L.tables)
            self._mark(ctx, profile, "encode Z", t0)
            merkle[Self.p, Self.H](ctx, self.arena, L.enc_z.code, row_z, Self.p.L(), L.tree_z)
            self._mark(ctx, profile, "merkle Z", t0)
            absorb[Self.p, Self.H](ctx, self.arena, T, DS_TREE_Z, root_offset[Self.H](L.tree_z, Self.p.L()), Self.H.DIGEST)
            self.proof.stage(self.arena, root_offset[Self.H](L.tree_z, Self.p.L()), Self.H.DIGEST)
            if S.products() > 0:
                absorb[Self.p, Self.H](ctx, self.arena, T, DS_TREE_Z, L.z2, S.products() * Self.p.h2() * e)
                self.proof.stage(self.arena, L.z2, S.products() * Self.p.h2() * e)
        squeeze_elements[Self.p, Self.H](ctx, self.arena, T, L.alpha, 1)
        self._mark(ctx, profile, "transcript Z", t0)

        # 7.4: the small grid, Q3 in the clear (sent with the Q root)
        var e2 = ext_pow[1](self.domains.omega2, h2 - 1)
        pi = 0
        for k in range(S.accumulators()):
            if Int(S.accs[k * ACC + 38]) == KIND_HORNER:
                continue
            small_grid_product[Self.p](ctx, self.arena, L.tables, L.z2 + pi * h2 * e,
                                       [(L.zval + k * N * e + (Self.p.h1() - 1) * e, Self.p.h1() * e), (L.n_end + pi * h2 * e, 16)],
                                       [(L.d_end + pi * h2 * e, 16)], L.sg, L.alpha, S.family_of(k), e2, pi == 0)
            pi += 1
        for g in range(S.wiring_products()):
            var wl = L.wlines + g * 4 * h2 * e
            small_grid_product[Self.p](ctx, self.arena, L.tables, L.z2 + pi * h2 * e, [(wl, 16), (wl + h2 * e, 16)],
                                       [(wl + 2 * h2 * e, 16), (wl + 3 * h2 * e, 16)], L.sg, L.alpha, get_u16(S.wires, g * WIRE + 4), e2, pi == 0)
            pi += 1
        for i in range(len(S.ends) // END):
            small_grid_end[Self.p](ctx, self.arena, L.tables, L.zval, S.columns_w, S.ends, i, L.sg, L.alpha, L.stage1, e2, pi == 0 and i == 0)
        if S.accumulators() > 0:
            small_grid_values[Self.p](ctx, self.arena, L.tables, L.sg, L.q3)
            self._mark(ctx, profile, "small grid", t0)

        # 8-10. residual grid, quotient, commit Q
        lde[Self.p](ctx, self.arena, L.enc_w.coeff, S.columns_w, L.tables, L.ltmp, L.lde)
        if S.columns_z > 0:
            lde[Self.p](ctx, self.arena, L.enc_z.coeff, S.columns_z, L.tables, L.ltmp, L.lde + S.columns_w * 4 * N * 2)
        if S.columns_p > 0:
            lde[Self.p](ctx, self.arena, L.pub_coeff, S.columns_p, L.tables, L.ltmp, L.lde + (S.columns_w + S.columns_z) * 4 * N * 2)
        self._mark(ctx, profile, "lde", t0)
        residual[Self.p](ctx, self.arena, L.lde, L.families, S.entries, L.tables, L.alpha, L.stage1, L.residual,
                         L.families_g, self.entries_g, L.accs, len(S.accs) // ACC, L.merge)
        self._mark(ctx, profile, "residual", t0)
        quotient[Self.p](ctx, self.arena, L.residual, L.tables, L.quotient, L.enc_q.trace)
        self._mark(ctx, profile, "quotient", t0)
        encode[Self.p](ctx, self.arena, L.enc_q, L.tables)
        self._mark(ctx, profile, "encode Q", t0)
        merkle[Self.p, Self.H](ctx, self.arena, L.enc_q.code, row_q, Self.p.L(), L.tree_q)
        self._mark(ctx, profile, "merkle Q", t0)
        absorb[Self.p, Self.H](ctx, self.arena, T, DS_TREE_Q, root_offset[Self.H](L.tree_q, Self.p.L()), Self.H.DIGEST)
        self.proof.stage(self.arena, root_offset[Self.H](L.tree_q, Self.p.L()), Self.H.DIGEST)
        if S.accumulators() > 0:
            absorb[Self.p, Self.H](ctx, self.arena, T, DS_TREE_Q, L.q3, 2 * h2 * e)
            self.proof.stage(self.arena, L.q3, 2 * h2 * e)
        squeeze_elements[Self.p, Self.H](ctx, self.arena, T, L.z, 2)                  # z = (z1, z2)
        self._mark(ctx, profile, "transcript Q", t0)

        # 11. openings at the P points
        build_queries[Self.p](ctx, self.arena, L.z, L.shifts, S.points, L.tables, self.domains, L.w_tab, L.w_z)
        self._mark(ctx, profile, "build_queries", t0)
        open[Self.p](ctx, self.arena, L.w_z, S.points, L.enc_w.stored, S.columns_w, L.open_partial, L.openings, S.columns())
        if S.columns_z > 0:
            open[Self.p](ctx, self.arena, L.w_z, S.points, L.enc_z.stored, S.columns_z, L.open_partial, L.openings + S.columns_w * e, S.columns())
        open[Self.p](ctx, self.arena, L.w_z, S.points, L.enc_q.stored, S.columns_q, L.open_partial, L.openings + (S.columns_w + S.columns_z) * e, S.columns())
        self._mark(ctx, profile, "open", t0)
        absorb[Self.p, Self.H](ctx, self.arena, T, DS_OPENINGS, L.openings, S.points * S.columns() * e)
        self.proof.stage(self.arena, L.openings, S.points * S.columns() * e)
        squeeze_elements[Self.p, Self.H](ctx, self.arena, T, L.beta_gamma, S.columns() + S.points)
        self._mark(ctx, profile, "transcript openings", t0)

        # 12. fold to the level-2 message
        fold[Self.p, False](ctx, self.arena, L.beta_gamma, L.enc_w.stored, S.columns_w, L.fold_y)
        if S.columns_z > 0:
            fold[Self.p, True](ctx, self.arena, L.beta_gamma + S.columns_w * e, L.enc_z.stored, S.columns_z, L.fold_y)
        fold[Self.p, True](ctx, self.arena, L.beta_gamma + (S.columns_w + S.columns_z) * e, L.enc_q.stored, S.columns_q, L.fold_y)
        self._mark(ctx, profile, "fold", t0)

        # 13. tail: each committed level opens the previous one
        var y = L.fold_y
        var y_len = N
        var running = L.running0
        if len(S.tail) > 0:
            running0[Self.p](ctx, self.arena, L.w_z, L.beta_gamma + S.columns() * e, S.points, L.running0)
        for i in range(len(S.tail)):
            var lvl = S.tail[i]
            var tl = L.tail[i]
            tail_encode(ctx, self.arena, y, lvl.rows, lvl.L // lvl.cosets, lvl.cosets, tl.etmp, tl.code, tl.rs)
            self._mark(ctx, profile, "tail encode " + String(i), t0)
            merkle[Self.p, Self.H](ctx, self.arena, tl.code, 8 * e, lvl.L, tl.tree)
            self._mark(ctx, profile, "tail merkle " + String(i), t0)
            absorb[Self.p, Self.H](ctx, self.arena, T, DS_TAIL_ROOT, root_offset[Self.H](tl.tree, lvl.L), Self.H.DIGEST)
            self.proof.stage(self.arena, root_offset[Self.H](tl.tree, lvl.L), Self.H.DIGEST)
            self._open_previous(ctx, i, T)
            self._mark(ctx, profile, "open previous " + String(i), t0)
            var count = self._prev_queries(i)
            var prev_dom = L.dom1 if i == 0 else L.tail[i - 1].dom
            var prev_L0 = Self.p.L0 if i == 0 else S.tail[i - 1].L // S.tail[i - 1].cosets
            points(ctx, self.arena, L.positions, count, prev_dom, prev_L0, L.pts)
            # the expected symbols v are the verifier's to compute from the opened rows (spec 9.3); nothing is sent
            squeeze_elements[Self.p, Self.H](ctx, self.arena, T, L.batch, self._v_count(i) + 1)   # batching scalars
            self._mark(ctx, profile, "transcript batch " + String(i), t0)
            tail_materialize[Self.p](ctx, self.arena, i == 0, running, L.batch, L.pts, count, y_len, tl.w_tilde, L.ptab)
            self._mark(ctx, profile, "materialize " + String(i), t0)
            for d in range(3):
                tail_round(ctx, self.arena, tl.w_tilde, y, y_len, d, L.r, L.partial, tl.rounds + d * 3 * e)
                absorb[Self.p, Self.H](ctx, self.arena, T, DS_TAIL_ROUND, tl.rounds + d * 3 * e, 3 * e)
                squeeze_elements[Self.p, Self.H](ctx, self.arena, T, L.r + d * e, 1)          # r_d
            self.proof.stage(self.arena, tl.rounds, 9 * e)
            self._mark(ctx, profile, "rounds " + String(i), t0)
            tail_fold(ctx, self.arena, y, lvl.rows, L.r, tl.y)
            tail_fold(ctx, self.arena, tl.w_tilde, lvl.rows, L.r, tl.running)
            self._mark(ctx, profile, "fold " + String(i), t0)
            y = tl.y
            y_len = lvl.rows
            running = tl.running

        # last: the clear vector, then open the last committed level
        absorb[Self.p, Self.H](ctx, self.arena, T, DS_CLEAR, y, y_len * e)
        self.proof.stage(self.arena, y, y_len * e)
        self._mark(ctx, profile, "transcript clear", t0)
        self._open_previous(ctx, len(S.tail), T)
        self._mark(ctx, profile, "open last", t0)
        var out = self.proof.finish()
        self._mark(ctx, profile, "finish", t0)
        return out^

    def _prev_queries(self, i: Int) -> Int:
        return Self.p.queries() if i == 0 else self.shape.tail[i - 1].queries

    def _v_count(self, i: Int) -> Int:
        return 4 * Self.p.n_cw() * Self.p.queries() if i == 0 else self.shape.tail[i - 1].queries

    def _open_previous(mut self, ctx: DeviceContext, i: Int, T: TranscriptLayout) raises:
        """Sample S on the level before tail level i (level 1 when i == 0), gather its multiproof(s),
        and stage them. The stage region is reused: the copy out is enqueued before the next gather."""
        ref L = self.layout
        ref S = self.shape
        if i == 0:
            squeeze_positions[Self.p, Self.H](ctx, self.arena, T, L.positions, Self.p.queries(), Self.p.L())
            for tree in [(L.enc_w.code, S.columns_w, L.tree_w), (L.enc_z.code, S.columns_z, L.tree_z), (L.enc_q.code, S.columns_q, L.tree_q)]:
                if tree[1] == 0:
                    continue                                    # no Z tree without accumulators
                var bound = query_gather[Self.p, Self.H](ctx, self.arena, tree[0], 4 * Self.p.n_cw() * tree[1], Self.p.L(),
                                                         tree[2], L.positions, Self.p.queries(), L.proof_stage)
                self.proof.stage(self.arena, L.proof_stage, bound, multiproof=True)
        else:
            var lvl = S.tail[i - 1]
            squeeze_positions[Self.p, Self.H](ctx, self.arena, T, L.positions, lvl.queries, lvl.L)
            var bound = query_gather[Self.p, Self.H](ctx, self.arena, L.tail[i - 1].code, 8 * Self.p.e, lvl.L, L.tail[i - 1].tree,
                                                     L.positions, lvl.queries, L.proof_stage)
            self.proof.stage(self.arena, L.proof_stage, bound, multiproof=True)


def proof_pool_bytes[p: Params, H: Hash](shape: Shape) -> Int:
    """Host staging for one proof: the fixed bytes at the largest prefix, every multiproof region
    bound, and the transcript prefix upload."""
    var n = shape.fixed_bytes[p, H.DIGEST](PREFIX_MAX) + PREFIX_MAX
    n += multiproof_region[H](4 * p.n_cw() * shape.columns_w, p.L(), p.queries())
    if shape.columns_z > 0:
        n += multiproof_region[H](4 * p.n_cw() * shape.columns_z, p.L(), p.queries())
    n += multiproof_region[H](4 * p.n_cw() * shape.columns_q, p.L(), p.queries())
    for lvl in shape.tail:
        n += multiproof_region[H](8 * p.e, lvl.L, lvl.queries)
    return n


def _put_u16(mut h: HostBuffer[DType.uint8], at: Int, v: Int):
    h[at] = UInt8(v & 255)
    h[at + 1] = UInt8(v >> 8)


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
    prover.arena.upload(ctx, prover.layout.enc_w.trace, prover.trace_host)


def load_advice[p: Params, H: Hash](ctx: DeviceContext, mut prover: Prover[p, H], advice: Span[UInt8, _]) raises:
    """Upload the advice indices: one u32 per row per lookup descriptor, in descriptor order (sort.mojo)."""
    if len(advice) != prover.shape.lookups() * p.N() * 4:
        raise Error("advice has the wrong size")
    _upload(ctx, prover.arena, prover.layout.idx, advice)


def load_public[p: Params, H: Hash](ctx: DeviceContext, mut prover: Prover[p, H], values: Span[UInt8, _]) raises:
    """Upload the public columns: one period of (h2 / m, h1) F values per column in order (docs/public-columns.md),
    tiled to H and transformed to the coefficients the LDE reads with the trace's `idft2` (ltmp as scratch)."""
    if len(values) != value_bytes(prover.shape.publics, p.h1(), p.h2()):
        raise Error("public values have the wrong size")
    for v in values:
        if v >= 127:
            raise Error("public values are F bytes below 127")
    _upload(ctx, prover.arena, prover.layout.pub_vals, tile_values(prover.shape.publics, values, p.h1(), p.h2()))
    if prover.shape.columns_p > 0:
        idft2[p](ctx, prover.arena, prover.layout.pub_vals, prover.layout.ltmp, prover.layout.pub_coeff, prover.shape.columns_p, prover.layout.tables)

"""Host orchestration (docs/design.md section 6): plan the arena once, then enqueue every stage of
spec section 10 in order on one stream. One synchronize at the end reads the proof bytes.

Milestone 1: W tree and Q tree on synthetic columns, residual and quotient on synthetic families,
openings at P points, the tail, and the clear vector. No Z tree (milestone 2), no frontend.

A stage that does not exist yet raises "not implemented: <kernel>" at the point where it would be
enqueued. Nothing fakes an output; the first end-to-end proof appears when the last raise is gone.
"""

from max.gpu.host import DeviceContext

from caracal7.params import Params
from caracal7.arena import Arena, Bump
from caracal7.tables import Domains, TableLayout, build_tables
from caracal7.encode import EncLayout, encode, rs_encode
from caracal7.transcript import TranscriptLayout, absorb, squeeze_elements, squeeze_positions
from caracal7.transcript import DS_PREFIX, DS_TREE_W, DS_TREE_Q, DS_OPENINGS, DS_TAIL_ROOT, DS_TAIL_V, DS_TAIL_ROUND, DS_CLEAR
from caracal7.proof import Shape, ProofWriter, TailLevel, VERSION
from caracal7.hash import Hash
from caracal7.merkle import merkle, query_gather, root_offset, tree_nodes, multiproof_region
from caracal7.residual import lde, residual, quotient
from caracal7.open import build_queries, open, fold
from caracal7.tail import tail_encode, expected_symbols, tail_materialize, tail_round, tail_fold



struct TailLayout(TrivialRegisterPassable):
    """Arena offsets of one committed tail level (spec 9.3), all E-valued."""
    var y: Int          # (slot, e)          message y_l
    var code: Int       # (s, 8, e)          leaf-major codeword rows
    var tree: Int       # (node, 32)
    var v: Int          # expected symbols for the previous level
    var w_tilde: Int    # (slot, e)          batched query
    var rounds: Int     # (3, 3, e)          sumcheck messages

    def __init__[p: Params, H: Hash](out self, mut bump: Bump, lvl: TailLevel, v_count: Int):
        self.y = bump.alloc(lvl.length * p.e)
        self.code = bump.alloc(lvl.L * 8 * p.e)
        self.tree = bump.alloc(tree_nodes(lvl.L) * H.DIGEST)
        self.v = bump.alloc(v_count * p.e)
        self.w_tilde = bump.alloc(lvl.length * p.e)
        self.rounds = bump.alloc(9 * p.e)


struct ProverLayout:
    """Every buffer of design section 3 as an arena offset. Built once per (Params, Shape)."""
    var tables: TableLayout
    var transcript: TranscriptLayout
    var enc_w: EncLayout            # witness tree: trace .. code
    var enc_q: EncLayout            # quotient tree: stored .. code (trace/coeff unused; quotient writes stored)
    var tree_w: Int                 # (node, 32), level 0 first, tree_nodes(L0) nodes
    var tree_q: Int
    var lde: Int                    # (column, G2, G1, 2)   witness columns on the residual grid
    var residual: Int               # (G2, G1, e)
    var quotient: Int               # (3, G2, G1, e)        A, B, Q2 in evaluation form
    var w_z: Int                    # (P, slot, e)          evaluation queries
    var openings: Int               # (P, column, e)
    var fold_y: Int                 # (slot, e)             y = sum beta_c stored(c), the level-2 message
    var positions: Int              # (queries, u32)        S_1
    var proof_stage: Int            # gathered rows and siblings, read back as proof bytes
    var tail: List[TailLayout]
    var bytes: Int

    def __init__[p: Params, H: Hash](out self, shape: Shape) raises:
        comptime N = p.N()
        comptime G = 4 * N
        var bump = Bump()
        self.tables = TableLayout.__init__[p](bump.alloc(0))
        _ = bump.alloc(self.tables.bytes)
        var largest = max(shape.columns() * p.e, p.queries() * 4)
        self.transcript = TranscriptLayout(bump, largest)
        self.enc_w = EncLayout.__init__[p](bump, shape.columns_w)
        self.enc_q = EncLayout.__init__[p](bump, shape.columns_q)
        self.tree_w = bump.alloc(tree_nodes(p.L()) * H.DIGEST)
        self.tree_q = bump.alloc(tree_nodes(p.L()) * H.DIGEST)
        self.lde = bump.alloc(shape.columns_w * G * 2)
        self.residual = bump.alloc(G * p.e)
        self.quotient = bump.alloc(3 * G * p.e)
        self.w_z = bump.alloc(shape.points * N * p.e)
        self.openings = bump.alloc(shape.points * shape.columns() * p.e)
        self.fold_y = bump.alloc(N * p.e)
        self.positions = bump.alloc(p.queries() * 4)
        self.proof_stage = bump.alloc(multiproof_region[H](4 * p.n_cw() * shape.columns_w, p.L(), p.queries()))
        self.tail = List[TailLayout]()
        for i in range(len(shape.tail)):
            var prev_q = p.queries() if i == 0 else shape.tail[i - 1].queries
            var v_count = 4 * p.n_cw() * prev_q if i == 0 else prev_q
            self.tail.append(TailLayout.__init__[p, H](bump, shape.tail[i], v_count))
        self.bytes = bump.used


struct Prover[p: Params, H: Hash]:
    var shape: Shape
    var layout: ProverLayout
    var arena: Arena
    var domains: Domains

    def __init__(out self, ctx: DeviceContext, var shape: Shape) raises:
        self.shape = shape^
        self.layout = ProverLayout.__init__[Self.p, Self.H](self.shape)
        self.arena = Arena(ctx, self.layout.bytes)
        self.domains = Domains.__init__[Self.p]()
        self.arena.upload(ctx, self.layout.tables.base, build_tables[Self.p](ctx, self.layout.tables, self.domains))

    def prove(self, ctx: DeviceContext, public_inputs: List[UInt8]) raises -> List[UInt8]:
        """Spec section 10 in order. The trace must already be in the arena at layout.enc_w.trace.
        Every call is an enqueue; the only host reads are the proof bytes at the end."""
        comptime N = Self.p.N()
        var base = self.arena.base()
        ref L = self.layout
        var T = L.transcript
        ref S = self.shape
        var row_w = 4 * Self.p.n_cw() * S.columns_w
        var row_q = 4 * Self.p.n_cw() * S.columns_q
        var proof = ProofWriter()

        # header and transcript prefix (statement-layer 6 step 1)
        proof.u32(Int(VERSION))
        proof.prefixed(public_inputs)
        absorb[Self.p, Self.H](ctx, base, T, DS_PREFIX, L.tables.base, L.tables.bytes)   # ponytail: the tables stand in for the params digest

        # 3. commit W
        encode[Self.p](ctx, base, L.enc_w, L.tables)
        merkle[Self.p, Self.H](ctx, base, L.enc_w.code, row_w, Self.p.L(), L.tree_w)
        absorb[Self.p, Self.H](ctx, base, T, DS_TREE_W, root_offset[Self.H](L.tree_w, Self.p.L()), Self.H.DIGEST)
        squeeze_elements[Self.p, Self.H](ctx, base, T, 3)                        # beta_1, delta, gamma (unused in milestone 1)

        # 8-10. residual grid, quotient, commit Q
        lde[Self.p](ctx, base, L.enc_w.coeff, S.columns_w, L.lde)
        residual[Self.p](ctx, base, L.lde, S.columns_w, L.tables.base, T.challenges, L.residual)
        quotient[Self.p](ctx, base, L.residual, L.quotient, L.enc_q.stored)
        rs_encode[Self.p](ctx, base, L.enc_q, L.tables)                       # pack is part of the quotient write
        merkle[Self.p, Self.H](ctx, base, L.enc_q.code, row_q, Self.p.L(), L.tree_q)
        absorb[Self.p, Self.H](ctx, base, T, DS_TREE_Q, root_offset[Self.H](L.tree_q, Self.p.L()), Self.H.DIGEST)
        squeeze_elements[Self.p, Self.H](ctx, base, T, 2)                        # z = (z1, z2)

        # 11. openings at the P points
        build_queries[Self.p](ctx, base, T.challenges, S.points, L.w_z)
        open[Self.p](ctx, base, L.w_z, S.points, L.enc_w.stored, S.columns_w, False, L.openings)
        open[Self.p](ctx, base, L.w_z, S.points, L.enc_q.stored, S.columns_q, True, L.openings + S.points * S.columns_w * Self.p.e)
        absorb[Self.p, Self.H](ctx, base, T, DS_OPENINGS, L.openings, S.points * S.columns() * Self.p.e)
        squeeze_elements[Self.p, Self.H](ctx, base, T, S.columns() + S.points)   # beta per column, gamma per point

        # 12. fold to the level-2 message
        fold[Self.p](ctx, base, T.challenges, L.enc_w.stored, S.columns_w, L.enc_q.stored, S.columns_q, L.fold_y)

        # 13. tail: each committed level opens the previous one
        var y = L.fold_y
        var y_len = N
        for i in range(len(S.tail)):
            var lvl = S.tail[i]
            var tl = L.tail[i]
            tail_encode[Self.p](ctx, base, y, lvl.rows, lvl.L, tl.code)
            merkle[Self.p, Self.H](ctx, base, tl.code, 8 * Self.p.e, lvl.L, tl.tree)
            absorb[Self.p, Self.H](ctx, base, T, DS_TAIL_ROOT, root_offset[Self.H](tl.tree, lvl.L), Self.H.DIGEST)
            self._open_previous(ctx, i, T, proof)
            expected_symbols[Self.p](ctx, base, y, L.positions, self._prev_queries(i), tl.v)
            absorb[Self.p, Self.H](ctx, base, T, DS_TAIL_V, tl.v, self._v_count(i) * Self.p.e)
            squeeze_elements[Self.p, Self.H](ctx, base, T, self._v_count(i) + 1)   # batching scalars
            tail_materialize[Self.p](ctx, base, y_len, T.challenges, tl.w_tilde)
            for d in range(3):
                tail_round[Self.p](ctx, base, tl.w_tilde, y, y_len, d, tl.rounds + d * 3 * Self.p.e)
                absorb[Self.p, Self.H](ctx, base, T, DS_TAIL_ROUND, tl.rounds + d * 3 * Self.p.e, 3 * Self.p.e)
                squeeze_elements[Self.p, Self.H](ctx, base, T, 1)                  # r_d
            tail_fold[Self.p](ctx, base, y, lvl.rows, T.challenges, tl.y)
            y = tl.y
            y_len = lvl.rows

        # last: the clear vector, then open the last committed level
        absorb[Self.p, Self.H](ctx, base, T, DS_CLEAR, y, y_len * Self.p.e)
        self._open_previous(ctx, len(S.tail), T, proof)
        ctx.synchronize()
        raise Error("not implemented: proof serialization")   # read back roots, openings, v, rounds, clear vector

    def _prev_queries(self, i: Int) -> Int:
        return Self.p.queries() if i == 0 else self.shape.tail[i - 1].queries

    def _v_count(self, i: Int) -> Int:
        return 4 * Self.p.n_cw() * Self.p.queries() if i == 0 else self.shape.tail[i - 1].queries

    def _open_previous(self, ctx: DeviceContext, i: Int, T: TranscriptLayout, mut proof: ProofWriter) raises:
        """Sample S on the level before tail level i (level 1 when i == 0) and gather its multiproof."""
        var base = self.arena.base()
        ref L = self.layout
        ref S = self.shape
        if i == 0:
            squeeze_positions[Self.p, Self.H](ctx, base, T, Self.p.queries(), Self.p.L())
            _ = query_gather[Self.p, Self.H](ctx, base, L.enc_w.code, 4 * Self.p.n_cw() * S.columns_w, Self.p.L(), L.tree_w,
                                   T.challenges, Self.p.queries(), L.proof_stage)
            _ = query_gather[Self.p, Self.H](ctx, base, L.enc_q.code, 4 * Self.p.n_cw() * S.columns_q, Self.p.L(), L.tree_q,
                                   T.challenges, Self.p.queries(), L.proof_stage)
        else:
            var lvl = S.tail[i - 1]
            squeeze_positions[Self.p, Self.H](ctx, base, T, lvl.queries, lvl.L)
            _ = query_gather[Self.p, Self.H](ctx, base, L.tail[i - 1].code, 8 * Self.p.e, lvl.L, L.tail[i - 1].tree,
                                   T.challenges, lvl.queries, L.proof_stage)


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

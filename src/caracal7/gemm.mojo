"""Byte GEMM mod 127 on SIMD lanes: the ladder climb of docs/design.md section 8 on the plain F case.

C[M, N] = A[M, K] . B[K, N] over F127, one byte per element, int32 accumulation,
lazy reduction (field.f_reduce) every REDUCE_EVERY products. The register-tile update is
backend.tile_mac, the same op the F2 skeleton (backend.gemm_f2) runs.

Three kernels, rungs 1, 4 and 5 of the ladder in design.md section 8:
    gemm127_naive   one thread per output
    gemm127_tiled   threadgroup tiles BM x BK, BK x BN; each thread owns a TM x TN register tile
    gemm127_vec     as tiled, plus 4-byte vector loads and a transposed A tile so the
                    register loads are vectors too (Boehm kernel 6). Row-major operands only.
"""

from std.math import ceildiv
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx
from max.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, row_major, stack_allocation

from caracal7.field import f_reduce
from caracal7.backend import BACKEND, Tile, tile_mac

comptime REDUCE_EVERY = BACKEND.max_terms   # 128 * 126^2 < 2^21, the f_reduce input bound
comptime TILE_DEFAULT = BACKEND.tile


@always_inline
def _reduce_acc(acc: Int32) -> Int32:
    return f_reduce(SIMD[DType.uint32, 1](acc.cast[DType.uint32]()))[0].cast[DType.int32]()


def gemm127_naive[
    M: Int, N: Int, K: Int,
    AL: TensorLayout, BL: TensorLayout, CL: TensorLayout,
](
    A: TileTensor[DType.uint8, AL, MutAnyOrigin],
    B: TileTensor[DType.uint8, BL, MutAnyOrigin],
    C: TileTensor[DType.uint8, CL, MutAnyOrigin],
):
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and C.flat_rank == 2
    var col = block_idx.x * 16 + thread_idx.x      # consecutive threads walk N: coalesced on B and C
    var row = block_idx.y * 16 + thread_idx.y
    if row >= M or col >= N:
        return
    var acc: Int32 = 0
    comptime for k in range(K):
        acc += rebind[Scalar[DType.uint8]](A[row, k]).cast[DType.int32]() * rebind[Scalar[DType.uint8]](B[k, col]).cast[DType.int32]()
        comptime if k % REDUCE_EVERY == REDUCE_EVERY - 1:
            acc = _reduce_acc(acc)
    C[row, col] = rebind[C.ElementType](_reduce_acc(acc).cast[DType.uint8]())


def gemm127_tiled[
    M: Int, N: Int, K: Int, T: Tile,
    AL: TensorLayout, BL: TensorLayout, CL: TensorLayout,
](
    A: TileTensor[DType.uint8, AL, MutAnyOrigin],
    B: TileTensor[DType.uint8, BL, MutAnyOrigin],
    C: TileTensor[DType.uint8, CL, MutAnyOrigin],
):
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and C.flat_rank == 2
    comptime assert M % T.BM == 0 and N % T.BN == 0 and K % T.BK == 0, "tile must divide the problem"
    comptime assert REDUCE_EVERY % T.BK == 0, "the lazy reduction cadence needs BK | REDUCE_EVERY"
    comptime BM = T.BM
    comptime BN = T.BN
    comptime BK = T.BK
    comptime TM = T.TM
    comptime TN = T.TN
    comptime THREADS = T.threads()
    comptime assert (BM * BK) % THREADS == 0 and (BK * BN) % THREADS == 0

    var tid = Int(thread_idx.x)
    var trow = tid // (BN // TN)         # this thread's tile row inside the block
    var tcol = tid % (BN // TN)
    var brow = Int(block_idx.y) * BM
    var bcol = Int(block_idx.x) * BN

    var As = stack_allocation[DType.uint8, address_space=AddressSpace.SHARED](row_major[BK, BM]())   # transposed
    var Bs = stack_allocation[DType.uint8, address_space=AddressSpace.SHARED](row_major[BK, BN]())
    var acc = InlineArray[SIMD[DType.int32, TN], TM](fill=SIMD[DType.int32, TN](0))   # register tile

    for kt in range(K // BK):
        # cooperative loads: consecutive threads take consecutive columns (coalesced)
        comptime for i in range(0, BM * BK, THREADS):
            var idx = i + tid
            As[idx % BK, idx // BK] = A[brow + idx // BK, kt * BK + idx % BK]
        comptime for i in range(0, BK * BN, THREADS):
            var idx = i + tid
            Bs[idx // BN, idx % BN] = B[kt * BK + idx // BN, bcol + idx % BN]
        barrier()
        comptime for k in range(BK):
            var a = As.ptr.unsafe_load[width=TM](k * BM + trow * TM).cast[DType.int32]()
            var b = Bs.ptr.unsafe_load[width=TN](k * BN + tcol * TN).cast[DType.int32]()
            tile_mac[BACKEND](a, b, acc)
        barrier()
        if (kt * BK) % REDUCE_EVERY >= REDUCE_EVERY - BK:
            comptime for i in range(TM):
                acc[i] = f_reduce(acc[i].cast[DType.uint32]()).cast[DType.int32]()

    comptime for i in range(TM):
        var out = f_reduce(acc[i].cast[DType.uint32]())
        comptime for j in range(TN):
            C[brow + trow * TM + i, bcol + tcol * TN + j] = rebind[C.ElementType](out[j])


def gemm127_vec[
    M: Int, N: Int, K: Int, T: Tile,
    AL: TensorLayout, BL: TensorLayout, CL: TensorLayout,
](
    A: TileTensor[DType.uint8, AL, MutAnyOrigin],
    B: TileTensor[DType.uint8, BL, MutAnyOrigin],
    C: TileTensor[DType.uint8, CL, MutAnyOrigin],
):
    """Operands are addressed through raw pointers with row-major strides K, N, N."""
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and C.flat_rank == 2
    comptime assert M % T.BM == 0 and N % T.BN == 0 and K % T.BK == 0, "tile must divide the problem"
    comptime assert REDUCE_EVERY % T.BK == 0, "the lazy reduction cadence needs BK | REDUCE_EVERY"
    comptime BM = T.BM
    comptime BN = T.BN
    comptime BK = T.BK
    comptime TM = T.TM
    comptime TN = T.TN
    comptime V = BACKEND.vec_bytes              # bytes per vector load
    comptime THREADS = T.threads()
    comptime assert BK % V == 0 and BN % V == 0 and TM % V == 0 and TN % V == 0
    comptime assert (BM * BK // V) % THREADS == 0 and (BK * BN // V) % THREADS == 0

    var tid = Int(thread_idx.x)
    var trow = tid // (BN // TN)
    var tcol = tid % (BN // TN)
    var brow = Int(block_idx.y) * BM
    var bcol = Int(block_idx.x) * BN

    var As = stack_allocation[DType.uint8, address_space=AddressSpace.SHARED](row_major[BK, BM]())   # transposed
    var Bs = stack_allocation[DType.uint8, address_space=AddressSpace.SHARED](row_major[BK, BN]())
    var acc = InlineArray[SIMD[DType.int32, TN], TM](fill=SIMD[DType.int32, TN](0))   # one row vector per i: keeps Metal from seeing a TM*TN-lane vector

    for kt in range(K // BK):
        comptime for i in range(0, BM * BK // V, THREADS):
            var idx = i + tid
            var r = idx // (BK // V)
            var k0 = (idx % (BK // V)) * V
            var v = A.ptr.unsafe_load[width=V]((brow + r) * K + kt * BK + k0)
            comptime for j in range(V):                # scatter into the transposed tile
                As.ptr.unsafe_store((k0 + j) * BM + r, v[j])
        comptime for i in range(0, BK * BN // V, THREADS):
            var idx = i + tid
            var k = idx // (BN // V)
            var n0 = (idx % (BN // V)) * V
            Bs.ptr.unsafe_store[width=V](k * BN + n0, B.ptr.unsafe_load[width=V]((kt * BK + k) * N + bcol + n0))
        barrier()
        comptime for k in range(BK):
            var a = As.ptr.unsafe_load[width=TM](k * BM + trow * TM).cast[DType.int32]()
            var b = Bs.ptr.unsafe_load[width=TN](k * BN + tcol * TN).cast[DType.int32]()
            tile_mac[BACKEND](a, b, acc)
        barrier()
        if (kt * BK) % REDUCE_EVERY >= REDUCE_EVERY - BK:
            comptime for i in range(TM):
                acc[i] = f_reduce(acc[i].cast[DType.uint32]()).cast[DType.int32]()

    comptime for i in range(TM):
        C.ptr.unsafe_store[width=TN]((brow + trow * TM + i) * N + bcol + tcol * TN,
                                     f_reduce(acc[i].cast[DType.uint32]()))


# ---- launchers: grid and block shapes live here, not at the call site ----

def launch_naive[M: Int, N: Int, K: Int, AL: TensorLayout, BL: TensorLayout, CL: TensorLayout](
    ctx: DeviceContext,
    A: TileTensor[DType.uint8, AL, MutAnyOrigin],
    B: TileTensor[DType.uint8, BL, MutAnyOrigin],
    C: TileTensor[DType.uint8, CL, MutAnyOrigin],
) raises:
    comptime kernel = gemm127_naive[M, N, K, AL, BL, CL]
    ctx.enqueue_function[kernel](A, B, C, grid_dim=(ceildiv(N, 16), ceildiv(M, 16)), block_dim=(16, 16))


def launch_tiled[M: Int, N: Int, K: Int, T: Tile, AL: TensorLayout, BL: TensorLayout, CL: TensorLayout](
    ctx: DeviceContext,
    A: TileTensor[DType.uint8, AL, MutAnyOrigin],
    B: TileTensor[DType.uint8, BL, MutAnyOrigin],
    C: TileTensor[DType.uint8, CL, MutAnyOrigin],
) raises:
    comptime kernel = gemm127_tiled[M, N, K, T, AL, BL, CL]
    ctx.enqueue_function[kernel](A, B, C, grid_dim=(N // T.BN, M // T.BM), block_dim=T.threads())


def launch_vec[M: Int, N: Int, K: Int, T: Tile, AL: TensorLayout, BL: TensorLayout, CL: TensorLayout](
    ctx: DeviceContext,
    A: TileTensor[DType.uint8, AL, MutAnyOrigin],
    B: TileTensor[DType.uint8, BL, MutAnyOrigin],
    C: TileTensor[DType.uint8, CL, MutAnyOrigin],
) raises:
    comptime kernel = gemm127_vec[M, N, K, T, AL, BL, CL]
    ctx.enqueue_function[kernel](A, B, C, grid_dim=(N // T.BN, M // T.BM), block_dim=T.threads())

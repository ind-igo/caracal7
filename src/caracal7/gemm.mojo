"""Byte GEMM mod 127 on SIMD lanes: the tile op of docs/design.md section 9, M1 path.

C[M, N] = A[M, K] . B[K, N] over F127, one byte per element, int32 accumulation,
lazy reduction (field.f_reduce) every REDUCE_EVERY products.

Two kernels, rungs 1 and 4 of the ladder in design.md section 8:
    gemm127_naive   one thread per output
    gemm127_tiled   threadgroup tiles BM x BK, BK x BN; each thread owns a TM x TN register tile
"""

from std.math import ceildiv
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx
from max.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, row_major, stack_allocation

from caracal7.field import f_reduce

comptime REDUCE_EVERY = 128   # 128 * 126^2 < 2^21, the f_reduce input bound


@fieldwise_init
struct Tile(TrivialRegisterPassable):
    var BM: Int
    var BN: Int
    var BK: Int
    var TM: Int
    var TN: Int

    def threads(self) -> Int:
        return (self.BM // self.TM) * (self.BN // self.TN)


comptime TILE_DEFAULT = Tile(BM=64, BN=64, BK=16, TM=4, TN=4)


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

    var As = stack_allocation[DType.uint8, address_space=AddressSpace.SHARED](row_major[BM, BK]())
    var Bs = stack_allocation[DType.uint8, address_space=AddressSpace.SHARED](row_major[BK, BN]())
    var acc = SIMD[DType.int32, TM * TN](0)      # register tile

    for kt in range(K // BK):
        # cooperative loads: consecutive threads take consecutive columns (coalesced)
        comptime for i in range(0, BM * BK, THREADS):
            var idx = i + tid
            As[idx // BK, idx % BK] = A[brow + idx // BK, kt * BK + idx % BK]
        comptime for i in range(0, BK * BN, THREADS):
            var idx = i + tid
            Bs[idx // BN, idx % BN] = B[kt * BK + idx // BN, bcol + idx % BN]
        barrier()
        comptime for k in range(BK):
            var regN = SIMD[DType.int32, TN]()
            comptime for j in range(TN):
                regN[j] = rebind[Scalar[DType.uint8]](Bs[k, tcol * TN + j]).cast[DType.int32]()
            comptime for i in range(TM):
                var a = rebind[Scalar[DType.uint8]](As[trow * TM + i, k]).cast[DType.int32]()
                comptime for j in range(TN):
                    acc[i * TN + j] += a * regN[j]
        barrier()
        if (kt * BK) % REDUCE_EVERY >= REDUCE_EVERY - BK:
            acc = f_reduce(acc.cast[DType.uint32]()).cast[DType.int32]()

    var out = f_reduce(acc.cast[DType.uint32]())
    comptime for i in range(TM):
        comptime for j in range(TN):
            C[brow + trow * TM + i, bcol + tcol * TN + j] = rebind[C.ElementType](out[i * TN + j])


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

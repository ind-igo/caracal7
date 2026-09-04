"""Backend and the tile op (docs/design.md section 9), SIMD-lane implementation for Apple M1 to M4.

`tile_mac` is the one hot op. On SIMD lanes it is the rank-1 update acc[i] += a[i] * b (mma_k = 1)
on int32 lanes, reduced lazily every `max_terms` products. An MMA backend (NVIDIA TensorCore, Apple
M5 MmaOpApple) replaces this function, the `Backend` value, and the fragment loads and k step of
the skeleton below (an MMA consumes mma_k columns of the tile per call); every stage above this
file is unchanged.

`gemm_f2` is the shared skeleton of every GEMM-shaped stage on F2 data (the grid DFTs, the
residual pass, the quotient interpolations): threadgroup tiles, register tiles, the F2 operands as
two byte planes in shared memory, four `tile_mac` per k for the complex product. Operands are read
through a `Loader`, so one kernel runs on a strided table, on an E-valued buffer seen as 8 F2 lanes
(D = 8), or on a family row gathered from the LDE. Every dimension is predicated: M, N, K can be
anything; the tile only has to be legal.
"""

from std.math import ceildiv
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx
from max.gpu.memory import AddressSpace
from layout import row_major, stack_allocation

from caracal7.field import F2, f_add, f_reduce_signed, WIDE_BIAS


@fieldwise_init
struct Tile(TrivialRegisterPassable):
    var BM: Int
    var BN: Int
    var BK: Int
    var TM: Int
    var TN: Int

    def threads(self) -> Int:
        return (self.BM // self.TM) * (self.BN // self.TN)


@fieldwise_init
struct Backend(TrivialRegisterPassable):
    """What differs between backends (design section 9). Only the fields a kernel reads today exist;
    a kernel that needs more adds a field here, never a device probe of its own."""
    var threadgroup_bytes: Int   # shared memory one block may use
    var mma_k: Int               # 1: rank-1 update on SIMD lanes; the MMA tile depth on MMA backends
    var max_terms: Int           # products one int32 lane accumulates before f_reduce_signed
    var vec_bytes: Int           # bytes per vector load
    var tile: Tile               # default tile


comptime SIMD_LANES = Backend(threadgroup_bytes=32768, mma_k=1, max_terms=128, vec_bytes=4,
                              tile=Tile(BM=64, BN=64, BK=16, TM=4, TN=4))
# 128 terms: signed F2 lanes move by at most 2 * 126^2 per term, 128 of them stay below WIDE_BIAS.
comptime BACKEND = SIMD_LANES   # ponytail: the only implementation; select by device family here when an MMA path lands
comptime LANE_TILE = Tile(BM=8, BN=128, BK=16, TM=8, TN=4)   # M = the 8 F2 lanes of E: residual, open, fold


@always_inline
def tile_mac[B: Backend, TM: Int, TN: Int, neg: Bool = False](
    a: SIMD[DType.int32, TM], b: SIMD[DType.int32, TN], mut acc: InlineArray[SIMD[DType.int32, TN], TM]
):
    """acc[i][j] += a[i] * b[j] (or -=). The (TM x mma_k) by (mma_k x TN) tile product with mma_k = 1."""
    comptime assert B.mma_k == 1, "only the SIMD-lane tile op exists"
    comptime for i in range(TM):
        comptime if neg:
            acc[i] -= a[i] * b
        else:
            acc[i] += a[i] * b


@always_inline
def tile_reduce[TM: Int, TN: Int](mut acc: InlineArray[SIMD[DType.int32, TN], TM]):
    comptime for i in range(TM):
        acc[i] = f_reduce_signed(acc[i]).cast[DType.int32]()


@fieldwise_init
struct Operands(TrivialRegisterPassable, DevicePassable):
    """Arena byte offsets and byte strides. A[m, k] and C[m, n] are strided; B goes through the
    Loader with the same fields. n splits as (n // D, n % D) with strides (hi, lo), so an E-valued
    buffer (point, 8 F2 lanes) is one operand with D = 8; D = 1 is the plain case (lo unused)."""
    var a: Int64
    var sa_m: Int64
    var sa_k: Int64
    var sa_z: Int64
    var b: Int64
    var sb_k: Int64
    var sb_hi: Int64
    var sb_lo: Int64
    var sb_z: Int64
    var c: Int64
    var sc_m: Int64
    var sc_hi: Int64
    var sc_lo: Int64
    var sc_z: Int64
    var aux0: Int64
    var aux1: Int64
    var aux2: Int64

    comptime device_type: AnyType = Self

    def _to_device_type(self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "Operands"


trait Loader:
    @staticmethod
    def load(base: Pointer[UInt8, MutAnyOrigin], o: Operands, k: Int, n_hi: Int, n_lo: Int, z: Int) -> F2:
        """B[k, n] with n = n_hi * D + n_lo."""
        ...


struct Strided(Loader):
    @staticmethod
    def load(base: Pointer[UInt8, MutAnyOrigin], o: Operands, k: Int, n_hi: Int, n_lo: Int, z: Int) -> F2:
        return base.unsafe_load[width=2](Int(o.b) + k * Int(o.sb_k) + n_hi * Int(o.sb_hi) + n_lo * Int(o.sb_lo) + z * Int(o.sb_z))


struct Bytes(Loader):
    """F values, one byte each, as F2 with a zero imaginary part."""
    @staticmethod
    def load(base: Pointer[UInt8, MutAnyOrigin], o: Operands, k: Int, n_hi: Int, n_lo: Int, z: Int) -> F2:
        var v = F2(0)
        v[0] = base[unsafe_offset=Int(o.b) + k * Int(o.sb_k) + n_hi * Int(o.sb_hi) + n_lo * Int(o.sb_lo) + z * Int(o.sb_z)]
        return v


def gemm_f2[B: Backend, T: Tile, L: Loader, D: Int, acc: Bool = False](
    base: Pointer[UInt8, MutAnyOrigin], o: Operands, M: Int32, N: Int32, K: Int32
):
    """C[m, n] = (C[m, n] if acc) + sum_k A[m, k] B[k, n] over F2, batch z = block_idx.z."""
    comptime BM = T.BM
    comptime BN = T.BN
    comptime BK = T.BK
    comptime TM = T.TM
    comptime TN = T.TN
    comptime THREADS = T.threads()
    comptime assert BM % TM == 0 and BN % TN == 0
    comptime assert B.max_terms % BK == 0, "the lazy reduction cadence needs BK | max_terms"
    comptime assert B.max_terms * 2 * 126 * 126 + 126 < Int(WIDE_BIAS), "signed F2 lanes overflow WIDE_BIAS before a reduction"
    comptime assert (BM * BK) % THREADS == 0 and (BK * BN) % THREADS == 0
    comptime assert 2 * (BK * BM + BK * BN) <= B.threadgroup_bytes

    var tid = Int(thread_idx.x)
    var trow = tid // (BN // TN)
    var tcol = tid % (BN // TN)
    var brow = Int(block_idx.y) * BM
    var bcol = Int(block_idx.x) * BN
    var z = Int(block_idx.z)
    var Mi = Int(M)
    var Ni = Int(N)
    var Ki = Int(K)

    var As0 = stack_allocation[DType.uint8, address_space=AddressSpace.SHARED](row_major[BK, BM]())   # transposed planes
    var As1 = stack_allocation[DType.uint8, address_space=AddressSpace.SHARED](row_major[BK, BM]())
    var Bs0 = stack_allocation[DType.uint8, address_space=AddressSpace.SHARED](row_major[BK, BN]())
    var Bs1 = stack_allocation[DType.uint8, address_space=AddressSpace.SHARED](row_major[BK, BN]())
    var re = InlineArray[SIMD[DType.int32, TN], TM](fill=SIMD[DType.int32, TN](0))
    var im = InlineArray[SIMD[DType.int32, TN], TM](fill=SIMD[DType.int32, TN](0))

    for kt in range(ceildiv(Ki, BK)):
        comptime for i in range(0, BM * BK, THREADS):
            var idx = i + tid
            var r = idx // BK
            var kk = idx % BK
            var v = F2(0)
            if brow + r < Mi and kt * BK + kk < Ki:
                v = base.unsafe_load[width=2](Int(o.a) + (brow + r) * Int(o.sa_m) + (kt * BK + kk) * Int(o.sa_k) + z * Int(o.sa_z))
            As0.ptr.unsafe_store(kk * BM + r, v[0])
            As1.ptr.unsafe_store(kk * BM + r, v[1])
        comptime for i in range(0, BK * BN, THREADS):
            var idx = i + tid
            var kk = idx // BN
            var n = bcol + idx % BN
            var v = F2(0)
            if n < Ni and kt * BK + kk < Ki:
                v = L.load(base, o, kt * BK + kk, n // D, n % D, z)
            Bs0.ptr.unsafe_store(kk * BN + idx % BN, v[0])
            Bs1.ptr.unsafe_store(kk * BN + idx % BN, v[1])
        barrier()
        comptime for kk in range(BK):
            var a0 = As0.ptr.unsafe_load[width=TM](kk * BM + trow * TM).cast[DType.int32]()
            var a1 = As1.ptr.unsafe_load[width=TM](kk * BM + trow * TM).cast[DType.int32]()
            var b0 = Bs0.ptr.unsafe_load[width=TN](kk * BN + tcol * TN).cast[DType.int32]()
            var b1 = Bs1.ptr.unsafe_load[width=TN](kk * BN + tcol * TN).cast[DType.int32]()
            tile_mac[B](a0, b0, re)
            tile_mac[B, neg=True](a1, b1, re)
            tile_mac[B](a0, b1, im)
            tile_mac[B](a1, b0, im)
        barrier()
        if ((kt + 1) * BK) % B.max_terms == 0:
            tile_reduce(re)
            tile_reduce(im)

    tile_reduce(re)
    tile_reduce(im)
    comptime for i in range(TM):
        var m = brow + trow * TM + i
        comptime for j in range(TN):
            var n = bcol + tcol * TN + j
            if m < Mi and n < Ni:
                var v = F2(UInt8(re[i][j]), UInt8(im[i][j]))
                var at = Int(o.c) + m * Int(o.sc_m) + (n // D) * Int(o.sc_hi) + (n % D) * Int(o.sc_lo) + z * Int(o.sc_z)
                comptime if acc:
                    v = f_add(v, base.unsafe_load[width=2](at))
                base.unsafe_store[width=2](at, v)


def launch_gemm_f2[B: Backend, T: Tile, L: Loader, D: Int, acc: Bool = False](
    ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin], o: Operands, M: Int, N: Int, K: Int, batch: Int = 1
) raises:
    comptime kernel = gemm_f2[B, T, L, D, acc]
    ctx.enqueue_function[kernel](base, o, Int32(M), Int32(N), Int32(K),
                                 grid_dim=(ceildiv(N, T.BN), ceildiv(M, T.BM), batch), block_dim=T.threads())


def strided(a: Int, sa_m: Int, sa_k: Int, b: Int, sb_k: Int, sb_hi: Int, sb_lo: Int,
            c: Int, sc_m: Int, sc_hi: Int, sc_lo: Int, sa_z: Int = 0, sb_z: Int = 0, sc_z: Int = 0) -> Operands:
    """Operands for the Strided loader, in Int."""
    return Operands(a=Int64(a), sa_m=Int64(sa_m), sa_k=Int64(sa_k), sa_z=Int64(sa_z),
                    b=Int64(b), sb_k=Int64(sb_k), sb_hi=Int64(sb_hi), sb_lo=Int64(sb_lo), sb_z=Int64(sb_z),
                    c=Int64(c), sc_m=Int64(sc_m), sc_hi=Int64(sc_hi), sc_lo=Int64(sc_lo), sc_z=Int64(sc_z),
                    aux0=0, aux1=0, aux2=0)

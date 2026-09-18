"""Backend and the tile op (docs/design.md section 9), SIMD-lane implementation for Apple M1 to M4.

Three backends, one per kind of tile op, selected by a build flag (`BACKEND`):
  LANES       `gemm_f2`: `tile_mac`, the rank-1 update acc[i] += a[i] * b on int32 lanes, reduced lazily
              every `max_terms` products. Any GPU; the only op on Apple M1 to M4. The default.
  APPLE_MMA   `gemm_f2_apple` (`-D CARACAL_APPLE_MMA`): `MmaOpApple`, the Apple GPU family 10 (M5) integer
              simdgroup MMA, int8 x int8 -> int32, 16 x 16 x 16 per simdgroup of 32 threads.
  NVIDIA_MMA  `gemm_f2_nvidia` (`-D CARACAL_NVIDIA_MMA`): `mma.sync.m16n8k32.s32.s8.s8.s32`, the NVIDIA
              integer tensor core (sm_80 and later), 16 x 8 x 32 per warp of 32 threads.
The MMA skeletons keep the operands, staging and predication of `gemm_f2` and replace the k step; every
stage above this file is unchanged. A is staged as int8 planes (re, im, -im), so the complex product is
four MMAs into two int32 accumulators, with the lane path's reduction cadence.

`gemm_f2` is the shared skeleton of every GEMM-shaped stage on F2 data (the grid DFTs, the
residual pass, the quotient interpolations): threadgroup tiles, register tiles, the F2 operands as
two byte planes in shared memory, four `tile_mac` per k for the complex product. Operands are read
through a `Loader`, so one kernel runs on a strided table, on an E-valued buffer seen as 8 F2 lanes
(D = 8), or on a family row gathered from the LDE. Every dimension is predicated: M, N, K can be
anything; the tile only has to be legal.
"""

from std.math import ceildiv
from std.sys import is_defined, _RegisterPackType, llvm_intrinsic
from std.sys.info import has_apple_gpu_accelerator
from std.sys._assembly import inlined_assembly
from std.memory import bitcast
from linalg.arch.apple.mma import MmaOpApple
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx
from max.gpu.memory import AddressSpace
from layout import row_major, stack_allocation

from core.field import F2, F4, f_add, f_reduce_signed, WIDE_BIAS
from core.bytes import Base
from core.arena import Arena


@fieldwise_init
struct Tile(TrivialRegisterPassable):
    var BM: Int
    var BN: Int
    var BK: Int
    var TM: Int                  # register tile rows per thread; on an MMA tile, rows per warp / simdgroup
    var TN: Int
    var mma: Bool                # a warp-level MMA skeleton instead of the lane skeleton

    def threads(self) -> Int:
        return (self.BM // self.TM) * (self.BN // self.TN) * (32 if self.mma else 1)


@fieldwise_init
struct Backend(TrivialRegisterPassable):
    """What differs between backends (design section 9). Only the fields a kernel reads today exist;
    a kernel that needs more adds a field here, never a device probe of its own."""
    var kind: Int                # KIND_LANES, KIND_APPLE_MMA, KIND_NVIDIA_MMA: which F2 skeleton launch_gemm_f2 runs
    var threadgroup_bytes: Int   # shared memory one block may use
    var mma_k: Int               # 1: rank-1 update on SIMD lanes; the MMA tile depth on MMA backends
    var max_terms: Int           # products one int32 lane accumulates before f_reduce_signed
    var vec_bytes: Int           # bytes per vector load
    var tile: Tile               # default tile
    var block: Int               # threads per block of the one-thread-per-element kernels
    var grind_threads: Int       # nonce search: threads, each walking nonces t, t + grind_threads, ...
    var grind_poll: Int          # iterations between a block's polls of the nonce found


comptime KIND_LANES = 0
comptime KIND_APPLE_MMA = 1
comptime KIND_NVIDIA_MMA = 2
comptime LANES = Backend(kind=KIND_LANES, threadgroup_bytes=32768, mma_k=1, max_terms=128, vec_bytes=4,
                         tile=Tile(BM=64, BN=64, BK=8, TM=4, TN=4, mma=False), block=256, grind_threads=8192, grind_poll=1)
# 128 terms: signed F2 lanes move by at most 2 * 126^2 per term, 128 of them stay below WIDE_BIAS.
comptime APPLE_MMA = Backend(kind=KIND_APPLE_MMA, threadgroup_bytes=32768, mma_k=16, max_terms=128, vec_bytes=16,
                             tile=Tile(BM=64, BN=64, BK=32, TM=32, TN=32, mma=True), block=256, grind_threads=8192, grind_poll=1)
comptime NVIDIA_MMA = Backend(kind=KIND_NVIDIA_MMA, threadgroup_bytes=49152, mma_k=32, max_terms=128, vec_bytes=16,
                              tile=Tile(BM=64, BN=64, BK=32, TM=32, TN=32, mma=True), block=256, grind_threads=32768, grind_poll=8)
# grind: a block leaves the search only once its nonces pass the one found, so up to grind_poll x grind_threads nonces
# are tried past the winner. M1 Pro, 20 bits, 20 paired seeds (2026-09-17): 8192 x 1 gives 2.1 ms a search, the old
# 32768 x 8 gave 4.0, 65536 x 8 gave 4.6. The NVIDIA pair is the old one, not measured there.
comptime BACKEND = NVIDIA_MMA if is_defined["CARACAL_NVIDIA_MMA"]() else (APPLE_MMA if is_defined["CARACAL_APPLE_MMA"]() else LANES)
# The lane build on an Apple host (M1..M4, GPU family 7..9): no integer simdgroup matrix, but the 8x8 fp16 -> fp32 one
# is there and runs at 2.3 T MAC/s on the M1 Pro, 3x the int32 lanes. Bytes < 127 are exact in fp16; their products are exact in the fp32 accumulator of the op.
comptime APPLE8 = BACKEND.kind == KIND_LANES and has_apple_gpu_accelerator()
# A build flag, not a device probe: host code reads BACKEND.tile for the launch shape, and `is_nvidia_gpu()` is
# only true inside device code; the Apple GPU family (M5 or not) is not visible at comptime at all.
comptime LANE_TILE = Tile(BM=8, BN=128, BK=16, TM=8, TN=4, mma=False)   # M = the 8 F2 lanes of E: residual, open, fold
comptime F4_TILE = Tile(BM=32, BN=32, BK=8, TM=2, TN=2, mma=False)   # gemm_f4: 4 x 4 register tiles spill (102 byte-GMAC/s), 2 x 4 gives 212, 2 x 2 gives 340
comptime F4_TERMS = 32   # F4 products per lane between reductions: a (1, i) lane moves by at most 8 * 126^2 per product


@always_inline
def tile_mac[B: Backend, TM: Int, TN: Int, neg: Bool = False](
    a: SIMD[DType.int32, TM], b: SIMD[DType.int32, TN], mut acc: InlineArray[SIMD[DType.int32, TN], TM]
):
    """acc[i][j] += a[i] * b[j] (or -=). The (TM x 1) by (1 x TN) tile product on lanes; valid on every backend."""
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
    var sb_zz: Int64
    var c: Int64
    var sc_m: Int64
    var sc_hi: Int64
    var sc_lo: Int64
    var sc_z: Int64
    var zd: Int64                # batch z = (z // zd, z % zd) when > 0: offsets zlo * s_z + zhi * s_zz per operand
    var sa_zz: Int64
    var sc_zz: Int64
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
    comptime real: Bool     # the imaginary part is always zero: half the tile products
    comptime kfast: Bool    # B is contiguous along k, not n: the tile load walks k in adjacent threads

    @staticmethod
    def load(base: Base, o: Operands, k: Int, n_hi: Int, n_lo: Int, z: Int) -> F2:
        """B[k, n] with n = n_hi * D + n_lo; z is the batch byte offset."""
        ...


struct Strided(Loader):
    comptime real = False
    comptime kfast = False

    @staticmethod
    def load(base: Base, o: Operands, k: Int, n_hi: Int, n_lo: Int, z: Int) -> F2:
        return base.unsafe_load[width=2, alignment=2](Int(o.b) + k * Int(o.sb_k) + n_hi * Int(o.sb_hi) + n_lo * Int(o.sb_lo) + z)


struct Bytes[kfast_: Bool = False](Loader):
    """F values, one byte each, as F2 with a zero imaginary part."""
    comptime real = True
    comptime kfast = Self.kfast_

    @staticmethod
    def load(base: Base, o: Operands, k: Int, n_hi: Int, n_lo: Int, z: Int) -> F2:
        var v = F2(0)
        v[0] = base[unsafe_offset=Int(o.b) + k * Int(o.sb_k) + n_hi * Int(o.sb_hi) + n_lo * Int(o.sb_lo) + z]
        return v


def gemm_f2[B: Backend, T: Tile, L: Loader, D: Int, acc: Bool = False](
    base: Base, o: Operands, M: Int32, N: Int32, K: Int32
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
    var zlo = z % Int(o.zd) if o.zd > 0 else z
    var zhi = z // Int(o.zd) if o.zd > 0 else 0
    var za = zlo * Int(o.sa_z) + zhi * Int(o.sa_zz)
    var zb = zlo * Int(o.sb_z) + zhi * Int(o.sb_zz)
    var zc = zlo * Int(o.sc_z) + zhi * Int(o.sc_zz)
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
                v = base.unsafe_load[width=2, alignment=2](Int(o.a) + (brow + r) * Int(o.sa_m) + (kt * BK + kk) * Int(o.sa_k) + za)
            As0.ptr.unsafe_store(kk * BM + r, v[0])
            As1.ptr.unsafe_store(kk * BM + r, v[1])
        comptime for i in range(0, BK * BN, THREADS):
            var idx = i + tid
            var kk: Int
            var nn: Int
            comptime if L.kfast:
                nn = idx // BK
                kk = idx % BK
            else:
                kk = idx // BN
                nn = idx % BN
            var n = bcol + nn
            var v = F2(0)
            if n < Ni and kt * BK + kk < Ki:
                v = L.load(base, o, kt * BK + kk, n // D, n % D, zb)
            Bs0.ptr.unsafe_store(kk * BN + nn, v[0])
            comptime if not L.real:
                Bs1.ptr.unsafe_store(kk * BN + nn, v[1])
        barrier()
        comptime for kk in range(BK):
            var a0 = As0.ptr.unsafe_load[width=TM](kk * BM + trow * TM).cast[DType.int32]()
            var a1 = As1.ptr.unsafe_load[width=TM](kk * BM + trow * TM).cast[DType.int32]()
            var b0 = Bs0.ptr.unsafe_load[width=TN](kk * BN + tcol * TN).cast[DType.int32]()
            tile_mac[B](a0, b0, re)
            tile_mac[B](a1, b0, im)
            comptime if not L.real:
                var b1 = Bs1.ptr.unsafe_load[width=TN](kk * BN + tcol * TN).cast[DType.int32]()
                tile_mac[B, neg=True](a1, b1, re)
                tile_mac[B](a0, b1, im)
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
                var at = Int(o.c) + m * Int(o.sc_m) + (n // D) * Int(o.sc_hi) + (n % D) * Int(o.sc_lo) + zc
                comptime if acc:
                    v = f_add(v, base.unsafe_load[width=2](at))
                base.unsafe_store[width=2](at, v)


@always_inline
def frag8(lane: Int) -> Tuple[Int, Int]:
    """The (row, column base) of a lane's two elements of an 8x8 simdgroup matrix (columns base, base + 1)."""
    return (((lane & 6) >> 1) + ((lane & 16) >> 2), ((lane & 1) << 1) + ((lane & 8) >> 1))


@always_inline
def mma8(a: SIMD[DType.float16, 2], b: SIMD[DType.float16, 2], c: SIMD[DType.float32, 2]) -> SIMD[DType.float32, 2]:
    """The 8x8 simdgroup matrix op of every Apple GPU: a @ b + c on fp16 fragments with fp32 accumulation, lane
    layout `frag8`. The AIR intrinsic takes 64-wide vectors; the per-lane pair sits at [0], [1]."""
    var aw = SIMD[DType.float16, 64](0)
    var bw = SIMD[DType.float16, 64](0)
    var cw = SIMD[DType.float32, 64](0)
    aw[0] = a[0]; aw[1] = a[1]
    bw[0] = b[0]; bw[1] = b[1]
    cw[0] = c[0]; cw[1] = c[1]
    var dw = llvm_intrinsic["llvm.air.simdgroup_matrix_8x8_multiply_accumulate", SIMD[DType.float32, 64]](aw, bw, cw)
    return SIMD[DType.float32, 2](dw[0], dw[1])


def gemm_f2_apple[B: Backend, T: Tile, L: Loader, D: Int, acc: Bool = False](
    base: Base, o: Operands, M: Int32, N: Int32, K: Int32
):
    """gemm_f2 on the Apple M5 simdgroup MMA: the k step is `MmaOpApple.mma` over the BK columns of the
    tile; re += a0 b0 + (-a1) b1, im += a1 b0 + a0 b1. Each simdgroup owns a (T.TM x T.TN) sub-tile; lane
    (rb, cb) of a 16 x 16 fragment holds columns cb..cb+3 of rows rb and rb + 8."""
    comptime BM = T.BM
    comptime BN = T.BN
    comptime BK = T.BK
    comptime SGM = T.TM
    comptime SGN = T.TN
    comptime NM = SGM // 16
    comptime NN = SGN // 16
    comptime THREADS = T.threads()
    comptime assert T.mma and B.kind == KIND_APPLE_MMA, "gemm_f2_apple runs on an MMA tile of APPLE_MMA"
    comptime assert BK % 16 == 0 and SGM % 16 == 0 and SGN % 16 == 0 and BM % SGM == 0 and BN % SGN == 0
    comptime assert B.max_terms % BK == 0, "the lazy reduction cadence needs BK | max_terms"
    comptime assert B.max_terms * 2 * 126 * 126 + 126 < Int(WIDE_BIAS), "signed F2 lanes overflow WIDE_BIAS before a reduction"
    comptime assert (BM * BK) % THREADS == 0 and (BK * BN) % THREADS == 0
    comptime assert 3 * BM * BK + 2 * BK * BN <= B.threadgroup_bytes
    comptime Op = MmaOpApple[DType.int32, DType.int8, NM, NN]

    var tid = Int(thread_idx.x)
    var sg = tid // 32
    var sg_row = sg // (BN // SGN)
    var sg_col = sg % (BN // SGN)
    var brow = Int(block_idx.y) * BM
    var bcol = Int(block_idx.x) * BN
    var z = Int(block_idx.z)
    var zlo = z % Int(o.zd) if o.zd > 0 else z
    var zhi = z // Int(o.zd) if o.zd > 0 else 0
    var za = zlo * Int(o.sa_z) + zhi * Int(o.sa_zz)
    var zb = zlo * Int(o.sb_z) + zhi * Int(o.sb_zz)
    var zc = zlo * Int(o.sc_z) + zhi * Int(o.sc_zz)
    var Mi = Int(M)
    var Ni = Int(N)
    var Ki = Int(K)

    var As0 = stack_allocation[DType.int8, address_space=AddressSpace.SHARED](row_major[BM, BK]())   # re
    var As1 = stack_allocation[DType.int8, address_space=AddressSpace.SHARED](row_major[BM, BK]())   # im
    var As2 = stack_allocation[DType.int8, address_space=AddressSpace.SHARED](row_major[BM, BK]())   # -im
    var Bs0 = stack_allocation[DType.int8, address_space=AddressSpace.SHARED](row_major[BK, BN]())
    var Bs1 = stack_allocation[DType.int8, address_space=AddressSpace.SHARED](row_major[BK, BN]())
    var op = Op()
    var re = Op.zero_accum()
    var im = Op.zero_accum()

    for kt in range(ceildiv(Ki, BK)):
        comptime for i in range(0, BM * BK, THREADS):
            var idx = i + tid
            var r = idx // BK
            var kk = idx % BK
            var v = F2(0)
            if brow + r < Mi and kt * BK + kk < Ki:
                v = base.unsafe_load[width=2, alignment=2](Int(o.a) + (brow + r) * Int(o.sa_m) + (kt * BK + kk) * Int(o.sa_k) + za)
            var a1 = Int8(v[1])
            As0.ptr.unsafe_store(idx, Int8(v[0]))
            As1.ptr.unsafe_store(idx, a1)
            As2.ptr.unsafe_store(idx, -a1)
        comptime for i in range(0, BK * BN, THREADS):
            var idx = i + tid
            var kk: Int
            var nn: Int
            comptime if L.kfast:
                nn = idx // BK
                kk = idx % BK
            else:
                kk = idx // BN
                nn = idx % BN
            var n = bcol + nn
            var v = F2(0)
            if n < Ni and kt * BK + kk < Ki:
                v = L.load(base, o, kt * BK + kk, n // D, n % D, zb)
            Bs0.ptr.unsafe_store(kk * BN + nn, Int8(v[0]))
            comptime if not L.real:
                Bs1.ptr.unsafe_store(kk * BN + nn, Int8(v[1]))
        barrier()
        var a0 = As0.tile[SGM, BK](sg_row, 0)
        var a1 = As1.tile[SGM, BK](sg_row, 0)
        var b0 = Bs0.tile[BK, SGN](0, sg_col)
        op.mma(re, a0, b0)
        op.mma(im, a1, b0)
        comptime if not L.real:
            var a2 = As2.tile[SGM, BK](sg_row, 0)
            var b1 = Bs1.tile[BK, SGN](0, sg_col)
            op.mma(re, a2, b1)
            op.mma(im, a0, b1)
        barrier()
        if ((kt + 1) * BK) % B.max_terms == 0:
            comptime for t in range(NM * NN):
                re[t] = f_reduce_signed(re[t]).cast[DType.int32]()
                im[t] = f_reduce_signed(im[t]).cast[DType.int32]()

    comptime for mi in range(NM):
        comptime for ni in range(NN):
            var fr = f_reduce_signed(re[mi * NN + ni])
            var fi = f_reduce_signed(im[mi * NN + ni])
            comptime for half in range(2):
                var m = brow + sg_row * SGM + mi * 16 + op.rb + half * 8
                comptime for j in range(4):
                    var n = bcol + sg_col * SGN + ni * 16 + op.cb + j
                    if m < Mi and n < Ni:
                        var v = F2(fr[half * 4 + j], fi[half * 4 + j])
                        var at = Int(o.c) + m * Int(o.sc_m) + (n // D) * Int(o.sc_hi) + (n % D) * Int(o.sc_lo) + zc
                        comptime if acc:
                            v = f_add(v, base.unsafe_load[width=2](at))
                        base.unsafe_store[width=2](at, v)


@always_inline
def mma_s8(mut d: SIMD[DType.int32, 4], a: SIMD[DType.uint32, 4], b: SIMD[DType.uint32, 2]):
    """d += A (16 x 32, int8, row) B (32 x 8, int8, col) on one warp: mma.sync.m16n8k32. Lane l holds, with
    g = l // 4 and t = l % 4: a[0] = A[g, 4t..4t+3], a[1] = A[g+8, 4t..], a[2] = A[g, 16+4t..], a[3] = A[g+8, 16+4t..];
    b[0] = B[4t..4t+3, g], b[1] = B[16+4t.., g]; d = (D[g, 2t], D[g, 2t+1], D[g+8, 2t], D[g+8, 2t+1])."""
    var r = inlined_assembly[
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        _RegisterPackType[Int32, Int32, Int32, Int32],
        constraints="=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r",
    ](a[0], a[1], a[2], a[3], b[0], b[1], d[0], d[1], d[2], d[3])
    d = SIMD[DType.int32, 4](r[0], r[1], r[2], r[3])


def gemm_f2_nvidia[B: Backend, T: Tile, L: Loader, D: Int, acc: Bool = False](
    base: Base, o: Operands, M: Int32, N: Int32, K: Int32
):
    """gemm_f2 on the NVIDIA integer tensor core: the k step is `mma_s8` per 16 x 8 x 32 fragment. A planes
    are staged (m, k) and B planes (n, k), k contiguous, so every fragment register is one 4-byte load.
    Each warp owns a (T.TM x T.TN) sub-tile of 16 x 8 fragments."""
    comptime BM = T.BM
    comptime BN = T.BN
    comptime BK = T.BK
    comptime WM = T.TM
    comptime WN = T.TN
    comptime NM = WM // 16
    comptime NN = WN // 8
    comptime THREADS = T.threads()
    comptime assert T.mma and B.kind == KIND_NVIDIA_MMA, "gemm_f2_nvidia runs on an MMA tile of NVIDIA_MMA"
    comptime assert BK % 32 == 0 and WM % 16 == 0 and WN % 8 == 0 and BM % WM == 0 and BN % WN == 0
    comptime assert B.max_terms % BK == 0, "the lazy reduction cadence needs BK | max_terms"
    comptime assert B.max_terms * 2 * 126 * 126 + 126 < Int(WIDE_BIAS), "signed F2 lanes overflow WIDE_BIAS before a reduction"
    comptime assert (BM * BK) % THREADS == 0 and (BK * BN) % THREADS == 0
    comptime assert 3 * BM * BK + 2 * BK * BN <= B.threadgroup_bytes

    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var g = lane // 4
    var t = lane % 4
    var warp = tid // 32
    var wrow = warp // (BN // WN)
    var wcol = warp % (BN // WN)
    var brow = Int(block_idx.y) * BM
    var bcol = Int(block_idx.x) * BN
    var z = Int(block_idx.z)
    var zlo = z % Int(o.zd) if o.zd > 0 else z
    var zhi = z // Int(o.zd) if o.zd > 0 else 0
    var za = zlo * Int(o.sa_z) + zhi * Int(o.sa_zz)
    var zb = zlo * Int(o.sb_z) + zhi * Int(o.sb_zz)
    var zc = zlo * Int(o.sc_z) + zhi * Int(o.sc_zz)
    var Mi = Int(M)
    var Ni = Int(N)
    var Ki = Int(K)

    var As0 = stack_allocation[DType.int8, address_space=AddressSpace.SHARED](row_major[BM, BK]())   # re
    var As1 = stack_allocation[DType.int8, address_space=AddressSpace.SHARED](row_major[BM, BK]())   # im
    var As2 = stack_allocation[DType.int8, address_space=AddressSpace.SHARED](row_major[BM, BK]())   # -im
    var Bs0 = stack_allocation[DType.int8, address_space=AddressSpace.SHARED](row_major[BN, BK]())   # (n, k)
    var Bs1 = stack_allocation[DType.int8, address_space=AddressSpace.SHARED](row_major[BN, BK]())
    var re = InlineArray[SIMD[DType.int32, 4], NM * NN](fill=SIMD[DType.int32, 4](0))
    var im = InlineArray[SIMD[DType.int32, 4], NM * NN](fill=SIMD[DType.int32, 4](0))

    @always_inline
    @parameter
    def frag_a(P: type_of(As0), row: Int, k0: Int) -> SIMD[DType.uint32, 4]:
        var r0 = (row + g) * BK + k0 + t * 4
        var r1 = (row + g + 8) * BK + k0 + t * 4
        return SIMD[DType.uint32, 4](
            bitcast[DType.uint32, 1](P.ptr.unsafe_load[width=4](r0)),
            bitcast[DType.uint32, 1](P.ptr.unsafe_load[width=4](r1)),
            bitcast[DType.uint32, 1](P.ptr.unsafe_load[width=4](r0 + 16)),
            bitcast[DType.uint32, 1](P.ptr.unsafe_load[width=4](r1 + 16)))

    @always_inline
    @parameter
    def frag_b(P: type_of(Bs0), col: Int, k0: Int) -> SIMD[DType.uint32, 2]:
        var r = (col + g) * BK + k0 + t * 4
        return SIMD[DType.uint32, 2](
            bitcast[DType.uint32, 1](P.ptr.unsafe_load[width=4](r)),
            bitcast[DType.uint32, 1](P.ptr.unsafe_load[width=4](r + 16)))

    for kt in range(ceildiv(Ki, BK)):
        comptime for i in range(0, BM * BK, THREADS):
            var idx = i + tid
            var r = idx // BK
            var kk = idx % BK
            var v = F2(0)
            if brow + r < Mi and kt * BK + kk < Ki:
                v = base.unsafe_load[width=2, alignment=2](Int(o.a) + (brow + r) * Int(o.sa_m) + (kt * BK + kk) * Int(o.sa_k) + za)
            var a1 = Int8(v[1])
            As0.ptr.unsafe_store(idx, Int8(v[0]))
            As1.ptr.unsafe_store(idx, a1)
            As2.ptr.unsafe_store(idx, -a1)
        comptime for i in range(0, BK * BN, THREADS):
            var idx = i + tid
            var kk: Int
            var nn: Int
            comptime if L.kfast:
                nn = idx // BK
                kk = idx % BK
            else:
                kk = idx // BN
                nn = idx % BN
            var n = bcol + nn
            var v = F2(0)
            if n < Ni and kt * BK + kk < Ki:
                v = L.load(base, o, kt * BK + kk, n // D, n % D, zb)
            Bs0.ptr.unsafe_store(nn * BK + kk, Int8(v[0]))
            comptime if not L.real:
                Bs1.ptr.unsafe_store(nn * BK + kk, Int8(v[1]))
        barrier()
        comptime for ks in range(0, BK, 32):
            comptime for mi in range(NM):
                var row = wrow * WM + mi * 16
                var a0 = frag_a(As0, row, ks)
                var a1 = frag_a(As1, row, ks)
                var a2 = frag_a(As2, row, ks)
                comptime for ni in range(NN):
                    var col = wcol * WN + ni * 8
                    var b0 = frag_b(Bs0, col, ks)
                    mma_s8(re[mi * NN + ni], a0, b0)
                    mma_s8(im[mi * NN + ni], a1, b0)
                    comptime if not L.real:
                        var b1 = frag_b(Bs1, col, ks)
                        mma_s8(re[mi * NN + ni], a2, b1)
                        mma_s8(im[mi * NN + ni], a0, b1)
        barrier()
        if ((kt + 1) * BK) % B.max_terms == 0:
            comptime for f in range(NM * NN):
                re[f] = f_reduce_signed(re[f]).cast[DType.int32]()
                im[f] = f_reduce_signed(im[f]).cast[DType.int32]()

    comptime for mi in range(NM):
        comptime for ni in range(NN):
            var fr = f_reduce_signed(re[mi * NN + ni])
            var fi = f_reduce_signed(im[mi * NN + ni])
            comptime for half in range(2):
                var m = brow + wrow * WM + mi * 16 + g + half * 8
                comptime for j in range(2):
                    var n = bcol + wcol * WN + ni * 8 + t * 2 + j
                    if m < Mi and n < Ni:
                        var v = F2(fr[half * 2 + j], fi[half * 2 + j])
                        var at = Int(o.c) + m * Int(o.sc_m) + (n // D) * Int(o.sc_hi) + (n % D) * Int(o.sc_lo) + zc
                        comptime if acc:
                            v = f_add(v, base.unsafe_load[width=2](at))
                        base.unsafe_store[width=2](at, v)


def launch_gemm_f2[B: Backend, T: Tile, L: Loader, D: Int, acc: Bool = False](
    ctx: DeviceContext, arena: Arena, o: Operands, M: Int, N: Int, K: Int, batch: Int = 1
) raises:
    comptime if T.mma and B.kind == KIND_NVIDIA_MMA:
        comptime kernel = gemm_f2_nvidia[B, T, L, D, acc]
        ctx.enqueue_function[kernel](arena.buf, o, Int32(M), Int32(N), Int32(K),
                                     grid_dim=(ceildiv(N, T.BN), ceildiv(M, T.BM), batch), block_dim=T.threads())
        return
    comptime kernel = gemm_f2_apple[B, T, L, D, acc] if T.mma else gemm_f2[B, T, L, D, acc]
    ctx.enqueue_function[kernel](arena.buf, o, Int32(M), Int32(N), Int32(K),
                                 grid_dim=(ceildiv(N, T.BN), ceildiv(M, T.BM), batch), block_dim=T.threads())


def strided(a: Int, sa_m: Int, sa_k: Int, b: Int, sb_k: Int, sb_hi: Int, sb_lo: Int,
            c: Int, sc_m: Int, sc_hi: Int, sc_lo: Int, sa_z: Int = 0, sb_z: Int = 0, sc_z: Int = 0,
            zd: Int = 0, sa_zz: Int = 0, sb_zz: Int = 0, sc_zz: Int = 0) -> Operands:
    """Operands for the Strided loader, in Int. With `zd` the batch splits as (z // zd, z % zd) with
    strides (s_zz, s_z) per operand; without, z is linear with stride s_z."""
    return Operands(a=Int64(a), sa_m=Int64(sa_m), sa_k=Int64(sa_k), sa_z=Int64(sa_z),
                    b=Int64(b), sb_k=Int64(sb_k), sb_hi=Int64(sb_hi), sb_lo=Int64(sb_lo), sb_z=Int64(sb_z), sb_zz=Int64(sb_zz),
                    c=Int64(c), sc_m=Int64(sc_m), sc_hi=Int64(sc_hi), sc_lo=Int64(sc_lo), sc_z=Int64(sc_z),
                    zd=Int64(zd), sa_zz=Int64(sa_zz), sc_zz=Int64(sc_zz),
                    aux0=0, aux1=0, aux2=0)


trait Loader4:
    @staticmethod
    def load(base: Base, o: Operands, k: Int, n_hi: Int, n_lo: Int, z: Int) -> F4:
        """B[k, n] as F4 (4 bytes), n = n_hi * D + n_lo."""
        ...


struct Strided4(Loader4):
    @staticmethod
    def load(base: Base, o: Operands, k: Int, n_hi: Int, n_lo: Int, z: Int) -> F4:
        return base.unsafe_load[width=4](Int(o.b) + k * Int(o.sb_k) + n_hi * Int(o.sb_hi) + n_lo * Int(o.sb_lo) + z)


def gemm_f4[B: Backend, T: Tile, L: Loader4, D: Int, acc: Bool = False](
    base: Base, o: Operands, M: Int32, N: Int32, K: Int32
):
    """C[m, n] = (C[m, n] if acc) + sum_k A[m, k] B[k, n] over F4 = F2[j], j^2 = 2 + i, coordinates
    (1, i, j, ij): the F2 skeleton with four byte planes and twenty `tile_mac` per k into four
    accumulator tiles; the j^2 = 2 + i term is folded per step with doubled A planes (more MACs, fewer
    live registers: six accumulator tiles spilled and ran 7x slower)."""
    comptime BM = T.BM
    comptime BN = T.BN
    comptime BK = T.BK
    comptime TM = T.TM
    comptime TN = T.TN
    comptime THREADS = T.threads()
    comptime assert BM % TM == 0 and BN % TN == 0
    comptime assert F4_TERMS % BK == 0, "the lazy reduction cadence needs BK | F4_TERMS"
    comptime assert F4_TERMS * 8 * 126 * 126 + 126 < Int(WIDE_BIAS), "F4 lanes overflow WIDE_BIAS before a reduction"
    comptime assert (BM * BK) % THREADS == 0 and (BK * BN) % THREADS == 0
    comptime assert 4 * (BK * BM + BK * BN) <= B.threadgroup_bytes

    var tid = Int(thread_idx.x)
    var trow = tid // (BN // TN)
    var tcol = tid % (BN // TN)
    var brow = Int(block_idx.y) * BM
    var bcol = Int(block_idx.x) * BN
    var z = Int(block_idx.z)
    var zlo = z % Int(o.zd) if o.zd > 0 else z
    var zhi = z // Int(o.zd) if o.zd > 0 else 0
    var za = zlo * Int(o.sa_z) + zhi * Int(o.sa_zz)
    var zb = zlo * Int(o.sb_z) + zhi * Int(o.sb_zz)
    var zc = zlo * Int(o.sc_z) + zhi * Int(o.sc_zz)
    var Mi = Int(M)
    var Ni = Int(N)
    var Ki = Int(K)

    var As0 = stack_allocation[DType.uint8, address_space=AddressSpace.SHARED](row_major[BK, BM]())
    var As1 = stack_allocation[DType.uint8, address_space=AddressSpace.SHARED](row_major[BK, BM]())
    var As2 = stack_allocation[DType.uint8, address_space=AddressSpace.SHARED](row_major[BK, BM]())
    var As3 = stack_allocation[DType.uint8, address_space=AddressSpace.SHARED](row_major[BK, BM]())
    var Bs0 = stack_allocation[DType.uint8, address_space=AddressSpace.SHARED](row_major[BK, BN]())
    var Bs1 = stack_allocation[DType.uint8, address_space=AddressSpace.SHARED](row_major[BK, BN]())
    var Bs2 = stack_allocation[DType.uint8, address_space=AddressSpace.SHARED](row_major[BK, BN]())
    var Bs3 = stack_allocation[DType.uint8, address_space=AddressSpace.SHARED](row_major[BK, BN]())
    var c0 = InlineArray[SIMD[DType.int32, TN], TM](fill=SIMD[DType.int32, TN](0))
    var c1 = InlineArray[SIMD[DType.int32, TN], TM](fill=SIMD[DType.int32, TN](0))
    var c2 = InlineArray[SIMD[DType.int32, TN], TM](fill=SIMD[DType.int32, TN](0))
    var c3 = InlineArray[SIMD[DType.int32, TN], TM](fill=SIMD[DType.int32, TN](0))

    for kt in range(ceildiv(Ki, BK)):
        comptime for i in range(0, BM * BK, THREADS):
            var idx = i + tid
            var r = idx // BK
            var kk = idx % BK
            var v = F4(0)
            if brow + r < Mi and kt * BK + kk < Ki:
                v = base.unsafe_load[width=4](Int(o.a) + (brow + r) * Int(o.sa_m) + (kt * BK + kk) * Int(o.sa_k) + za)
            As0.ptr.unsafe_store(kk * BM + r, v[0])
            As1.ptr.unsafe_store(kk * BM + r, v[1])
            As2.ptr.unsafe_store(kk * BM + r, v[2])
            As3.ptr.unsafe_store(kk * BM + r, v[3])
        comptime for i in range(0, BK * BN, THREADS):
            var idx = i + tid
            var kk = idx // BN
            var n = bcol + idx % BN
            var v = F4(0)
            if n < Ni and kt * BK + kk < Ki:
                v = L.load(base, o, kt * BK + kk, n // D, n % D, zb)
            Bs0.ptr.unsafe_store(kk * BN + idx % BN, v[0])
            Bs1.ptr.unsafe_store(kk * BN + idx % BN, v[1])
            Bs2.ptr.unsafe_store(kk * BN + idx % BN, v[2])
            Bs3.ptr.unsafe_store(kk * BN + idx % BN, v[3])
        barrier()
        comptime for kk in range(BK):
            var a0 = As0.ptr.unsafe_load[width=TM](kk * BM + trow * TM).cast[DType.int32]()
            var a1 = As1.ptr.unsafe_load[width=TM](kk * BM + trow * TM).cast[DType.int32]()
            var a2 = As2.ptr.unsafe_load[width=TM](kk * BM + trow * TM).cast[DType.int32]()
            var a3 = As3.ptr.unsafe_load[width=TM](kk * BM + trow * TM).cast[DType.int32]()
            var b0 = Bs0.ptr.unsafe_load[width=TN](kk * BN + tcol * TN).cast[DType.int32]()
            var b1 = Bs1.ptr.unsafe_load[width=TN](kk * BN + tcol * TN).cast[DType.int32]()
            var b2 = Bs2.ptr.unsafe_load[width=TN](kk * BN + tcol * TN).cast[DType.int32]()
            var b3 = Bs3.ptr.unsafe_load[width=TN](kk * BN + tcol * TN).cast[DType.int32]()
            # (a_lo + a_hi j)(b_lo + b_hi j) = a_lo b_lo + (2 + i) a_hi b_hi + (a_lo b_hi + a_hi b_lo) j
            tile_mac[B](a0, b0, c0)
            tile_mac[B, neg=True](a1, b1, c0)
            tile_mac[B](a2 + a2, b2, c0)             # 2 (a2 b2 - a3 b3) - (a2 b3 + a3 b2)
            tile_mac[B, neg=True](a3 + a3, b3, c0)
            tile_mac[B, neg=True](a2, b3, c0)
            tile_mac[B, neg=True](a3, b2, c0)
            tile_mac[B](a0, b1, c1)
            tile_mac[B](a1, b0, c1)
            tile_mac[B](a2 + a2, b3, c1)             # 2 (a2 b3 + a3 b2) + (a2 b2 - a3 b3)
            tile_mac[B](a3 + a3, b2, c1)
            tile_mac[B](a2, b2, c1)
            tile_mac[B, neg=True](a3, b3, c1)
            tile_mac[B](a0, b2, c2)
            tile_mac[B, neg=True](a1, b3, c2)
            tile_mac[B](a2, b0, c2)
            tile_mac[B, neg=True](a3, b1, c2)
            tile_mac[B](a0, b3, c3)
            tile_mac[B](a1, b2, c3)
            tile_mac[B](a2, b1, c3)
            tile_mac[B](a3, b0, c3)
        barrier()
        if ((kt + 1) * BK) % F4_TERMS == 0:
            tile_reduce(c0)
            tile_reduce(c1)
            tile_reduce(c2)
            tile_reduce(c3)

    tile_reduce(c0)
    tile_reduce(c1)
    tile_reduce(c2)
    tile_reduce(c3)
    comptime for i in range(TM):
        var m = brow + trow * TM + i
        comptime for j in range(TN):
            var n = bcol + tcol * TN + j
            if m < Mi and n < Ni:
                var v = F4(UInt8(c0[i][j]), UInt8(c1[i][j]), UInt8(c2[i][j]), UInt8(c3[i][j]))
                var at = Int(o.c) + m * Int(o.sc_m) + (n // D) * Int(o.sc_hi) + (n % D) * Int(o.sc_lo) + zc
                comptime if acc:
                    v = f_add(v, base.unsafe_load[width=4](at))
                base.unsafe_store[width=4](at, v)


def launch_gemm_f4[B: Backend, T: Tile, L: Loader4, D: Int, acc: Bool = False](
    ctx: DeviceContext, arena: Arena, o: Operands, M: Int, N: Int, K: Int, batch: Int = 1
) raises:
    comptime kernel = gemm_f4[B, T, L, D, acc]
    ctx.enqueue_function[kernel](arena.buf, o, Int32(M), Int32(N), Int32(K),
                                 grid_dim=(ceildiv(N, T.BN), ceildiv(M, T.BM), batch), block_dim=T.threads())

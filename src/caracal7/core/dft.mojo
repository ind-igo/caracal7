"""The axis-2 DFT as three radix stages, each a `gemm_f2` launch (docs/decisions.md).
A dense DFT of length n costs n multiply-adds per output; on the chain axis (2 h2 = 1792 for ECDSA)
that made the LDE quadratic in the chain count. With n = n1 n2 n3 (n1 the odd part, n2 <= n3 the
two halves of the power of two) the cost is n1 + n2 + n3 per output, three passes over memory.
Index split: input k = k1 + n1 k2 + n1 n2 k3, output j = jj + n2 n3 j1 with jj = j3 + n3 j2.
    stage 3   Y3[j3][k1 + n1 k2] = sum_k3 T3[j3, k3] x[k]                      T3 = s root^(n1 n2 j3 k3)
    stage 2   Y2[jj][k1]         = sum_k2 T2[j3][j2, k2] Y3[j3][k1 + n1 k2]    T2 = root^(n1 jj k2)
    stage 1   X[j]               = sum_k1 T1[jj][j1, k1] Y2[jj][k1]            T1 = root^(j k1)
The twists of Cooley-Tukey are folded into the per-batch tables T2 and T1, so each stage is one
batched GEMM with the batch split as (column, j3) or (column, jj) through `Operands.zd`.
Rows hold W F2 values (the axis-1 index); every n-row buffer has column stride n W 2 bytes."""
from max.gpu.host import DeviceContext

from caracal7.core.params import Params
from caracal7.core.backend import BACKEND, Tile, Strided, launch_gemm_f2, strided
from caracal7.core.arena import Arena


@fieldwise_init
struct DftPlan(TrivialRegisterPassable):
    """n = n1 n2 n3 and the nonzero inputs k3 of stage 3 (k_in = n1 n2 k3)."""
    var n: Int
    var n1: Int
    var n2: Int
    var n3: Int
    var k3: Int

    def __init__(out self, n: Int, k_in: Int):
        var odd = n
        while odd % 2 == 0:
            odd //= 2
        var a = 0
        var t = n // odd
        while t > 1:
            t //= 2
            a += 1
        self.n = n
        self.n1 = odd
        self.n2 = 1 << (a // 2)
        self.n3 = 1 << (a - a // 2)
        self.k3 = k_in // (self.n1 * self.n2)

    def t3(self) -> Int:
        return 0

    def t2(self) -> Int:
        return self.n3 * self.k3 * 2

    def t1(self) -> Int:
        return self.t2() + self.n3 * self.n2 * self.n2 * 2

    def bytes(self) -> Int:
        return self.t1() + self.n2 * self.n3 * self.n1 * self.n1 * 2


def radix_tile[r: Int]() -> Tile:
    """A tile whose M side fits a radix-r stage (M = K = r)."""
    comptime if r <= 8:
        return Tile(BM=8, BN=128, BK=8, TM=8, TN=4)
    elif r <= 16:
        return Tile(BM=16, BN=64, BK=16, TM=4, TN=4)
    elif r <= 32:
        return Tile(BM=32, BN=64, BK=32, TM=4, TN=4)
    else:
        return BACKEND.tile


def dft_axis2[p: Params, forward: Bool](ctx: DeviceContext, arena: Arena,
                                        src: Int, dst: Int, W: Int, columns: Int, tab: Int) raises:
    """src rows k < h2 (per column, W F2 each, column stride h2 W 2) -> dst rows j < n, n = 2 h2
    forward (points of G2) or h2 inverse (coefficients). `src` is the stage-2 scratch and is
    overwritten; it must hold n rows per column. `tab` is the DftPlan table (tables.mojo)."""
    comptime h2 = p.h2()
    comptime plan = DftPlan(2 * h2, h2) if forward else DftPlan(h2, h2)
    comptime n = plan.n
    comptime n1 = plan.n1
    comptime n2 = plan.n2
    comptime n3 = plan.n3
    comptime k3 = plan.k3
    comptime assert n1 * n2 * k3 == h2, "the input length must split as n1 n2 k3"
    var R = W * 2
    launch_gemm_f2[BACKEND, radix_tile[n3](), Strided, 1](ctx, arena, strided(
        a=tab + plan.t3(), sa_m=k3 * 2, sa_k=2,
        b=src, sb_k=n1 * n2 * R, sb_hi=2, sb_lo=0, sb_z=h2 * R,
        c=dst, sc_m=n1 * n2 * R, sc_hi=2, sc_lo=0, sc_z=n * R), n3, n1 * n2 * W, k3, batch=columns)
    launch_gemm_f2[BACKEND, radix_tile[n2](), Strided, 1](ctx, arena, strided(
        a=tab + plan.t2(), sa_m=n2 * 2, sa_k=2, sa_z=n2 * n2 * 2,
        b=dst, sb_k=n1 * R, sb_hi=2, sb_lo=0, sb_z=n1 * n2 * R, sb_zz=n * R,
        c=src, sc_m=n3 * n1 * R, sc_hi=2, sc_lo=0, sc_z=n1 * R, sc_zz=n * R, zd=n3), n2, n1 * W, n2, batch=columns * n3)
    # ponytail: at n1 = 1 this stage is a copy; skip it when a grid with a power-of-two chain axis matters
    launch_gemm_f2[BACKEND, radix_tile[n1](), Strided, 1](ctx, arena, strided(
        a=tab + plan.t1(), sa_m=n1 * 2, sa_k=2, sa_z=n1 * n1 * 2,
        b=src, sb_k=R, sb_hi=2, sb_lo=0, sb_z=n1 * R, sb_zz=n * R,
        c=dst, sc_m=n2 * n3 * R, sc_hi=2, sc_lo=0, sc_z=R, sc_zz=n * R, zd=n2 * n3), n1, W, n1, batch=columns * n2 * n3)

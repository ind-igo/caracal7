"""Fields: F = F127 on bytes, the quadratic tower F2, F4, and E: F4[v] / (v^5 - G5) = F_(127^20) by
default, or the quadratic tower F8, F16 = F_(127^16) when `E16` below is True (measurement only: the
soundness ledger puts that field under 96 bits). A build flag cannot drive it: `is_defined` stays
symbolic inside types and the E widths would not unify, so the switch is this one constant.

Storage: one byte per F coordinate, values 0..126. A level-k tower element is SIMD[uint8, 2^k]; the low
half is the coordinate on 1, the high half on the level generator g_k, with g_k^2 = C_k (a non-square in
the level-(k-1) field).

E at e = 20 is level 5 of the ext_* functions: five F4 limbs a_0 + a_1 v + ... + a_4 v^4, limb m on lanes
[4 m, 4 m + 4), so F4 acts limb-wise on E. The GPU only has power-of-two vectors, so an E value sits in
32 lanes with lanes 20..31 zero; it is stored as E_BYTES = 20 bytes (bytes.Buf[E_BYTES] splits the access
into 16 + 4). v^5 = G5 = i + j is irreducible over F4 because 5 divides |F4*| and G5 is not a fifth power
(test_field checks G5^((127^4 - 1) / 5) != 1). The q-Frobenius (q = 127^4) fixes F4 and sends v to
ZETA v with ZETA = G5^((q - 1) / 5) a fifth root of unity, so a conjugate is a limb-wise scaling and the
norm to F4 is the product of the five conjugates (ext_inv0).

Tower constants (docs/decisions.md):
    k=1  F2 = F[i],   i^2 = -1
    k=2  F4 = F2[j],  j^2 = 2 + i
    k=3  F8 = F4[u],  u^2 = j        (E16 only)
    k=4  E  = F8[y],  y^2 = u        (E16 only)
    E20  F4[v],       v^5 = i + j
"""

from std.math import min, floor
from std.sys.info import is_gpu

comptime P: Int = 127
comptime E16: Bool = False                # True: the 16-coordinate tower instead of the 20-coordinate E
comptime E_LEVEL: Int = 4 if E16 else 5    # the E level of the ext_* functions: 1 << E_LEVEL register lanes
comptime E_WIDTH: Int = 1 << E_LEVEL
comptime E_BYTES: Int = 16 if E16 else 20  # coordinates stored per E value; lanes E_BYTES.. are zero
comptime E_LIMBS: Int = 5
comptime E_DFT_V: Int = 4 if (E_BYTES // 2) % 4 == 0 else 2   # F2 lanes per thread when dft_axis walks E lines of E_BYTES // 2 F2 lanes

comptime F2 = SIMD[DType.uint8, 2]
comptime F4 = SIMD[DType.uint8, 4]
comptime E = SIMD[DType.uint8, E_WIDTH]
comptime E20 = SIMD[DType.uint8, 32]       # the e = 20 register form, named so its helpers compile under E16 too
comptime E16T = SIMD[DType.uint8, 16]      # the e = 16 tower form

comptime C1 = SIMD[DType.uint8, 1](126)                 # -1
comptime C2 = SIMD[DType.uint8, 2](2, 1)                # 2 + i
comptime C3 = SIMD[DType.uint8, 4](0, 0, 1, 0)          # j
comptime C4 = SIMD[DType.uint8, 8](0, 0, 0, 0, 1, 0, 0, 0)  # u (E16)
comptime G5 = F4(0, 1, 1, 0)                            # i + j: v^5
# ZETA^m for m < 5 (test_field checks ZETA1 = G5^((127^4 - 1) / 5) and the powers)
comptime ZETA1 = F4(95, 77, 62, 76)
comptime ZETA2 = F4(95, 50, 76, 112)
comptime ZETA3 = F4(95, 50, 51, 15)
comptime ZETA4 = F4(95, 77, 65, 51)


# ---- F127 on byte lanes -------------------------------------------------

@always_inline
def f_reduce[w: SIMDLength](x: SIMD[DType.uint32, w]) -> SIMD[DType.uint8, w]:
    """Reduce lanes below 2^21 to 0..126: three rounds of (x & 127) + (x >> 7), then 127 -> 0."""
    var r = (x & 127) + (x >> 7)
    r = (r & 127) + (r >> 7)
    r = (r & 127) + (r >> 7)
    return wcast[DType.uint8](min(r, r - 127))      # r in [0, 254): unsigned wrap picks the reduced value


comptime WIDE_BIAS: Int32 = 127 * (1 << 15)   # added before reducing a signed accumulator; |acc| must stay below it
comptime F4_MAC_MAX = 30                       # F4 MACs one wide accumulator holds: 30 * 8 * 126^2 < WIDE_BIAS


@always_inline
def f_reduce_signed[w: SIMDLength](x: SIMD[DType.int32, w]) -> SIMD[DType.uint8, w]:
    """Reduce signed lanes with |x| < WIDE_BIAS to 0..126."""
    var r = wcast[DType.uint32](x + WIDE_BIAS)     # < 2^23
    r = (r & 127) + (r >> 7)
    return f_reduce(r)


@always_inline
def f4_mac_wide(mut acc: SIMD[DType.int32, 4], a: SIMD[DType.uint8, 4], b: SIMD[DType.uint8, 4]):
    """acc += a * b in F4 without reduction, as signed int32 coordinates.
    Tower: i^2 = -1, j^2 = 2 + i; coordinates (1, i, j, ij). Reduce with f_reduce_signed after at most
    F4_MAC_MAX calls."""
    var x = a.cast[DType.int32]()
    var y = b.cast[DType.int32]()
    var p02 = x[0] * y[2] - x[1] * y[3] + x[2] * y[0] - x[3] * y[1]
    var p03 = x[0] * y[3] + x[1] * y[2] + x[2] * y[1] + x[3] * y[0]
    var q0 = x[2] * y[2] - x[3] * y[3]             # (a2 + a3 i)(b2 + b3 i), real
    var q1 = x[2] * y[3] + x[3] * y[2]             # imaginary
    acc[0] += x[0] * y[0] - x[1] * y[1] + 2 * q0 - q1
    acc[1] += x[0] * y[1] + x[1] * y[0] + 2 * q1 + q0
    acc[2] += p02
    acc[3] += p03


# ---- fp32 lanes: the ALU-bound kernels keep values as float32 (exact integers below 2^24). A value
# is "centered" when |x| <= 63 (190 after reducing a large one); tables and buffers stay canonical.

comptime V2 = SIMD[DType.float32, 2]
comptime V4 = SIMD[DType.float32, 4]
comptime FP_INV = Float32(1.0 / 127.0)
comptime FP_ROUND = Float32(12582912.0)        # 1.5 * 2^23: (v + FP_ROUND) - FP_ROUND is round(v) for |v| < 2^22


@always_inline
def wcast[sd: DType, w: SIMDLength, //, dt: DType](x: SIMD[sd, w]) -> SIMD[dt, w]:
    """SIMD.cast, 16 lanes at a time past 16: Metal's compiler crashes on a 32-lane conversion (a v32
    is fine in arithmetic). Every cast of a value that can be E-wide goes through here."""
    comptime if w > 16:
        var lo = wcast[dt](x.slice[w // 2]())
        var hi = wcast[dt](x.slice[w // 2, offset = w // 2]())
        return rebind[SIMD[dt, w]](lo.join(hi))
    else:
        return x.cast[dt]()


@always_inline
def to_f32[w: SIMDLength](x: SIMD[DType.uint8, w]) -> SIMD[DType.float32, w]:
    return wcast[DType.float32](x)


@always_inline
def to_u8[w: SIMDLength](x: SIMD[DType.float32, w]) -> SIMD[DType.uint8, w]:
    return wcast[DType.uint8](x)


@always_inline
def fp_reduce[w: SIMDLength](x: SIMD[DType.float32, w]) -> SIMD[DType.float32, w]:
    """The centered residue x - 127 round(x / 127): |result| <= 63 for |x| < 4 M, <= 190 for |x| < 2^24
    (measured: 64). The rounding relies on (v + FP_ROUND) - FP_ROUND being evaluated as written; no
    fast-math reassociation."""
    var q = (x * FP_INV + FP_ROUND) - FP_ROUND
    return q.fma(SIMD[DType.float32, w](-127.0), x)


@always_inline
def fp_center[w: SIMDLength](t: SIMD[DType.uint8, w]) -> SIMD[DType.float32, w]:
    """A canonical value as a float in [-63, 63]; twiddles load this way so products stay small."""
    var f = to_f32(t)
    return f - 127.0 * floor((f + 63.5) * FP_INV)


@always_inline
def fp_canonical[w: SIMDLength](x: SIMD[DType.float32, w]) -> SIMD[DType.uint8, w]:
    """The store form 0..126 of a value with |x| < 2^24: the centered residue r, |r| <= 190, then
    r - 127 floor((r + 1/2) / 127), exact at r = +-127 where floor(r / 127) could round wrong."""
    var r = fp_reduce(x)
    return to_u8(r - 127.0 * floor((r + 0.5) * FP_INV))


@always_inline
def fp_mul_f2(a: V2, b: V2) -> V2:
    """a * b in F2: (a0 + a1 i)(b0 + b1 i), 4 products."""
    return V2(a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


@always_inline
def fp_mul2(w: V2, y: V4) -> V4:
    """w * y for w in F2: (w0 + w1 i)(y0 + y1 i + y2 j + y3 ij), 8 products."""
    var ys = V4(y[1], y[0], y[3], y[2])
    return (V4(w[1]) * V4(-1.0, 1.0, -1.0, 1.0)).fma(ys, V4(w[0]) * y)


@always_inline
def fp_mul4(x: V4, y: V4) -> V4:
    """x * y in F4, the tower of f4_mac_wide; |result| <= 8 |x| |y| per coordinate."""
    var p02 = x[0] * y[2] - x[1] * y[3] + x[2] * y[0] - x[3] * y[1]
    var p03 = x[0] * y[3] + x[1] * y[2] + x[2] * y[1] + x[3] * y[0]
    var q0 = x[2] * y[2] - x[3] * y[3]
    var q1 = x[2] * y[3] + x[3] * y[2]
    return V4(x[0] * y[0] - x[1] * y[1] + 2.0 * q0 - q1, x[0] * y[1] + x[1] * y[0] + 2.0 * q1 + q0, p02, p03)


@always_inline
def fp_const_mul[k: Int](x: SIMD[DType.float32, 1 << (k - 1)]) -> SIMD[DType.float32, 1 << (k - 1)]:
    """C_k x in the level-(k-1) field: -x, (2 + i) x, j x. Every lane of the result is at most
    three lanes of x in size."""
    comptime if k == 1:
        return -x
    elif k == 2:
        return rebind[SIMD[DType.float32, 1 << (k - 1)]](V2(2.0 * x[0] - x[1], x[0] + 2.0 * x[1]))
    elif k == 3:
        return rebind[SIMD[DType.float32, 1 << (k - 1)]](V4(2.0 * x[2] - x[3], x[2] + 2.0 * x[3], x[0], x[1]))
    else:
        comptime assert k == 4, "tower levels are 1..4"
        var lo = x.slice[4]()
        var hi = x.slice[4, offset=4]()
        return rebind[SIMD[DType.float32, 1 << (k - 1)]](fp_const_mul[3](hi).join(lo))


@always_inline
def fp_g5_mul(x: V4) -> V4:
    """G5 x = (i + j) x in F4 on float lanes: (-x1 + 2 x2 - x3, x0 + x2 + 2 x3, x0 - x3, x1 + x2).
    Every lane of the result is at most four lanes of x in size."""
    return V4(2.0 * x[2] - x[1] - x[3], x[0] + x[2] + 2.0 * x[3], x[0] - x[3], x[1] + x[2])


comptime EF = SIMD[DType.float32, E_WIDTH]      # E on float lanes
comptime EF20 = SIMD[DType.float32, 32]
comptime VH = SIMD[DType.float32, E_WIDTH // 2] # one plane of E (the real or the imaginary F2 parts) on float lanes
comptime BH = SIMD[DType.uint8, E_WIDTH // 2]   # the same plane as bytes


# E as two planes: the F2 lanes of an E value are (re, im) pairs; the GEMM skeleton and the lane kernels
# hold the two planes apart. `rebind` because the compiler does not unify `W // 2` with deinterleave's width.

@always_inline
def e_planes(x: E) -> Tuple[BH, BH]:
    var d = x.deinterleave()
    return (rebind[BH](d[0]), rebind[BH](d[1]))


@always_inline
def e_merge(re: BH, im: BH) -> E:
    return rebind[E](re.interleave(im))


@always_inline
def ef_planes(x: EF) -> Tuple[VH, VH]:
    var d = x.deinterleave()
    return (rebind[VH](d[0]), rebind[VH](d[1]))


@always_inline
def ef_merge(re: VH, im: VH) -> EF:
    return rebind[EF](re.interleave(im))


@always_inline
def fp_limb[m: Int](a: EF20) -> V4:
    return a.slice[4, offset = 4 * m]()


@always_inline
def fp_e_mul(a: EF20, b: EF20) -> EF20:
    """Five-limb schoolbook on float lanes, no reduction, output limb by limb (25 F4 products, |lane| <=
    8 |x| |y| each, then v^5 = G5 on the wrapped sum). Lane bound 136 |x| |y| per lane: with canonical
    operands (<= 126) every lane is below 2.2 M, with centered operands (<= 63) below 0.55 M; fp_reduce
    needs the sum of such products a caller accumulates to stay below 4 M for a centered result."""
    var r = EF20(0)
    comptime for t in range(E_LIMBS):
        var lo = V4(0)
        var hi = V4(0)
        comptime for m in range(E_LIMBS):
            comptime n = t - m
            comptime if n >= 0:
                lo += fp_mul4(fp_limb[m](a), fp_limb[n](b))
            elif t + E_LIMBS - m < E_LIMBS:
                hi += fp_mul4(fp_limb[m](a), fp_limb[t + E_LIMBS - m](b))
        comptime if t < E_LIMBS - 1:
            lo += fp_g5_mul(hi)
        comptime for l in range(4):
            r[4 * t + l] = lo[l]
    return r


@always_inline
def fp_ext_mul[k: Int](a: SIMD[DType.float32, 1 << k], b: SIMD[DType.float32, 1 << k]) -> SIMD[DType.float32, 1 << k]:
    """Schoolbook on the tower with float lanes, no reduction: (a0 + a1 g)(b0 + b1 g) = (a0 b0 + C a1 b1)
    + (a0 b1 + a1 b0) g. Lane bound 2^(2k - 1) P for k >= 1 and a base product bound P; at E_LEVEL the
    five-limb product of fp_e_mul."""
    comptime if k == 0:
        return a * b
    elif k == 5:
        return rebind[SIMD[DType.float32, 1 << k]](fp_e_mul(rebind[EF20](a), rebind[EF20](b)))
    else:
        comptime assert k <= 4, "tower levels are 0..4 and E20 is level 5"
        comptime h = 1 << (k - 1)
        var a0 = a.slice[h]()
        var a1 = a.slice[h, offset=h]()
        var b0 = b.slice[h]()
        var b1 = b.slice[h, offset=h]()
        var lo = fp_ext_mul[k - 1](a0, b0) + fp_const_mul[k](fp_ext_mul[k - 1](a1, b1))
        var hi = fp_ext_mul[k - 1](a0, b1) + fp_ext_mul[k - 1](a1, b0)
        return rebind[SIMD[DType.float32, 1 << k]](lo.join(hi))


@always_inline
def fp_ext_pow[k: Int](a: SIMD[DType.uint8, 1 << k], n: Int) -> SIMD[DType.float32, 1 << k]:
    """a^n on float lanes, reduced after every product: |result| <= 190."""
    var base = to_f32(a)
    var acc = SIMD[DType.float32, 1 << k](0)
    acc[0] = 1
    var m = n
    while m > 0:
        if m & 1:
            acc = fp_reduce(fp_ext_mul[k](acc, base))
        base = fp_reduce(fp_ext_mul[k](base, base))
        m >>= 1
    return acc


@always_inline
def f_add[w: SIMDLength](a: SIMD[DType.uint8, w], b: SIMD[DType.uint8, w]) -> SIMD[DType.uint8, w]:
    var s = a + b                                   # < 254, no overflow
    return min(s, s - 127)


@always_inline
def f_sub[w: SIMDLength](a: SIMD[DType.uint8, w], b: SIMD[DType.uint8, w]) -> SIMD[DType.uint8, w]:
    var s = a + 127 - b                             # in [1, 253]
    return min(s, s - 127)


@always_inline
def f_neg[w: SIMDLength](a: SIMD[DType.uint8, w]) -> SIMD[DType.uint8, w]:
    var s = 127 - a                                 # in [1, 127]
    return min(s, s - 127)


@always_inline
def f_mul[w: SIMDLength](a: SIMD[DType.uint8, w], b: SIMD[DType.uint8, w]) -> SIMD[DType.uint8, w]:
    var p = wcast[DType.uint16](a) * wcast[DType.uint16](b)   # < 2^14
    var r = (p & 127) + (p >> 7)
    r = (r & 127) + (r >> 7)
    return wcast[DType.uint8](min(r, r - 127))


def f_pow[w: SIMDLength](a: SIMD[DType.uint8, w], n: Int) -> SIMD[DType.uint8, w]:
    var base = a
    var acc = SIMD[DType.uint8, w](1)
    var k = n
    while k > 0:
        if k & 1:
            acc = f_mul(acc, base)
        base = f_mul(base, base)
        k >>= 1
    return acc


def f_inv(a: Scalar[DType.uint8]) raises -> Scalar[DType.uint8]:
    if a == 0:
        raise Error("f_inv(0)")
    return f_pow(a, 125)


# ---- Quadratic tower (k <= 2) and E (k == E_LEVEL) ---------------------

@always_inline
def _level_const[k: Int]() -> SIMD[DType.uint8, 1 << (k - 1)]:
    """C_k as an element of the level-(k-1) field."""
    comptime if k == 1:
        return rebind[SIMD[DType.uint8, 1 << (k - 1)]](C1)
    elif k == 2:
        return rebind[SIMD[DType.uint8, 1 << (k - 1)]](C2)
    elif k == 3:
        return rebind[SIMD[DType.uint8, 1 << (k - 1)]](C3)
    else:
        return rebind[SIMD[DType.uint8, 1 << (k - 1)]](C4)


@always_inline
def _zeta_pow[m: Int]() -> F4:
    """ZETA^(m mod 5)."""
    comptime if m % 5 == 0:
        return F4(1, 0, 0, 0)
    elif m % 5 == 1:
        return ZETA1
    elif m % 5 == 2:
        return ZETA2
    elif m % 5 == 3:
        return ZETA3
    else:
        return ZETA4


@always_inline
def limb[m: Int](a: E20) -> F4:
    """Limb m: the F4 coefficient of v^m."""
    return a.slice[4, offset = 4 * m]()


@always_inline
def e_from_limbs(l0: F4, l1: F4, l2: F4, l3: F4, l4: F4) -> E20:
    """Lane stores, not SIMD.insert: Metal has no llvm.vector.insert intrinsic."""
    var r = E20(0)
    comptime for l in range(4):
        r[l] = l0[l]
        r[4 + l] = l1[l]
        r[8 + l] = l2[l]
        r[12 + l] = l3[l]
        r[16 + l] = l4[l]
    return r


@always_inline
def _e_mul(a: E20, b: E20) -> E20:
    """Five-limb schoolbook, output limb by limb: c_t = sum_{m + n = t} a_m b_n + G5 sum_{m + n = t + 5} a_m b_n,
    each sum in one wide F4 accumulator (at most 5 MACs). No arrays: Metal compiles register code only."""
    var r = E20(0)
    comptime for t in range(E_LIMBS):
        var lo = SIMD[DType.int32, 4](0)
        var hi = SIMD[DType.int32, 4](0)
        comptime for m in range(E_LIMBS):
            comptime n = t - m
            comptime if n >= 0:
                f4_mac_wide(lo, limb[m](a), limb[n](b))
            elif t + E_LIMBS - m < E_LIMBS:
                f4_mac_wide(hi, limb[m](a), limb[t + E_LIMBS - m](b))
        var v = f_reduce_signed(lo)
        comptime if t < E_LIMBS - 1:
            v = f_add(v, ext_mul[2](G5, f_reduce_signed(hi)))
        comptime for l in range(4):
            r[4 * t + l] = v[l]
    return r


@always_inline
def ext_mul[k: Int](a: SIMD[DType.uint8, 1 << k], b: SIMD[DType.uint8, 1 << k]) -> SIMD[DType.uint8, 1 << k]:
    """Schoolbook on the tower: (a0 + a1 g)(b0 + b1 g) = (a0 b0 + C a1 b1) + (a0 b1 + a1 b0) g; the
    five-limb product at E_LEVEL."""
    comptime if k == 0:
        return f_mul(a, b)
    elif k == 5 and not is_gpu():
        # host: the float-lane product (25 fp_mul4 on NEON) beats the wide-int MACs; the result is exact below 2^24
        return rebind[SIMD[DType.uint8, 1 << k]](fp_canonical(fp_e_mul(to_f32(rebind[E20](a)), to_f32(rebind[E20](b)))))
    elif k == 5:
        return rebind[SIMD[DType.uint8, 1 << k]](_e_mul(rebind[E20](a), rebind[E20](b)))
    elif k == 4 and not is_gpu():
        return rebind[SIMD[DType.uint8, 1 << k]](e_mul_power(rebind[E16T](a), rebind[E16T](b)))   # ponytail: host only; try on device when measured
    else:
        comptime assert k <= 4, "tower levels are 0..4 and E20 is level 5"
        comptime h = 1 << (k - 1)
        var a0 = a.slice[h]()
        var a1 = a.slice[h, offset=h]()
        var b0 = b.slice[h]()
        var b1 = b.slice[h, offset=h]()
        var lo = f_add(ext_mul[k - 1](a0, b0), ext_mul[k - 1](_level_const[k](), ext_mul[k - 1](a1, b1)))
        var hi = f_add(ext_mul[k - 1](a0, b1), ext_mul[k - 1](a1, b0))
        return rebind[SIMD[DType.uint8, 1 << k]](lo.join(hi))


@always_inline
def e_frobenius[i: Int](a: E20) -> E20:
    """The i-th q-Frobenius conjugate a^(q^i), q = 127^4: limb m scaled by ZETA^(i m)."""
    return e_from_limbs(limb[0](a), ext_mul[2](_zeta_pow[i](), limb[1](a)), ext_mul[2](_zeta_pow[2 * i](), limb[2](a)),
                        ext_mul[2](_zeta_pow[3 * i](), limb[3](a)), ext_mul[2](_zeta_pow[4 * i](), limb[4](a)))


@always_inline
def e_scale(s: F4, a: E20) -> E20:
    """s * a for s in F4: limb-wise."""
    return e_from_limbs(ext_mul[2](s, limb[0](a)), ext_mul[2](s, limb[1](a)), ext_mul[2](s, limb[2](a)),
                        ext_mul[2](s, limb[3](a)), ext_mul[2](s, limb[4](a)))


@always_inline
def ext_conj[k: Int](a: SIMD[DType.uint8, 1 << k]) -> SIMD[DType.uint8, 1 << k]:
    """a0 - a1 g: the level-k conjugate (1 <= k <= 4)."""
    comptime h = 1 << (k - 1)
    var a0 = a.slice[h]()
    var a1 = a.slice[h, offset=h]()
    return rebind[SIMD[DType.uint8, 1 << k]](a0.join(f_neg(a1)))


@always_inline
def ext_norm[k: Int](a: SIMD[DType.uint8, 1 << k]) -> SIMD[DType.uint8, 1 << (k - 1)]:
    """N(a) = a0^2 - C a1^2 in the level-(k-1) field (1 <= k <= 4)."""
    comptime h = 1 << (k - 1)
    var a0 = a.slice[h]()
    var a1 = a.slice[h, offset=h]()
    return f_sub(ext_mul[k - 1](a0, a0), ext_mul[k - 1](_level_const[k](), ext_mul[k - 1](a1, a1)))


def ext_inv0[k: Int](a: SIMD[DType.uint8, 1 << k]) -> SIMD[DType.uint8, 1 << k]:
    """Norm descent: a^-1 = conj(a) / N(a) on the tower; at E_LEVEL the product of the four nontrivial
    Frobenius conjugates over the norm in F4. 0 -> 0 (a^125 = 0 in F). The kernel form."""
    comptime if k == 0:
        return f_pow(a, 125)
    elif k == 5:
        var x = rebind[E20](a)
        var others = ext_mul[5](ext_mul[5](e_frobenius[1](x), e_frobenius[2](x)), ext_mul[5](e_frobenius[3](x), e_frobenius[4](x)))
        var n = limb[0](ext_mul[5](x, others))            # the norm lies in F4: limbs 1..4 are zero
        return rebind[SIMD[DType.uint8, 1 << k]](e_scale(ext_inv0[2](n), others))
    else:
        comptime h = 1 << (k - 1)
        var n_inv = ext_inv0[k - 1](ext_norm[k](a))
        var c = ext_conj[k](a)
        var c0 = c.slice[h]()
        var c1 = c.slice[h, offset=h]()
        return rebind[SIMD[DType.uint8, 1 << k]](ext_mul[k - 1](c0, n_inv).join(ext_mul[k - 1](c1, n_inv)))


def ext_inv[k: Int](a: SIMD[DType.uint8, 1 << k]) raises -> SIMD[DType.uint8, 1 << k]:
    """Raises on zero."""
    if a == SIMD[DType.uint8, 1 << k](0):
        raise Error("ext_inv(0)")
    return ext_inv0[k](a)


def ext_pow[k: Int](a: SIMD[DType.uint8, 1 << k], n: Int) -> SIMD[DType.uint8, 1 << k]:
    var base = a
    var acc = SIMD[DType.uint8, 1 << k](0)
    acc[0] = 1
    var m = n
    while m > 0:
        if m & 1:
            acc = ext_mul[k](acc, base)
        base = ext_mul[k](base, base)
        m >>= 1
    return acc


@always_inline
def ext_embed[k: Int, w: SIMDLength](a: SIMD[DType.uint8, w]) -> SIMD[DType.uint8, 1 << k]:
    """Zero-pad a lower-level element into level k."""
    var r = SIMD[DType.uint8, 1 << k](0)
    comptime for t in range(w):
        r[t] = a[t]
    return r


@always_inline
def ext_one[k: Int]() -> SIMD[DType.uint8, 1 << k]:
    """The multiplicative identity of level k."""
    return ext_embed[k](SIMD[DType.uint8, 1](1))


# ---- E16 product in the power basis (host) ------------------------------
# F16 = F[y] / (y^16 - 4 y^8 + 5): u = y^2, j = y^4, i = y^8 - 2. Tower index t = t0 + 2 t1 + 4 t2 + 8 t3
# is i^t0 j^t1 u^t2 y^t3, so its t0 = 0 part has y-exponent 4 t1 + 2 t2 + t3. One 16 x 16 convolution in
# 32-bit lanes replaces the 625 scalar products of the recursive schoolbook (324 -> 58 ns on the host).

@always_inline
def _texp(m: Int) -> Int:
    """Tower index (t0 = 0) whose y-exponent is m."""
    return 2 * ((m & 1) * 4 + (m & 2) + (m >> 2))


@always_inline
def e_to_power(a: E16T) -> E16T:
    """Lane m of the result is the coefficient of y^m."""
    var ev = SIMD[DType.uint8, 8]()
    var od = SIMD[DType.uint8, 8]()
    comptime for m in range(8):
        ev[m] = a[_texp(m)]
        od[m] = a[_texp(m) + 1]
    var lo = f_sub(ev, f_add(od, od))       # y^m: a_even - 2 a_odd (i = y^8 - 2)
    return rebind[E16T](lo.join(od))        # y^(m+8): a_odd


@always_inline
def e_from_power(p: E16T) -> E16T:
    var lo = p.slice[8]()
    var hi = p.slice[8, offset=8]()
    var ev = f_add(lo, f_add(hi, hi))
    var a = E16T()
    comptime for m in range(8):
        a[_texp(m)] = ev[m]
        a[_texp(m) + 1] = hi[m]
    return a


@always_inline
def e_mul_power(a: E16T, b: E16T) -> E16T:
    var pa = e_to_power(a)
    var b32 = e_to_power(b).cast[DType.uint32]().join(SIMD[DType.uint32, 16](0))
    var acc = SIMD[DType.uint32, 32](0)
    comptime for m in range(16):
        acc += (b32 * UInt32(pa[m])).shift_right[m]()     # lanes < 16 * 126^2
    var c = f_reduce(acc).cast[DType.uint32]()
    var top = c.slice[8, offset=24]()                      # y^16 = 4 y^8 + 122
    var mid = c.slice[8, offset=16]() + top * 4
    var lo8 = c.slice[8, offset=8]() + top * 122 + mid * 4
    var lo0 = c.slice[8]() + mid * 122                     # lanes < 2^17
    return e_from_power(f_reduce(lo0.join(lo8)))

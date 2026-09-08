"""Fields: F = F127 on bytes, and the quadratic tower F2, F4, F8, E = F16.

Storage: one byte per F coordinate, values 0..126. A level-k tower element is
SIMD[uint8, 2^k]; the low half is the coordinate on 1, the high half on the
level generator g_k, with g_k^2 = C_k (a non-square in the level-(k-1) field).

Tower constants (docs/decisions.md):
    k=1  F2 = F[i],   i^2 = -1
    k=2  F4 = F2[j],  j^2 = 2 + i
    k=3  F8 = F4[u],  u^2 = j
    k=4  E  = F8[y],  y^2 = u
"""

from std.math import min
from std.sys.info import is_gpu

comptime P: Int = 127
comptime E_LEVEL: Int = 4          # E = F_{127^16}
comptime E_WIDTH: Int = 1 << E_LEVEL

comptime F2 = SIMD[DType.uint8, 2]
comptime F4 = SIMD[DType.uint8, 4]
comptime E = SIMD[DType.uint8, 16]

comptime C1 = SIMD[DType.uint8, 1](126)                 # -1
comptime C2 = SIMD[DType.uint8, 2](2, 1)                # 2 + i
comptime C3 = SIMD[DType.uint8, 4](0, 0, 1, 0)          # j
comptime C4 = SIMD[DType.uint8, 8](0, 0, 0, 0, 1, 0, 0, 0)  # u


# ---- F127 on byte lanes -------------------------------------------------

@always_inline
def f_reduce[w: SIMDLength](x: SIMD[DType.uint32, w]) -> SIMD[DType.uint8, w]:
    """Reduce lanes below 2^21 to 0..126: three rounds of (x & 127) + (x >> 7), then 127 -> 0."""
    var r = (x & 127) + (x >> 7)
    r = (r & 127) + (r >> 7)
    r = (r & 127) + (r >> 7)
    return min(r, r - 127).cast[DType.uint8]()      # r in [0, 254): unsigned wrap picks the reduced value


comptime WIDE_BIAS: Int32 = 127 * (1 << 15)   # added before reducing a signed accumulator; |acc| must stay below it
comptime F4_MAC_MAX = 30                       # F4 MACs one wide accumulator holds: 30 * 8 * 126^2 < WIDE_BIAS


@always_inline
def f_reduce_signed[w: SIMDLength](x: SIMD[DType.int32, w]) -> SIMD[DType.uint8, w]:
    """Reduce signed lanes with |x| < WIDE_BIAS to 0..126."""
    var r = (x + WIDE_BIAS).cast[DType.uint32]()   # < 2^23
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
    var p = a.cast[DType.uint16]() * b.cast[DType.uint16]()   # < 2^14
    var r = (p & 127) + (p >> 7)
    r = (r & 127) + (r >> 7)
    return min(r, r - 127).cast[DType.uint8]()


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


# ---- Quadratic tower ----------------------------------------------------

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
def ext_mul[k: Int](a: SIMD[DType.uint8, 1 << k], b: SIMD[DType.uint8, 1 << k]) -> SIMD[DType.uint8, 1 << k]:
    """Schoolbook on the tower: (a0 + a1 g)(b0 + b1 g) = (a0 b0 + C a1 b1) + (a0 b1 + a1 b0) g."""
    comptime if k == 0:
        return f_mul(a, b)
    elif k == E_LEVEL and not is_gpu():
        return rebind[SIMD[DType.uint8, 1 << k]](e_mul_power(rebind[E](a), rebind[E](b)))   # ponytail: host only; try on device when measured
    else:
        comptime h = 1 << (k - 1)
        var a0 = a.slice[h]()
        var a1 = a.slice[h, offset=h]()
        var b0 = b.slice[h]()
        var b1 = b.slice[h, offset=h]()
        var lo = f_add(ext_mul[k - 1](a0, b0), ext_mul[k - 1](_level_const[k](), ext_mul[k - 1](a1, b1)))
        var hi = f_add(ext_mul[k - 1](a0, b1), ext_mul[k - 1](a1, b0))
        return rebind[SIMD[DType.uint8, 1 << k]](lo.join(hi))


# ---- E product in the power basis (host) ---------------------------------
# E = F[y] / (y^16 - 4 y^8 + 5): u = y^2, j = y^4, i = y^8 - 2. Tower index t = t0 + 2 t1 + 4 t2 + 8 t3
# is i^t0 j^t1 u^t2 y^t3, so its t0 = 0 part has y-exponent 4 t1 + 2 t2 + t3. One 16 x 16 convolution in
# 32-bit lanes replaces the 625 scalar products of the recursive schoolbook (324 -> 58 ns on the host).

@always_inline
def _texp(m: Int) -> Int:
    """Tower index (t0 = 0) whose y-exponent is m."""
    return 2 * ((m & 1) * 4 + (m & 2) + (m >> 2))


@always_inline
def e_to_power(a: E) -> E:
    """Lane m of the result is the coefficient of y^m."""
    var ev = SIMD[DType.uint8, 8]()
    var od = SIMD[DType.uint8, 8]()
    comptime for m in range(8):
        ev[m] = a[_texp(m)]
        od[m] = a[_texp(m) + 1]
    var lo = f_sub(ev, f_add(od, od))       # y^m: a_even - 2 a_odd (i = y^8 - 2)
    return rebind[E](lo.join(od))           # y^(m+8): a_odd


@always_inline
def e_from_power(p: E) -> E:
    var lo = p.slice[8]()
    var hi = p.slice[8, offset=8]()
    var ev = f_add(lo, f_add(hi, hi))
    var a = E()
    comptime for m in range(8):
        a[_texp(m)] = ev[m]
        a[_texp(m) + 1] = hi[m]
    return a


@always_inline
def e_mul_power(a: E, b: E) -> E:
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


@always_inline
def ext_conj[k: Int](a: SIMD[DType.uint8, 1 << k]) -> SIMD[DType.uint8, 1 << k]:
    """a0 - a1 g: the level-k conjugate (k >= 1)."""
    comptime h = 1 << (k - 1)
    var a0 = a.slice[h]()
    var a1 = a.slice[h, offset=h]()
    return rebind[SIMD[DType.uint8, 1 << k]](a0.join(f_neg(a1)))


@always_inline
def ext_norm[k: Int](a: SIMD[DType.uint8, 1 << k]) -> SIMD[DType.uint8, 1 << (k - 1)]:
    """N(a) = a0^2 - C a1^2 in the level-(k-1) field (k >= 1)."""
    comptime h = 1 << (k - 1)
    var a0 = a.slice[h]()
    var a1 = a.slice[h, offset=h]()
    return f_sub(ext_mul[k - 1](a0, a0), ext_mul[k - 1](_level_const[k](), ext_mul[k - 1](a1, a1)))


def ext_inv0[k: Int](a: SIMD[DType.uint8, 1 << k]) -> SIMD[DType.uint8, 1 << k]:
    """Norm descent: a^-1 = conj(a) / N(a); 0 -> 0 (a^125 = 0 in F). The kernel form."""
    comptime if k == 0:
        return f_pow(a, 125)
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

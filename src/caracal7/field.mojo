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
    else:
        comptime h = 1 << (k - 1)
        var a0 = a.slice[h]()
        var a1 = a.slice[h, offset=h]()
        var b0 = b.slice[h]()
        var b1 = b.slice[h, offset=h]()
        var lo = f_add(ext_mul[k - 1](a0, b0), ext_mul[k - 1](_level_const[k](), ext_mul[k - 1](a1, b1)))
        var hi = f_add(ext_mul[k - 1](a0, b1), ext_mul[k - 1](a1, b0))
        return rebind[SIMD[DType.uint8, 1 << k]](lo.join(hi))


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


def ext_inv[k: Int](a: SIMD[DType.uint8, 1 << k]) raises -> SIMD[DType.uint8, 1 << k]:
    """Norm descent: a^-1 = conj(a) / N(a). Raises on zero."""
    comptime if k == 0:
        return f_inv(a[0])
    else:
        comptime h = 1 << (k - 1)
        var n_inv = ext_inv[k - 1](ext_norm[k](a))
        var c = ext_conj[k](a)
        var c0 = c.slice[h]()
        var c1 = c.slice[h, offset=h]()
        return rebind[SIMD[DType.uint8, 1 << k]](ext_mul[k - 1](c0, n_inv).join(ext_mul[k - 1](c1, n_inv)))


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


# ---- E x F2 wide MAC (residual stage): E over F2 has coordinates (2k, 2k+1) = (u, v) of one F2 slot ----
comptime E_MAC_MAX = 64      # 64 * 2 * 126^2 < WIDE_BIAS


def e_mac_f2_wide(mut ev: SIMD[DType.int32, 8], mut od: SIMD[DType.int32, 8], a: E, b: F2):
    """(ev, od) += a * b with a in E (even lanes u, odd lanes v) and b = b0 + b1 i in F2."""
    var ae: SIMD[DType.uint8, 8]
    var ao: SIMD[DType.uint8, 8]
    ae, ao = a.deinterleave()
    var ae32 = ae.cast[DType.int32]()
    var ao32 = ao.cast[DType.int32]()
    var b0 = Int32(b[0])
    var b1 = Int32(b[1])
    ev += ae32 * b0 - ao32 * b1
    od += ae32 * b1 + ao32 * b0


def e_wide_reduce(mut ev: SIMD[DType.int32, 8], mut od: SIMD[DType.int32, 8]) -> E:
    """Reduce the two lane sets to F and interleave back to E; the accumulators restart from the result."""
    var re = f_reduce_signed(ev)
    var ro = f_reduce_signed(od)
    ev = re.cast[DType.int32]()
    od = ro.cast[DType.int32]()
    return re.interleave(ro)

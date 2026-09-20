"""The tensor form of the tail's queries (pcs/tensor.mojo) against the direct definitions."""

from std.testing import assert_equal, assert_true, TestSuite

from core.field import F4, E, f_add, f_sub, f_mul, ext_mul, ext_pow, ext_embed, E_LEVEL, E_BYTES
from core.params import CLIENT, Params
from core.tables import Domains
from core.bytes import host_base, list_e, Buf
from pcs.open import slot_weight, host_table
from pcs.encode import pack_index
from pcs.tail import fold8_host
from pcs.tensor import Unit, query_units, consistency_units, row_units, clear_value, f4_dual, f4_trace, f4_frob

comptime p = CLIENT.grid(72, 32)      # a1 = 3, m1 = 9, a2 = 5, m2 = 1
comptime q = CLIENT.grid(64, 24)      # a1 = 6, m1 = 1, a2 = 3, m2 = 3
comptime b = CLIENT.grid(24, 24)      # both odd parts: m1 = m2 = 3
comptime c = CLIENT.grid(12, 24)      # a1 = 2, m1 = 3, a2 = 3, m2 = 3


def _e(seed: Int) -> E:
    var v = E(0)
    for k in range(E_BYTES):
        v[k] = UInt8((seed * 31 + k * 17 + 5) % 127)
    return v


def _materialize(units: List[Unit], first: Int, digits: Int, m: Int) -> List[UInt8]:
    """The units as a vector over (digits from `first`, r)."""
    var count = digits - first
    var out = List[UInt8](length=m * (1 << count) * E_BYTES, fill=0)
    for i in range(len(units)):
        var m1 = len(units[i].s1)
        for idx in range(1 << count):
            for r2 in range(len(units[i].s2)):
                for r1 in range(m1):
                    var at = idx + (1 << count) * (r1 + m1 * r2)
                    var s = f_add(list_e(out, at), units[i].at(first, idx, count, r1, r2))
                    for t in range(E_BYTES):
                        out[at * E_BYTES + t] = s[t]
    return out^


def _check_query[pp: Params](z1: E, z2: E) raises:
    var d = Domains.__init__[pp]()
    var units = List[Unit]()
    query_units[pp](z1, z2, _e(9), d.rho1, d.rho2, units)
    var w = _materialize(units, 0, pp.a1 + pp.a2, pp.m1 * pp.m2)
    var tab = host_table[pp](z1, z2, d.rho1, d.rho2)
    var bad = 0
    for slot in range(pp.N()):
        var want = ext_mul[E_LEVEL](_e(9), slot_weight[pp](slot, host_base(tab), Buf[E_BYTES](0), 0, d.rho1, d.rho2))
        if list_e(w, slot) != want:
            bad += 1
    assert_equal(bad, 0)


def test_query_units_match_slot_weight() raises:
    _check_query[p](_e(1), _e(2))
    _check_query[q](_e(3), _e(4))
    _check_query[b](_e(7), _e(8))
    _check_query[b](E(0), _e(8))      # a zero coordinate: ext_inv0 keeps every product right


def test_trace_form_gives_the_coordinates() raises:
    var dual = f4_dual()
    for k in range(4):
        var bk = F4(0)
        bk[k] = 1
        for tau in range(4):
            var v = ext_mul[2](dual[tau], bk)
            var acc = F4(0)
            for j in range(4):
                acc = f_add(acc, f4_frob(v, j))
            assert_equal(Int(acc[1]) + Int(acc[2]) + Int(acc[3]), 0)
            assert_equal(Int(acc[0]), 1 if tau == k else 0)


def _check_consistency[pp: Params]() raises:
    var d = Domains.__init__[pp]()
    var pt = d.level1.point(7)
    var weights = InlineArray[E, 4](fill=E(0))
    for tau in range(4):
        weights[tau] = _e(20 + tau)
    var units = List[Unit]()
    consistency_units[pp](pt, weights, f4_dual(), 0, units)
    var w = _materialize(units, 0, pp.a1 + pp.a2, pp.m1 * pp.m2)
    var bad = 0
    for slot in range(pp.N()):
        var i: Int
        var j: Int
        i, j = pack_index[pp](slot)
        var bj = F4(0)
        bj[j] = 1
        var v = ext_mul[2](bj, ext_pow[2](pt, i))
        var want = E(0)
        for tau in range(4):
            want = f_add(want, f_mul(weights[tau], E(v[tau])))
        if list_e(w, slot) != want:
            bad += 1
    assert_equal(bad, 0)


def test_consistency_units_match_the_coordinate_functionals() raises:
    _check_consistency[p]()
    _check_consistency[q]()
    _check_consistency[b]()


def test_row_units_are_powers() raises:
    var d = Domains.__init__[q]()
    var pt = d.level1.point(11)
    var units = List[Unit]()
    comptime D = q.a1 + q.a2
    row_units(pt, _e(5), 3, D, q.m1, q.m2, units)
    var w = _materialize(units, 3, D, q.m1 * q.m2)
    var pw = ext_embed[E_LEVEL](F4(1, 0, 0, 0))
    var be = ext_embed[E_LEVEL](pt)
    for row in range(q.m1 * q.m2 * (1 << (D - 3))):
        assert_true(list_e(w, row) == ext_mul[E_LEVEL](_e(5), pw))
        pw = ext_mul[E_LEVEL](pw, be)


def _check_fold[q: Params]() raises:
    comptime D = q.a1 + q.a2
    var d = Domains.__init__[q]()
    var units = List[Unit]()
    query_units[q](_e(3), _e(4), _e(6), d.rho1, d.rho2, units)
    var full = _materialize(units, 0, D, q.m1 * q.m2)
    var r = List[UInt8]()
    for k in range(3):
        var rk = _e(40 + k)
        for t in range(E_BYTES):
            r.append(rk[t])
    var folded = fold8_host(full, q.N() // 8, r)
    for i in range(len(units)):
        for k in range(3):
            units[i].fold(k, list_e(r, k))
    var w = _materialize(units, 3, D, q.m1 * q.m2)
    assert_equal(len(w), len(folded))
    assert_true(w == folded)
    # the clear check is the same sum against a vector
    var y = List[UInt8]()
    for i in range(len(folded) // E_BYTES):
        var v = _e(100 + i)
        for t in range(E_BYTES):
            y.append(v[t])
    var lhs = E(0)
    for i in range(len(y) // E_BYTES):
        lhs = f_add(lhs, ext_mul[E_LEVEL](list_e(folded, i), list_e(y, i)))
    assert_true(clear_value(units, y, 3, D) == lhs)


def test_fold_matches_the_vector_fold() raises:
    _check_fold[q]()      # the folded digits are t and two of x1', twisted by r1 (m1 = 1)
    _check_fold[b]()      # a1 = 3: t, both digits of x1' twisted by r1 over m1 = 3; x2 stays, twisted by r2
    _check_fold[c]()      # a1 = 2: t, the one digit of x1', then the first digit of x2, twisted by r2


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

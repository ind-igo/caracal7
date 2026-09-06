"""Blake3 against the official test vectors (input byte i = i mod 251) and the keyed mode."""

from std.testing import assert_equal, TestSuite

from caracal7.core.bytes import host_base
from caracal7.core.hash import Blake3


def _pattern(n: Int) -> List[UInt8]:
    var l = List[UInt8](capacity=n)
    for i in range(n):
        l.append(UInt8(i % 251))
    return l^


def _hex(l: List[UInt8]) -> String:
    var s = String()
    for b in l:
        for d in [Int(b >> 4), Int(b & 15)]:
            s += chr(48 + d) if d < 10 else chr(87 + d)
    return s^


def _leaf(n: Int) raises -> String:
    var m = _pattern(n)
    var out = List[UInt8](length=32, fill=0)
    Blake3.leaf(host_base(m), n, host_base(out))
    return _hex(out)


def test_vectors() raises:
    assert_equal(_leaf(0), "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262")
    assert_equal(_leaf(1), "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213")
    assert_equal(_leaf(63), "e9bc37a594daad83be9470df7f7b3798297c3d834ce80ba85d6e207627b7db7b")
    assert_equal(_leaf(64), "4eed7141ea4a5cd4b788606bd23f46e212af9cacebacdc7d1f4c6dc7f2511b98")
    assert_equal(_leaf(65), "de1e5fa0be70df6d2be8fffd0e99ceaa8eb6e8c93a63f2d8d1c30ecb6b263dee")
    assert_equal(_leaf(1024), "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7")
    assert_equal(_leaf(1025), "d00278ae47eb27b34faecf67b4fe263f82d5412916c1ffd97c8cb7fb814b8444")
    assert_equal(_leaf(2048), "e776b6028c7cd22a4d0ba182a8bf62205d2ef576467e838ed6f2529b85fba24a")
    assert_equal(_leaf(2049), "5f4d72f40d7a5f82b15ca2b2e44b1de3c2ef86c426c95c1af0b6879522563030")
    assert_equal(_leaf(3072), "b98cb0ff3623be03326b373de6b9095218513e64f1ee2edd2525c7ad1e5cffd2")


def test_node_is_hash_of_concatenation() raises:
    var m = _pattern(64)
    var out = List[UInt8](length=32, fill=0)
    Blake3.node(host_base(m), host_base(m[32:]), host_base(out))
    assert_equal(_hex(out), _leaf(64))


def test_absorb_is_keyed_hash() raises:
    # keyed(key = bytes 0..31, message = pattern(100)); absorb prepends ds, so pattern(100) = ds || pattern(100)[1:]
    var key = List[UInt8](capacity=32)
    for i in range(32):
        key.append(UInt8(i))
    var m = _pattern(100)
    var tail = List[UInt8](capacity=99)
    for i in range(1, 100):
        tail.append(m[i])
    Blake3.absorb(host_base(key), 0, host_base(tail), 99)
    assert_equal(_hex(key), "2ea26063087b8022ad6417194c7f35f75c3baa86f93326c5df51bbb841d1552f")


def test_squeeze_is_keyed_hash_of_counter() raises:
    # blake3_keyed(key = bytes 0..31, message = LE64(0))
    var key = List[UInt8](capacity=32)
    for i in range(32):
        key.append(UInt8(i))
    var out = List[UInt8](length=32, fill=0)
    Blake3.squeeze(host_base(key), 0, host_base(out))
    assert_equal(_hex(out), "782c6ee8963e660954892cf37288ab697922f0fe3744dfa8b08162ac6438e8c1")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

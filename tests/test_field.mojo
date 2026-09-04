from std.testing import assert_equal, TestSuite
from caracal7.field import add


def test_add_wraps() raises:
    assert_equal(add(126, 3), 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

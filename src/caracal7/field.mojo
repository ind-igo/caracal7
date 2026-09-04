"""F127 base field. Placeholder until milestone 1 lands the tower E."""

comptime P: UInt8 = 127


def add(a: UInt8, b: UInt8) -> UInt8:
    var s = a + b
    return s - P if s >= P else s

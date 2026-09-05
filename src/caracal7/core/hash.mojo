"""Hash interface: the Merkle tree and the transcript are parameterized on `H: Hash`, so the hash is a
comptime choice next to Params. Blake3 is the default; a second implementation (Keccak, Poseidon for
recursion) conforms to the same trait and every kernel above it is unchanged.

The functions are plain Mojo with no allocation, so the same code runs inside a kernel (one thread
per Merkle node, one thread for the transcript) and on the host for the verifier.
"""

from caracal7.core.bytes import Base

trait Hash:
    comptime DIGEST: Int          # output bytes, 32 for Blake3
    comptime BLOCK: Int           # input block bytes, 64 for Blake3
    comptime NAME: StringSpan[ImmStaticOrigin]

    @staticmethod
    def leaf(src: Base, bytes: Int, out_ptr: Base):
        """Digest of one Merkle leaf: `bytes` bytes at `src` -> DIGEST bytes at `out_ptr`."""
        ...

    @staticmethod
    def node(left: Base, right: Base, out_ptr: Base):
        """Digest of two child digests."""
        ...

    @staticmethod
    def absorb(state: Base, ds: UInt8, src: Base, bytes: Int):
        """Transcript step: state <- H_keyed(key=state, ds || src[0:bytes]). Serial by definition."""
        ...

    @staticmethod
    def squeeze(state: Base, counter: Int, out_ptr: Base):
        """DIGEST bytes of challenge material for block `counter` from the current state."""
        ...


# ---------------------------------------------------------------------------------------------------
# Blake3 (https://github.com/BLAKE3-team/BLAKE3-specs). Hash and keyed_hash modes, 32-byte output.

comptime _W = SIMD[DType.uint32, 16]
comptime _CV = SIMD[DType.uint32, 8]
comptime _IV = _CV(0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19)
comptime _CHUNK_START: UInt32 = 1
comptime _CHUNK_END: UInt32 = 2
comptime _PARENT: UInt32 = 4
comptime _ROOT: UInt32 = 8
comptime _KEYED: UInt32 = 16
comptime _STACK = 24                 # 2^24 chunks = 16 GiB per message: no absorb can reach it


def _rotr(x: UInt32, n: Int) -> UInt32:
    return (x >> UInt32(n)) | (x << UInt32(32 - n))


def _g(mut s: _W, a: Int, b: Int, c: Int, d: Int, mx: UInt32, my: UInt32):
    s[a] = s[a] + s[b] + mx
    s[d] = _rotr(s[d] ^ s[a], 16)
    s[c] = s[c] + s[d]
    s[b] = _rotr(s[b] ^ s[c], 12)
    s[a] = s[a] + s[b] + my
    s[d] = _rotr(s[d] ^ s[a], 8)
    s[c] = s[c] + s[d]
    s[b] = _rotr(s[b] ^ s[c], 7)


def _compress(cv: _CV, block: _W, counter: UInt64, block_len: UInt32, flags: UInt32) -> _CV:
    var s = _IV.join(_IV)
    comptime for i in range(8):
        s[i] = cv[i]
    s[12] = UInt32(counter & 0xFFFFFFFF)
    s[13] = UInt32(counter >> 32)
    s[14] = block_len
    s[15] = flags
    var m = block
    comptime for _ in range(7):
        _g(s, 0, 4, 8, 12, m[0], m[1])
        _g(s, 1, 5, 9, 13, m[2], m[3])
        _g(s, 2, 6, 10, 14, m[4], m[5])
        _g(s, 3, 7, 11, 15, m[6], m[7])
        _g(s, 0, 5, 10, 15, m[8], m[9])
        _g(s, 1, 6, 11, 12, m[10], m[11])
        _g(s, 2, 7, 8, 13, m[12], m[13])
        _g(s, 3, 4, 9, 14, m[14], m[15])
        m = m.shuffle[2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]()
    return s.slice[8]() ^ s.slice[8, offset=8]()


def _load_words[n: Int](src: Base) -> SIMD[DType.uint32, n]:
    var w = SIMD[DType.uint32, n](0)
    comptime for i in range(4 * n):
        w[i // 4] |= UInt32(src[unsafe_offset=i]) << UInt32(8 * (i % 4))
    return w


def _store_cv(cv: _CV, dst: Base):
    comptime for i in range(32):
        dst[unsafe_offset=i] = UInt8((cv[i // 4] >> UInt32(8 * (i % 4))) & 255)


def _parent(key: _CV, left: _CV, right: _CV, flags: UInt32) -> _CV:
    return _compress(key, left.join(right), 0, 64, flags | _PARENT)


def _hash(key: _CV, flags0: UInt32, prefix: Int, ds: UInt8, src: Base, bytes: Int,
          dst: Base):
    """Blake3 of the message (ds if prefix else nothing) || src[0:bytes] under `key` and mode flags."""
    var msg_len = bytes + prefix
    var n_chunks = max(1, (msg_len + 1023) // 1024)
    var stack = InlineArray[_CV, _STACK](fill=_CV(0))
    var depth = 0
    var cv = key
    for k in range(n_chunks):
        cv = key
        var chunk_len = min(1024, msg_len - k * 1024)
        var n_blocks = max(1, (chunk_len + 63) // 64)
        for b in range(n_blocks):
            var start = k * 1024 + b * 64
            var n = min(64, msg_len - start)
            var w = _W(0)
            for i in range(n):
                var j = start + i
                var byte = ds if (prefix == 1 and j == 0) else src[unsafe_offset=j - prefix]
                w[i // 4] |= UInt32(byte) << UInt32(8 * (i % 4))
            var f = flags0
            if b == 0:
                f |= _CHUNK_START
            if b == n_blocks - 1:
                f |= _CHUNK_END
                if n_chunks == 1:
                    f |= _ROOT
            cv = _compress(cv, w, UInt64(k), UInt32(n), f)
        if k < n_chunks - 1:
            var total = k + 1
            while total & 1 == 0:
                depth -= 1
                cv = _parent(key, stack[depth], cv, flags0)
                total >>= 1
            stack[depth] = cv
            depth += 1
        else:
            while depth > 0:
                depth -= 1
                cv = _parent(key, stack[depth], cv, flags0 | (_ROOT if depth == 0 else 0))
    _store_cv(cv, dst)


struct Blake3(Hash):
    comptime DIGEST = 32
    comptime BLOCK = 64
    comptime NAME = "blake3"

    @staticmethod
    def leaf(src: Base, bytes: Int, out_ptr: Base):
        _hash(_IV, 0, 0, 0, src, bytes, out_ptr)

    @staticmethod
    def node(left: Base, right: Base, out_ptr: Base):
        # blake3(left || right): one 64-byte block, so one compression
        var w = _load_words[8](left).join(_load_words[8](right))
        _store_cv(_compress(_IV, w, 0, 64, _CHUNK_START | _CHUNK_END | _ROOT), out_ptr)

    @staticmethod
    def absorb(state: Base, ds: UInt8, src: Base, bytes: Int):
        _hash(_load_words[8](state), _KEYED, 1, ds, src, bytes, state)

    @staticmethod
    def squeeze(state: Base, counter: Int, out_ptr: Base):
        # blake3_keyed(key=state, message=LE64(counter))
        var w = _W(0)
        w[0] = UInt32(counter & 0xFFFFFFFF)
        w[1] = UInt32(counter >> 32)
        _store_cv(_compress(_load_words[8](state), w, 0, 8, _KEYED | _CHUNK_START | _CHUNK_END | _ROOT), out_ptr)

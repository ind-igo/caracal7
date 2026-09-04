"""Hash interface: the Merkle tree and the transcript are parameterized on `H: Hash`, so the hash is a
comptime choice next to Params. Blake3 is the default; a second implementation (Keccak, Poseidon for
recursion) conforms to the same trait and every kernel above it is unchanged.

The functions are device code: a Merkle kernel calls `H.compress` per node, the transcript kernel
calls it per absorbed block. A host implementation of the same trait serves the verifier.
"""

trait Hash:
    comptime DIGEST: Int          # output bytes, 32 for Blake3
    comptime BLOCK: Int           # input block bytes, 64 for Blake3
    comptime NAME: StringSpan[ImmStaticOrigin]

    @staticmethod
    def leaf(src: Pointer[UInt8, MutAnyOrigin], bytes: Int, out_ptr: Pointer[UInt8, MutAnyOrigin]):
        """Digest of one Merkle leaf: `bytes` bytes at `src` -> DIGEST bytes at `out_ptr`."""
        ...

    @staticmethod
    def node(left: Pointer[UInt8, MutAnyOrigin], right: Pointer[UInt8, MutAnyOrigin], out_ptr: Pointer[UInt8, MutAnyOrigin]):
        """Digest of two child digests."""
        ...

    @staticmethod
    def absorb(state: Pointer[UInt8, MutAnyOrigin], ds: UInt8, src: Pointer[UInt8, MutAnyOrigin], bytes: Int):
        """Transcript step: state <- H(state || ds || src[0:bytes]). Serial by definition."""
        ...

    @staticmethod
    def squeeze(state: Pointer[UInt8, MutAnyOrigin], counter: Int, out_ptr: Pointer[UInt8, MutAnyOrigin]):
        """DIGEST bytes of challenge material for block `counter` from the current state."""
        ...


struct Blake3(Hash):
    comptime DIGEST = 32
    comptime BLOCK = 64
    comptime NAME = "blake3"

    @staticmethod
    def leaf(src: Pointer[UInt8, MutAnyOrigin], bytes: Int, out_ptr: Pointer[UInt8, MutAnyOrigin]):
        pass    # not implemented: blake3 compression on device (merkle milestone)

    @staticmethod
    def node(left: Pointer[UInt8, MutAnyOrigin], right: Pointer[UInt8, MutAnyOrigin], out_ptr: Pointer[UInt8, MutAnyOrigin]):
        pass    # not implemented

    @staticmethod
    def absorb(state: Pointer[UInt8, MutAnyOrigin], ds: UInt8, src: Pointer[UInt8, MutAnyOrigin], bytes: Int):
        pass    # not implemented

    @staticmethod
    def squeeze(state: Pointer[UInt8, MutAnyOrigin], counter: Int, out_ptr: Pointer[UInt8, MutAnyOrigin]):
        pass    # not implemented

"""The C API for the csp-benchmarks Rust track (csp-rust/): a session is a workload with its prover
built (the prepare), `c7_prove` is the timed step. Build with `mojo build --emit shared-lib`.

Handles are heap addresses; every call names the target and size again because the session type is
the grid, a compile-time parameter. Sessions share the one `c7_runtime` device context. Inputs are raw bytes: sha256 and keccak take the message then the
32-byte digest; poseidon takes `size` little-endian u32 elements below 2^31 - 1; ecdsa takes e, x_Q,
y_Q, r, s as 32-byte big-endian words. Targets: 0 sha256, 1 keccak, 2 poseidon, 3 ecdsa."""

from max.gpu.host import DeviceContext
from std.ffi import external_call

from core.params import CLIENT, Params
from core.hash import Blake3
from workload import Workload, Session
from workloads.sha256 import Sha256
from workloads.keccak import Keccak
from workloads.poseidon import Poseidon
from workloads.ecdsa import Ecdsa, Point
from workloads.bigint import Big

comptime OPEN = 0
comptime PROVE = 1
comptime VERIFY = 2
comptime CELLS = 3
comptime PREP = 4
comptime CLOSE = 5

comptime Bytes = ImmPointer[UInt8, ImmUntrackedOrigin]
comptime MutBytes = MutPointer[UInt8, MutUntrackedOrigin]
comptime ABSENT = 8                     # a placeholder address for the pointer an op does not read; never dereferenced


def _bytes(data: Bytes, at: Int, n: Int) -> List[UInt8]:
    var out_list = List[UInt8](capacity=n)
    for i in range(n):
        out_list.append(data[unsafe_offset=at + i])
    return out_list^


def _word(data: Bytes, at: Int) -> Big:
    """A 32-byte big-endian word."""
    var le = List[UInt8](capacity=32)
    for i in range(32):
        le.append(data[unsafe_offset=at + 31 - i])
    return Big.from_bytes(le)


def _leak[T: Movable & Deinitable](var v: T) -> Int:
    """Move `v` to the heap and return its address; the pointer is rebuilt with `unsafe_from_address`."""
    var one = List[T](capacity=1)
    one.append(v^)
    var a = one.unsafe_take_allocation()
    return Int(a^.unsafe_leak())


def _op[p: Params, W: Workload & Movable & Deinitable](op: Int, handle: Int, var w: Optional[W], var public: List[UInt8],
                                          buf: MutBytes, n: Int) raises -> Int:
    """One session type; `w` is only read by OPEN (with `handle` the runtime), `buf`/`n` by PROVE (output,
    capacity) and VERIFY (proof)."""
    comptime S = Session[p, Blake3, W]
    if op == OPEN:
        var rt = MutPointer[DeviceContext, MutUntrackedOrigin](unsafe_from_address=handle)
        return _leak(S(rt[], w.take(), public^))
    var ptr = MutPointer[S, MutUntrackedOrigin](unsafe_from_address=handle)
    if op == PROVE:
        var proof = ptr[].prove()
        if len(proof) > n:
            return -2
        for i in range(len(proof)):
            buf[unsafe_offset=i] = proof[i]
        return len(proof)
    if op == VERIFY:
        var proof = List[UInt8](capacity=n)
        for i in range(n):
            proof.append(buf[unsafe_offset=i])
        return 1 if ptr[].verify(proof^) else 0
    if op == CELLS:
        return ptr[].cells()
    if op == PREP:
        return ptr[].preprocessing_bytes()
    if op == CLOSE:
        ptr.unsafe_deinit_pointee()
        ptr.unsafe_free()
        return 0
    raise Error("unknown op")


def _hash_public(data: Bytes, size: Int) -> List[UInt8]:
    return _bytes(data, 0, size + 32)


def _route(op: Int, target: Int, size: Int, handle: Int, data: Bytes, buf: MutBytes, n: Int) raises -> Int:
    if target == 0:
        var w = Optional[Sha256](Sha256(_bytes(data, 0, size))) if op == OPEN else Optional[Sha256]()
        var pub = _hash_public(data, size) if op == OPEN else List[UInt8]()
        if size <= 128:
            return _op[CLIENT.grid(32, 193), Sha256](op, handle, w^, pub^, buf, n)
        if size <= 256:
            return _op[CLIENT.grid(32, 321), Sha256](op, handle, w^, pub^, buf, n)
        if size <= 512:
            return _op[CLIENT.grid(32, 577), Sha256](op, handle, w^, pub^, buf, n)
        if size <= 1024:
            return _op[CLIENT.grid(32, 1089), Sha256](op, handle, w^, pub^, buf, n)
        if size <= 2048:
            return _op[CLIENT.grid(32, 2113), Sha256](op, handle, w^, pub^, buf, n)
        raise Error("sha256 sizes: up to 2048 bytes")
    if target == 1:
        var w = Optional[Keccak](Keccak(_bytes(data, 0, size))) if op == OPEN else Optional[Keccak]()
        var pub = _hash_public(data, size) if op == OPEN else List[UInt8]()
        if size <= 128:
            return _op[CLIENT.grid(64, 24), Keccak](op, handle, w^, pub^, buf, n)
        if size <= 256:
            return _op[CLIENT.grid(64, 48), Keccak](op, handle, w^, pub^, buf, n)
        if size <= 512:
            return _op[CLIENT.grid(64, 96), Keccak](op, handle, w^, pub^, buf, n)
        if size <= 1024:
            return _op[CLIENT.grid(64, 192), Keccak](op, handle, w^, pub^, buf, n)
        if size <= 2048:
            return _op[CLIENT.grid(64, 384), Keccak](op, handle, w^, pub^, buf, n)
        raise Error("keccak sizes: up to 2048 bytes")
    if target == 2:
        var w = Optional[Poseidon]()
        if op == OPEN:
            var elems = List[Int](capacity=size)
            for i in range(size):
                var v = 0
                for b in range(4):
                    v |= Int(data[unsafe_offset=4 * i + b]) << (8 * b)
                elems.append(v)
            w = Optional[Poseidon](Poseidon(elems^))
        if size <= 8:
            return _op[CLIENT.grid(64, 368), Poseidon](op, handle, w^, List[UInt8](), buf, n)
        if size <= 16:
            return _op[CLIENT.grid(64, 720), Poseidon](op, handle, w^, List[UInt8](), buf, n)
        raise Error("poseidon sizes: 1 to 16 elements")
    if target == 3:
        var w = Optional[Ecdsa]()
        if op == OPEN:
            w = Optional[Ecdsa](Ecdsa(_word(data, 96), _word(data, 128), _word(data, 0), Point(_word(data, 32), _word(data, 64), False)))
        return _op[CLIENT.grid(144, 576), Ecdsa](op, handle, w^, List[UInt8](), buf, n)
    raise Error("targets: 0 sha256, 1 keccak, 2 poseidon, 3 ecdsa")


def _call(op: Int, target: Int, size: Int, handle: Int, data: Bytes, buf: MutBytes, n: Int) -> Int:
    try:
        return _route(op, target, size, handle, data, buf, n)
    except e:
        print("caracal7:", e)
        return -1


@export("c7_runtime")
def c7_runtime() abi("C") -> Int:
    """The device context every session shares (kernels compile once per context), or -1. Never freed.
    First creates the Mojo runtime's CPU device if the process has none: a Mojo `main` does that before
    user code, a C caller does not, and `parallelize` (the trace writer) faults without it."""
    if external_call["KGEN_CompilerRT_AsyncRT_GetCurrentCPUDevice", Int]() == 0:
        _ = external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()
    try:
        return _leak(DeviceContext())
    except e:
        print("caracal7:", e)
        return -1


@export("c7_open")
def c7_open(runtime: Int, target: Int, size: Int, data: Bytes) abi("C") -> Int:
    """A session handle, or -1. `data` holds the inputs as the module docstring lays them out."""
    return _call(OPEN, target, size, runtime, data, MutBytes(unsafe_from_address=ABSENT), 0)


@export("c7_prove")
def c7_prove(handle: Int, target: Int, size: Int, dst: MutBytes, cap: Int) abi("C") -> Int:
    """The proof into `dst`; its length, -2 if `cap` is too small, -1 on error."""
    return _call(PROVE, target, size, handle, Bytes(unsafe_from_address=ABSENT), dst, cap)


@export("c7_verify")
def c7_verify(handle: Int, target: Int, size: Int, proof: MutBytes, n: Int) abi("C") -> Int:
    """1 if the proof verifies, 0 if not, -1 on error."""
    return _call(VERIFY, target, size, handle, Bytes(unsafe_from_address=ABSENT), proof, n)


@export("c7_cells")
def c7_cells(handle: Int, target: Int, size: Int) abi("C") -> Int:
    return _call(CELLS, target, size, handle, Bytes(unsafe_from_address=ABSENT), MutBytes(unsafe_from_address=ABSENT), 0)


@export("c7_preprocessing_bytes")
def c7_preprocessing_bytes(handle: Int, target: Int, size: Int) abi("C") -> Int:
    return _call(PREP, target, size, handle, Bytes(unsafe_from_address=ABSENT), MutBytes(unsafe_from_address=ABSENT), 0)


@export("c7_close")
def c7_close(handle: Int, target: Int, size: Int) abi("C") -> Int:
    return _call(CLOSE, target, size, handle, Bytes(unsafe_from_address=ABSENT), MutBytes(unsafe_from_address=ABSENT), 0)

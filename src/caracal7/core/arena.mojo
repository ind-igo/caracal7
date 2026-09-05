"""Arena: the one device allocation of a prover instance (docs/design.md rule 6).

Every buffer is an offset into the arena, assigned by `Bump` at setup. Launchers take the `Arena` and pass its
`DeviceBuffer` at enqueue; the kernel receives the base pointer (`bytes.Base`) plus `Buf[W]` offsets. The
host never holds a raw device pointer. Nothing is allocated after setup.
"""

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

comptime ALIGN = 256


struct Bump(TrivialRegisterPassable):
    """Bump-pointer planner: hand out aligned offsets, remember the total."""
    var used: Int

    def __init__(out self):
        self.used = 0

    def alloc(mut self, bytes: Int) -> Int:
        var off = self.used
        self.used += (bytes + ALIGN - 1) // ALIGN * ALIGN
        return off


struct Arena:
    var buf: DeviceBuffer[DType.uint8]
    var bytes: Int

    def __init__(out self, ctx: DeviceContext, bytes: Int) raises:
        self.buf = ctx.enqueue_create_buffer[DType.uint8](bytes)
        self.bytes = bytes

    def upload(self, ctx: DeviceContext, off: Int, host: HostBuffer[DType.uint8]) raises:
        """Copy a host buffer into the arena at `off`. Setup only."""
        ctx.enqueue_copy(self.buf.create_sub_buffer[DType.uint8](off, len(host)), host)

    def download(self, ctx: DeviceContext, off: Int, host: HostBuffer[DType.uint8]) raises:
        """Copy arena bytes at `off` to a host buffer. Proof output and tests only."""
        ctx.enqueue_copy(host, self.buf.create_sub_buffer[DType.uint8](off, len(host)))

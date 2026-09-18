"""Arena: the one device allocation of a prover instance (docs/design.md rule 6).

Every buffer is an offset into the arena, assigned by `Bump` at setup. Launchers take the `Arena` and pass its
`DeviceBuffer` at enqueue; the kernel receives the base pointer (`bytes.Base`) plus `Buf[W]` offsets. The
host never holds a raw device pointer. Nothing is allocated after setup.
"""

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

comptime ALIGN = 256


# Prover stages, in protocol order: the lifetime of a region is a closed stage interval. Regions whose
# lifetimes never meet may share bytes (Bump.plan). Every stage past the fold is a tail level: ST_TAIL + i.
comptime ST_LOAD = 0        # before prove: the trace, advice and public uploads
comptime ST_SORT = 1
comptime ST_W = 2           # commit W
comptime ST_ACC = 3         # the Z stage
comptime ST_Z = 4           # commit Z
comptime ST_SG = 5          # the small grid
comptime ST_LDE = 6
comptime ST_RES = 7
comptime ST_QUO = 8
comptime ST_Q = 9           # commit Q
comptime ST_OPEN = 10
comptime ST_FOLD = 11
comptime ST_RUN0 = 12       # the level-2 running query
comptime ST_TAIL = 13
comptime ST_END = 1 << 30   # lives to the end of the proof


struct Bump(Movable):
    """Arena planner. `alloc` hands out sequential aligned offsets and records each region's stage
    lifetime. `plan` then packs the recorded regions so that regions with disjoint lifetimes share
    bytes; a second construction pass in replay mode hands out the packed offsets in the same order.
    Without `plan` the offsets are the plain bump layout, which is what the unit tests use."""
    var used: Int
    var sizes: List[Int]
    var first: List[Int]
    var last: List[Int]
    var planned: List[Int]
    var cursor: Int

    def __init__(out self):
        self.used = 0
        self.sizes = List[Int]()
        self.first = List[Int]()
        self.last = List[Int]()
        self.planned = List[Int]()
        self.cursor = 0

    def alloc(mut self, bytes: Int, first: Int = ST_LOAD, last: Int = ST_END) -> Int:
        """A region live from stage `first` through stage `last`. A zero-byte region is a marker for
        the next region's offset (the base of a table block allocated right after it)."""
        var size = (bytes + ALIGN - 1) // ALIGN * ALIGN
        if len(self.planned) > 0:
            var off = self.planned[self.cursor]
            self.cursor += 1
            return off
        var off = self.used
        self.used += size
        self.sizes.append(size)
        self.first.append(first)
        self.last.append(last)
        return off

    def plan(mut self):
        """Pack the recorded regions: first fit by decreasing size, a region only avoiding the placed
        regions whose lifetimes meet its own. Switches the planner to replay mode."""
        var n = len(self.sizes)
        var order = List[Int]()
        for i in range(n):
            order.append(i)
        # insertion sort by size, decreasing; markers (size 0) go last and take the next region's offset
        for i in range(1, n):
            var j = i
            while j > 0 and self.sizes[order[j - 1]] < self.sizes[order[j]]:
                var t = order[j]; order[j] = order[j - 1]; order[j - 1] = t
                j -= 1
        self.planned = List[Int](length=n, fill=-1)
        var total = 0
        for k in range(n):
            var i = order[k]
            if self.sizes[i] == 0:
                break
            var off = 0
            var moved = True
            while moved:
                moved = False
                for m in range(k):
                    var j = order[m]
                    if self.first[i] > self.last[j] or self.first[j] > self.last[i]:
                        continue
                    if off < self.planned[j] + self.sizes[j] and self.planned[j] < off + self.sizes[i]:
                        off = self.planned[j] + self.sizes[j]
                        moved = True
            self.planned[i] = off
            total = max(total, off + self.sizes[i])
        for i in range(n - 1, -1, -1):                      # descending: a run of markers resolves to the same offset
            if self.sizes[i] == 0:
                self.planned[i] = self.planned[i + 1] if i + 1 < n else total
        self.used = total
        self.cursor = 0


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

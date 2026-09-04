"""Device-resident Fiat-Shamir transcript (docs/design.md section 6, spec 9.4).

An `H: Hash` state lives in the arena. `absorb` hashes a device buffer under a domain separator;
`squeeze` writes challenges next to it: E elements as e bytes, positions as uniform integers below L.
Every kernel that needs a challenge reads it from the arena. The host never sees a challenge.

Not implemented yet: every entry point raises. The interface is fixed here so prover.mojo and
verifier.mojo are written against it.
"""

from max.gpu.host import DeviceContext

from caracal7.params import Params
from caracal7.arena import Bump
from caracal7.hash import Hash

# Domain separators, one per line of spec 9.4, in transcript order.
comptime DS_PREFIX: UInt8 = 0       # protocol version, tower constants, grid, domains and rates, public inputs
comptime DS_TREE_W: UInt8 = 1       # -> beta_1, delta, gamma (stage 1)
comptime DS_TREE_Z: UInt8 = 2       # with Z2 -> alpha
comptime DS_TREE_Q: UInt8 = 3       # with Q3 -> z
comptime DS_OPENINGS: UInt8 = 4     # alpha_{c,p} -> beta (E^columns), gamma (E^P)
comptime DS_TAIL_ROOT: UInt8 = 5    # Mat(y_l) root -> S_{l-1}
comptime DS_TAIL_V: UInt8 = 6       # expected symbols v -> batching scalars
comptime DS_TAIL_ROUND: UInt8 = 7   # sumcheck message s_i -> r_i
comptime DS_CLEAR: UInt8 = 8        # y_ell in the clear -> S_{ell-1}

comptime STATE_BYTES = 64           # H.DIGEST state plus a counter, device resident


struct TranscriptLayout(TrivialRegisterPassable):
    """Arena offsets: the hash state and one challenge buffer sized for the largest squeeze."""
    var state: Int
    var challenges: Int
    var challenge_bytes: Int

    def __init__(out self, mut bump: Bump, challenge_bytes: Int):
        self.state = bump.alloc(STATE_BYTES)
        self.challenges = bump.alloc(challenge_bytes)
        self.challenge_bytes = challenge_bytes


def absorb[p: Params, H: Hash](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin], t: TranscriptLayout,
                      ds: UInt8, src: Int, bytes: Int) raises:
    """Hash `bytes` at arena offset `src` into the state under separator `ds`."""
    raise Error("not implemented: transcript.absorb")


def squeeze_elements[p: Params, H: Hash](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin], t: TranscriptLayout,
                                count: Int) raises:
    """Write `count` E elements (e bytes each) to t.challenges."""
    raise Error("not implemented: transcript.squeeze_elements")


def squeeze_positions[p: Params, H: Hash](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin], t: TranscriptLayout,
                                 count: Int, below: Int) raises:
    """Write `count` uniform positions below `below` (u32 each) to t.challenges."""
    raise Error("not implemented: transcript.squeeze_positions")

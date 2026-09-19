"""The verifier's predicates on a disclosed 32-byte window of DG1 (`sod.mojo`): the TD3 machine-readable zone's
nationality, date of birth, sex and expiry date, then age and validity on a date. DG1 is 61 5B 5F 1F 58 then
the 88 MRZ characters; line 2 starts at character 44. The default window starts at the nationality so that the
document number (line 2, characters 0..8) stays hidden; the optional data after the expiry is disclosed, and
in some states it carries a national identity number (any 32-byte window holding nationality, birth and
expiry discloses one of the two). The fields are TD3's: a DG1 of another length is refused (TD1 is 95
bytes with the birth date at byte 35). Two-digit years: a holder older than 100 reads as a child."""

comptime MRZ = 5                    # DG1 offset of the MRZ
comptime TD3 = MRZ + 88              # DG1 length of a TD3 passport
comptime NATIONALITY = MRZ + 54
comptime BIRTH = MRZ + 57
comptime SEX = MRZ + 64
comptime EXPIRY = MRZ + 65
comptime FIELDS_END = EXPIRY + 6
comptime WINDOW_OFFSET = NATIONALITY   # the default disclosed window: DG1 bytes 59 .. 90


@fieldwise_init
struct Date(ImplicitlyCopyable, Movable, Equatable, Writable):
    var y: Int
    var m: Int
    var d: Int

    def __lt__(self, o: Self) -> Bool:
        if self.y != o.y:
            return self.y < o.y
        if self.m != o.m:
            return self.m < o.m
        return self.d < o.d


@fieldwise_init
struct Mrz(Copyable, Movable):
    var nationality: String
    var birth: Date
    var sex: String
    var expiry: Date

    def age_on(self, today: Date) -> Int:
        var a = today.y - self.birth.y
        if Date(today.y, today.m, today.d) < Date(today.y, self.birth.m, self.birth.d):
            a -= 1
        return a

    def expired_on(self, today: Date) -> Bool:
        return self.expiry < today


def _digits(w: List[UInt8], at: Int, n: Int) raises -> Int:
    var v = 0
    for i in range(n):
        var c = Int(w[at + i])
        if c < 48 or c > 57:
            raise Error("the MRZ field is not decimal")
        v = 10 * v + c - 48
    return v


def _date(w: List[UInt8], at: Int, century: Int) raises -> Date:
    var d = Date(century + _digits(w, at, 2), _digits(w, at + 2, 2), _digits(w, at + 4, 2))
    if d.m < 1 or d.m > 12 or d.d < 1 or d.d > 31:
        raise Error("the MRZ date is not a calendar date")
    return d


def mrz_fields(window: List[UInt8], offset: Int, length: Int, today: Date) raises -> Mrz:
    """The fields of the window disclosed at DG1 byte `offset` of a DG1 of `length` bytes (the pinned public
    input; TD3 only). Two-digit years: the expiry is 20yy; the birth is 20yy when that is not after `today`,
    else 19yy."""
    if length != TD3:
        raise Error("the DG1 is not a TD3 passport's")
    if offset > NATIONALITY or offset + len(window) < FIELDS_END:
        raise Error("the disclosed window does not hold the MRZ fields")
    var at = NATIONALITY - offset
    var nat = String()
    for i in range(3):
        nat += String(chr(Int(window[at + i])))
    var birth = _date(window, BIRTH - offset, 2000)
    if today < birth:
        birth.y -= 100
    return Mrz(nat, birth, String(chr(Int(window[SEX - offset]))), _date(window, EXPIRY - offset, 2000))

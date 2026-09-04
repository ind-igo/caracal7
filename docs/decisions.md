# Decisions

Implementation decisions that the spec does not settle. Dated, newest last.

- 2026-09-04: uv with `max[all]` from PyPI (stable, Mojo 1.0.0); the pixi setup was replaced at Indigo's request. Tests run with `mojo run -I src`, one `TestSuite` runner per file, since `mojo test` no longer exists.

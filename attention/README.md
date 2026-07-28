# Attention implementation area

`attention/` contains the implementation surface for CCO’s current exact
attention-foundation phase and older research prototypes.

- `foundation.py` defines the exact candidate entry point measured by the
  protected benchmark.
- `reference.py` is an explicit development reference.
- `hybrid.py` contains local, spectral, correlation, and landmark prototypes.
- `benchmark.py` is a one-case development diagnostic, not an official score.

The protected workload, artifact generation, and PR decision logic live under
`eval/` and `benchmarks/`. Candidate PRs cannot modify those paths.

The current benchmark is still `draft`. See the root `README.md` for public
phase status and accepted contribution scope.

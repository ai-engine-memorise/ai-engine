# Future: Full Hexagonal Migration

Status: **deferred / reference**. The current build (see [`recsys-architecture.md`](./recsys-architecture.md))
deliberately ports only `EventSource`, `ContentStore`, `EmbeddingModel` and keeps scoring/fusion as
pure functions. This doc records the full ports-and-adapters end-state and **when + how** to migrate to
it — so the decision is intentional, not forgotten.

## Why we did NOT do this now

Abstraction pays only when ≥2 real implementations exist behind a port. Today:

- engagement / signal building / scorers / fusion each have **one** implementation.
- they are already pure functions → already testable without a Protocol.

Wrapping them in Protocols now = indirection tax (extra hop, extra file, ripple-on-change) with no
payoff. Premature. We pay that cost only when a second implementation actually shows up.

## Full end-state (target if/when triggers hit)

Promote these from pure functions to plugin Protocols:

```python
class EngagementScorer(Protocol):
    def score(self, events: list[InteractionEvent], content: Content, cfg: RecConfig) -> EngagementScore: ...

class CandidateGenerator(Protocol):
    name: str
    def generate(self, signals: UserSignals, store: ContentStore, cfg: RecConfig) -> list[Candidate]: ...

class Scorer(Protocol):
    name: str
    def score(self, cand: Candidate, signals: UserSignals, content: Content) -> float: ...  # [0,1]

class Ranker(Protocol):
    def rank(self, scored: list[ScoredCandidate], cfg: RecConfig, *, limit: int) -> list[ScoredCandidate]: ...
```

Plus:

- **Registry** — `Recommender` takes `list[CandidateGenerator]`, `list[Scorer]`, a `Ranker`, all by
  config name. Add a generator/scorer without touching the orchestrator.
- **Config-driven assembly** — `RecConfig` names which generators/scorers/ranker are active + weights.
  Swap strategies per experiment without code change.
- **Per-port adapter contract test suites** — one shared test run against every impl of each port.

## Migration triggers (promote a pure fn to a Protocol when...)

| Component        | Promote when...                                                                 |
|------------------|---------------------------------------------------------------------------------|
| `EngagementScorer` | a 2nd scoring scheme appears (e.g. ML-learned engagement vs rule-based)        |
| `CandidateGenerator` | generators become swappable per experiment, or >4 of them                   |
| `Scorer`         | scorers are toggled/reweighted per request via config, or 3rd-party scorers add |
| `Ranker`         | a learned LTR ranker lands beside weighted+MMR (the big one — see below)        |
| anything         | you need A/B of two strategies live, selected by config name                     |

The **learned ranker** is the most likely first real trigger: once impression/click data accumulates,
an LTR `Ranker` joins `WeightedFusionRanker` → now 2 impls → the `Ranker` Protocol earns its keep.

## Migration steps (mechanical, low-risk because types already exist)

The pure-function version is designed so this is a lift, not a rewrite:

1. Wrap the existing pure fn in a class implementing the new Protocol (`def score(...)` delegates to
   the fn). Zero logic change.
2. Add the Protocol to `contracts/ports.py`.
3. Change `Recommender` to accept the port (list) instead of calling the fn directly.
4. Move the registry/active-set into `RecConfig`.
5. Add the second implementation — the thing that triggered the migration.
6. Add a shared contract test; run it against both impls.

Because the **domain models and pure fns don't change**, existing unit/golden tests stay green. Only
wiring moves. This is the payoff of "typed models throughout, pure fns first": migration is cheap and
deferred until justified.

## Anti-goals (do NOT over-port)

- Don't Protocol-wrap a thing with one impl and no second on the roadmap.
- Don't add a registry before there's anything to register.
- Don't abstract `EmbeddingModel` into multi-vendor config if fastembed is forever — the fake is the
  only reason its Protocol exists; keep it that minimal.

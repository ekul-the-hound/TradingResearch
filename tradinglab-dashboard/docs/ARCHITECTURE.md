# TradingLab Dashboard — Implementation Architecture

> Prompt 3 deliverable. Design only — no visual pages built. Defines the folder
> structure, route map, typed domain models, and data-access abstraction for the
> canonical React + TS frontend described in `docs/AUDIT.md`.

---

## 1. Folder structure

```
frontend/
├─ index.html
├─ src/
│  ├─ app/
│  │  ├─ App.tsx                 # router + shell composition
│  │  ├─ routes.tsx              # route table (single source of route defs)
│  │  └─ shell/
│  │     ├─ AppShell.tsx         # 232px nav + 44px status bar + content grid
│  │     ├─ LeftNav.tsx
│  │     └─ SystemStatusBar.tsx  # (built in Prompt 5)
│  │
│  ├─ design/
│  │  ├─ tokens.css              # color/space/radius/type CSS variables
│  │  ├─ tokens.ts               # typed token accessors
│  │  ├─ globals.css             # resets, tabular-nums, base type
│  │  └─ Preview.tsx             # design-system story page (Prompt 4)
│  │
│  ├─ primitives/                # reusable, domain-agnostic UI (Prompt 4)
│  │  ├─ AppPanel.tsx
│  │  ├─ PanelHeader.tsx
│  │  ├─ StatusChip.tsx
│  │  ├─ TruthLabel.tsx
│  │  ├─ MetricValue.tsx
│  │  ├─ MetricDefinitionTooltip.tsx
│  │  ├─ EmptyState.tsx
│  │  ├─ ErrorState.tsx
│  │  ├─ LoadingState.tsx
│  │  ├─ UnavailableState.tsx
│  │  ├─ DataFreshnessLabel.tsx
│  │  ├─ DenseDataTable.tsx
│  │  ├─ FilterBar.tsx
│  │  ├─ SectionTitle.tsx
│  │  ├─ WarningBanner.tsx
│  │  ├─ InlineEvidenceBadge.tsx
│  │  ├─ ChartFrame.tsx
│  │  ├─ SeverityIndicator.tsx
│  │  └─ index.ts
│  │
│  ├─ domain/                    # domain-specific components (Prompts 6–13)
│  │  ├─ research-command/
│  │  ├─ strategy-lab/
│  │  ├─ strategy-detail/
│  │  ├─ portfolio-builder/
│  │  ├─ challenge-readiness/
│  │  ├─ discovery-inbox/
│  │  ├─ data-integrity/
│  │  └─ execution/
│  │
│  ├─ models/                    # typed domain models (this doc, §3)
│  │  ├─ truth.ts                # truth-label enums/unions
│  │  ├─ strategy.ts
│  │  ├─ pipeline.ts
│  │  ├─ backtest.ts
│  │  ├─ validation.ts
│  │  ├─ cost.ts
│  │  ├─ provenance.ts
│  │  ├─ integrity.ts
│  │  ├─ ftmo.ts
│  │  ├─ portfolio.ts
│  │  ├─ queue.ts
│  │  ├─ system.ts
│  │  └─ index.ts
│  │
│  ├─ data/                      # data-access abstraction (§4)
│  │  ├─ ResearchRepository.ts   # the interface every page depends on
│  │  ├─ RepositoryProvider.tsx  # React context; selects impl
│  │  ├─ result.ts               # Loadable<T> wrapper (loading/error/etc.)
│  │  ├─ mock/
│  │  │  ├─ MockResearchRepository.ts
│  │  │  └─ fixtures/            # DEV FIXTURE data, clearly marked
│  │  └─ sqlite/                 # read-only adapters (Prompt 14)
│  │     └─ SqliteResearchRepository.ts
│  │
│  ├─ charts/                    # chart utilities (thin wrappers over lib)
│  │  ├─ ChartTheme.ts
│  │  ├─ EquityCurve.tsx
│  │  ├─ DrawdownChart.tsx
│  │  ├─ Distribution.tsx
│  │  └─ CorrelationMatrix.tsx
│  │
│  ├─ lib/
│  │  ├─ format.ts               # ONE canonical formatter per metric type
│  │  └─ hooks.ts
│  │
│  └─ main.tsx
│
└─ tests/                        # mirrors src/; Vitest + RTL
```

**Separation guarantee (CLAUDE.md §7):** `domain/` and `app/` import from
`data/ResearchRepository` (interface) and `models/` only — never from a concrete
repository. Pages cannot know whether data is mock or SQLite.

---

## 2. Route map

| Route                       | Page                | Primary question |
| --------------------------- | ------------------- | ---------------- |
| `/research-command`         | Research Command    | What happened in the last run, where are strategies in the funnel, what blocks me, what do I review next? |
| `/strategy-lab`             | Strategy Lab        | Which strategies deserve research/validation/selection/rejection, and what's the evidence? |
| `/strategies/:strategyId`   | Strategy Detail     | Reject, investigate, validate further, promote to selection, or prep for paper? |
| `/portfolio-builder`        | Portfolio Builder   | Which validated strategies *improve the portfolio* after correlation/risk/cost/DD/FTMO constraints? |
| `/challenge-readiness`      | Challenge Readiness | Is this appropriate for an FTMO-style challenge under the *modeled* rules, and where are the limits? |
| `/discovery-inbox`          | Discovery Inbox     | Which ideas are worth review, which are redundant, which are blocked by missing data? |
| `/data-integrity`           | Data & Integrity    | Can I trust the datasets, provenance, validation boundaries, and config behind these conclusions? |
| `/execution`                | Execution           | (Future) session/broker readiness — defaults to **Execution Offline**. |

Default route → `/research-command`. Strategy Detail tabs are deep-linkable:
`/strategies/:id#performance|validation|risk|challenge|provenance`.

Per-route inputs/outputs/truth-labels are specified inline in each build prompt
(6–13); not duplicated here to keep one source of truth.

---

## 3. Typed domain models

All truth labels are shared unions so the UI can render them uniformly.

```ts
// models/truth.ts
export type OperatingMode = 'RESEARCH' | 'BACKTEST' | 'PAPER' | 'DEMO' | 'LIVE';
export type MarketMode   = 'HISTORICAL' | 'DELAYED' | 'REAL-TIME' | 'UNKNOWN';
export type HoldoutState = 'SEALED' | 'UNSEALED' | 'UNKNOWN';
export type ReturnsProvenance =
  | 'REAL' | 'UNVERIFIED' | 'SYNTHETIC_RISK' | 'UNAVAILABLE';
export type ComplianceBasis =
  | 'AUTHORITATIVE' | 'PROXY' | 'INCOMPLETE' | 'UNKNOWN';
export type BrokerState = 'OFFLINE' | 'NOT_CONFIGURED' | 'CONNECTED';
export type Status =
  | 'PASS' | 'FAIL' | 'WARNING' | 'INFO'
  | 'UNKNOWN' | 'PROXY' | 'INCOMPLETE' | 'OFFLINE';
export type Severity = 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
export type LifecycleStage =
  | 'DISCOVERED' | 'CODE_VALID' | 'BACKTESTED' | 'COST_ADJUSTED'
  | 'VALIDATED' | 'PORTFOLIO_CANDIDATE' | 'PAPER' | 'LIVE'
  | 'REJECTED' | 'RETIRED';

// A value that may not be knowable. Never collapse to 0 or "".
export type Unknowable<T> =
  | { kind: 'value'; value: T }
  | { kind: 'unavailable'; reason: string }
  | { kind: 'unknown'; reason?: string }
  | { kind: 'proxy'; value: T; note: string };
```

```ts
// models/strategy.ts
export interface StrategySummary {
  strategyId: string;            // copyable
  name: string;
  symbol: string;
  timeframe: string;
  version: string;
  origin: 'manual' | 'discovered' | 'mutation' | 'UNKNOWN';
  stage: LifecycleStage;
  lastRunAt: string | null;      // ISO; null => never run
  netSharpe: Unknowable<number>;
  netReturnPct: Unknowable<number>;
  maxDrawdownPct: Unknowable<number>;
  tradeCount: Unknowable<number>;
  pbo: Unknowable<number>;
  deflatedSharpe: Unknowable<number>;
  robustness: Unknowable<number>;
  parameterStability: Unknowable<number>;
  diversificationSignal: Unknowable<number>;
  ftmoFit: Unknowable<Status>;
  returnsProvenance: ReturnsProvenance;
  holdout: HoldoutState;
  evidence: StrategyEvidence;
}

export type EvidenceKey =
  | 'DATA' | 'COST' | 'HOLDOUT' | 'REAL_RETURNS' | 'MANUAL_GATES'
  | 'OVERFITTING' | 'ROBUSTNESS' | 'PARAMETER_STABILITY'
  | 'PORTFOLIO_FIT' | 'CHALLENGE_FIT';

export type StrategyEvidence = Record<EvidenceKey, {
  status: Status;
  tooltip: string;
}>;
```

```ts
// models/backtest.ts  (mirrors database.py backtest_results/backtest_trades)
export interface BacktestResult {
  id: number;
  strategyName: string;
  variantId: string | null;
  symbol: string;
  timeframe: string | null;
  startDate: string; endDate: string;
  barsTested: number | null;
  totalReturnPct: number | null;
  sharpeRatio: number | null;
  maxDrawdownPct: number | null;
  totalTrades: number | null;
  winRate: number | null;
  profitFactor: number | null;
  dataFingerprint: string | null;   // NULL => predates provenance tracking
  dataRows: number | null;
  dataFirst: string | null; dataLast: string | null;
  codeFingerprint: string | null;
  timestamp: string;
  returnsProvenance: ReturnsProvenance;  // derived from CanonicalResult, not UI
}

export interface BacktestTrade {
  backtestId: number;
  entryDate: string; exitDate: string;
  entryPrice: number; exitPrice: number;
  size: number; pnl: number; returnPct: number;
  durationBars: number; isLong: boolean;
}
```

```ts
// models/pipeline.ts
export interface PipelineRun {
  runId: string;
  startedAt: string | null;
  completedAt: string | null;
  durationSec: number | null;
  inputCount: Unknowable<number>;
  survivorCount: Unknowable<number>;
  topCandidateId: string | null;
  primaryBlocker: string | null;
  dataFingerprint: string | null;
  holdout: HoldoutState;
  costProfileName: string;
  returnsProvenance: ReturnsProvenance;
  status: 'SUCCESS' | 'FAILED' | 'PARTIAL' | 'NO_RUNS';
}

export interface FunnelStage {
  stage: LifecycleStage;
  count: Unknowable<number>;
  blocked: Unknowable<number>;
  rejected: Unknowable<number>;
  definition: string;
}
export type PipelineFunnel = FunnelStage[];
```

```ts
// models/validation.ts
export interface ValidationResult {
  strategyId: string;
  manualGates: {
    sharpe: { value: number | null; threshold: number; status: Status };
    trades: { value: number | null; threshold: number; status: Status };
    maxDrawdown: { value: number | null; threshold: number; status: Status };
  };
  holdout: { fraction: number; state: HoldoutState };
  pbo: Unknowable<number>;
  deflatedSharpe: Unknowable<number>;
  cscv: Unknowable<number>;
  permutationPValue: Unknowable<number>;
  lookaheadScan: Unknowable<Status>;
  parameterStability: Unknowable<number>;
  verdict: 'PROMOTE' | 'INVESTIGATE' | 'REJECT' | 'INCOMPLETE' | 'UNKNOWN';
  // NOTE: verdict is read from backend evidence — never a fabricated composite.
}
```

```ts
// models/cost.ts
export interface CostProfile {
  name: string;                 // e.g. "Pessimistic Manual"
  spreadPips: number;           // 2
  slippagePips: number;         // 1
  swaps: 'INTRADAY_NONE' | 'MODELED' | 'UNKNOWN';
}
```

```ts
// models/provenance.ts
export interface DataProvenance {
  datasetFingerprint: string | null;
  source: string;               // "HistData"
  symbol: string; timeframe: string;
  dateRange: { first: string; last: string } | null;
  barCount: number | null;
  timezoneVerified: Unknowable<boolean>;
  codeFingerprint: string | null;
  parentIds: string[]; childIds: string[];
  returnsStatus: ReturnsProvenance;
  integrityWarnings: string[];
}
```

```ts
// models/integrity.ts
export interface IntegrityStatus {
  warningCount: number;
  holdoutFraction: number;
  holdout: HoldoutState;
  syntheticRiskResultIds: number[];
  missingFingerprintRunIds: string[];
  dependencies: DependencyHealth[];
}
export interface DependencyHealth {
  name: 'SearXNG' | 'Ollama' | 'DataPath' | 'SQLite' | string;
  state: 'OK' | 'DOWN' | 'NOT_CHECKED';  // default NOT_CHECKED, never fake OK
  detail: string | null;
}
```

```ts
// models/ftmo.ts  (mirrors firm_rules / ftmo_compliance / consistency_rule)
export interface FTMOComplianceStatus {
  basis: ComplianceBasis;       // PROXY until wired to FTMOComplianceChecker
  isFullyModelled: boolean;     // from FirmRules
  modelledRules: string[];
  unmodelledRules: string[];    // e.g. CONSISTENCY_RULE
  limits: {
    maxDailyLossPct: number;    // 0.05
    maxTotalDrawdownPct: number;// 0.10
    profitTargetPct: Unknowable<number>;
    minTradingDays: number;     // 4
  };
  breachReasons: string[];
  drawdownHeadroom: Unknowable<number>;
  targetProgress: Unknowable<number>;
  daysTraded: Unknowable<number>;
  consistency: ConsistencyState;
}
export type ConsistencyState =
  | { evaluated: false; reason: 'NO_THRESHOLD_CONFIGURED' }
  | { evaluated: true; passed: boolean; threshold: number;
      bestDayShare: number; bestDayDate: string };

// models/ftmo.ts (cont.)
export interface ChallengeSimulationResult {
  available: boolean;
  nSimulations: number;
  pPass: Unknowable<number>;
  p95WorstDayPct: Unknowable<number>;
  drawdownDistribution: number[] | null;
  breachReasonDistribution: Record<string, number> | null;
  baselineVsStress: { baselinePPass: number; stressPPass: number } | null;
  inputAssumptions: string;     // path count, fee basis — never a bare number
}
```

```ts
// models/portfolio.ts
export interface PortfolioCandidate {
  strategyId: string;
  eligible: boolean;
  exclusionReason: string | null; // rejected / no overlap / missing returns…
  evidence: StrategyEvidence;
}
export interface PortfolioComputation {
  componentRunIds: string[];
  overlapWindow: { first: string; last: string } | null;
  combinedSharpe: Unknowable<number>;
  combinedMaxDrawdownPct: Unknowable<number>;
  combinedReturnPct: Unknowable<number>;
  correlationMatrix: Unknowable<number[][]>;
  tailRisk: Unknowable<{ var95: number; cvar95: number; tailRatio: number;
                         skew: number; kurtosis: number }>;
  status: 'COMPUTED' | 'PARTIAL' | 'INSUFFICIENT_OVERLAP' | 'NO_CANDIDATES';
}
```

```ts
// models/queue.ts
export interface ResearchQueueItem {
  id: string;
  severity: Severity;
  title: string;
  reason: string;
  affects: { kind: 'strategy' | 'run'; id: string };
  sourceLabel: string;
  suggestedAction: string;
  destination: string | null;   // route to navigate
}
```

```ts
// models/system.ts
export interface SystemStatus {
  operatingMode: OperatingMode;
  marketMode: MarketMode;
  dataSource: string | null;
  lastRunAt: string | null;      // actual only; never inferred
  datasetFingerprint: string | null;
  holdout: HoldoutState;
  costProfile: CostProfile | null;
  broker: BrokerState;           // default OFFLINE / NOT_CONFIGURED
  integrityWarningCount: number;
  isFixture: boolean;            // drives the DEV FIXTURE marker
}
```

---

## 4. Data-access abstraction

A single interface. Pages depend on it; concrete impls are swapped by provider.

```ts
// data/result.ts
export type Loadable<T> =
  | { state: 'loading' }
  | { state: 'ready'; data: T; isFixture: boolean }
  | { state: 'empty' }              // query ran, no rows
  | { state: 'unavailable'; reason: string } // table/db/column absent
  | { state: 'error'; error: string };
```

```ts
// data/ResearchRepository.ts
export interface ResearchRepository {
  readonly isFixture: boolean;      // true => everything is DEV FIXTURE

  getSystemStatus(): Promise<Loadable<SystemStatus>>;
  getLatestRun(): Promise<Loadable<PipelineRun>>;
  getFunnel(): Promise<Loadable<PipelineFunnel>>;
  getResearchQueue(): Promise<Loadable<ResearchQueueItem[]>>;

  listStrategies(filter?: StrategyFilter): Promise<Loadable<StrategySummary[]>>;
  getStrategy(id: string): Promise<Loadable<StrategySummary>>;
  getBacktestResult(id: string): Promise<Loadable<BacktestResult>>;
  getBacktestTrades(id: string): Promise<Loadable<BacktestTrade[]>>;
  getValidation(id: string): Promise<Loadable<ValidationResult>>;
  getProvenance(id: string): Promise<Loadable<DataProvenance>>;

  getFTMOCompliance(id: string): Promise<Loadable<FTMOComplianceStatus>>;
  getChallengeSimulation(id: string): Promise<Loadable<ChallengeSimulationResult>>;

  getPortfolioCandidates(): Promise<Loadable<PortfolioCandidate[]>>;
  computePortfolio(ids: string[]): Promise<Loadable<PortfolioComputation>>;

  getIntegrityStatus(): Promise<Loadable<IntegrityStatus>>;
  getDiscoveryInbox(): Promise<Loadable<DiscoveryInbox>>;
}
```

- `MockResearchRepository` sets `isFixture = true`; every `Loadable` it returns
  carries `isFixture: true`, which the UI turns into a visible **DEV FIXTURE**
  marker. **Fixtures cannot masquerade as production data.**
- `SqliteResearchRepository` (Prompt 14) sets `isFixture = false`, reads the
  databases from `docs/AUDIT.md §3–§5`, and returns `unavailable` (never `0`)
  when a table/column/db is missing.
- `RepositoryProvider.tsx` exposes the repo via context; a `useRepository()` hook
  is the only way pages reach data.

### Surfacing UNKNOWN / UNAVAILABLE / PROXY / UNVERIFIED

Two layers, both mandatory:
1. **Container level:** `Loadable` distinguishes loading / empty / unavailable /
   error — rendered by the `*State` primitives.
2. **Field level:** `Unknowable<T>` distinguishes value / unavailable / unknown /
   proxy per cell — rendered by `MetricValue` + `TruthLabel`. A missing number is
   never shown as `0` or blank; a proxy value always renders its `note`.

---

## 5. Phase-one minimum viable scope

1. Prompt 4 — shell, tokens, all primitives, `/preview` story page (mock only).
2. Prompt 5 — `SystemStatusBar` bound to `getSystemStatus()`.
3. Prompt 6 — `/research-command` (funnel + latest run + queue + health strip +
   truth ribbon) against `MockResearchRepository`.

Everything reads through the interface from day one, so Prompt 14 swaps
implementations without touching a single page.

---

## 6. Constraints honored

- No live trading API. No absent DB field pretended to exist.
- Desktop-first, research-first.
- **One canonical metric definition** lives in `lib/format.ts` + backend results;
  components never re-derive a metric.
- `multi_objective_optimizer` output is typed and labeled **Pareto Selection**,
  never "optimization."

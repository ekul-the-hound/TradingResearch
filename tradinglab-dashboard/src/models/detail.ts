import type { HoldoutState, ReturnsProvenance, Status, Unknowable } from './truth';
import type { StrategySummary } from './strategy';

export interface EquityPoint {
  t: string; // ISO date
  gross: number | null; // account %, null if not available
  net: number; // account %
}

export interface TradeRow {
  entryDate: string;
  exitDate: string;
  isLong: boolean;
  entryPrice: number;
  exitPrice: number;
  pnl: number;
  returnPct: number;
  durationBars: number;
}

export interface PerformanceEvidence {
  // equity curve; gross present only if both gross and net exist
  equity: Unknowable<EquityPoint[]>;
  hasGross: boolean;
  headline: {
    netReturnPct: Unknowable<number>;
    sharpe: Unknowable<number>;
    maxDrawdownPct: Unknowable<number>;
    tradeCount: Unknowable<number>;
    winRatePct: Unknowable<number>;
    profitFactor: Unknowable<number>;
    expectancy: Unknowable<number>;
  };
  period: { first: string; last: string } | null;
  trades: Unknowable<TradeRow[]>;
  returnsProvenance: ReturnsProvenance;
}

export interface GateCheck {
  value: number | null;
  threshold: number;
  comparator: '>=' | '<=';
  status: Status;
}

export interface ValidationEvidence {
  manualGates: {
    sharpe: GateCheck;
    trades: GateCheck;
    maxDrawdown: GateCheck;
  };
  holdout: { fraction: number; state: HoldoutState };
  pbo: Unknowable<number>;
  deflatedSharpe: Unknowable<number>;
  cscv: Unknowable<number>;
  permutationPValue: Unknowable<number>;
  walkForward: Unknowable<Status>;
  lookaheadScan: Unknowable<Status>;
  prohibitedPatternScan: Unknowable<Status>;
  parameterStability: Unknowable<number>;
  // Plain-language verdict from available evidence — never a fabricated score.
  verdict:
    | 'PROMOTE'
    | 'INVESTIGATE'
    | 'REJECT'
    | 'INCOMPLETE'
    | 'UNKNOWN';
  verdictReason: string;
}

export interface RiskEvidence {
  var95Pct: Unknowable<number>;
  cvar95Pct: Unknowable<number>;
  tailRatio: Unknowable<number>;
  skew: Unknowable<number>;
  kurtosis: Unknowable<number>;
  maeMfe: Unknowable<{ avgMaePct: number; avgMfePct: number }>;
  // Stop/target analysis is in-sample until independently validated.
  stopTargetInSample: Unknowable<{ note: string }>;
  capacity: Unknowable<{ note: string }>;
  drawdownSeries: Unknowable<{ t: string; ddPct: number }[]>;
}

export interface ProvenanceEvidence {
  datasetFingerprint: string | null;
  dataSource: string | null;
  symbol: string;
  timeframe: string;
  dateRange: { first: string; last: string } | null;
  barCount: number | null;
  timezoneVerified: Unknowable<boolean>;
  codeFingerprint: string | null;
  parentIds: string[];
  childIds: string[];
  runConfig: Record<string, string> | null;
  costProfile: { name: string; spreadPips: number; slippagePips: number } | null;
  resultTables: string[]; // e.g. ["backtest_results#42", "backtest_trades"]
  returnsStatus: ReturnsProvenance;
  integrityWarnings: string[];
}

export interface StrategyDetail {
  summary: StrategySummary;
  performance: PerformanceEvidence;
  validation: ValidationEvidence;
  risk: RiskEvidence;
  provenance: ProvenanceEvidence;
  // challenge fit is reused from the ftmo models (Prompt 11 owns full page)
}

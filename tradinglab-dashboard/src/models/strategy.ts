import type {
  HoldoutState,
  LifecycleStage,
  ReturnsProvenance,
  Status,
  Unknowable,
} from './truth';

export type EvidenceKey =
  | 'DATA'
  | 'COST'
  | 'HOLDOUT'
  | 'REAL_RETURNS'
  | 'MANUAL_GATES'
  | 'OVERFITTING'
  | 'ROBUSTNESS'
  | 'PARAMETER_STABILITY'
  | 'PORTFOLIO_FIT'
  | 'CHALLENGE_FIT';

export const EVIDENCE_KEYS: EvidenceKey[] = [
  'DATA',
  'COST',
  'HOLDOUT',
  'REAL_RETURNS',
  'MANUAL_GATES',
  'OVERFITTING',
  'ROBUSTNESS',
  'PARAMETER_STABILITY',
  'PORTFOLIO_FIT',
  'CHALLENGE_FIT',
];

export const EVIDENCE_LABEL: Record<EvidenceKey, string> = {
  DATA: 'DATA',
  COST: 'COST',
  HOLDOUT: 'HOLDOUT',
  REAL_RETURNS: 'REAL RETURNS',
  MANUAL_GATES: 'MANUAL GATES',
  OVERFITTING: 'OVERFITTING',
  ROBUSTNESS: 'ROBUSTNESS',
  PARAMETER_STABILITY: 'PARAMETER STABILITY',
  PORTFOLIO_FIT: 'PORTFOLIO FIT',
  CHALLENGE_FIT: 'CHALLENGE FIT',
};

export interface EvidenceItem {
  status: Status;
  tooltip: string;
}

export type StrategyEvidence = Record<EvidenceKey, EvidenceItem>;

export type StrategyOrigin = 'manual' | 'discovered' | 'mutation' | 'UNKNOWN';

export interface StrategySummary {
  strategyId: string; // copyable
  name: string;
  symbol: string;
  timeframe: string;
  version: string;
  origin: StrategyOrigin;
  stage: LifecycleStage;
  discoverySource: string | null; // e.g. "web extraction", "manual"
  lastRunAt: string | null; // ISO; null => never run

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

  // data-quality context
  dataFingerprint: string | null;
  dataSource: string | null;
  testWindow: { first: string; last: string } | null;
  timezoneVerified: Unknowable<boolean>;

  evidence: StrategyEvidence;
}

export interface StrategyFilter {
  stage?: LifecycleStage | 'ALL';
  symbol?: string | 'ALL';
  timeframe?: string | 'ALL';
  holdout?: HoldoutState | 'ALL';
  returnsProvenance?: ReturnsProvenance | 'ALL';
  manualGate?: Status | 'ALL';
  validation?: Status | 'ALL';
  ftmo?: Status | 'ALL';
  discoverySource?: string | 'ALL';
  evidenceComplete?: boolean;
  text?: string;
}

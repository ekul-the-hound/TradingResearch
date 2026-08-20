import type { HoldoutState, Status, Unknowable } from './truth';

export interface DatasetRecord {
  fingerprint: string | null; // null => predates provenance tracking
  source: string; // "HistData"
  symbol: string;
  timeframe: string;
  dateRange: { first: string; last: string } | null;
  barCount: number | null;
  timezoneVerified: Unknowable<boolean>;
  usedByResultCount: number;
}

export interface ProvenanceCoverage {
  totalResults: number;
  withFingerprint: number;
  missingFingerprint: number; // NULL rows: predate provenance fix
  withCodeFingerprint: number;
  // result ids that predate provenance tracking (NULL fingerprint)
  missingFingerprintResultIds: string[];
}

export interface HoldoutStatus {
  fraction: number; // DEFAULT_HOLDOUT_FRACTION
  state: HoldoutState;
  cutoffDate: string | null;
  sealedResultCount: Unknowable<number>;
  unsealedResultCount: Unknowable<number>;
}

export interface ReturnsProvenanceLedger {
  real: number;
  unverified: number;
  syntheticRisk: number;
  unavailable: number;
  // strategy/result ids flagged synthetic-risk (require attention)
  syntheticRiskResultIds: string[];
  allowSyntheticFlag: boolean; // canonical_result.ALLOW_SYNTHETIC_RETURNS
}

export interface DependencyHealth {
  name: 'SearXNG' | 'Ollama' | 'DataPath' | 'SQLite' | string;
  // default NOT_CHECKED — never a fabricated green OK
  state: 'OK' | 'DOWN' | 'NOT_CHECKED';
  detail: string | null;
  lastCheckedAt: string | null;
}

export interface ConfigFreeze {
  frozen: boolean;
  hash: Unknowable<string>;
  costProfileName: string;
  holdoutFraction: number;
  gateSummary: string; // "Sharpe≥0.5, Trades≥20, DD≤30%"
  driftDetected: Unknowable<boolean>;
  keys: { key: string; value: string }[];
}

export interface IntegrityStatus {
  warningCount: number;
  datasets: DatasetRecord[];
  provenanceCoverage: ProvenanceCoverage;
  holdout: HoldoutStatus;
  returnsLedger: ReturnsProvenanceLedger;
  dependencies: DependencyHealth[];
  configFreeze: ConfigFreeze;
  // overall banner
  overallStatus: Status;
}

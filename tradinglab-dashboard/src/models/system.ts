import type {
  BrokerState,
  HoldoutState,
  MarketMode,
  OperatingMode,
} from './truth';

export interface CostProfile {
  name: string; // e.g. "Pessimistic Manual"
  spreadPips: number; // 2
  slippagePips: number; // 1
  swaps: 'INTRADAY_NONE' | 'MODELED' | 'UNKNOWN';
}

export interface SystemStatus {
  operatingMode: OperatingMode;
  marketMode: MarketMode;
  dataSource: string | null; // "HistData"; null => UNKNOWN
  lastRunAt: string | null; // actual ISO only; never inferred
  datasetFingerprint: string | null; // present => copyable; null => absent
  holdout: HoldoutState;
  costProfile: CostProfile | null; // null => UNKNOWN
  broker: BrokerState; // defaults OFFLINE / NOT_CONFIGURED
  integrityWarningCount: number;
}

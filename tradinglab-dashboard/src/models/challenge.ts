import type { Status, Unknowable } from './truth';
import type {
  ChallengeSimulationResult,
  ConsistencyState,
  FTMOComplianceStatus,
} from './ftmo';

export interface FirmProfile {
  firm: string; // "FTMO"
  accountSize: number | null;
  phase: 'CHALLENGE' | 'VERIFICATION' | 'FUNDED' | 'UNKNOWN';
  profitTargetPct: number;
  maxDailyLossPct: number;
  maxTotalDrawdownPct: number;
  minTradingDays: number;
  // consistency rule is only modeled once a threshold is configured
  consistencyThresholdPct: number | null;
}

export interface RuleModelRow {
  rule: string;
  modeled: boolean;
  basis: 'AUTHORITATIVE' | 'PROXY' | 'INCOMPLETE' | 'NOT_MODELED';
  note: string;
}

export interface ChallengeReadiness {
  firm: FirmProfile;
  // which rules are actually implemented vs. display-only
  ruleModel: RuleModelRow[];
  // candidate strategies assessed for challenge fit (proxy)
  perStrategy: {
    strategyId: string;
    name: string;
    ftmoFit: Unknowable<Status>;
    basis: 'PROXY' | 'AUTHORITATIVE' | 'INCOMPLETE';
    note: string;
  }[];
  consistency: ConsistencyState;
  overallBasis: 'PROXY' | 'AUTHORITATIVE' | 'INCOMPLETE';
  readinessStatus: Status; // never an unconditional PASS while PROXY
}

export type { ChallengeSimulationResult, FTMOComplianceStatus };

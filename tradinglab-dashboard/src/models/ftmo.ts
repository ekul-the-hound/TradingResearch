import type { ComplianceBasis, Unknowable } from './truth';

export type ConsistencyState =
  | { evaluated: false; reason: 'NO_THRESHOLD_CONFIGURED' }
  | {
      evaluated: true;
      passed: boolean;
      threshold: number;
      bestDayShare: number;
      bestDayDate: string;
    };

export interface FTMOComplianceStatus {
  basis: ComplianceBasis; // PROXY until wired to FTMOComplianceChecker
  isFullyModelled: boolean;
  modelledRules: string[];
  unmodelledRules: string[]; // e.g. CONSISTENCY_RULE
  limits: {
    maxDailyLossPct: number; // 0.05
    maxTotalDrawdownPct: number; // 0.10
    profitTargetPct: Unknowable<number>;
    minTradingDays: number; // 4
  };
  breachReasons: string[];
  drawdownHeadroom: Unknowable<number>;
  targetProgress: Unknowable<number>;
  daysTraded: Unknowable<number>;
  consistency: ConsistencyState;
}

export interface ChallengeSimulationResult {
  available: boolean;
  nSimulations: number;
  pPass: Unknowable<number>;
  p95WorstDayPct: Unknowable<number>;
  drawdownDistribution: number[] | null;
  breachReasonDistribution: Record<string, number> | null;
  baselineVsStress: { baselinePPass: number; stressPPass: number } | null;
  inputAssumptions: string; // path count, fee basis — never a bare number
}

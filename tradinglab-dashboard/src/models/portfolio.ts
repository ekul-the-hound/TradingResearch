import type { Status, Unknowable } from './truth';
import type { StrategyEvidence } from './strategy';

export interface PortfolioCandidate {
  strategyId: string;
  name: string;
  symbol: string;
  timeframe: string;
  eligible: boolean;
  exclusionReason: string | null; // why it can't be selected
  netSharpe: Unknowable<number>;
  netReturnPct: Unknowable<number>;
  maxDrawdownPct: Unknowable<number>;
  returnsReal: boolean; // only REAL-returns candidates can be combined
  evidence: StrategyEvidence;
}

export interface CorrelationMatrix {
  ids: string[];
  labels: string[];
  values: number[][]; // symmetric; diagonal = 1
}

export interface TailRisk {
  var95: number;
  cvar95: number;
  tailRatio: number;
  skew: number;
  kurtosis: number;
}

export interface PortfolioComputation {
  componentIds: string[];
  overlapWindow: { first: string; last: string } | null;
  // The result of NSGA-II is a Pareto SELECTION over an already-backtested pool.
  // It is NOT parameter optimization. This label must be surfaced in the UI.
  method: 'PARETO_SELECTION';
  combinedSharpe: Unknowable<number>;
  combinedMaxDrawdownPct: Unknowable<number>;
  combinedReturnPct: Unknowable<number>;
  correlation: Unknowable<CorrelationMatrix>;
  tailRisk: Unknowable<TailRisk>;
  // FTMO-aware constraints evaluated on the combined portfolio (proxy basis).
  ftmoConstraint: {
    basis: 'PROXY' | 'AUTHORITATIVE' | 'INCOMPLETE';
    maxTotalDrawdownPct: number;
    combinedDrawdownHeadroom: Unknowable<number>;
    status: Status;
  };
  status: 'COMPUTED' | 'PARTIAL' | 'INSUFFICIENT_OVERLAP' | 'NO_CANDIDATES';
  notes: string[];
}

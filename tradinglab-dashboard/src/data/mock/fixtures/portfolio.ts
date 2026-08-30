import type {
  CorrelationMatrix,
  PortfolioCandidate,
  PortfolioComputation,
  TailRisk,
} from '../../../models/portfolio';
import { proxy, unavailable, val } from '../../../models/truth';
import { FIXTURE_STRATEGIES } from './strategies';
import {
  blockedComputation,
  evaluateCombineGate,
} from '../../../domain/portfolio-builder/combine';

// DEVELOPMENT FIXTURE. Not from any database.

// Derive candidates from the strategy pool. In the current (zero-survivor) pool,
// only REAL-returns strategies are eligible; the synthetic-risk one is excluded
// with an explicit reason. This honestly reflects that no portfolio can form.
export const FIXTURE_CANDIDATES: PortfolioCandidate[] = FIXTURE_STRATEGIES.map(
  (s) => {
    const returnsReal = s.returnsProvenance === 'REAL';
    let eligible = true;
    let reason: string | null = null;

    if (s.stage === 'REJECTED') {
      eligible = false;
      reason = 'Strategy is rejected.';
    } else if (!returnsReal) {
      eligible = false;
      reason =
        s.returnsProvenance === 'SYNTHETIC_RISK'
          ? 'Returns are synthetic-risk; cannot enter a portfolio.'
          : 'Returns unavailable — not yet backtested.';
    } else if (s.stage === 'DISCOVERED' || s.stage === 'BACKTESTED') {
      eligible = false;
      reason = 'Not yet validated — below portfolio-candidate stage.';
    }

    return {
      strategyId: s.strategyId,
      name: s.name,
      symbol: s.symbol,
      timeframe: s.timeframe,
      eligible,
      exclusionReason: reason,
      netSharpe: s.netSharpe,
      netReturnPct: s.netReturnPct,
      maxDrawdownPct: s.maxDrawdownPct,
      returnsReal,
      evidence: s.evidence,
    };
  },
);

function fakeCorrelation(ids: string[], labels: string[]): CorrelationMatrix {
  const n = ids.length;
  const values: number[][] = [];
  for (let i = 0; i < n; i++) {
    values[i] = [];
    for (let j = 0; j < n; j++) {
      values[i][j] = i === j ? 1 : Number((0.2 + ((i + j) % 3) * 0.15).toFixed(2));
    }
  }
  return { ids, labels, values };
}

const TAIL: TailRisk = {
  var95: -2.4,
  cvar95: -3.9,
  tailRatio: 0.95,
  skew: -0.22,
  kurtosis: 3.9,
};

export function computePortfolioFixture(ids: string[]): PortfolioComputation {
  const selected = FIXTURE_CANDIDATES.filter((c) => ids.includes(c.strategyId));
  const gate = evaluateCombineGate(selected);

  if (!gate.canCombine) {
    return blockedComputation(gate);
  }

  // Combinable path (would only trigger if the pool had ≥2 REAL eligible ones).
  const usable = selected.filter((c) => c.eligible && c.returnsReal);
  const labels = usable.map((c) => c.name);
  return {
    componentIds: gate.usableIds,
    overlapWindow: { first: '2019-01-01', last: '2024-06-30' },
    method: 'PARETO_SELECTION',
    combinedSharpe: val(0.71),
    combinedMaxDrawdownPct: val(16.2),
    combinedReturnPct: val(9.8),
    correlation: val(fakeCorrelation(gate.usableIds, labels)),
    tailRisk: val(TAIL),
    ftmoConstraint: {
      basis: 'PROXY',
      maxTotalDrawdownPct: 10,
      combinedDrawdownHeadroom: proxy(
        -6.2,
        'Proxy: combined summary drawdown vs. 10% limit, not FTMOComplianceChecker.',
      ),
      status: 'WARNING',
    },
    status: 'COMPUTED',
    notes: [
      'Selection performed by NSGA-II over an already-backtested pool (Pareto selection, not parameter optimization).',
    ],
  };
}

// Used when nothing is selected yet.
export const EMPTY_COMPUTATION: PortfolioComputation = {
  componentIds: [],
  overlapWindow: null,
  method: 'PARETO_SELECTION',
  combinedSharpe: unavailable('No candidates selected.'),
  combinedMaxDrawdownPct: unavailable('No candidates selected.'),
  combinedReturnPct: unavailable('No candidates selected.'),
  correlation: unavailable('No candidates selected.'),
  tailRisk: unavailable('No candidates selected.'),
  ftmoConstraint: {
    basis: 'PROXY',
    maxTotalDrawdownPct: 10,
    combinedDrawdownHeadroom: unavailable('No candidates selected.'),
    status: 'INCOMPLETE',
  },
  status: 'NO_CANDIDATES',
  notes: ['Select candidates to evaluate a portfolio.'],
};

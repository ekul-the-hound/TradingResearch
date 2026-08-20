import type {
  PortfolioCandidate,
  PortfolioComputation,
} from '../../models/portfolio';
import { unavailable } from '../../models/truth';

// Determine whether a set of selected candidates can be combined at all.
// The rule: every selected candidate must be eligible AND have REAL returns,
// and there must be at least two of them. Otherwise the combination is blocked
// with an explicit reason — never fabricated combined metrics.
export interface CombineGate {
  canCombine: boolean;
  status: PortfolioComputation['status'];
  reason: string;
  usableIds: string[];
}

export function evaluateCombineGate(
  selected: PortfolioCandidate[],
): CombineGate {
  if (selected.length === 0) {
    return {
      canCombine: false,
      status: 'NO_CANDIDATES',
      reason: 'No candidates selected.',
      usableIds: [],
    };
  }
  const ineligible = selected.filter((c) => !c.eligible);
  const synthetic = selected.filter((c) => c.eligible && !c.returnsReal);
  const usable = selected.filter((c) => c.eligible && c.returnsReal);

  if (usable.length < 2) {
    const bits: string[] = [];
    if (ineligible.length)
      bits.push(`${ineligible.length} ineligible`);
    if (synthetic.length)
      bits.push(`${synthetic.length} lack REAL returns`);
    return {
      canCombine: false,
      status: usable.length === 0 ? 'NO_CANDIDATES' : 'PARTIAL',
      reason:
        `Need at least two combinable candidates with REAL returns. ` +
        (bits.length ? `Excluded: ${bits.join(', ')}.` : 'Select more candidates.'),
      usableIds: usable.map((c) => c.strategyId),
    };
  }

  return {
    canCombine: true,
    status: 'COMPUTED',
    reason: `${usable.length} combinable candidates with REAL returns.`,
    usableIds: usable.map((c) => c.strategyId),
  };
}

// A blocked computation object with no fabricated numbers.
export function blockedComputation(
  gate: CombineGate,
): PortfolioComputation {
  return {
    componentIds: gate.usableIds,
    overlapWindow: null,
    method: 'PARETO_SELECTION',
    combinedSharpe: unavailable(gate.reason),
    combinedMaxDrawdownPct: unavailable(gate.reason),
    combinedReturnPct: unavailable(gate.reason),
    correlation: unavailable(gate.reason),
    tailRisk: unavailable(gate.reason),
    ftmoConstraint: {
      basis: 'PROXY',
      maxTotalDrawdownPct: 10,
      combinedDrawdownHeadroom: unavailable(gate.reason),
      status: 'INCOMPLETE',
    },
    status: gate.status,
    notes: [gate.reason],
  };
}

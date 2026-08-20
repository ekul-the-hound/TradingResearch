import type { GateCheck, ValidationEvidence } from '../../models/detail';
import type { Status } from '../../models/truth';

export const GATE_THRESHOLDS = {
  minSharpe: 0.5,
  minTrades: 20,
  maxDrawdownPct: 30,
} as const;

export function gate(
  value: number | null,
  threshold: number,
  comparator: '>=' | '<=',
): GateCheck {
  let status: Status;
  if (value === null) status = 'UNKNOWN';
  else if (comparator === '>=') status = value >= threshold ? 'PASS' : 'FAIL';
  else status = value <= threshold ? 'PASS' : 'FAIL';
  return { value, threshold, comparator, status };
}

// Plain-language verdict derived ONLY from present evidence. Never invents a
// composite score. Order of precedence: hard gate fails → reject; missing gate
// evidence → incomplete; overfit → investigate; else promote-eligible.
export function deriveVerdict(
  gates: ValidationEvidence['manualGates'],
  pboKind: 'value' | 'proxy' | 'unavailable' | 'unknown',
  pboValue: number | null,
): { verdict: ValidationEvidence['verdict']; reason: string } {
  const gateList = [gates.sharpe, gates.trades, gates.maxDrawdown];

  if (gateList.some((g) => g.status === 'FAIL')) {
    const failed = gateList
      .filter((g) => g.status === 'FAIL')
      .map((g) => `${g.comparator}${g.threshold}`)
      .join(', ');
    return {
      verdict: 'REJECT',
      reason: `Fails manual gate(s): ${failed}.`,
    };
  }
  if (gateList.some((g) => g.status === 'UNKNOWN')) {
    return {
      verdict: 'INCOMPLETE',
      reason: 'One or more manual gates have no evaluated value.',
    };
  }
  if (pboKind === 'value' && pboValue !== null && pboValue > 0.5) {
    return {
      verdict: 'INVESTIGATE',
      reason: `PBO ${pboValue.toFixed(2)} > 0.5 suggests overfitting despite passing gates.`,
    };
  }
  if (pboKind === 'unknown' || pboKind === 'unavailable') {
    return {
      verdict: 'INVESTIGATE',
      reason:
        'Manual gates pass, but overfitting evidence (PBO/DSR) is not available.',
    };
  }
  return {
    verdict: 'PROMOTE',
    reason: 'Passes manual gates with acceptable overfitting evidence.',
  };
}

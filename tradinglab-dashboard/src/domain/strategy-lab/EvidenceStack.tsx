import type { StrategyEvidence } from '../../models/strategy';
import { EVIDENCE_KEYS, EVIDENCE_LABEL } from '../../models/strategy';
import { StatusChip } from '../../primitives';

// Compact evidence stack. `dense` shows short two-letter codes for the table;
// full mode shows readable labels.
export function EvidenceStack({
  evidence,
  dense = false,
}: {
  evidence: StrategyEvidence;
  dense?: boolean;
}) {
  return (
    <span className="tl-evstack" role="group" aria-label="Evidence">
      {EVIDENCE_KEYS.map((k) => {
        const item = evidence[k];
        const label = dense ? shortCode(k) : EVIDENCE_LABEL[k];
        return (
          <StatusChip
            key={k}
            status={item.status}
            label={label}
            title={`${EVIDENCE_LABEL[k]}: ${item.tooltip}`}
          />
        );
      })}
    </span>
  );
}

function shortCode(k: string): string {
  const map: Record<string, string> = {
    DATA: 'DA',
    COST: 'CO',
    HOLDOUT: 'HO',
    REAL_RETURNS: 'RR',
    MANUAL_GATES: 'MG',
    OVERFITTING: 'OF',
    ROBUSTNESS: 'RB',
    PARAMETER_STABILITY: 'PS',
    PORTFOLIO_FIT: 'PF',
    CHALLENGE_FIT: 'CF',
  };
  return map[k] ?? k.slice(0, 2);
}

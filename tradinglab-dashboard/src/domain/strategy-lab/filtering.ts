import type { StrategyFilter, StrategySummary } from '../../models/strategy';
import { EVIDENCE_KEYS } from '../../models/strategy';

// A strategy's evidence is "complete" when no evidence item is UNKNOWN or
// INCOMPLETE (it may still be FAIL — that's known, not missing).
export function isEvidenceComplete(s: StrategySummary): boolean {
  return EVIDENCE_KEYS.every((k) => {
    const st = s.evidence[k].status;
    return st !== 'UNKNOWN' && st !== 'INCOMPLETE';
  });
}

function evidenceStatus(s: StrategySummary, key: keyof StrategySummary['evidence']) {
  return s.evidence[key].status;
}

export function applyFilter(
  rows: StrategySummary[],
  f: StrategyFilter,
): StrategySummary[] {
  return rows.filter((s) => {
    if (f.stage && f.stage !== 'ALL' && s.stage !== f.stage) return false;
    if (f.symbol && f.symbol !== 'ALL' && s.symbol !== f.symbol) return false;
    if (f.timeframe && f.timeframe !== 'ALL' && s.timeframe !== f.timeframe)
      return false;
    if (f.holdout && f.holdout !== 'ALL' && s.holdout !== f.holdout) return false;
    if (
      f.returnsProvenance &&
      f.returnsProvenance !== 'ALL' &&
      s.returnsProvenance !== f.returnsProvenance
    )
      return false;
    if (
      f.manualGate &&
      f.manualGate !== 'ALL' &&
      evidenceStatus(s, 'MANUAL_GATES') !== f.manualGate
    )
      return false;
    if (
      f.ftmo &&
      f.ftmo !== 'ALL' &&
      s.ftmoFit.kind === 'value' &&
      s.ftmoFit.value !== f.ftmo
    )
      return false;
    if (
      f.discoverySource &&
      f.discoverySource !== 'ALL' &&
      s.discoverySource !== f.discoverySource
    )
      return false;
    if (f.evidenceComplete && !isEvidenceComplete(s)) return false;
    if (f.text) {
      const t = f.text.toLowerCase();
      const hay = `${s.name} ${s.strategyId} ${s.symbol}`.toLowerCase();
      if (!hay.includes(t)) return false;
    }
    return true;
  });
}

export interface SavedViewDef {
  id: string;
  label: string;
  filter: StrategyFilter;
  // extra predicate the flat filter can't express
  predicate?: (s: StrategySummary) => boolean;
}

export const SAVED_VIEWS: SavedViewDef[] = [
  {
    id: 'needs-review',
    label: 'Needs Review',
    filter: {},
    predicate: (s) =>
      s.evidence.MANUAL_GATES.status === 'WARNING' ||
      s.evidence.OVERFITTING.status === 'INCOMPLETE' ||
      s.stage === 'BACKTESTED',
  },
  {
    id: 'validation-ready',
    label: 'Validation Ready',
    filter: { stage: 'COST_ADJUSTED' },
  },
  {
    id: 'evidence-incomplete',
    label: 'Evidence Incomplete',
    filter: {},
    predicate: (s) => !isEvidenceComplete(s),
  },
  {
    id: 'portfolio-candidates',
    label: 'Portfolio Candidates',
    filter: { stage: 'PORTFOLIO_CANDIDATE' },
  },
  {
    id: 'rejected',
    label: 'Rejected',
    filter: { stage: 'REJECTED' },
  },
  {
    id: 'manual-ideas',
    label: 'Manual Ideas',
    filter: {},
    predicate: (s) => s.origin === 'manual',
  },
  {
    id: 'integrity-warnings',
    label: 'Data/Integrity Warnings',
    filter: {},
    predicate: (s) =>
      s.returnsProvenance === 'SYNTHETIC_RISK' ||
      s.dataFingerprint === null ||
      s.evidence.DATA.status === 'WARNING' ||
      s.evidence.DATA.status === 'FAIL',
  },
];

export function applySavedView(
  rows: StrategySummary[],
  view: SavedViewDef,
): StrategySummary[] {
  const base = applyFilter(rows, view.filter);
  return view.predicate ? base.filter(view.predicate) : base;
}

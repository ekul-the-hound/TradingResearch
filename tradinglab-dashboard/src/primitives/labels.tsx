import type { Severity, Status } from '../models/truth';
import { StatusChip } from './StatusChip';

// TruthLabel: KEY: value chip for status-bar / truth-ribbon dimensions.
type TruthTone = 'neutral' | 'pass' | 'fail' | 'warn' | 'info' | 'research';
const TONE_COLOR: Record<TruthTone, string> = {
  neutral: 'var(--c-text-2)',
  pass: 'var(--c-pass)',
  fail: 'var(--c-fail)',
  warn: 'var(--c-warn)',
  info: 'var(--c-info)',
  research: 'var(--c-research)',
};

export function TruthLabel({
  label,
  value,
  tone = 'neutral',
  title,
}: {
  label: string;
  value: string;
  tone?: TruthTone;
  title?: string;
}) {
  return (
    <span className="tl-truth" title={title}>
      <span className="tl-truth__key">{label}</span>
      <span className="tl-truth__val" style={{ color: TONE_COLOR[tone] }}>
        {value}
      </span>
    </span>
  );
}

// Compact evidence badge (used in evidence stacks).
export function InlineEvidenceBadge({
  label,
  status,
  title,
}: {
  label: string;
  status: Status;
  title?: string;
}) {
  return <StatusChip status={status} label={label} title={title} />;
}

export function SeverityIndicator({
  severity,
  showLabel = true,
}: {
  severity: Severity;
  showLabel?: boolean;
}) {
  const cls = severity.toLowerCase();
  return (
    <span className={`tl-sev tl-sev--${cls}`} aria-label={`Severity ${severity}`}>
      <span className="tl-sev__dot" aria-hidden />
      {showLabel && severity}
    </span>
  );
}

export function DataFreshnessLabel({
  timestamp,
  stale = false,
}: {
  timestamp: string | null;
  stale?: boolean;
}) {
  return (
    <span className={`tl-fresh ${stale ? 'tl-fresh--stale' : ''}`}>
      <span aria-hidden>{stale ? '▲' : '◷'}</span>
      {timestamp ? `Last run ${timestamp}` : 'Last run UNKNOWN'}
    </span>
  );
}

import type { Status } from '../models/truth';

// Every status carries a glyph + text, never color alone (CLAUDE.md §6/§8).
const GLYPH: Record<Status, string> = {
  PASS: '✓',
  FAIL: '✕',
  WARNING: '▲',
  INFO: 'i',
  UNKNOWN: '?',
  PROXY: '≈',
  INCOMPLETE: '◐',
  OFFLINE: '○',
};

const CLASS: Record<Status, string> = {
  PASS: 'tl-chip--pass',
  FAIL: 'tl-chip--fail',
  WARNING: 'tl-chip--warning',
  INFO: 'tl-chip--info',
  UNKNOWN: 'tl-chip--unknown',
  PROXY: 'tl-chip--proxy',
  INCOMPLETE: 'tl-chip--incomplete',
  OFFLINE: 'tl-chip--offline',
};

export function StatusChip({
  status,
  label,
  title,
}: {
  status: Status;
  label?: string;
  title?: string;
}) {
  return (
    <span
      className={`tl-chip ${CLASS[status]}`}
      role="status"
      title={title}
      aria-label={`${status}${label ? `: ${label}` : ''}`}
    >
      <span className="tl-chip__glyph" aria-hidden>
        {GLYPH[status]}
      </span>
      {label ?? status}
    </span>
  );
}

import type { ReactNode } from 'react';
import type { Status, Unknowable } from '../../models/truth';
import { StatusChip } from '../../primitives';

// Render an Unknowable numeric cell: value / UNKNOWN / UNAVAILABLE / ≈proxy.
// Never blank, never 0-as-missing.
export function numCell(
  u: Unknowable<number>,
  fmt: (v: number) => string,
): ReactNode {
  switch (u.kind) {
    case 'value':
      return fmt(u.value);
    case 'proxy':
      return (
        <span title={u.note}>
          {fmt(u.value)} <span className="tl-cell-proxy">≈</span>
        </span>
      );
    case 'unavailable':
      return <span className="tl-cell-missing" title={u.reason}>N/A</span>;
    case 'unknown':
      return (
        <span className="tl-cell-missing" title={u.reason ?? 'Unknown'}>
          ?
        </span>
      );
  }
}

// Sort key for an Unknowable numeric: missing sinks to an extreme.
export function numSort(u: Unknowable<number>, missingLow = true): number {
  if (u.kind === 'value' || u.kind === 'proxy') return u.value;
  return missingLow ? -Infinity : Infinity;
}

export function statusCell(u: Unknowable<Status>): ReactNode {
  switch (u.kind) {
    case 'value':
      return <StatusChip status={u.value} />;
    case 'proxy':
      return <StatusChip status={u.value} label={`${u.value} ≈`} title={u.note} />;
    case 'unavailable':
      return <span className="tl-cell-missing" title={u.reason}>N/A</span>;
    case 'unknown':
      return (
        <span className="tl-cell-missing" title={u.reason ?? 'Unknown'}>
          ?
        </span>
      );
  }
}

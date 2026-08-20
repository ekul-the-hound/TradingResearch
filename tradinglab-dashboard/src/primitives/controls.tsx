import type { ReactNode } from 'react';
import type { Status } from '../models/truth';
import { StatusChip } from './StatusChip';
import { UnavailableState, EmptyState, LoadingState, ErrorState } from './states';

export interface SavedView {
  id: string;
  label: string;
}

export function FilterBar({
  children,
  savedViews,
  activeView,
  onSelectView,
}: {
  children?: ReactNode;
  savedViews?: SavedView[];
  activeView?: string;
  onSelectView?: (id: string) => void;
}) {
  return (
    <div className="tl-filterbar" role="search">
      {savedViews && savedViews.length > 0 && (
        <div className="tl-filterbar__group" role="group" aria-label="Saved views">
          {savedViews.map((v) => (
            <button
              key={v.id}
              className="tl-savedview"
              aria-pressed={activeView === v.id}
              onClick={() => onSelectView?.(v.id)}
            >
              {v.label}
            </button>
          ))}
        </div>
      )}
      {children}
    </div>
  );
}

// ChartFrame reserves space for title, unit, time basis, status, and
// empty/error/loading/unavailable states (CLAUDE.md §5).
type ChartState =
  | { state: 'ready' }
  | { state: 'loading' }
  | { state: 'empty'; message?: string }
  | { state: 'unavailable'; reason: string }
  | { state: 'error'; error: string };

export function ChartFrame({
  title,
  unit,
  timeBasis,
  status,
  statusLabel,
  frameState = { state: 'ready' },
  children,
}: {
  title: string;
  unit?: string;
  timeBasis?: string;
  status?: Status;
  statusLabel?: string;
  frameState?: ChartState;
  children?: ReactNode;
}) {
  return (
    <figure className="tl-chartframe" style={{ margin: 0 }}>
      <figcaption className="tl-chartframe__hdr">
        <span className="tl-chartframe__title">
          {title}
          {unit && <span className="tl-chartframe__unit">({unit})</span>}
        </span>
        {status && (
          <span className="tl-chartframe__meta">
            <StatusChip status={status} label={statusLabel} />
          </span>
        )}
      </figcaption>
      <div className="tl-chartframe__plot">
        {frameState.state === 'ready' && children}
        {frameState.state === 'loading' && <LoadingState rows={2} />}
        {frameState.state === 'empty' && (
          <EmptyState title="No data in range" message={frameState.message} />
        )}
        {frameState.state === 'unavailable' && (
          <UnavailableState reason={frameState.reason} />
        )}
        {frameState.state === 'error' && <ErrorState message={frameState.error} />}
      </div>
      {timeBasis && <div className="tl-chartframe__basis">{timeBasis}</div>}
    </figure>
  );
}

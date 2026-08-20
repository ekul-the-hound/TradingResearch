import { useState, type ReactNode } from 'react';
import type { Unknowable } from '../models/truth';
import { DASH } from '../lib/format';

// Renders a metric with a label, honoring Unknowable states. A missing value
// shows UNKNOWN/UNAVAILABLE — never 0 or blank (CLAUDE.md §3/§6).
export function MetricValue<T>({
  label,
  metric,
  render,
  sub,
  tone,
  definition,
}: {
  label: string;
  metric: Unknowable<T>;
  render: (v: T) => string;
  sub?: string;
  tone?: 'pos' | 'neg';
  definition?: DefinitionProps;
}) {
  let valueEl: ReactNode;
  let valueClass = 'tl-metric__value';

  switch (metric.kind) {
    case 'value':
      valueEl = render(metric.value);
      if (tone) valueClass += ` tl-metric__value--${tone}`;
      break;
    case 'proxy':
      valueEl = (
        <>
          {render(metric.value)}
          <span
            title={metric.note}
            style={{ color: 'var(--c-text-muted)', fontSize: 11, marginLeft: 4 }}
          >
            ≈ PROXY
          </span>
        </>
      );
      break;
    case 'unavailable':
      valueEl = 'UNAVAILABLE';
      valueClass += ' tl-metric__value--muted';
      break;
    case 'unknown':
      valueEl = 'UNKNOWN';
      valueClass += ' tl-metric__value--muted';
      break;
  }

  const reason =
    metric.kind === 'unavailable' || metric.kind === 'unknown'
      ? metric.reason
      : undefined;

  return (
    <div className="tl-metric">
      <span className="tl-metric__label">
        {label}
        {definition && <MetricDefinitionTooltip {...definition} />}
      </span>
      <span className={valueClass} title={reason}>
        {valueEl ?? DASH}
      </span>
      {sub && <span className="tl-metric__sub">{sub}</span>}
    </div>
  );
}

export interface DefinitionProps {
  term: string;
  definition: string;
  basis?: string; // calculation / period / source basis
}

export function MetricDefinitionTooltip({
  term,
  definition,
  basis,
}: DefinitionProps) {
  const [open, setOpen] = useState(false);
  return (
    <span
      className="tl-deftip"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
      tabIndex={0}
      role="button"
      aria-label={`Definition of ${term}`}
    >
      <span className="tl-deftip__marker" aria-hidden>
        ?
      </span>
      {open && (
        <dl className="tl-deftip__pop" role="tooltip">
          <dt>{term}</dt>
          <dd>{definition}</dd>
          {basis && <dd className="tl-deftip__pop-basis">{basis}</dd>}
        </dl>
      )}
    </span>
  );
}

import type { Bucket, HealthModule } from '../../models/queue';
import { AppPanel, StatusChip, UnavailableState } from '../../primitives';

const TONE_COLOR: Record<NonNullable<Bucket['tone']>, string> = {
  pass: 'var(--c-pass)',
  fail: 'var(--c-fail)',
  warn: 'var(--c-warn)',
  info: 'var(--c-info)',
  research: 'var(--c-research)',
  neutral: 'var(--c-text-muted)',
};

function MiniBars({ buckets }: { buckets: Bucket[] }) {
  const max = Math.max(...buckets.map((b) => b.value), 1);
  const srText = buckets.map((b) => `${b.label}: ${b.value}`).join(', ');
  return (
    <>
      <span className="tl-sr-only">{srText}</span>
      <div className="tl-health__bars" aria-hidden>
        {buckets.map((b) => (
          <div className="tl-health__bar-col" key={b.label} title={`${b.label}: ${b.value}`}>
            <div
              className="tl-health__bar"
              style={{
                height: `${Math.max(6, (b.value / max) * 100)}%`,
                background: TONE_COLOR[b.tone ?? 'neutral'],
              }}
            />
            <span className="tl-health__bar-label">{b.label}</span>
          </div>
        ))}
      </div>
    </>
  );
}

function Module({ m }: { m: HealthModule }) {
  return (
    <AppPanel flush className="tl-health__mod">
      <div className="tl-health__hdr">
        <span className="tl-health__title">{m.title}</span>
        <StatusChip status={m.status} />
      </div>
      <div className="tl-health__body">
        {m.primary.kind === 'unavailable' || m.primary.kind === 'unknown' ? (
          <UnavailableState
            title={m.primary.kind === 'unknown' ? 'Unknown' : 'Unavailable'}
            reason={m.primary.reason ?? 'Not wired.'}
          />
        ) : (
          <>
            <div className="tl-health__primary">
              <span className="tl-health__primary-val">
                {m.primary.value.value}
              </span>
              <span className="tl-health__primary-label">
                {m.primary.value.label}
              </span>
            </div>
            {m.secondary && (
              <div className="tl-health__secondary">
                <span>{m.secondary.label}</span>
                <span className="tl-health__secondary-val">
                  {m.secondary.value}
                </span>
              </div>
            )}
            {m.buckets && m.buckets.length > 0 && <MiniBars buckets={m.buckets} />}
          </>
        )}
      </div>
      <div className="tl-health__src">Source: {m.sourceLabel}</div>
    </AppPanel>
  );
}

export function ResearchHealthStrip({ modules }: { modules: HealthModule[] }) {
  return (
    <div className="tl-health">
      {modules.map((m) => (
        <Module key={m.key} m={m} />
      ))}
    </div>
  );
}

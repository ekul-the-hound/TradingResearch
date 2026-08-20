export function LoadingState({ rows = 3 }: { rows?: number }) {
  return (
    <div className="tl-state" role="status" aria-live="polite" aria-busy="true">
      <div style={{ width: '100%', display: 'grid', gap: 8 }}>
        {Array.from({ length: rows }).map((_, i) => (
          <div
            key={i}
            className="tl-skel"
            style={{ width: `${90 - i * 12}%`, margin: '0 auto' }}
          />
        ))}
      </div>
      <span className="tl-state__msg">Loading…</span>
    </div>
  );
}

export function EmptyState({
  title = 'No records',
  message,
}: {
  title?: string;
  message?: string;
}) {
  return (
    <div className="tl-state">
      <span className="tl-state__glyph" aria-hidden>
        ∅
      </span>
      <span className="tl-state__title">{title}</span>
      {message && <span className="tl-state__msg">{message}</span>}
    </div>
  );
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div className="tl-state tl-state--error" role="alert">
      <span className="tl-state__glyph" aria-hidden>
        ✕
      </span>
      <span className="tl-state__title">Error</span>
      <span className="tl-state__msg">{message}</span>
    </div>
  );
}

// Distinct from Empty: data source/table/column is absent or not wired.
export function UnavailableState({
  title = 'Data unavailable',
  reason,
}: {
  title?: string;
  reason: string;
}) {
  return (
    <div className="tl-state tl-state--unavailable">
      <span className="tl-state__glyph" aria-hidden>
        ⊘
      </span>
      <span className="tl-state__title">{title}</span>
      <span className="tl-state__msg">{reason}</span>
    </div>
  );
}

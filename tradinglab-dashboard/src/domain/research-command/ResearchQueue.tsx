import { useNavigate } from 'react-router-dom';
import type { ResearchQueueItem } from '../../models/queue';
import type { Severity } from '../../models/truth';
import { EmptyState, SeverityIndicator } from '../../primitives';

const SEV_ORDER: Record<Severity, number> = {
  CRITICAL: 0,
  HIGH: 1,
  MEDIUM: 2,
  LOW: 3,
};

export function ResearchQueue({ items }: { items: ResearchQueueItem[] }) {
  const navigate = useNavigate();
  if (items.length === 0) {
    return (
      <EmptyState
        title="Queue clear"
        message="No integrity, evidence, or run blockers require action."
      />
    );
  }
  const sorted = [...items].sort(
    (a, b) => SEV_ORDER[a.severity] - SEV_ORDER[b.severity],
  );

  return (
    <ul className="tl-queue" aria-label="Research queue">
      {sorted.map((item) => {
        const clickable = !!item.destination;
        return (
          <li
            key={item.id}
            className="tl-queue__item"
            data-clickable={clickable || undefined}
            tabIndex={clickable ? 0 : undefined}
            onClick={clickable ? () => navigate(item.destination!) : undefined}
            onKeyDown={
              clickable
                ? (e) => {
                    if (e.key === 'Enter') navigate(item.destination!);
                  }
                : undefined
            }
          >
            <div className="tl-queue__top">
              <SeverityIndicator severity={item.severity} />
              <span className="tl-queue__title">{item.title}</span>
              <span className="tl-queue__source">{item.sourceLabel}</span>
            </div>
            <div className="tl-queue__reason">{item.reason}</div>
            <div className="tl-queue__foot">
              {item.affects && (
                <span className="tl-queue__affects mono">
                  {item.affects.id}
                </span>
              )}
              <span className="tl-queue__action">→ {item.suggestedAction}</span>
            </div>
          </li>
        );
      })}
    </ul>
  );
}

import type { DependencyHealth } from '../../models/integrity';
import { StatusChip } from '../../primitives';
import { ts } from '../../lib/format';

const STATE_MAP: Record<
  DependencyHealth['state'],
  { status: 'PASS' | 'FAIL' | 'UNKNOWN'; label: string }
> = {
  OK: { status: 'PASS', label: 'OK' },
  DOWN: { status: 'FAIL', label: 'DOWN' },
  NOT_CHECKED: { status: 'UNKNOWN', label: 'NOT CHECKED' },
};

export function DependencyHealthStrip({
  deps,
}: {
  deps: DependencyHealth[];
}) {
  return (
    <div className="tl-deps">
      {deps.map((d) => {
        const m = STATE_MAP[d.state];
        return (
          <div className="tl-dep" key={d.name}>
            <div className="tl-dep__top">
              <span className="tl-dep__name">{d.name}</span>
              <StatusChip status={m.status} label={m.label} />
            </div>
            {d.detail && <div className="tl-dep__detail">{d.detail}</div>}
            <div className="tl-dep__checked">
              {d.lastCheckedAt ? `Checked ${ts(d.lastCheckedAt)}` : 'Never probed'}
            </div>
          </div>
        );
      })}
    </div>
  );
}

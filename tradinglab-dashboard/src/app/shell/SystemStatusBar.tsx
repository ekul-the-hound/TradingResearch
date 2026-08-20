import { useEffect, useState } from 'react';
import type { Loadable } from '../../models/truth';
import type { SystemStatus } from '../../models/system';
import { useRepository } from '../../data/useRepository';
import { TruthLabel } from '../../primitives';
import { toSegments, type StatusSegment } from './statusMapping';

function CopyableSegment({ seg }: { seg: StatusSegment }) {
  const [copied, setCopied] = useState(false);
  if (!seg.copyable) {
    return <TruthLabel label={seg.label} value={seg.value} tone={seg.tone} title={seg.title} />;
  }
  return (
    <button
      type="button"
      className="tl-statusbar__copy"
      title={copied ? 'Copied' : seg.title}
      onClick={() => {
        navigator.clipboard?.writeText(seg.copyable!);
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1200);
      }}
    >
      <TruthLabel
        label={seg.label}
        value={copied ? 'copied ✓' : seg.value}
        tone={seg.tone}
      />
    </button>
  );
}

export function SystemStatusBar({ isFixture }: { isFixture?: boolean }) {
  const repo = useRepository();
  const [loadable, setLoadable] = useState<Loadable<SystemStatus>>({
    state: 'loading',
  });

  useEffect(() => {
    let alive = true;
    repo
      .getSystemStatus()
      .then((r) => alive && setLoadable(r))
      .catch(
        (e) =>
          alive &&
          setLoadable({ state: 'error', error: String(e?.message ?? e) }),
      );
    return () => {
      alive = false;
    };
  }, [repo]);

  return (
    <div className="tl-statusarea tl-statusbar" role="status" aria-label="System status">
      <StatusBarBody loadable={loadable} isFixture={isFixture ?? repo.isFixture} />
    </div>
  );
}

function StatusBarBody({
  loadable,
  isFixture,
}: {
  loadable: Loadable<SystemStatus>;
  isFixture: boolean;
}) {
  if (loadable.state === 'loading') {
    return <span className="tl-statusbar__note">Reading system status…</span>;
  }
  if (loadable.state === 'error') {
    return (
      <span className="tl-statusbar__note tl-statusbar__note--err">
        System status error: {loadable.error}
      </span>
    );
  }
  if (loadable.state === 'unavailable') {
    return (
      <span className="tl-statusbar__note">
        System status unavailable — {loadable.reason}
      </span>
    );
  }
  if (loadable.state === 'empty') {
    return <span className="tl-statusbar__note">No system status recorded.</span>;
  }

  const segments = toSegments(loadable.data);
  return (
    <>
      <div className="tl-statusbar__segs">
        {segments.map((seg) => (
          <CopyableSegment key={seg.key} seg={seg} />
        ))}
      </div>
      {isFixture && (
        <span className="tl-statusbar__fixture" title="Values are development fixtures, not database-sourced">
          DEV FIXTURE
        </span>
      )}
    </>
  );
}

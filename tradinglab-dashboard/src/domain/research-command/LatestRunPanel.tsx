import type { PipelineRun } from '../../models/pipeline';
import type { ReturnsProvenance, Status, Unknowable } from '../../models/truth';
import { StatusChip, TruthLabel, WarningBanner } from '../../primitives';
import { int, shortFingerprint, ts } from '../../lib/format';

function u(v: Unknowable<number>): string {
  if (v.kind === 'value' || v.kind === 'proxy') return int(v.value);
  return v.kind === 'unavailable' ? 'UNAVAILABLE' : 'UNKNOWN';
}

const RUN_STATUS: Record<PipelineRun['status'], { status: Status; label: string }> =
  {
    SUCCESS: { status: 'PASS', label: 'SUCCESS' },
    FAILED: { status: 'FAIL', label: 'FAILED' },
    PARTIAL: { status: 'WARNING', label: 'PARTIAL' },
    NO_RUNS: { status: 'UNKNOWN', label: 'NO RUNS' },
  };

const PROV: Record<ReturnsProvenance, { status: Status; label: string }> = {
  REAL: { status: 'PASS', label: 'REAL' },
  UNVERIFIED: { status: 'WARNING', label: 'UNVERIFIED' },
  SYNTHETIC_RISK: { status: 'FAIL', label: 'SYNTHETIC RISK' },
  UNAVAILABLE: { status: 'UNKNOWN', label: 'UNAVAILABLE' },
};

function Field({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="tl-run__field">
      <span className="tl-run__field-label">{label}</span>
      <span className="tl-run__field-value">{value}</span>
    </div>
  );
}

export function LatestRunPanel({ run }: { run: PipelineRun }) {
  if (run.status === 'NO_RUNS') {
    return (
      <div style={{ padding: 'var(--s-4)' }}>
        <WarningBanner tone="info">
          No pipeline runs recorded yet. Run <code>run_pipeline.py</code> to
          populate results.
        </WarningBanner>
      </div>
    );
  }

  const rs = RUN_STATUS[run.status];
  const prov = PROV[run.returnsProvenance];
  const duration =
    run.durationSec != null ? `${Math.round(run.durationSec / 60)}m` : '—';

  return (
    <div className="tl-run">
      <div className="tl-run__head">
        <span className="tl-run__id mono">{run.runId}</span>
        <StatusChip status={rs.status} label={rs.label} />
      </div>

      <div className="tl-run__grid">
        <Field label="Completed" value={ts(run.completedAt)} />
        <Field label="Duration" value={duration} />
        <Field label="Inputs" value={u(run.inputCount)} />
        <Field
          label="Survivors"
          value={
            <span
              style={{
                color:
                  run.survivorCount.kind === 'value' &&
                  run.survivorCount.value === 0
                    ? 'var(--c-warn)'
                    : undefined,
              }}
            >
              {u(run.survivorCount)}
            </span>
          }
        />
        <Field
          label="Top candidate"
          value={
            run.topCandidateId ? (
              <span className="mono">{run.topCandidateId}</span>
            ) : (
              '—'
            )
          }
        />
        <Field
          label="Fingerprint"
          value={
            run.dataFingerprint ? (
              <span className="mono">{shortFingerprint(run.dataFingerprint)}</span>
            ) : (
              'NONE'
            )
          }
        />
      </div>

      <div className="tl-run__truth">
        <TruthLabel
          label="Holdout"
          value={run.holdout}
          tone={run.holdout === 'SEALED' ? 'pass' : 'neutral'}
        />
        <TruthLabel label="Cost" value={run.costProfileName} />
        <TruthLabel label="Returns" value={prov.label} tone={
          run.returnsProvenance === 'REAL' ? 'pass'
          : run.returnsProvenance === 'SYNTHETIC_RISK' ? 'fail'
          : 'warn'
        } />
      </div>

      {run.primaryBlocker && (
        <div className="tl-run__blocker">
          <span className="tl-run__blocker-label">Primary blocker</span>
          <span className="tl-run__blocker-text">{run.primaryBlocker}</span>
        </div>
      )}
    </div>
  );
}

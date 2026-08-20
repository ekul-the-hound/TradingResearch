import type { GateCheck, ValidationEvidence } from '../../models/detail';
import type { Status, Unknowable } from '../../models/truth';
import {
  AppPanel,
  MetricValue,
  SectionTitle,
  StatusChip,
  WarningBanner,
} from '../../primitives';

function GateRow({ label, g, unit = '' }: { label: string; g: GateCheck; unit?: string }) {
  return (
    <div className="tl-gate">
      <span className="tl-gate__label">{label}</span>
      <span className="tl-gate__val">
        {g.value === null ? 'UNKNOWN' : `${g.value}${unit}`}
        <span className="tl-gate__thresh">
          {' '}
          ({g.comparator} {g.threshold}
          {unit})
        </span>
      </span>
      <StatusChip status={g.status} />
    </div>
  );
}

function StatCell({
  label,
  u,
  fmt = (v: number) => v.toFixed(2),
}: {
  label: string;
  u: Unknowable<number>;
  fmt?: (v: number) => string;
}) {
  return <MetricValue label={label} metric={u} render={fmt} />;
}

function ScanChip({ label, u }: { label: string; u: Unknowable<Status> }) {
  return (
    <div className="tl-scan">
      <span className="tl-scan__label">{label}</span>
      {u.kind === 'value' ? (
        <StatusChip status={u.value} />
      ) : (
        <StatusChip status="UNKNOWN" label={u.kind === 'unavailable' ? 'N/A' : 'not run'} />
      )}
    </div>
  );
}

const VERDICT_STATUS: Record<ValidationEvidence['verdict'], Status> = {
  PROMOTE: 'PASS',
  INVESTIGATE: 'WARNING',
  REJECT: 'FAIL',
  INCOMPLETE: 'INCOMPLETE',
  UNKNOWN: 'UNKNOWN',
};

export function ValidationTab({ v }: { v: ValidationEvidence }) {
  return (
    <>
      <AppPanel>
        <div className="tl-verdict">
          <div className="tl-verdict__head">
            <SectionTitle>Current promotion verdict</SectionTitle>
            <StatusChip status={VERDICT_STATUS[v.verdict]} label={v.verdict} />
          </div>
          <p className="tl-verdict__reason">{v.verdictReason}</p>
          <p className="tl-verdict__note">
            Verdict is derived only from available evidence. No composite score is
            fabricated.
          </p>
        </div>
      </AppPanel>

      <div style={{ height: 'var(--s-4)' }} />

      <div className="tl-grid">
        <div style={{ gridColumn: 'span 6' }}>
          <AppPanel>
            <SectionTitle>Manual gates</SectionTitle>
            <div className="tl-gates">
              <GateRow label="Sharpe ≥ 0.5" g={v.manualGates.sharpe} />
              <GateRow label="Trades ≥ 20" g={v.manualGates.trades} />
              <GateRow label="Max DD ≤ 30%" g={v.manualGates.maxDrawdown} unit="%" />
            </div>
            <div className="tl-holdout">
              <span>Holdout</span>
              <StatusChip
                status={v.holdout.state === 'SEALED' ? 'PASS' : 'UNKNOWN'}
                label={`${v.holdout.state} (${v.holdout.fraction.toFixed(2)})`}
              />
            </div>
          </AppPanel>
        </div>

        <div style={{ gridColumn: 'span 6' }}>
          <AppPanel>
            <SectionTitle>Overfitting & significance</SectionTitle>
            <div className="tl-dh-metrics">
              <StatCell label="PBO" u={v.pbo} />
              <StatCell label="Deflated Sharpe" u={v.deflatedSharpe} />
              <StatCell label="CSCV" u={v.cscv} />
              <StatCell label="Permutation p" u={v.permutationPValue} fmt={(x) => x.toFixed(3)} />
              <StatCell label="Param stability" u={v.parameterStability} />
            </div>
          </AppPanel>
        </div>

        <div style={{ gridColumn: 'span 12' }}>
          <AppPanel>
            <SectionTitle>Static & robustness scans</SectionTitle>
            <div className="tl-scans">
              <ScanChip label="Walk-forward" u={v.walkForward} />
              <ScanChip label="Lookahead scan" u={v.lookaheadScan} />
              <ScanChip label="Prohibited patterns" u={v.prohibitedPatternScan} />
            </div>
          </AppPanel>
        </div>
      </div>

      {v.parameterStability.kind !== 'value' && (
        <>
          <div style={{ height: 'var(--s-4)' }} />
          <WarningBanner tone="warning">
            <strong>No parameter-sweep data.</strong> Stability/sensitivity
            heatmaps are only shown from actual sweep results — none exist for
            this strategy.
          </WarningBanner>
        </>
      )}
    </>
  );
}

import { useCallback } from 'react';
import { useRepository } from '../../data/useRepository';
import { useLoadable } from '../../lib/hooks';
import {
  AppPanel,
  type Column,
  DenseDataTable,
  LoadableView,
  PanelHeader,
  SectionTitle,
  StatusChip,
  WarningBanner,
} from '../../primitives';
import type { DatasetRecord, IntegrityStatus } from '../../models/integrity';
import { DependencyHealthStrip } from './DependencyHealthStrip';
import { int, shortFingerprint } from '../../lib/format';
import './data-integrity.css';

const DATASET_COLS: Column<DatasetRecord>[] = [
  {
    key: 'fp',
    header: 'Fingerprint',
    render: (d) =>
      d.fingerprint ? (
        <span className="mono">{shortFingerprint(d.fingerprint)}</span>
      ) : (
        <span className="tl-di-missing">NONE (predates tracking)</span>
      ),
  },
  { key: 'src', header: 'Source', render: (d) => d.source },
  { key: 'sym', header: 'Symbol', render: (d) => d.symbol, sortValue: (d) => d.symbol },
  { key: 'tf', header: 'TF', render: (d) => d.timeframe },
  {
    key: 'range',
    header: 'Date range',
    render: (d) => (d.dateRange ? `${d.dateRange.first} → ${d.dateRange.last}` : 'UNKNOWN'),
  },
  {
    key: 'bars',
    header: 'Bars',
    numeric: true,
    sortValue: (d) => d.barCount ?? -1,
    render: (d) => (d.barCount != null ? int(d.barCount) : 'UNKNOWN'),
  },
  {
    key: 'tz',
    header: 'TZ verified',
    render: (d) =>
      d.timezoneVerified.kind === 'value' ? (
        <StatusChip status={d.timezoneVerified.value ? 'PASS' : 'FAIL'} label={d.timezoneVerified.value ? 'YES' : 'NO'} />
      ) : (
        <StatusChip status="UNKNOWN" label="UNKNOWN" />
      ),
  },
  {
    key: 'used',
    header: 'Used by',
    numeric: true,
    sortValue: (d) => d.usedByResultCount,
    render: (d) => `${int(d.usedByResultCount)} results`,
  },
];

function CoverageBar({ data }: { data: IntegrityStatus }) {
  const c = data.provenanceCoverage;
  const pctWith = c.totalResults ? (c.withFingerprint / c.totalResults) * 100 : 0;
  return (
    <div>
      <div className="tl-cov__row">
        <span>Provenance fingerprint coverage</span>
        <span className="mono">
          {c.withFingerprint} / {c.totalResults} ({pctWith.toFixed(0)}%)
        </span>
      </div>
      <div className="tl-cov__track" aria-hidden>
        <div
          className="tl-cov__fill tl-cov__fill--pass"
          style={{ width: `${pctWith}%` }}
        />
      </div>
      {c.missingFingerprint > 0 && (
        <p className="tl-di-note">
          {c.missingFingerprint} result(s) have NULL fingerprints — they predate
          provenance tracking. This NULL is the intended signal, not an error.
        </p>
      )}
    </div>
  );
}

export function DataIntegrityPage() {
  const repo = useRepository();
  const loadable = useLoadable(
    useCallback(() => repo.getIntegrityStatus(), [repo]),
  );
  const badge = repo.isFixture ? <StatusChip status="INFO" label="DEV FIXTURE" /> : undefined;

  return (
    <>
      <h1 className="tl-page-title">Data & Integrity</h1>
      <p className="tl-page-sub">
        Can I trust the datasets, provenance, validation boundaries, and config
        behind these conclusions?
      </p>

      <LoadableView loadable={loadable} emptyTitle="No integrity data">
        {(data) => (
          <>
            {data.overallStatus === 'WARNING' && (
              <WarningBanner tone="warning">
                <strong>{data.warningCount} open integrity warning(s).</strong>{' '}
                Synthetic-return risk and missing provenance fingerprints are
                present. Details below.
              </WarningBanner>
            )}

            <div style={{ height: 'var(--s-4)' }} />

            {/* Returns ledger + holdout */}
            <div className="tl-grid">
              <div style={{ gridColumn: 'span 6' }}>
                <AppPanel>
                  <SectionTitle>Returns provenance ledger</SectionTitle>
                  <div className="tl-ledger">
                    <LedgerCell label="Real" value={data.returnsLedger.real} tone="pass" />
                    <LedgerCell label="Unverified" value={data.returnsLedger.unverified} tone="warn" />
                    <LedgerCell label="Synthetic risk" value={data.returnsLedger.syntheticRisk} tone="fail" />
                    <LedgerCell label="Unavailable" value={data.returnsLedger.unavailable} tone="neutral" />
                  </div>
                  <p className="tl-di-note">
                    ALLOW_SYNTHETIC_RETURNS ={' '}
                    <span className="mono">
                      {String(data.returnsLedger.allowSyntheticFlag)}
                    </span>
                    .{' '}
                    {data.returnsLedger.syntheticRiskResultIds.length > 0 && (
                      <>
                        Flagged:{' '}
                        {data.returnsLedger.syntheticRiskResultIds.join(', ')}.
                      </>
                    )}
                  </p>
                </AppPanel>
              </div>

              <div style={{ gridColumn: 'span 6' }}>
                <AppPanel>
                  <div className="tl-verdict__head">
                    <SectionTitle>Holdout boundary</SectionTitle>
                    <StatusChip
                      status={data.holdout.state === 'SEALED' ? 'PASS' : 'UNKNOWN'}
                      label={data.holdout.state}
                    />
                  </div>
                  <div className="tl-ledger">
                    <LedgerCell
                      label="Fraction"
                      value={data.holdout.fraction}
                      fmt={(v) => v.toFixed(2)}
                    />
                    <LedgerCell
                      label="Cutoff"
                      valueText={data.holdout.cutoffDate ?? 'UNKNOWN'}
                    />
                    <LedgerCell
                      label="Sealed"
                      value={
                        data.holdout.sealedResultCount.kind === 'value'
                          ? data.holdout.sealedResultCount.value
                          : undefined
                      }
                    />
                  </div>
                </AppPanel>
              </div>
            </div>

            <div style={{ height: 'var(--s-4)' }} />

            {/* Dataset registry */}
            <AppPanel flush>
              <PanelHeader
                title="Dataset registry"
                subtitle="HistData datasets and their fingerprints"
                meta={badge}
              />
              <CoverageWrap data={data} />
              <DenseDataTable
                columns={DATASET_COLS}
                rows={data.datasets}
                getRowId={(d) =>
                  d.fingerprint ?? `null-${d.symbol}-${d.timeframe}-${d.usedByResultCount}`
                }
                initialSortKey="used"
              />
            </AppPanel>

            <div style={{ height: 'var(--s-5)' }} />

            {/* Dependency health */}
            <SectionTitle>Dependency health</SectionTitle>
            <WarningBanner tone="info">
              These services are <strong>not probed from the dashboard</strong>.
              Their status shows NOT CHECKED rather than a fabricated green OK.
            </WarningBanner>
            <div style={{ height: 'var(--s-2)' }} />
            <DependencyHealthStrip deps={data.dependencies} />

            <div style={{ height: 'var(--s-5)' }} />

            {/* Config freeze */}
            <AppPanel flush>
              <PanelHeader
                title="Config freeze"
                subtitle="Frozen research configuration"
                meta={
                  <StatusChip
                    status={data.configFreeze.frozen ? 'PASS' : 'WARNING'}
                    label={data.configFreeze.frozen ? 'FROZEN' : 'UNFROZEN'}
                  />
                }
              />
              <div style={{ padding: 'var(--s-4)' }}>
                {data.configFreeze.driftDetected.kind === 'value' &&
                  data.configFreeze.driftDetected.value && (
                    <WarningBanner tone="critical">
                      Config drift detected against the frozen hash.
                    </WarningBanner>
                  )}
                <table className="tl-cfg-table">
                  <tbody>
                    {data.configFreeze.keys.map((k) => (
                      <tr key={k.key}>
                        <th className="mono">{k.key}</th>
                        <td className="mono">{k.value}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </AppPanel>
          </>
        )}
      </LoadableView>
    </>
  );
}

function CoverageWrap({ data }: { data: IntegrityStatus }) {
  return (
    <div style={{ padding: 'var(--s-4)', borderBottom: 'var(--border)' }}>
      <CoverageBar data={data} />
    </div>
  );
}

function LedgerCell({
  label,
  value,
  valueText,
  tone = 'neutral',
  fmt = (v: number) => int(v),
}: {
  label: string;
  value?: number;
  valueText?: string;
  tone?: 'pass' | 'fail' | 'warn' | 'neutral';
  fmt?: (v: number) => string;
}) {
  const color =
    tone === 'pass'
      ? 'var(--c-pass)'
      : tone === 'fail'
        ? 'var(--c-fail)'
        : tone === 'warn'
          ? 'var(--c-warn)'
          : 'var(--c-text)';
  return (
    <div className="tl-ledger__cell">
      <span className="tl-ledger__label">{label}</span>
      <span className="tl-ledger__value" style={{ color }}>
        {valueText ?? (value != null ? fmt(value) : 'UNKNOWN')}
      </span>
    </div>
  );
}

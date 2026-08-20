import {
  AppPanel,
  PanelHeader,
  SectionTitle,
  WarningBanner,
  StatusChip,
  TruthLabel,
  InlineEvidenceBadge,
  SeverityIndicator,
  DataFreshnessLabel,
  MetricValue,
  LoadingState,
  EmptyState,
  ErrorState,
  UnavailableState,
  DenseDataTable,
  FilterBar,
  ChartFrame,
  type Column,
  type SavedView,
} from '../primitives';
import type { Status } from '../models/truth';
import { val, unknown, unavailable, proxy } from '../models/truth';
import { pct, ratio, int, ts } from '../lib/format';

const ALL_STATUS: Status[] = [
  'PASS',
  'FAIL',
  'WARNING',
  'INFO',
  'UNKNOWN',
  'PROXY',
  'INCOMPLETE',
  'OFFLINE',
];

interface DemoRow {
  id: string;
  name: string;
  symbol: string;
  tf: string;
  sharpe: number | null;
  dd: number | null;
  trades: number | null;
  provenance: string;
}

const DEMO_ROWS: DemoRow[] = [
  {
    id: 'EURUSD_H1_MeanRev_v07',
    name: 'EURUSD_H1_MeanRev',
    symbol: 'EUR/USD',
    tf: '1H',
    sharpe: 0.62,
    dd: 18.4,
    trades: 41,
    provenance: 'REAL',
  },
  {
    id: 'GBPUSD_H4_Break_v03',
    name: 'GBPUSD_H4_Break',
    symbol: 'GBP/USD',
    tf: '4H',
    sharpe: -0.11,
    dd: 33.2,
    trades: 12,
    provenance: 'SYNTHETIC_RISK',
  },
  {
    id: 'EURUSD_H4_ADX_v01',
    name: 'EURUSD_H4_ADX',
    symbol: 'EUR/USD',
    tf: '4H',
    sharpe: null,
    dd: null,
    trades: null,
    provenance: 'UNAVAILABLE',
  },
];

const COLS: Column<DemoRow>[] = [
  { key: 'name', header: 'Strategy', render: (r) => r.name, sortValue: (r) => r.name },
  { key: 'symbol', header: 'Symbol', render: (r) => r.symbol },
  { key: 'tf', header: 'TF', render: (r) => r.tf },
  {
    key: 'sharpe',
    header: 'Net Sharpe',
    title: 'Sharpe ratio, net of pessimistic cost profile',
    numeric: true,
    sortValue: (r) => r.sharpe ?? -Infinity,
    render: (r) => (r.sharpe == null ? 'UNKNOWN' : ratio(r.sharpe)),
  },
  {
    key: 'dd',
    header: 'Max DD',
    numeric: true,
    sortValue: (r) => r.dd ?? Infinity,
    render: (r) => (r.dd == null ? 'UNKNOWN' : pct(r.dd, 1)),
  },
  {
    key: 'trades',
    header: 'Trades',
    numeric: true,
    sortValue: (r) => r.trades ?? -Infinity,
    render: (r) => (r.trades == null ? 'UNKNOWN' : int(r.trades)),
  },
  {
    key: 'prov',
    header: 'Returns',
    render: (r) =>
      r.provenance === 'REAL' ? (
        <StatusChip status="PASS" label="REAL" />
      ) : r.provenance === 'SYNTHETIC_RISK' ? (
        <StatusChip status="FAIL" label="SYNTH RISK" />
      ) : (
        <StatusChip status="UNKNOWN" label="UNAVAIL" />
      ),
  },
];

const SAVED_VIEWS: SavedView[] = [
  { id: 'needs-review', label: 'Needs Review' },
  { id: 'validation-ready', label: 'Validation Ready' },
  { id: 'evidence-incomplete', label: 'Evidence Incomplete' },
];

function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <AppPanel>
      <SectionTitle>{title}</SectionTitle>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center' }}>
        {children}
      </div>
    </AppPanel>
  );
}

export function Preview() {
  return (
    <>
      <h1 className="tl-page-title">Design System</h1>
      <p className="tl-page-sub">
        All TradingLab primitives with realistic research-system labels. This
        route is a development preview.
      </p>

      <WarningBanner tone="info">
        <strong>DEV FIXTURE.</strong> Every value on this page is illustrative and
        not sourced from any database.
      </WarningBanner>

      <div className="tl-grid" style={{ marginTop: 16 }}>
        <div style={{ gridColumn: 'span 12' }}>
          <Group title="Status chips (label + glyph, never color-only)">
            {ALL_STATUS.map((s) => (
              <StatusChip key={s} status={s} />
            ))}
          </Group>
        </div>

        <div style={{ gridColumn: 'span 6' }}>
          <Group title="Truth labels">
            <TruthLabel label="Mode" value="BACKTEST" tone="info" />
            <TruthLabel label="Data" value="HISTORICAL" />
            <TruthLabel label="Holdout" value="SEALED" tone="pass" />
            <TruthLabel label="Returns" value="SYNTHETIC RISK" tone="fail" />
            <TruthLabel label="Compliance" value="PROXY" tone="warn" />
            <TruthLabel label="Broker" value="OFFLINE" tone="warn" />
          </Group>
        </div>

        <div style={{ gridColumn: 'span 6' }}>
          <Group title="Evidence stack">
            <InlineEvidenceBadge label="DATA" status="PASS" />
            <InlineEvidenceBadge label="COST" status="PASS" />
            <InlineEvidenceBadge label="HOLDOUT" status="PASS" />
            <InlineEvidenceBadge label="REAL RETURNS" status="FAIL" />
            <InlineEvidenceBadge label="MANUAL GATES" status="WARNING" />
            <InlineEvidenceBadge label="OVERFITTING" status="INCOMPLETE" />
            <InlineEvidenceBadge label="ROBUSTNESS" status="UNKNOWN" />
            <InlineEvidenceBadge label="CHALLENGE FIT" status="PROXY" />
          </Group>
        </div>

        <div style={{ gridColumn: 'span 6' }}>
          <Group title="Severity">
            <SeverityIndicator severity="CRITICAL" />
            <SeverityIndicator severity="HIGH" />
            <SeverityIndicator severity="MEDIUM" />
            <SeverityIndicator severity="LOW" />
          </Group>
        </div>

        <div style={{ gridColumn: 'span 6' }}>
          <Group title="Freshness (actual timestamp only)">
            <DataFreshnessLabel timestamp={ts('2026-01-14T19:42:00Z')} />
            <DataFreshnessLabel timestamp={ts('2025-11-02T08:10:00Z')} stale />
            <DataFreshnessLabel timestamp={null} />
          </Group>
        </div>

        <div style={{ gridColumn: 'span 12' }}>
          <Group title="Metric values (Unknowable states)">
            <div style={{ minWidth: 120 }}>
              <MetricValue
                label="Net Sharpe"
                metric={val(0.62)}
                render={(v) => ratio(v)}
                sub="n=41 trades"
                definition={{
                  term: 'Net Sharpe',
                  definition:
                    'Annualized return over volatility, net of the pessimistic cost profile.',
                  basis: 'Basis: 2-pip spread, 1-pip slippage, intraday/no swaps.',
                }}
              />
            </div>
            <div style={{ minWidth: 120 }}>
              <MetricValue
                label="Net Return"
                metric={val(-3.2)}
                render={(v) => pct(v)}
                tone="neg"
              />
            </div>
            <div style={{ minWidth: 120 }}>
              <MetricValue
                label="P(pass)"
                metric={proxy(0.41, 'From display proxy, not FTMOComplianceChecker.')}
                render={(v) => pct(v * 100, 0)}
              />
            </div>
            <div style={{ minWidth: 140 }}>
              <MetricValue
                label="Consistency headroom"
                metric={unknown('Consistency threshold not configured.')}
                render={(v: number) => pct(v)}
              />
            </div>
            <div style={{ minWidth: 140 }}>
              <MetricValue
                label="Capacity"
                metric={unavailable('Insufficient liquidity data.')}
                render={(v: number) => int(v)}
              />
            </div>
          </Group>
        </div>

        <div style={{ gridColumn: 'span 8' }}>
          <AppPanel flush>
            <PanelHeader
              title="Dense data table"
              subtitle="Sortable, keyboard-navigable, tabular numerals"
              meta={<StatusChip status="INFO" label="DEV FIXTURE" />}
            />
            <FilterBar savedViews={SAVED_VIEWS} activeView="needs-review">
              <div className="tl-filterbar__group">
                <label htmlFor="tf">Timeframe</label>
                <select id="tf" className="tl-select" defaultValue="all">
                  <option value="all">All</option>
                  <option>1H</option>
                  <option>4H</option>
                </select>
              </div>
            </FilterBar>
            <DenseDataTable
              columns={COLS}
              rows={DEMO_ROWS}
              getRowId={(r) => r.id}
              onRowActivate={() => {}}
              initialSortKey="sharpe"
            />
          </AppPanel>
        </div>

        <div style={{ gridColumn: 'span 4' }}>
          <ChartFrame
            title="Equity curve"
            unit="account %"
            timeBasis="2019-01 → 2024-06 · HistData · SEALED holdout"
            status="INFO"
            statusLabel="BACKTEST"
          >
            <div
              style={{
                height: 160,
                display: 'grid',
                placeItems: 'center',
                color: 'var(--c-text-muted)',
                fontSize: 11,
              }}
            >
              (chart utilities land in later prompts)
            </div>
          </ChartFrame>
        </div>

        <div style={{ gridColumn: 'span 3' }}>
          <AppPanel flush>
            <PanelHeader title="Loading" />
            <LoadingState />
          </AppPanel>
        </div>
        <div style={{ gridColumn: 'span 3' }}>
          <AppPanel flush>
            <PanelHeader title="Empty" />
            <EmptyState
              title="No strategies"
              message="No records match the current filters."
            />
          </AppPanel>
        </div>
        <div style={{ gridColumn: 'span 3' }}>
          <AppPanel flush>
            <PanelHeader title="Unavailable" />
            <UnavailableState reason="lineage.db not found or not wired." />
          </AppPanel>
        </div>
        <div style={{ gridColumn: 'span 3' }}>
          <AppPanel flush>
            <PanelHeader title="Error" />
            <ErrorState message="Query failed: no such column data_fingerprint." />
          </AppPanel>
        </div>

        <div style={{ gridColumn: 'span 12' }}>
          <Group title="Banners">
            <div style={{ display: 'grid', gap: 8, width: '100%' }}>
              <WarningBanner tone="critical">
                <strong>Synthetic-returns risk.</strong> 1 result exposes returns
                that were not verified via require_returns().
              </WarningBanner>
              <WarningBanner tone="warning">
                <strong>Consistency threshold unset.</strong> Challenge headroom
                cannot be computed until configured.
              </WarningBanner>
            </div>
          </Group>
        </div>
      </div>
    </>
  );
}

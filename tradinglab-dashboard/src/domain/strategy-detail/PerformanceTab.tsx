import type { PerformanceEvidence } from '../../models/detail';
import type { Unknowable } from '../../models/truth';
import {
  AppPanel,
  ChartFrame,
  type Column,
  DenseDataTable,
  MetricValue,
  PanelHeader,
  SectionTitle,
} from '../../primitives';
import { LineChart } from '../../charts/LineChart';
import { int, pct, ratio } from '../../lib/format';
import type { TradeRow } from '../../models/detail';

const TRADE_COLS: Column<TradeRow>[] = [
  { key: 'entry', header: 'Entry', render: (t) => t.entryDate, sortValue: (t) => t.entryDate },
  { key: 'exit', header: 'Exit', render: (t) => t.exitDate },
  { key: 'dir', header: 'Dir', render: (t) => (t.isLong ? 'Long' : 'Short') },
  { key: 'ep', header: 'Entry Px', numeric: true, render: (t) => t.entryPrice.toFixed(5) },
  { key: 'xp', header: 'Exit Px', numeric: true, render: (t) => t.exitPrice.toFixed(5) },
  {
    key: 'pnl',
    header: 'P&L',
    numeric: true,
    sortValue: (t) => t.pnl,
    render: (t) => (
      <span style={{ color: t.pnl >= 0 ? 'var(--c-pass)' : 'var(--c-fail)' }}>
        {t.pnl.toFixed(2)}
      </span>
    ),
  },
  { key: 'ret', header: 'Ret %', numeric: true, sortValue: (t) => t.returnPct, render: (t) => pct(t.returnPct, 2) },
  { key: 'bars', header: 'Bars', numeric: true, render: (t) => int(t.durationBars) },
];

function chartStateFor(u: Unknowable<unknown>):
  | { state: 'ready' }
  | { state: 'unavailable'; reason: string }
  | { state: 'empty' } {
  if (u.kind === 'value' || u.kind === 'proxy') return { state: 'ready' };
  if (u.kind === 'unavailable') return { state: 'unavailable', reason: u.reason };
  return { state: 'unavailable', reason: u.reason ?? 'Unknown' };
}

export function PerformanceTab({ perf }: { perf: PerformanceEvidence }) {
  const eq = perf.equity;
  const series =
    eq.kind === 'value'
      ? [
          {
            label: 'Net',
            color: 'var(--c-info)',
            points: eq.value.map((p, i) => ({ x: i, y: p.net })),
          },
          ...(perf.hasGross
            ? [
                {
                  label: 'Gross',
                  color: 'var(--c-text-muted)',
                  points: eq.value.map((p, i) => ({ x: i, y: p.gross ?? p.net })),
                },
              ]
            : []),
        ]
      : [];

  const periodTxt = perf.period
    ? `${perf.period.first} → ${perf.period.last}`
    : 'period UNKNOWN';

  return (
    <>
      <div className="tl-grid">
        <div style={{ gridColumn: 'span 8' }}>
          <ChartFrame
            title="Equity curve"
            unit="account %"
            timeBasis={`${periodTxt} · net${perf.hasGross ? ' vs gross' : ''} · returns ${perf.returnsProvenance}`}
            status={perf.returnsProvenance === 'REAL' ? 'PASS' : 'WARNING'}
            statusLabel={perf.returnsProvenance === 'REAL' ? 'REAL' : 'SYNTH RISK'}
            frameState={chartStateFor(eq)}
          >
            {series.length > 0 && <LineChart series={series} yUnit="%" />}
          </ChartFrame>
        </div>
        <div style={{ gridColumn: 'span 4' }}>
          <AppPanel>
            <SectionTitle>Headline (net of costs)</SectionTitle>
            <div className="tl-dh-metrics">
              <MetricValue label="Net return" metric={perf.headline.netReturnPct} render={(v) => pct(v, 1)} />
              <MetricValue label="Sharpe" metric={perf.headline.sharpe} render={(v) => ratio(v)} />
              <MetricValue label="Max DD" metric={perf.headline.maxDrawdownPct} render={(v) => pct(v, 1)} />
              <MetricValue
                label="Trades"
                metric={perf.headline.tradeCount}
                render={(v) => int(v)}
                sub="sample size"
              />
              <MetricValue label="Win rate" metric={perf.headline.winRatePct} render={(v) => pct(v, 1)} />
              <MetricValue label="Profit factor" metric={perf.headline.profitFactor} render={(v) => ratio(v)} />
              <MetricValue label="Expectancy" metric={perf.headline.expectancy} render={(v) => ratio(v)} />
            </div>
          </AppPanel>
        </div>
      </div>

      <div style={{ height: 'var(--s-4)' }} />

      <AppPanel flush>
        <PanelHeader title="Trade blotter" subtitle="From backtest_trades where present" />
        {perf.trades.kind === 'value' ? (
          perf.trades.value.length > 0 ? (
            <DenseDataTable
              columns={TRADE_COLS}
              rows={perf.trades.value}
              getRowId={(t) => `${t.entryDate}-${t.exitDate}-${t.entryPrice}`}
              initialSortKey="entry"
            />
          ) : (
            <div className="tl-detail-empty">No trades in this result.</div>
          )
        ) : (
          <div className="tl-detail-empty">
            {perf.trades.kind === 'unavailable'
              ? `Trade list unavailable — ${perf.trades.reason}`
              : perf.trades.kind === 'unknown'
                ? `Trade list not available — ${perf.trades.reason ?? 'unknown'}`
                : 'Trade list not available.'}
          </div>
        )}
      </AppPanel>
    </>
  );
}

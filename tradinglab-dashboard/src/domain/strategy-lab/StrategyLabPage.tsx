import { useCallback, useMemo, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useRepository } from '../../data/useRepository';
import { useLoadable } from '../../lib/hooks';
import {
  AppPanel,
  DenseDataTable,
  FilterBar,
  LoadableView,
  PanelHeader,
  StatusChip,
  WarningBanner,
} from '../../primitives';
import type { StrategySummary } from '../../models/strategy';
import { buildColumns } from './columns';
import { SAVED_VIEWS, applyFilter, applySavedView } from './filtering';
import './strategy-lab.css';

const ALL = 'ALL';

export function StrategyLabPage() {
  const repo = useRepository();
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const stageParam = params.get('stage') ?? ALL;

  const [activeView, setActiveView] = useState<string | null>(null);
  const [symbol, setSymbol] = useState(ALL);
  const [tf, setTf] = useState(ALL);
  const [prov, setProv] = useState(ALL);
  const [text, setText] = useState('');

  const loadable = useLoadable(
    useCallback(() => repo.listStrategies(), [repo]),
  );
  const columns = useMemo(() => buildColumns(), []);

  const rows = useMemo(
    () => (loadable.state === 'ready' ? loadable.data : []),
    [loadable],
  );

  const filtered: StrategySummary[] = useMemo(() => {
    let base = rows;
    if (activeView) {
      const view = SAVED_VIEWS.find((v) => v.id === activeView);
      if (view) base = applySavedView(base, view);
    }
    return applyFilter(base, {
      stage: stageParam === ALL ? 'ALL' : (stageParam as never),
      symbol: symbol === ALL ? 'ALL' : symbol,
      timeframe: tf === ALL ? 'ALL' : tf,
      returnsProvenance: prov === ALL ? 'ALL' : (prov as never),
      text: text || undefined,
    });
  }, [rows, activeView, stageParam, symbol, tf, prov, text]);

  const symbols = useMemo(
    () => Array.from(new Set(rows.map((r) => r.symbol))).sort(),
    [rows],
  );
  const tfs = useMemo(
    () => Array.from(new Set(rows.map((r) => r.timeframe))).sort(),
    [rows],
  );

  const fixtureBadge = repo.isFixture ? (
    <StatusChip status="INFO" label="DEV FIXTURE" />
  ) : undefined;

  return (
    <>
      <h1 className="tl-page-title">Strategy Lab</h1>
      <p className="tl-page-sub">
        Which strategies deserve research, validation, selection, or rejection —
        and what is the evidence?
      </p>

      <WarningBanner tone="info">
        Read-only. Metrics that an authoritative source has not returned show as
        UNKNOWN / N/A, never as fabricated values. FTMO fit is a display PROXY
        until wired to FTMOComplianceChecker.
      </WarningBanner>

      <div style={{ height: 'var(--s-4)' }} />

      <FilterBar
        savedViews={SAVED_VIEWS.map((v) => ({ id: v.id, label: v.label }))}
        activeView={activeView ?? undefined}
        onSelectView={(id) => setActiveView((cur) => (cur === id ? null : id))}
      >
        <div className="tl-filterbar__group">
          <label htmlFor="f-stage">Stage</label>
          <select
            id="f-stage"
            className="tl-select"
            value={stageParam}
            onChange={(e) => {
              const v = e.target.value;
              if (v === ALL) params.delete('stage');
              else params.set('stage', v);
              setParams(params, { replace: true });
            }}
          >
            <option value={ALL}>All</option>
            <option value="DISCOVERED">Discovered</option>
            <option value="CODE_VALID">Code Valid</option>
            <option value="BACKTESTED">Backtested</option>
            <option value="COST_ADJUSTED">Cost Adjusted</option>
            <option value="VALIDATED">Validated</option>
            <option value="PORTFOLIO_CANDIDATE">Portfolio Candidate</option>
            <option value="REJECTED">Rejected</option>
          </select>
        </div>
        <div className="tl-filterbar__group">
          <label htmlFor="f-sym">Symbol</label>
          <select
            id="f-sym"
            className="tl-select"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
          >
            <option value={ALL}>All</option>
            {symbols.map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </div>
        <div className="tl-filterbar__group">
          <label htmlFor="f-tf">TF</label>
          <select
            id="f-tf"
            className="tl-select"
            value={tf}
            onChange={(e) => setTf(e.target.value)}
          >
            <option value={ALL}>All</option>
            {tfs.map((t) => (
              <option key={t}>{t}</option>
            ))}
          </select>
        </div>
        <div className="tl-filterbar__group">
          <label htmlFor="f-prov">Returns</label>
          <select
            id="f-prov"
            className="tl-select"
            value={prov}
            onChange={(e) => setProv(e.target.value)}
          >
            <option value={ALL}>All</option>
            <option value="REAL">Real</option>
            <option value="UNVERIFIED">Unverified</option>
            <option value="SYNTHETIC_RISK">Synthetic risk</option>
            <option value="UNAVAILABLE">Unavailable</option>
          </select>
        </div>
        <div className="tl-filterbar__group" style={{ marginLeft: 'auto' }}>
          <input
            className="tl-input"
            placeholder="Filter name / id…"
            value={text}
            onChange={(e) => setText(e.target.value)}
            aria-label="Text filter"
          />
        </div>
      </FilterBar>

      <div style={{ height: 'var(--s-3)' }} />

      <AppPanel flush>
        <PanelHeader
          title="Strategies"
          subtitle={`${filtered.length} of ${rows.length} shown`}
          meta={fixtureBadge}
        />
        <LoadableView
          loadable={loadable}
          emptyTitle="No strategy records"
          emptyMessage="No strategies found. Run discovery and backtests to populate."
        >
          {() =>
            filtered.length === 0 ? (
              <div style={{ padding: 'var(--s-5)', textAlign: 'center', color: 'var(--c-text-muted)' }}>
                No strategies match the current filters.
              </div>
            ) : (
              <DenseDataTable
                columns={columns}
                rows={filtered}
                getRowId={(r) => r.strategyId}
                onRowActivate={(r) => navigate(`/strategies/${r.strategyId}`)}
                initialSortKey="sharpe"
              />
            )
          }
        </LoadableView>
      </AppPanel>

      <p className="tl-lab-note">
        Sample size (trade count) is shown alongside performance. Click a row to
        open its evidence dossier. Metric definitions are on column hover.
      </p>
    </>
  );
}

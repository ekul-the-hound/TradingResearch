import { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useRepository } from '../../data/useRepository';
import { useLoadable } from '../../lib/hooks';
import {
  AppPanel,
  LoadableView,
  MetricValue,
  PanelHeader,
  SectionTitle,
  StatusChip,
  UnavailableState,
  WarningBanner,
} from '../../primitives';
import type {
  PortfolioCandidate,
  PortfolioComputation,
} from '../../models/portfolio';
import type { Loadable } from '../../models/truth';
import { CorrelationMatrixView } from './CorrelationMatrixView';
import { pct, ratio } from '../../lib/format';
import './portfolio-builder.css';

function CandidateRow({
  c,
  checked,
  onToggle,
  onOpen,
}: {
  c: PortfolioCandidate;
  checked: boolean;
  onToggle: () => void;
  onOpen: () => void;
}) {
  return (
    <tr className={c.eligible ? '' : 'tl-cand--ineligible'}>
      <td>
        <input
          type="checkbox"
          checked={checked}
          disabled={!c.eligible}
          onChange={onToggle}
          aria-label={`Select ${c.name}`}
        />
      </td>
      <td>
        <button className="tl-linklike mono" onClick={onOpen}>
          {c.name}
        </button>
      </td>
      <td>{c.symbol}</td>
      <td>{c.timeframe}</td>
      <td className="tl-td--num">
        {c.netSharpe.kind === 'value' ? ratio(c.netSharpe.value) : '—'}
      </td>
      <td className="tl-td--num">
        {c.maxDrawdownPct.kind === 'value' ? pct(c.maxDrawdownPct.value, 1) : '—'}
      </td>
      <td>
        {c.eligible ? (
          <StatusChip status="PASS" label="ELIGIBLE" />
        ) : (
          <StatusChip status="FAIL" label="EXCLUDED" title={c.exclusionReason ?? ''} />
        )}
      </td>
      <td className="tl-cand__reason">{c.exclusionReason ?? ''}</td>
    </tr>
  );
}

function ResultPanel({ comp }: { comp: PortfolioComputation }) {
  const blocked = comp.status !== 'COMPUTED';
  return (
    <AppPanel flush>
      <PanelHeader
        title="Combined portfolio"
        subtitle="Metrics computed only from REAL-returns candidates over their overlap window"
        meta={<StatusChip status="INFO" label="PARETO SELECTION" />}
      />
      <div style={{ padding: 'var(--s-4)' }}>
        <WarningBanner tone="info">
          Selection uses NSGA-II over an already-backtested pool — this is{' '}
          <strong>Pareto selection, not parameter optimization</strong>. No
          sweep is performed here.
        </WarningBanner>

        <div style={{ height: 'var(--s-3)' }} />

        {blocked ? (
          <UnavailableState
            title={
              comp.status === 'NO_CANDIDATES'
                ? 'No combinable portfolio'
                : comp.status === 'PARTIAL'
                  ? 'Not enough combinable candidates'
                  : 'Insufficient overlap'
            }
            reason={comp.notes[0] ?? 'Cannot combine the current selection.'}
          />
        ) : (
          <>
            <div className="tl-dh-metrics">
              <MetricValue label="Combined Sharpe" metric={comp.combinedSharpe} render={(v) => ratio(v)} />
              <MetricValue label="Combined return" metric={comp.combinedReturnPct} render={(v) => pct(v, 1)} />
              <MetricValue label="Combined max DD" metric={comp.combinedMaxDrawdownPct} render={(v) => pct(v, 1)} />
              <MetricValue
                label="FTMO DD headroom"
                metric={comp.ftmoConstraint.combinedDrawdownHeadroom}
                render={(v) => pct(v, 1)}
              />
            </div>
            {comp.overlapWindow && (
              <p className="tl-di-note">
                Overlap window: {comp.overlapWindow.first} →{' '}
                {comp.overlapWindow.last}. FTMO constraint basis:{' '}
                {comp.ftmoConstraint.basis}.
              </p>
            )}
            <div style={{ height: 'var(--s-4)' }} />
            {comp.correlation.kind === 'value' ? (
              <>
                <SectionTitle>Correlation matrix</SectionTitle>
                <CorrelationMatrixView m={comp.correlation.value} />
              </>
            ) : (
              <UnavailableState
                title="Correlation unavailable"
                reason={
                  comp.correlation.kind === 'unavailable'
                    ? comp.correlation.reason
                    : 'Not computed.'
                }
              />
            )}
          </>
        )}
      </div>
    </AppPanel>
  );
}

export function PortfolioBuilderPage() {
  const repo = useRepository();
  const navigate = useNavigate();
  const candidates = useLoadable(
    useCallback(() => repo.getPortfolioCandidates(), [repo]),
  );

  const [selected, setSelected] = useState<string[]>([]);
  const [comp, setComp] = useState<Loadable<PortfolioComputation>>({
    state: 'loading',
  });

  useEffect(() => {
    let alive = true;
    repo.computePortfolio(selected).then((r) => {
      if (alive) setComp(r);
    });
    return () => {
      alive = false;
    };
  }, [repo, selected]);

  const rows = useMemo(
    () => (candidates.state === 'ready' ? candidates.data : []),
    [candidates],
  );
  const eligibleCount = useMemo(
    () => rows.filter((c) => c.eligible).length,
    [rows],
  );

  const badge = repo.isFixture ? <StatusChip status="INFO" label="DEV FIXTURE" /> : undefined;

  function toggle(id: string) {
    setSelected((cur) =>
      cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id],
    );
  }

  return (
    <>
      <h1 className="tl-page-title">Portfolio Builder</h1>
      <p className="tl-page-sub">
        Which validated strategies improve the portfolio after correlation, risk,
        cost, drawdown, and FTMO constraints?
      </p>

      {eligibleCount === 0 && (
        <WarningBanner tone="warning">
          <strong>No eligible candidates.</strong> The current pool has no
          validated, REAL-returns strategies at portfolio-candidate stage, so no
          portfolio can be formed. This reflects the real state of the pool.
        </WarningBanner>
      )}

      <div style={{ height: 'var(--s-4)' }} />

      <AppPanel flush>
        <PanelHeader
          title="Candidates"
          subtitle={`${eligibleCount} eligible of ${rows.length}. Only REAL-returns, validated strategies can be combined.`}
          meta={badge}
        />
        <LoadableView loadable={candidates} emptyTitle="No candidates">
          {(data) => (
            <div className="tl-table-wrap">
              <table className="tl-table" aria-label="Portfolio candidates">
                <thead>
                  <tr>
                    <th />
                    <th>Strategy</th>
                    <th>Symbol</th>
                    <th>TF</th>
                    <th>Net Sharpe</th>
                    <th>Max DD</th>
                    <th>Status</th>
                    <th>Exclusion reason</th>
                  </tr>
                </thead>
                <tbody>
                  {data.map((c) => (
                    <CandidateRow
                      key={c.strategyId}
                      c={c}
                      checked={selected.includes(c.strategyId)}
                      onToggle={() => toggle(c.strategyId)}
                      onOpen={() => navigate(`/strategies/${c.strategyId}`)}
                    />
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </LoadableView>
      </AppPanel>

      <div style={{ height: 'var(--s-4)' }} />

      <LoadableView loadable={comp} emptyTitle="No computation">
        {(data) => <ResultPanel comp={data} />}
      </LoadableView>
    </>
  );
}

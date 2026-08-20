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
  ChallengeReadiness,
  RuleModelRow,
} from '../../models/challenge';
import type { ChallengeSimulationResult } from '../../models/ftmo';
import type { Loadable, Status, Unknowable } from '../../models/truth';
import { pct, prob } from '../../lib/format';
import './challenge-readiness.css';

const BASIS_STATUS: Record<RuleModelRow['basis'], Status> = {
  AUTHORITATIVE: 'PASS',
  PROXY: 'PROXY',
  INCOMPLETE: 'INCOMPLETE',
  NOT_MODELED: 'FAIL',
};

function fitCell(u: Unknowable<Status>) {
  if (u.kind === 'value') return <StatusChip status={u.value} />;
  if (u.kind === 'proxy')
    return <StatusChip status={u.value} label={`${u.value} ≈`} title={u.note} />;
  return <StatusChip status="UNKNOWN" label={u.kind === 'unavailable' ? 'N/A' : '?'} />;
}

function SimPanel({
  sim,
  strategyName,
}: {
  sim: Loadable<ChallengeSimulationResult>;
  strategyName: string | null;
}) {
  return (
    <AppPanel>
      <SectionTitle>Challenge simulation{strategyName ? ` — ${strategyName}` : ''}</SectionTitle>
      <LoadableView loadable={sim} emptyTitle="Select a strategy">
        {(s) =>
          s.available ? (
            <>
              <div className="tl-dh-metrics">
                <MetricValue label="P(pass)" metric={s.pPass} render={(v) => prob(v)} />
                <MetricValue label="P95 worst day" metric={s.p95WorstDayPct} render={(v) => pct(v, 2)} />
                <MetricValue
                  label="Simulations"
                  metric={{ kind: 'value', value: s.nSimulations }}
                  render={(v) => String(v)}
                />
              </div>
              <p className="tl-di-note">{s.inputAssumptions}</p>
            </>
          ) : (
            <UnavailableState
              title="No simulation run"
              reason={s.inputAssumptions}
            />
          )
        }
      </LoadableView>
    </AppPanel>
  );
}

export function ChallengeReadinessPage() {
  const repo = useRepository();
  const navigate = useNavigate();
  const readiness = useLoadable(
    useCallback(() => repo.getChallengeReadiness(), [repo]),
  );

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [simResult, setSimResult] = useState<Loadable<ChallengeSimulationResult>>({
    state: 'empty',
  });

  useEffect(() => {
    if (!selectedId) return;
    let alive = true;
    repo.getChallengeSimulation(selectedId).then((r) => {
      if (alive) setSimResult(r);
    });
    return () => {
      alive = false;
    };
  }, [repo, selectedId]);

  // When nothing is selected, present the empty state without a synchronous
  // setState inside an effect.
  const sim: Loadable<ChallengeSimulationResult> = selectedId
    ? simResult
    : { state: 'empty' };

  const data = readiness.state === 'ready' ? readiness.data : null;
  const selectedName = useMemo(
    () => data?.perStrategy.find((p) => p.strategyId === selectedId)?.name ?? null,
    [data, selectedId],
  );
  const badge = repo.isFixture ? <StatusChip status="INFO" label="DEV FIXTURE" /> : undefined;

  return (
    <>
      <h1 className="tl-page-title">Challenge Readiness</h1>
      <p className="tl-page-sub">
        Is this appropriate for an FTMO-style challenge under the modeled rules,
        and where are the limits?
      </p>

      <WarningBanner tone="warning">
        <strong>Compliance basis is PROXY.</strong> Rule checks are derived from
        summary statistics, not FTMOComplianceChecker against per-trade data.
        Readiness is never shown as an unconditional pass while on a proxy basis.
      </WarningBanner>

      <div style={{ height: 'var(--s-4)' }} />

      <LoadableView loadable={readiness} emptyTitle="No readiness data">
        {(d: ChallengeReadiness) => (
          <>
            {/* Firm profile */}
            <AppPanel flush>
              <PanelHeader
                title={`${d.firm.firm} — ${d.firm.phase}`}
                subtitle={
                  d.firm.accountSize
                    ? `Account ${d.firm.accountSize.toLocaleString('en-US')}`
                    : 'Account size unknown'
                }
                meta={badge}
              />
              <div style={{ padding: 'var(--s-4)' }}>
                <div className="tl-dh-metrics">
                  <MetricValue
                    label="Profit target"
                    metric={{ kind: 'value', value: d.firm.profitTargetPct }}
                    render={(v) => pct(v, 0)}
                  />
                  <MetricValue
                    label="Max daily loss"
                    metric={{ kind: 'value', value: d.firm.maxDailyLossPct }}
                    render={(v) => pct(v, 0)}
                  />
                  <MetricValue
                    label="Max total DD"
                    metric={{ kind: 'value', value: d.firm.maxTotalDrawdownPct }}
                    render={(v) => pct(v, 0)}
                  />
                  <MetricValue
                    label="Min trading days"
                    metric={{ kind: 'value', value: d.firm.minTradingDays }}
                    render={(v) => String(v)}
                  />
                  <MetricValue
                    label="Consistency threshold"
                    metric={
                      d.firm.consistencyThresholdPct == null
                        ? { kind: 'unknown', reason: 'Not configured.' }
                        : { kind: 'value', value: d.firm.consistencyThresholdPct }
                    }
                    render={(v) => pct(v, 0)}
                  />
                </div>
              </div>
            </AppPanel>

            <div style={{ height: 'var(--s-4)' }} />

            {/* Rule model */}
            <AppPanel flush>
              <PanelHeader
                title="Rule model"
                subtitle="Which rules are actually modeled, and on what basis"
              />
              <div className="tl-table-wrap">
                <table className="tl-table" aria-label="Rule model">
                  <thead>
                    <tr>
                      <th>Rule</th>
                      <th>Modeled</th>
                      <th>Basis</th>
                      <th>Note</th>
                    </tr>
                  </thead>
                  <tbody>
                    {d.ruleModel.map((r) => (
                      <tr key={r.rule}>
                        <td>{r.rule}</td>
                        <td>
                          <StatusChip
                            status={r.modeled ? 'PASS' : 'FAIL'}
                            label={r.modeled ? 'YES' : 'NO'}
                          />
                        </td>
                        <td>
                          <StatusChip status={BASIS_STATUS[r.basis]} label={r.basis} />
                        </td>
                        <td className="tl-cand__reason">{r.note}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </AppPanel>

            <div style={{ height: 'var(--s-4)' }} />

            {/* Consistency rule */}
            <AppPanel>
              <SectionTitle>Consistency rule</SectionTitle>
              {d.consistency.evaluated === false ? (
                <WarningBanner tone="warning">
                  <strong>Configuration required.</strong> The consistency
                  threshold is not set, so best-day-share headroom cannot be
                  computed. No number is shown. This rule may structurally
                  prohibit burst-style strategies once configured.
                </WarningBanner>
              ) : (
                <MetricValue
                  label="Best-day share"
                  metric={{ kind: 'value', value: d.consistency.bestDayShare * 100 }}
                  render={(v) => pct(v, 1)}
                />
              )}
            </AppPanel>

            <div style={{ height: 'var(--s-4)' }} />

            {/* Per-strategy fit + simulation */}
            <div className="tl-grid">
              <div style={{ gridColumn: 'span 7' }}>
                <AppPanel flush>
                  <PanelHeader
                    title="Per-strategy challenge fit"
                    subtitle="Proxy assessment. Click a row to simulate."
                  />
                  <div className="tl-table-wrap">
                    <table className="tl-table" aria-label="Per-strategy challenge fit">
                      <thead>
                        <tr>
                          <th>Strategy</th>
                          <th>FTMO fit</th>
                          <th>Basis</th>
                          <th>Note</th>
                        </tr>
                      </thead>
                      <tbody>
                        {d.perStrategy.map((p) => (
                          <tr
                            key={p.strategyId}
                            data-clickable="true"
                            role="button"
                            tabIndex={0}
                            aria-selected={selectedId === p.strategyId}
                            aria-label={`Simulate ${p.name}`}
                            onClick={() => setSelectedId(p.strategyId)}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter' || e.key === ' ') {
                                e.preventDefault();
                                setSelectedId(p.strategyId);
                              }
                            }}
                          >
                            <td>
                              <button
                                className="tl-linklike mono"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  navigate(`/strategies/${p.strategyId}#challenge`);
                                }}
                              >
                                {p.name}
                              </button>
                            </td>
                            <td>{fitCell(p.ftmoFit)}</td>
                            <td>
                              <StatusChip
                                status={p.basis === 'PROXY' ? 'PROXY' : p.basis === 'INCOMPLETE' ? 'INCOMPLETE' : 'PASS'}
                                label={p.basis}
                              />
                            </td>
                            <td className="tl-cand__reason">{p.note}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </AppPanel>
              </div>
              <div style={{ gridColumn: 'span 5' }}>
                <SimPanel sim={sim} strategyName={selectedName} />
              </div>
            </div>
          </>
        )}
      </LoadableView>
    </>
  );
}

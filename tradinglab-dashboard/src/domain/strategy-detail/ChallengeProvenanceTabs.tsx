import type { StrategyDetail } from '../../models/detail';
import type {
  ChallengeSimulationResult,
  FTMOComplianceStatus,
} from '../../models/ftmo';
import type { Loadable } from '../../models/truth';
import {
  AppPanel,
  LoadableView,
  MetricValue,
  SectionTitle,
  StatusChip,
  WarningBanner,
} from '../../primitives';
import { pct, prob } from '../../lib/format';

export function ChallengeFitTab({
  ftmo,
  sim,
}: {
  ftmo: Loadable<FTMOComplianceStatus>;
  sim: Loadable<ChallengeSimulationResult>;
}) {
  return (
    <>
      <WarningBanner tone="warning">
        <strong>Compliance basis is PROXY.</strong> These values are display
        proxies derived from summary statistics, not from FTMOComplianceChecker
        run against per-trade data. Do not treat as authoritative.
      </WarningBanner>

      <div style={{ height: 'var(--s-4)' }} />

      <LoadableView loadable={ftmo} emptyTitle="Strategy not found">
        {(f) => (
          <div className="tl-grid">
            <div style={{ gridColumn: 'span 6' }}>
              <AppPanel>
                <div className="tl-verdict__head">
                  <SectionTitle>Rule model</SectionTitle>
                  <StatusChip status="PROXY" label={f.basis} />
                </div>
                <div className="tl-dh-metrics">
                  <MetricValue
                    label="Max daily loss"
                    metric={{ kind: 'value', value: f.limits.maxDailyLossPct * 100 }}
                    render={(v) => pct(v, 0)}
                  />
                  <MetricValue
                    label="Max total DD"
                    metric={{ kind: 'value', value: f.limits.maxTotalDrawdownPct * 100 }}
                    render={(v) => pct(v, 0)}
                  />
                  <MetricValue label="Profit target" metric={f.limits.profitTargetPct} render={(v) => pct(v, 0)} />
                  <MetricValue
                    label="Min trading days"
                    metric={{ kind: 'value', value: f.limits.minTradingDays }}
                    render={(v) => String(v)}
                  />
                </div>
                <p className="tl-detail-note" style={{ marginTop: 8 }}>
                  Modelled: {f.modelledRules.join(', ')}.{' '}
                  <span style={{ color: 'var(--c-warn)' }}>
                    Unmodelled: {f.unmodelledRules.join(', ')}.
                  </span>
                </p>
              </AppPanel>
            </div>

            <div style={{ gridColumn: 'span 6' }}>
              <AppPanel>
                <SectionTitle>Proxy headroom</SectionTitle>
                <div className="tl-dh-metrics">
                  <MetricValue label="Drawdown headroom" metric={f.drawdownHeadroom} render={(v) => pct(v, 1)} />
                  <MetricValue label="Target progress" metric={f.targetProgress} render={(v) => pct(v, 1)} />
                  <MetricValue label="Days traded" metric={f.daysTraded} render={(v) => String(v)} />
                </div>
                {f.breachReasons.length > 0 && (
                  <ul className="tl-breaches">
                    {f.breachReasons.map((r, i) => (
                      <li key={i}>{r}</li>
                    ))}
                  </ul>
                )}
              </AppPanel>
            </div>

            <div style={{ gridColumn: 'span 12' }}>
              <AppPanel>
                <SectionTitle>Consistency rule</SectionTitle>
                {f.consistency.evaluated === false ? (
                  <WarningBanner tone="warning">
                    <strong>Configuration required.</strong> The consistency
                    threshold is not defined, so consistency headroom cannot be
                    computed. No numeric value is shown.
                  </WarningBanner>
                ) : (
                  <MetricValue
                    label="Best-day share"
                    metric={{ kind: 'value', value: f.consistency.bestDayShare * 100 }}
                    render={(v) => pct(v, 1)}
                  />
                )}
              </AppPanel>
            </div>
          </div>
        )}
      </LoadableView>

      <div style={{ height: 'var(--s-4)' }} />

      <AppPanel>
        <SectionTitle>Challenge simulation</SectionTitle>
        <LoadableView loadable={sim} emptyTitle="Strategy not found">
          {(s) =>
            s.available ? (
              <div className="tl-dh-metrics">
                <MetricValue label="P(pass)" metric={s.pPass} render={(v) => prob(v)} />
                <MetricValue label="P95 worst day" metric={s.p95WorstDayPct} render={(v) => pct(v, 2)} />
              </div>
            ) : (
              <p className="tl-detail-note">
                No simulation output. {s.inputAssumptions}
              </p>
            )
          }
        </LoadableView>
      </AppPanel>
    </>
  );
}

export function ProvenanceTab({ detail }: { detail: StrategyDetail }) {
  const p = detail.provenance;
  const rows: [string, string][] = [
    ['Dataset fingerprint', p.datasetFingerprint ?? 'NONE (predates provenance)'],
    ['Data source', p.dataSource ?? 'UNKNOWN'],
    ['Symbol / timeframe', `${p.symbol} · ${p.timeframe}`],
    ['Date range', p.dateRange ? `${p.dateRange.first} → ${p.dateRange.last}` : 'UNKNOWN'],
    ['Bar count', p.barCount != null ? p.barCount.toLocaleString('en-US') : 'UNKNOWN'],
    [
      'Timezone verified',
      p.timezoneVerified.kind === 'value'
        ? p.timezoneVerified.value
          ? 'YES'
          : 'NO'
        : 'UNKNOWN',
    ],
    ['Code fingerprint', p.codeFingerprint ?? 'UNKNOWN'],
    ['Parent lineage', p.parentIds.length ? p.parentIds.join(', ') : 'none'],
    ['Child lineage', p.childIds.length ? p.childIds.join(', ') : 'none'],
    ['Cost profile', p.costProfile ? `${p.costProfile.name} (${p.costProfile.spreadPips}p/${p.costProfile.slippagePips}p)` : 'UNKNOWN'],
    ['Result tables', p.resultTables.join(', ')],
    ['Returns status', p.returnsStatus],
  ];

  return (
    <>
      <AppPanel flush>
        <table className="tl-prov-table">
          <tbody>
            {rows.map(([k, val]) => (
              <tr key={k}>
                <th>{k}</th>
                <td className="mono">{val}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </AppPanel>

      {p.integrityWarnings.length > 0 && (
        <>
          <div style={{ height: 'var(--s-4)' }} />
          <SectionTitle>Integrity warnings</SectionTitle>
          {p.integrityWarnings.map((w, i) => (
            <div key={i} style={{ marginBottom: 6 }}>
              <WarningBanner tone="critical">{w}</WarningBanner>
            </div>
          ))}
        </>
      )}
    </>
  );
}

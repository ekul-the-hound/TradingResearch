import type { RiskEvidence } from '../../models/detail';
import type { Unknowable } from '../../models/truth';
import {
  AppPanel,
  ChartFrame,
  MetricValue,
  SectionTitle,
  UnavailableState,
} from '../../primitives';
import { LineChart } from '../../charts/LineChart';
import { pct } from '../../lib/format';

function frameState(u: Unknowable<unknown>) {
  if (u.kind === 'value' || u.kind === 'proxy') return { state: 'ready' as const };
  return {
    state: 'unavailable' as const,
    reason: u.kind === 'unavailable' ? u.reason : (u.reason ?? 'Unknown'),
  };
}

export function RiskTab({ risk }: { risk: RiskEvidence }) {
  const dd = risk.drawdownSeries;
  const ddSeries =
    dd.kind === 'value'
      ? [
          {
            label: 'Drawdown',
            color: 'var(--c-fail)',
            points: dd.value.map((p, i) => ({ x: i, y: p.ddPct })),
          },
        ]
      : [];

  return (
    <>
      <ChartFrame
        title="Drawdown profile"
        unit="account %"
        timeBasis="Underwater equity over the test window"
        status="INFO"
        statusLabel="BACKTEST"
        frameState={frameState(dd)}
      >
        {ddSeries.length > 0 && <LineChart series={ddSeries} height={140} yUnit="%" />}
      </ChartFrame>

      <div style={{ height: 'var(--s-4)' }} />

      <div className="tl-grid">
        <div style={{ gridColumn: 'span 6' }}>
          <AppPanel>
            <SectionTitle>Tail risk (real return series required)</SectionTitle>
            <div className="tl-dh-metrics">
              <MetricValue label="VaR 95%" metric={risk.var95Pct} render={(v) => pct(v, 2)} />
              <MetricValue label="CVaR 95%" metric={risk.cvar95Pct} render={(v) => pct(v, 2)} />
              <MetricValue label="Tail ratio" metric={risk.tailRatio} render={(v) => v.toFixed(2)} />
              <MetricValue label="Skew" metric={risk.skew} render={(v) => v.toFixed(2)} />
              <MetricValue label="Kurtosis" metric={risk.kurtosis} render={(v) => v.toFixed(2)} />
            </div>
          </AppPanel>
        </div>

        <div style={{ gridColumn: 'span 6' }}>
          <AppPanel>
            <SectionTitle>MAE / MFE & hidden loss</SectionTitle>
            {risk.maeMfe.kind === 'value' ? (
              <div className="tl-dh-metrics">
                <MetricValue
                  label="Avg MAE"
                  metric={{ kind: 'value', value: risk.maeMfe.value.avgMaePct }}
                  render={(v) => pct(v, 2)}
                />
                <MetricValue
                  label="Avg MFE"
                  metric={{ kind: 'value', value: risk.maeMfe.value.avgMfePct }}
                  render={(v) => pct(v, 2)}
                />
              </div>
            ) : (
              <UnavailableState
                reason={
                  risk.maeMfe.kind === 'unavailable'
                    ? risk.maeMfe.reason
                    : risk.maeMfe.kind === 'unknown'
                      ? (risk.maeMfe.reason ?? 'Requires intrabar data.')
                      : 'Requires intrabar data.'
                }
              />
            )}
          </AppPanel>
        </div>

        <div style={{ gridColumn: 'span 6' }}>
          <AppPanel>
            <SectionTitle>Stop / target analysis</SectionTitle>
            {risk.stopTargetInSample.kind === 'value' ? (
              <p className="tl-detail-note">
                <span className="tl-insample">IN-SAMPLE</span>{' '}
                {risk.stopTargetInSample.value.note} Marked in-sample until
                independently validated.
              </p>
            ) : (
              <UnavailableState reason="Stop/target analysis unavailable." />
            )}
          </AppPanel>
        </div>

        <div style={{ gridColumn: 'span 6' }}>
          <AppPanel>
            <SectionTitle>Capacity / liquidity</SectionTitle>
            {risk.capacity.kind === 'value' ? (
              <p className="tl-detail-note">{risk.capacity.value.note}</p>
            ) : (
              <UnavailableState
                title="Not measured"
                reason={
                  risk.capacity.kind === 'unavailable'
                    ? risk.capacity.reason
                    : 'Insufficient data — not extrapolated.'
                }
              />
            )}
          </AppPanel>
        </div>
      </div>
    </>
  );
}

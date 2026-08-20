import { useCallback, useState } from 'react';
import { useRepository } from '../../data/useRepository';
import { useLoadable } from '../../lib/hooks';
import {
  AppPanel,
  LoadableView,
  MetricValue,
  PanelHeader,
  SectionTitle,
  StatusChip,
  WarningBanner,
} from '../../primitives';
import type {
  DiscoveredStrategy,
  DiscoveryInbox,
  UntestableIdea,
} from '../../models/discovery';
import { ratio, ts } from '../../lib/format';
import './discovery-inbox.css';

const STATUS_CHIP: Record<
  DiscoveredStrategy['status'],
  { s: 'PASS' | 'FAIL' | 'WARNING' | 'INFO' | 'UNKNOWN'; label: string }
> = {
  NEW: { s: 'INFO', label: 'NEW' },
  REVIEWED: { s: 'INFO', label: 'REVIEWED' },
  PROMOTED: { s: 'PASS', label: 'PROMOTED' },
  REJECTED: { s: 'FAIL', label: 'REJECTED' },
  DUPLICATE: { s: 'WARNING', label: 'DUPLICATE' },
};

function QualityBars({ data }: { data: DiscoveryInbox }) {
  const max = Math.max(...data.qualityDistribution.map((b) => b.count), 1);
  const srText = data.qualityDistribution
    .map((b) => `${b.label}: ${b.count}`)
    .join(', ');
  return (
    <>
      <span className="tl-sr-only">Quality distribution — {srText}</span>
      <div className="tl-qbars" aria-hidden>
        {data.qualityDistribution.map((b) => (
          <div className="tl-qbar-col" key={b.label} title={`${b.label}: ${b.count}`}>
            <div
              className="tl-qbar"
              style={{ height: `${Math.max(6, (b.count / max) * 100)}%` }}
            />
            <span className="tl-qbar-label">{b.label}</span>
          </div>
        ))}
      </div>
    </>
  );
}

function flags(s: DiscoveredStrategy): string {
  const f: string[] = [];
  if (s.hasMath) f.push('math');
  if (s.hasCode) f.push('code');
  if (s.hasExplicitParams) f.push('params');
  if (s.hasBacktest) f.push('backtest');
  return f.length ? f.join(' · ') : 'none';
}

function StrategyRow({ s }: { s: DiscoveredStrategy }) {
  const chip = STATUS_CHIP[s.status];
  return (
    <tr className={s.status === 'DUPLICATE' ? 'tl-dup' : ''}>
      <td>
        <div className="tl-disc-name">{s.name}</div>
        <div className="tl-disc-summary">{s.summary}</div>
      </td>
      <td className="tl-td--num">
        {s.qualityScore.kind === 'value' ? (
          <span
            style={{
              color:
                s.qualityScore.value >= 3
                  ? 'var(--c-pass)'
                  : s.qualityScore.value >= 2
                    ? 'var(--c-warn)'
                    : 'var(--c-fail)',
            }}
          >
            {ratio(s.qualityScore.value, 1)}
          </span>
        ) : (
          <span className="tl-di-missing">?</span>
        )}
      </td>
      <td className="tl-disc-flags">{flags(s)}</td>
      <td>{s.indicators.join(', ') || '—'}</td>
      <td className="mono tl-disc-model">{s.modelUsed ?? '—'}</td>
      <td>
        {s.status === 'DUPLICATE' && s.duplicateOf ? (
          <span title={`Duplicate of ${s.duplicateOf}`}>
            <StatusChip status="WARNING" label="DUP" />{' '}
            <span className="mono tl-disc-dupof">→ {s.duplicateOf}</span>
          </span>
        ) : (
          <StatusChip status={chip.s} label={chip.label} />
        )}
      </td>
    </tr>
  );
}

function IdeaCard({ idea }: { idea: UntestableIdea }) {
  return (
    <div className="tl-idea">
      <div className="tl-idea__top">
        <span className="tl-idea__title">{idea.title}</span>
        {idea.category && <span className="tl-idea__cat">{idea.category}</span>}
      </div>
      <p className="tl-idea__desc">{idea.description}</p>
      <div className="tl-idea__why">
        <span className="tl-idea__label">Blocked by</span> {idea.whyUntestable}
      </div>
      <div className="tl-idea__need">
        <span className="tl-idea__label">Data needed</span> {idea.dataNeeded}
      </div>
      <div className="tl-idea__foot">
        {idea.confidence.kind === 'value' && (
          <span>confidence {(idea.confidence.value * 100).toFixed(0)}%</span>
        )}
        {idea.effort && <span>effort {idea.effort}</span>}
        {idea.timeframe && <span>{idea.timeframe}</span>}
      </div>
    </div>
  );
}

export function DiscoveryInboxPage() {
  const repo = useRepository();
  const [tab, setTab] = useState<'candidates' | 'ideas'>('candidates');
  const loadable = useLoadable(
    useCallback(() => repo.getDiscoveryInbox(), [repo]),
  );
  const badge = repo.isFixture ? <StatusChip status="INFO" label="DEV FIXTURE" /> : undefined;

  return (
    <>
      <h1 className="tl-page-title">Discovery Inbox</h1>
      <p className="tl-page-sub">
        Which ideas are worth review, which are redundant, and which are blocked
        by missing data?
      </p>

      <LoadableView loadable={loadable} emptyTitle="No discovery data">
        {(data) => (
          <>
            {data.meanQuality.kind === 'value' && data.meanQuality.value < 3 && (
              <WarningBanner tone="warning">
                <strong>Mean discovery quality is {ratio(data.meanQuality.value, 1)}/5.</strong>{' '}
                The pool is dominated by generic textbook indicators. Strategy
                quality is the binding constraint on passing the gates.
              </WarningBanner>
            )}

            <div style={{ height: 'var(--s-4)' }} />

            {/* Health strip */}
            <div className="tl-grid">
              <div style={{ gridColumn: 'span 4' }}>
                <AppPanel>
                  <SectionTitle>Discovery quality</SectionTitle>
                  <MetricValue
                    label="Mean quality"
                    metric={data.meanQuality}
                    render={(v) => `${ratio(v, 1)} / 5`}
                  />
                  <div style={{ height: 'var(--s-2)' }} />
                  <QualityBars data={data} />
                </AppPanel>
              </div>
              <div style={{ gridColumn: 'span 4' }}>
                <AppPanel>
                  <SectionTitle>Semantic dedup</SectionTitle>
                  <div className="tl-dh-metrics">
                    <MetricValue label="Total" metric={{ kind: 'value', value: data.dedup.total }} render={String} />
                    <MetricValue label="Unique" metric={{ kind: 'value', value: data.dedup.unique }} render={String} />
                    <MetricValue label="Duplicates" metric={{ kind: 'value', value: data.dedup.duplicates }} render={String} tone="neg" />
                  </div>
                  <p className="tl-di-note">Method: {data.dedup.method}</p>
                </AppPanel>
              </div>
              <div style={{ gridColumn: 'span 4' }}>
                <AppPanel>
                  <SectionTitle>Discovery run</SectionTitle>
                  <div className="tl-verdict__head">
                    <span className="tl-di-note">Pipeline status</span>
                    <StatusChip status={data.pipelineStatus} />
                  </div>
                  <p className="tl-di-note">
                    Last discovery:{' '}
                    {data.lastDiscoveryAt ? ts(data.lastDiscoveryAt) : 'UNKNOWN'}
                  </p>
                  <p className="tl-di-note">
                    Source models: {data.sourceModels.join(', ')}
                  </p>
                </AppPanel>
              </div>
            </div>

            <div style={{ height: 'var(--s-4)' }} />

            {/* Tabs */}
            <nav className="tl-tabs" role="tablist" aria-label="Discovery inbox sections">
              <button
                role="tab"
                aria-selected={tab === 'candidates'}
                tabIndex={tab === 'candidates' ? 0 : -1}
                className={`tl-tab ${tab === 'candidates' ? 'is-active' : ''}`}
                onClick={() => setTab('candidates')}
              >
                Candidates ({data.strategies.length})
              </button>
              <button
                role="tab"
                aria-selected={tab === 'ideas'}
                tabIndex={tab === 'ideas' ? 0 : -1}
                className={`tl-tab ${tab === 'ideas' ? 'is-active' : ''}`}
                onClick={() => setTab('ideas')}
              >
                Untestable Ideas ({data.untestableIdeas.length})
              </button>
            </nav>

            {tab === 'candidates' ? (
              <AppPanel flush>
                <PanelHeader
                  title="Extracted strategy candidates"
                  subtitle="From source_extractor + strategy_inbox; dedup via FAISS"
                  meta={badge}
                />
                <div className="tl-table-wrap">
                  <table className="tl-table" aria-label="Extracted strategy candidates">
                    <thead>
                      <tr>
                        <th>Candidate</th>
                        <th>Quality</th>
                        <th>Flags</th>
                        <th>Indicators</th>
                        <th>Model</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.strategies.map((s) => (
                        <StrategyRow key={s.id} s={s} />
                      ))}
                    </tbody>
                  </table>
                </div>
              </AppPanel>
            ) : (
              <AppPanel flush>
                <PanelHeader
                  title="Untestable ideas backlog"
                  subtitle="Ideas blocked by missing data (algorithm_ideas.ideas)"
                  meta={badge}
                />
                <div className="tl-ideas">
                  {data.untestableIdeas.map((idea) => (
                    <IdeaCard key={idea.id} idea={idea} />
                  ))}
                </div>
              </AppPanel>
            )}
          </>
        )}
      </LoadableView>
    </>
  );
}

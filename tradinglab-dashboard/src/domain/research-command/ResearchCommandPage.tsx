import { useCallback } from 'react';
import { useRepository } from '../../data/useRepository';
import { useLoadable } from '../../lib/hooks';
import {
  AppPanel,
  LoadableView,
  PanelHeader,
  SectionTitle,
  StatusChip,
} from '../../primitives';
import { StrategyFunnel } from './StrategyFunnel';
import { LatestRunPanel } from './LatestRunPanel';
import { ResearchQueue } from './ResearchQueue';
import { ResearchHealthStrip } from './ResearchHealthStrip';
import { TruthRibbon } from './TruthRibbon';
import './research-command.css';

export function ResearchCommandPage() {
  const repo = useRepository();
  const status = useLoadable(useCallback(() => repo.getSystemStatus(), [repo]));
  const run = useLoadable(useCallback(() => repo.getLatestRun(), [repo]));
  const funnel = useLoadable(useCallback(() => repo.getFunnel(), [repo]));
  const queue = useLoadable(useCallback(() => repo.getResearchQueue(), [repo]));
  const health = useLoadable(useCallback(() => repo.getResearchHealth(), [repo]));

  const fixtureBadge = repo.isFixture ? (
    <StatusChip status="INFO" label="DEV FIXTURE" />
  ) : undefined;

  return (
    <>
      <h1 className="tl-page-title">Research Command</h1>
      <p className="tl-page-sub">
        Research pipeline, strategy evidence, and integrity status
      </p>

      {/* Truth ribbon directly below the title */}
      {status.state === 'ready' && (
        <TruthRibbon
          status={status.data}
          run={run.state === 'ready' ? run.data : null}
        />
      )}

      {/* Dominant funnel */}
      <AppPanel flush className="tl-rc-funnel-panel">
        <PanelHeader
          title="Strategy Funnel"
          subtitle="Discovery → promotion. Click a stage to filter Strategy Lab."
          meta={fixtureBadge}
        />
        <div style={{ padding: 'var(--s-4)' }}>
          <LoadableView loadable={funnel} emptyTitle="No funnel data">
            {(data) => <StrategyFunnel funnel={data} />}
          </LoadableView>
        </div>
      </AppPanel>

      <div className="tl-grid" style={{ marginTop: 'var(--s-4)' }}>
        {/* Latest run */}
        <div style={{ gridColumn: 'span 7' }}>
          <AppPanel flush>
            <PanelHeader
              title="Latest Pipeline Run"
              subtitle="Most recent run_pipeline.py execution"
              meta={fixtureBadge}
            />
            <LoadableView
              loadable={run}
              emptyTitle="No runs yet"
              emptyMessage="Run run_pipeline.py to populate."
            >
              {(data) => <LatestRunPanel run={data} />}
            </LoadableView>
          </AppPanel>
        </div>

        {/* Research queue */}
        <div style={{ gridColumn: 'span 5' }}>
          <AppPanel flush style={{ height: '100%' }}>
            <PanelHeader
              title="Research Queue"
              subtitle="Rule-driven blockers, most severe first"
              meta={fixtureBadge}
            />
            <div style={{ padding: 'var(--s-2)' }}>
              <LoadableView loadable={queue}>
                {(data) => <ResearchQueue items={data} />}
              </LoadableView>
            </div>
          </AppPanel>
        </div>
      </div>

      {/* Health strip */}
      <div style={{ marginTop: 'var(--s-5)' }}>
        <SectionTitle>Research Health</SectionTitle>
        <LoadableView loadable={health} emptyTitle="No health data">
          {(data) => <ResearchHealthStrip modules={data.modules} />}
        </LoadableView>
      </div>
    </>
  );
}

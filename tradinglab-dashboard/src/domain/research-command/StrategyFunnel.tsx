import { useNavigate } from 'react-router-dom';
import type { FunnelStage, PipelineFunnel } from '../../models/pipeline';
import type { LifecycleStage, Unknowable } from '../../models/truth';
import { MetricDefinitionTooltip } from '../../primitives';
import { int } from '../../lib/format';

const STAGE_LABEL: Record<LifecycleStage, string> = {
  DISCOVERED: 'Discovered',
  CODE_VALID: 'Code Valid',
  BACKTESTED: 'Backtested',
  COST_ADJUSTED: 'Cost Adjusted',
  VALIDATED: 'Validated',
  PORTFOLIO_CANDIDATE: 'Portfolio Candidate',
  PAPER: 'Paper',
  LIVE: 'Live',
  REJECTED: 'Rejected',
  RETIRED: 'Retired',
};

function countText(u: Unknowable<number>): string {
  if (u.kind === 'value') return int(u.value);
  if (u.kind === 'proxy') return int(u.value);
  return u.kind === 'unavailable' ? 'N/A' : '?';
}

function StageCell({ stage }: { stage: FunnelStage }) {
  const navigate = useNavigate();
  const isLive = stage.stage === 'LIVE';
  const blocked =
    stage.blocked.kind === 'value' && stage.blocked.value > 0
      ? stage.blocked.value
      : 0;
  const rejected =
    stage.rejected.kind === 'value' && stage.rejected.value > 0
      ? stage.rejected.value
      : 0;

  return (
    <button
      className="tl-funnel__stage"
      data-live={isLive || undefined}
      onClick={() =>
        navigate(`/strategy-lab?stage=${stage.stage}`)
      }
      title={stage.definition}
    >
      <span className="tl-funnel__stage-name">
        {STAGE_LABEL[stage.stage]}
        <MetricDefinitionTooltip
          term={STAGE_LABEL[stage.stage]}
          definition={stage.definition}
        />
      </span>
      <span className="tl-funnel__count">{countText(stage.count)}</span>
      <span className="tl-funnel__sub">
        {blocked > 0 && (
          <span className="tl-funnel__tag tl-funnel__tag--blocked">
            {blocked} blocked
          </span>
        )}
        {rejected > 0 && (
          <span className="tl-funnel__tag tl-funnel__tag--rejected">
            {rejected} rejected
          </span>
        )}
        {blocked === 0 && rejected === 0 && (
          <span className="tl-funnel__tag tl-funnel__tag--muted">—</span>
        )}
      </span>
    </button>
  );
}

export function StrategyFunnel({ funnel }: { funnel: PipelineFunnel }) {
  return (
    <div className="tl-funnel" role="list" aria-label="Strategy funnel">
      {funnel.map((stage, i) => (
        <div className="tl-funnel__item" role="listitem" key={stage.stage}>
          <StageCell stage={stage} />
          {i < funnel.length - 1 && (
            <span className="tl-funnel__arrow" aria-hidden>
              ›
            </span>
          )}
        </div>
      ))}
    </div>
  );
}

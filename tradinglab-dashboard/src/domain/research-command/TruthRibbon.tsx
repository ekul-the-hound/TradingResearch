import type { SystemStatus } from '../../models/system';
import type { PipelineRun } from '../../models/pipeline';
import type { ReturnsProvenance } from '../../models/truth';
import { TruthLabel } from '../../primitives';

const PROV_LABEL: Record<ReturnsProvenance, string> = {
  REAL: 'REAL',
  UNVERIFIED: 'UNVERIFIED',
  SYNTHETIC_RISK: 'SYNTHETIC RISK',
  UNAVAILABLE: 'UNAVAILABLE',
};

// A compact, non-noisy ribbon: data / result type / returns / holdout / cost /
// broker. Reads from system status, overlaid with the latest run's provenance.
export function TruthRibbon({
  status,
  run,
}: {
  status: SystemStatus;
  run: PipelineRun | null;
}) {
  const prov = run?.returnsProvenance ?? 'UNAVAILABLE';
  return (
    <div className="tl-ribbon" role="note" aria-label="Truth ribbon">
      <TruthLabel label="Data" value={status.marketMode} />
      <TruthLabel label="Result" value="BACKTEST / RESEARCH" tone="info" />
      <TruthLabel
        label="Returns"
        value={PROV_LABEL[prov]}
        tone={
          prov === 'REAL' ? 'pass' : prov === 'SYNTHETIC_RISK' ? 'fail' : 'warn'
        }
      />
      <TruthLabel
        label="Holdout"
        value={status.holdout}
        tone={status.holdout === 'SEALED' ? 'pass' : 'neutral'}
      />
      <TruthLabel
        label="Cost"
        value={status.costProfile ? status.costProfile.name : 'UNKNOWN'}
      />
      <TruthLabel
        label="Broker"
        value={status.broker === 'NOT_CONFIGURED' ? 'NOT CONFIGURED' : status.broker}
        tone="warn"
      />
    </div>
  );
}

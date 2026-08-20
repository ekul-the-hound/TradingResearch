import { useState } from 'react';
import { Link } from 'react-router-dom';
import type { StrategyDetail } from '../../models/detail';
import type { ReturnsProvenance } from '../../models/truth';
import { TruthLabel } from '../../primitives';
import { shortFingerprint, ts } from '../../lib/format';

const PROV_TONE: Record<ReturnsProvenance, 'pass' | 'warn' | 'fail' | 'neutral'> = {
  REAL: 'pass',
  UNVERIFIED: 'warn',
  SYNTHETIC_RISK: 'fail',
  UNAVAILABLE: 'neutral',
};
const PROV_LABEL: Record<ReturnsProvenance, string> = {
  REAL: 'REAL',
  UNVERIFIED: 'UNVERIFIED',
  SYNTHETIC_RISK: 'SYNTHETIC RISK',
  UNAVAILABLE: 'UNAVAILABLE',
};

export function DetailHeader({ detail }: { detail: StrategyDetail }) {
  const s = detail.summary;
  const p = detail.provenance;
  const [copied, setCopied] = useState(false);

  return (
    <div className="tl-dh">
      <div className="tl-dh__top">
        <div>
          <h1 className="tl-page-title" style={{ margin: 0 }}>
            {s.name}{' '}
            <span className="mono" style={{ fontSize: 13, color: 'var(--c-text-muted)' }}>
              {s.version}
            </span>
          </h1>
          <div className="tl-dh__id">
            <span className="mono">{s.strategyId}</span>
            <button
              className="tl-dh__copy"
              onClick={() => {
                navigator.clipboard?.writeText(s.strategyId);
                setCopied(true);
                setTimeout(() => setCopied(false), 1200);
              }}
            >
              {copied ? 'copied ✓' : 'copy'}
            </button>
          </div>
        </div>
        <div className="tl-dh__meta">
          <TruthLabel label="Symbol" value={`${s.symbol} · ${s.timeframe}`} />
          <TruthLabel label="Stage" value={s.stage.replace(/_/g, ' ')} tone="info" />
          <TruthLabel label="Origin" value={s.origin} />
        </div>
      </div>

      <div className="tl-dh__truth">
        <TruthLabel
          label="Returns"
          value={PROV_LABEL[p.returnsStatus]}
          tone={PROV_TONE[p.returnsStatus]}
        />
        <TruthLabel
          label="Holdout"
          value={s.holdout}
          tone={s.holdout === 'SEALED' ? 'pass' : 'neutral'}
        />
        <TruthLabel
          label="Fingerprint"
          value={p.datasetFingerprint ? shortFingerprint(p.datasetFingerprint) : 'NONE'}
        />
        <TruthLabel
          label="Window"
          value={p.dateRange ? `${p.dateRange.first} → ${p.dateRange.last}` : 'UNKNOWN'}
        />
        <TruthLabel
          label="Cost"
          value={p.costProfile ? p.costProfile.name : 'UNKNOWN'}
        />
        <TruthLabel label="Last eval" value={s.lastRunAt ? ts(s.lastRunAt) : 'UNKNOWN'} />
        {p.parentIds.length > 0 && (
          <span className="tl-truth">
            <span className="tl-truth__key">PARENT</span>
            <Link className="mono" to={`/strategies/${p.parentIds[0]}`}>
              {p.parentIds[0]}
            </Link>
          </span>
        )}
      </div>
    </div>
  );
}

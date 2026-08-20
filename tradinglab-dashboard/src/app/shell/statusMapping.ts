import type {
  BrokerState,
  HoldoutState,
  MarketMode,
  OperatingMode,
} from '../../models/truth';
import type { CostProfile, SystemStatus } from '../../models/system';
import { shortFingerprint, ts } from '../../lib/format';

export type Tone = 'neutral' | 'pass' | 'fail' | 'warn' | 'info' | 'research';

export interface StatusSegment {
  key: string;
  label: string;
  value: string;
  tone: Tone;
  title?: string; // hover detail
  copyable?: string; // full value to copy, if different from display
}

const MODE_TONE: Record<OperatingMode, Tone> = {
  RESEARCH: 'info',
  BACKTEST: 'info',
  PAPER: 'research',
  DEMO: 'research',
  LIVE: 'pass',
};

const MARKET_LABEL: Record<MarketMode, string> = {
  HISTORICAL: 'HISTORICAL',
  DELAYED: 'DELAYED',
  'REAL-TIME': 'REAL-TIME',
  UNKNOWN: 'UNKNOWN',
};
const MARKET_TONE: Record<MarketMode, Tone> = {
  HISTORICAL: 'neutral',
  DELAYED: 'warn',
  'REAL-TIME': 'pass',
  UNKNOWN: 'neutral',
};

const HOLDOUT_TONE: Record<HoldoutState, Tone> = {
  SEALED: 'pass',
  UNSEALED: 'fail',
  UNKNOWN: 'neutral',
};

// Broker never defaults to a green "connected" look (CLAUDE.md §2/§3).
const BROKER_LABEL: Record<BrokerState, string> = {
  OFFLINE: 'OFFLINE',
  NOT_CONFIGURED: 'NOT CONFIGURED',
  CONNECTED: 'CONNECTED',
};
const BROKER_TONE: Record<BrokerState, Tone> = {
  OFFLINE: 'warn',
  NOT_CONFIGURED: 'warn',
  CONNECTED: 'pass',
};

function costLabel(cost: CostProfile | null): { value: string; title?: string } {
  if (!cost) return { value: 'UNKNOWN' };
  const swaps =
    cost.swaps === 'INTRADAY_NONE'
      ? 'intraday/no swaps'
      : cost.swaps === 'MODELED'
        ? 'swaps modeled'
        : 'swaps UNKNOWN';
  return {
    value: `${cost.name} · ${cost.spreadPips}p / ${cost.slippagePips}p`,
    title: `${cost.name}: ${cost.spreadPips}-pip spread, ${cost.slippagePips}-pip slippage, ${swaps}`,
  };
}

// Convert a SystemStatus into ordered, left-to-right display segments. This is
// the whole truth-mapping surface — kept pure so it can be tested directly.
export function toSegments(status: SystemStatus): StatusSegment[] {
  const cost = costLabel(status.costProfile);
  return [
    {
      key: 'mode',
      label: 'Mode',
      value: status.operatingMode,
      tone: MODE_TONE[status.operatingMode],
    },
    {
      key: 'market',
      label: 'Data',
      value: MARKET_LABEL[status.marketMode],
      tone: MARKET_TONE[status.marketMode],
    },
    {
      key: 'source',
      label: 'Source',
      value: status.dataSource ?? 'UNKNOWN',
      tone: 'neutral',
    },
    {
      key: 'lastrun',
      label: 'Last run',
      value: status.lastRunAt ? ts(status.lastRunAt) : 'UNKNOWN',
      tone: 'neutral',
      title: status.lastRunAt
        ? 'Actual timestamp of the most recent successful pipeline run.'
        : 'No successful run timestamp is available.',
    },
    {
      key: 'fingerprint',
      label: 'Dataset',
      value: status.datasetFingerprint
        ? shortFingerprint(status.datasetFingerprint)
        : 'NONE',
      tone: 'neutral',
      copyable: status.datasetFingerprint ?? undefined,
      title: status.datasetFingerprint
        ? 'Click to copy full dataset fingerprint.'
        : 'No dataset fingerprint recorded.',
    },
    {
      key: 'holdout',
      label: 'Holdout',
      value: status.holdout,
      tone: HOLDOUT_TONE[status.holdout],
    },
    {
      key: 'cost',
      label: 'Cost',
      value: cost.value,
      tone: status.costProfile ? 'neutral' : 'neutral',
      title: cost.title,
    },
    {
      key: 'broker',
      label: 'Broker',
      value: BROKER_LABEL[status.broker],
      tone: BROKER_TONE[status.broker],
    },
    {
      key: 'warnings',
      label: 'Integrity',
      value:
        status.integrityWarningCount > 0
          ? `${status.integrityWarningCount} warning${status.integrityWarningCount === 1 ? '' : 's'}`
          : 'clear',
      tone: status.integrityWarningCount > 0 ? 'warn' : 'pass',
    },
  ];
}

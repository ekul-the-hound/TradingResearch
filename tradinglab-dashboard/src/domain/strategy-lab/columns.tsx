import type { Column } from '../../primitives';
import type { StrategySummary } from '../../models/strategy';
import { int, pct, ratio, ts } from '../../lib/format';
import { numCell, numSort, statusCell } from './cells';
import { EvidenceStack } from './EvidenceStack';

const STAGE_SHORT: Record<string, string> = {
  DISCOVERED: 'Discovered',
  CODE_VALID: 'Code Valid',
  BACKTESTED: 'Backtested',
  COST_ADJUSTED: 'Cost Adj.',
  VALIDATED: 'Validated',
  PORTFOLIO_CANDIDATE: 'Portfolio',
  PAPER: 'Paper',
  LIVE: 'Live',
  REJECTED: 'Rejected',
  RETIRED: 'Retired',
};

const PROV_STATUS = {
  REAL: 'PASS',
  UNVERIFIED: 'WARNING',
  SYNTHETIC_RISK: 'FAIL',
  UNAVAILABLE: 'UNKNOWN',
} as const;

export function buildColumns(): Column<StrategySummary>[] {
  return [
    {
      key: 'name',
      header: 'Strategy',
      sortValue: (r) => r.name,
      render: (r) => (
        <span>
          <span style={{ fontWeight: 600 }}>{r.name}</span>{' '}
          <span className="mono" style={{ color: 'var(--c-text-muted)', fontSize: 10 }}>
            {r.version}
          </span>
        </span>
      ),
    },
    { key: 'symbol', header: 'Symbol', sortValue: (r) => r.symbol, render: (r) => r.symbol },
    { key: 'tf', header: 'TF', sortValue: (r) => r.timeframe, render: (r) => r.timeframe },
    {
      key: 'stage',
      header: 'Stage',
      sortValue: (r) => r.stage,
      render: (r) => STAGE_SHORT[r.stage] ?? r.stage,
    },
    {
      key: 'evidence',
      header: 'Evidence',
      render: (r) => <EvidenceStack evidence={r.evidence} dense />,
    },
    {
      key: 'sharpe',
      header: 'Net Sharpe',
      title: 'Sharpe ratio, net of pessimistic cost profile (2p/1p, intraday).',
      numeric: true,
      sortValue: (r) => numSort(r.netSharpe),
      render: (r) => numCell(r.netSharpe, (v) => ratio(v)),
    },
    {
      key: 'ret',
      header: 'Net Ret %',
      title: 'Net total return over the test window.',
      numeric: true,
      sortValue: (r) => numSort(r.netReturnPct),
      render: (r) => numCell(r.netReturnPct, (v) => pct(v, 1)),
    },
    {
      key: 'dd',
      header: 'Max DD %',
      title: 'Maximum drawdown. Manual gate rejects > 30%.',
      numeric: true,
      sortValue: (r) => numSort(r.maxDrawdownPct, false),
      render: (r) => numCell(r.maxDrawdownPct, (v) => pct(v, 1)),
    },
    {
      key: 'trades',
      header: 'Trades',
      title: 'Trade count (sample size). Manual gate requires ≥ 20.',
      numeric: true,
      sortValue: (r) => numSort(r.tradeCount),
      render: (r) => numCell(r.tradeCount, (v) => int(v)),
    },
    {
      key: 'pbo',
      header: 'PBO',
      title: 'Probability of Backtest Overfitting (CSCV). > 0.5 is overfit.',
      numeric: true,
      sortValue: (r) => numSort(r.pbo, false),
      render: (r) => numCell(r.pbo, (v) => v.toFixed(2)),
    },
    {
      key: 'dsr',
      header: 'DSR',
      title: 'Deflated Sharpe Ratio (multiple-testing adjusted).',
      numeric: true,
      sortValue: (r) => numSort(r.deflatedSharpe),
      render: (r) => numCell(r.deflatedSharpe, (v) => v.toFixed(2)),
    },
    {
      key: 'robust',
      header: 'Robust',
      title: 'Robustness score from perturbation tests.',
      numeric: true,
      sortValue: (r) => numSort(r.robustness),
      render: (r) => numCell(r.robustness, (v) => v.toFixed(2)),
    },
    {
      key: 'pstab',
      header: 'Param Stab',
      title: 'Parameter stability across neighboring parameter sets.',
      numeric: true,
      sortValue: (r) => numSort(r.parameterStability),
      render: (r) => numCell(r.parameterStability, (v) => v.toFixed(2)),
    },
    {
      key: 'div',
      header: 'Diversif.',
      title: 'Diversification / correlation signal vs. existing pool.',
      numeric: true,
      sortValue: (r) => numSort(r.diversificationSignal),
      render: (r) => numCell(r.diversificationSignal, (v) => v.toFixed(2)),
    },
    {
      key: 'ftmo',
      header: 'FTMO Fit',
      title:
        'FTMO fit. PROXY until wired to FTMOComplianceChecker against trades.',
      render: (r) => statusCell(r.ftmoFit),
    },
    {
      key: 'prov',
      header: 'Returns',
      title: 'Returns provenance from canonical_result.',
      render: (r) => (
        <span
          className={`tl-provtag tl-provtag--${PROV_STATUS[r.returnsProvenance].toLowerCase()}`}
          title={r.returnsProvenance}
        >
          {r.returnsProvenance === 'SYNTHETIC_RISK'
            ? 'SYNTH RISK'
            : r.returnsProvenance}
        </span>
      ),
    },
    {
      key: 'holdout',
      header: 'Holdout',
      sortValue: (r) => r.holdout,
      render: (r) => r.holdout,
    },
    {
      key: 'lastrun',
      header: 'Last Run',
      title: 'Actual timestamp of last evaluation.',
      sortValue: (r) => r.lastRunAt ?? '',
      render: (r) => (r.lastRunAt ? ts(r.lastRunAt) : 'never'),
    },
  ];
}

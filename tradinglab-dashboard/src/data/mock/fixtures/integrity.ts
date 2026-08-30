import type { IntegrityStatus } from '../../../models/integrity';
import { unknown, val } from '../../../models/truth';

// DEVELOPMENT FIXTURE. Not from any database. Dependency health defaults to
// NOT_CHECKED — the UI must never fabricate a green OK for a service it did not
// actually probe (CLAUDE.md §2, AUDIT §9).

export const FIXTURE_INTEGRITY: IntegrityStatus = {
  warningCount: 2,
  overallStatus: 'WARNING',
  datasets: [
    {
      fingerprint: '8f2a1c9d4b7e3a10',
      source: 'HistData',
      symbol: 'EUR/USD',
      timeframe: '1H',
      dateRange: { first: '2019-01-01', last: '2024-06-30' },
      barCount: 34_320,
      timezoneVerified: val(true),
      usedByResultCount: 46,
    },
    {
      fingerprint: 'b7c1e2a9f3d40851',
      source: 'HistData',
      symbol: 'GBP/USD',
      timeframe: '4H',
      dateRange: { first: '2019-01-01', last: '2024-06-30' },
      barCount: 8_580,
      timezoneVerified: val(true),
      usedByResultCount: 31,
    },
    {
      fingerprint: null,
      source: 'HistData',
      symbol: 'GBP/USD',
      timeframe: '4H',
      dateRange: { first: '2019-01-01', last: '2024-06-30' },
      barCount: null,
      timezoneVerified: unknown('Not persisted per result.'),
      usedByResultCount: 15,
    },
  ],
  provenanceCoverage: {
    totalResults: 92,
    withFingerprint: 77,
    missingFingerprint: 15,
    withCodeFingerprint: 77,
    missingFingerprintResultIds: ['backtest_results#12', 'backtest_results#33'],
  },
  holdout: {
    fraction: 0.2,
    state: 'SEALED',
    cutoffDate: '2023-05-01',
    sealedResultCount: val(92),
    unsealedResultCount: val(0),
  },
  returnsLedger: {
    real: 90,
    unverified: 0,
    syntheticRisk: 1,
    unavailable: 1,
    syntheticRiskResultIds: ['GBPUSD_H4_Break_v03'],
    allowSyntheticFlag: false,
  },
  dependencies: [
    {
      name: 'SearXNG',
      state: 'NOT_CHECKED',
      detail: 'Docker service — not probed from the dashboard.',
      lastCheckedAt: null,
    },
    {
      name: 'Ollama',
      state: 'NOT_CHECKED',
      detail: 'Local + cloud inference — not probed.',
      lastCheckedAt: null,
    },
    {
      name: 'DataPath',
      state: 'NOT_CHECKED',
      detail: 'E:\\TradingData — existence not verified from here.',
      lastCheckedAt: null,
    },
    {
      name: 'SQLite',
      state: 'NOT_CHECKED',
      detail: 'results/backtest_results.db and related — not opened read-only yet.',
      lastCheckedAt: null,
    },
  ],
  configFreeze: {
    frozen: true,
    hash: val('cfg_5a1b2c3d'),
    costProfileName: 'Pessimistic Manual',
    holdoutFraction: 0.2,
    gateSummary: 'Sharpe≥0.5, Trades≥20, DD≤30%',
    driftDetected: val(false),
    keys: [
      { key: 'DEFAULT_HOLDOUT_FRACTION', value: '0.20' },
      { key: 'COST_SPREAD_PIPS', value: '2' },
      { key: 'COST_SLIPPAGE_PIPS', value: '1' },
      { key: 'SWAPS', value: 'intraday / none' },
      { key: 'MIN_SHARPE', value: '0.5' },
      { key: 'MIN_TRADES', value: '20' },
      { key: 'MAX_DRAWDOWN_PCT', value: '30' },
      { key: 'ALLOW_SYNTHETIC_RETURNS', value: 'False' },
    ],
  },
};

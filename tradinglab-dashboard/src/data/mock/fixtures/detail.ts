import type {
  EquityPoint,
  StrategyDetail,
  TradeRow,
} from '../../../models/detail';
import type {
  ChallengeSimulationResult,
  FTMOComplianceStatus,
} from '../../../models/ftmo';
import { proxy, unavailable, unknown, val } from '../../../models/truth';
import { FIXTURE_STRATEGIES } from './strategies';
import { deriveVerdict, gate, GATE_THRESHOLDS } from '../../../domain/strategy-detail/verdict';

// DEVELOPMENT FIXTURE. Not from any database.

function equity(seed: number, n = 66, drift = 0.12): EquityPoint[] {
  const out: EquityPoint[] = [];
  let net = 0;
  let gross = 0;
  let x = seed;
  const start = new Date('2019-01-01').getTime();
  for (let i = 0; i < n; i++) {
    x = (x * 9301 + 49297) % 233280;
    const r = x / 233280 - 0.5;
    net += drift + r * 2.4;
    gross += drift + 0.35 + r * 2.4;
    out.push({
      t: new Date(start + i * 30 * 864e5).toISOString().slice(0, 10),
      gross: Number(gross.toFixed(2)),
      net: Number(net.toFixed(2)),
    });
  }
  return out;
}

const TRADES_A: TradeRow[] = Array.from({ length: 8 }).map((_, i) => ({
  entryDate: `2023-0${(i % 9) + 1}-05`,
  exitDate: `2023-0${(i % 9) + 1}-06`,
  isLong: i % 2 === 0,
  entryPrice: 1.085 + i * 0.001,
  exitPrice: 1.088 + i * 0.001,
  pnl: (i % 3 === 0 ? -1 : 1) * (40 + i * 6),
  returnPct: (i % 3 === 0 ? -1 : 1) * (0.3 + i * 0.05),
  durationBars: 6 + i,
}));

function buildDetailA(): StrategyDetail {
  const summary = FIXTURE_STRATEGIES.find(
    (s) => s.strategyId === 'EURUSD_H1_MeanRev_v07',
  )!;
  const gates = {
    sharpe: gate(0.62, GATE_THRESHOLDS.minSharpe, '>='),
    trades: gate(41, GATE_THRESHOLDS.minTrades, '>='),
    maxDrawdown: gate(18.4, GATE_THRESHOLDS.maxDrawdownPct, '<='),
  };
  const v = deriveVerdict(gates, 'value', 0.34);
  return {
    summary,
    performance: {
      equity: val(equity(101)),
      hasGross: true,
      headline: {
        netReturnPct: val(7.4),
        sharpe: val(0.62),
        maxDrawdownPct: val(18.4),
        tradeCount: val(41),
        winRatePct: val(58.5),
        profitFactor: val(1.32),
        expectancy: val(0.18),
      },
      period: { first: '2019-01-01', last: '2024-06-30' },
      trades: val(TRADES_A),
      returnsProvenance: 'REAL',
    },
    validation: {
      manualGates: gates,
      holdout: { fraction: 0.2, state: 'SEALED' },
      pbo: val(0.34),
      deflatedSharpe: val(0.21),
      cscv: val(0.34),
      permutationPValue: val(0.04),
      walkForward: unknown('Walk-forward not run.'),
      lookaheadScan: val('PASS'),
      prohibitedPatternScan: val('PASS'),
      parameterStability: unknown('No sweep data.'),
      verdict: v.verdict,
      verdictReason: v.reason,
    },
    risk: {
      var95Pct: val(-2.1),
      cvar95Pct: val(-3.4),
      tailRatio: val(0.92),
      skew: val(-0.31),
      kurtosis: val(4.2),
      maeMfe: unknown('Intrabar data not persisted for this result.'),
      stopTargetInSample: val({ note: 'ATR stop; in-sample until validated.' }),
      capacity: unavailable('Insufficient liquidity/impact data.'),
      drawdownSeries: val(
        equity(7).map((p) => ({ t: p.t, ddPct: Math.min(0, p.net - 8) })),
      ),
    },
    provenance: {
      datasetFingerprint: '8f2a1c9d4b7e3a10',
      dataSource: 'HistData',
      symbol: 'EUR/USD',
      timeframe: '1H',
      dateRange: { first: '2019-01-01', last: '2024-06-30' },
      barCount: 34_320,
      timezoneVerified: val(true),
      codeFingerprint: 'c0de1234abcd5678',
      parentIds: ['EURUSD_H1_MeanRev_v06'],
      childIds: [],
      runConfig: { holdout: '0.20', cost: 'Pessimistic Manual' },
      costProfile: { name: 'Pessimistic Manual', spreadPips: 2, slippagePips: 1 },
      resultTables: ['backtest_results#842', 'backtest_trades'],
      returnsStatus: 'REAL',
      integrityWarnings: [],
    },
  };
}

function buildDetailB(): StrategyDetail {
  const summary = FIXTURE_STRATEGIES.find(
    (s) => s.strategyId === 'GBPUSD_H4_Break_v03',
  )!;
  const gates = {
    sharpe: gate(-0.11, GATE_THRESHOLDS.minSharpe, '>='),
    trades: gate(12, GATE_THRESHOLDS.minTrades, '>='),
    maxDrawdown: gate(33.2, GATE_THRESHOLDS.maxDrawdownPct, '<='),
  };
  const v = deriveVerdict(gates, 'unknown', null);
  return {
    summary,
    performance: {
      equity: unavailable('Returns are synthetic-risk; equity curve withheld.'),
      hasGross: false,
      headline: {
        netReturnPct: val(-3.2),
        sharpe: val(-0.11),
        maxDrawdownPct: val(33.2),
        tradeCount: val(12),
        winRatePct: val(41.7),
        profitFactor: val(0.88),
        expectancy: val(-0.14),
      },
      period: { first: '2019-01-01', last: '2024-06-30' },
      trades: unknown('Trade list not persisted for this result.'),
      returnsProvenance: 'SYNTHETIC_RISK',
    },
    validation: {
      manualGates: gates,
      holdout: { fraction: 0.2, state: 'SEALED' },
      pbo: unknown('PBO not run.'),
      deflatedSharpe: unknown('DSR not run.'),
      cscv: unknown('CSCV not run.'),
      permutationPValue: unknown('Not run.'),
      walkForward: unknown('Not run.'),
      lookaheadScan: val('PASS'),
      prohibitedPatternScan: val('PASS'),
      parameterStability: unknown('No sweep data.'),
      verdict: v.verdict,
      verdictReason: v.reason,
    },
    risk: {
      var95Pct: unavailable('Real return series required.'),
      cvar95Pct: unavailable('Real return series required.'),
      tailRatio: unavailable('Real return series required.'),
      skew: unavailable('Real return series required.'),
      kurtosis: unavailable('Real return series required.'),
      maeMfe: unavailable('No intrabar data.'),
      stopTargetInSample: val({ note: 'Fixed target; in-sample.' }),
      capacity: unavailable('Insufficient data.'),
      drawdownSeries: unavailable('Real return series required.'),
    },
    provenance: {
      datasetFingerprint: null,
      dataSource: 'HistData',
      symbol: 'GBP/USD',
      timeframe: '4H',
      dateRange: { first: '2019-01-01', last: '2024-06-30' },
      barCount: null,
      timezoneVerified: unknown('Not persisted per result.'),
      codeFingerprint: null,
      parentIds: ['GBPUSD_H4_Break_v02'],
      childIds: [],
      runConfig: null,
      costProfile: { name: 'Pessimistic Manual', spreadPips: 2, slippagePips: 1 },
      resultTables: ['backtest_results#803'],
      returnsStatus: 'SYNTHETIC_RISK',
      integrityWarnings: [
        'No dataset fingerprint recorded (predates provenance tracking).',
        'Returns not verified via require_returns().',
      ],
    },
  };
}

export const FIXTURE_DETAILS: Record<string, StrategyDetail> = {
  EURUSD_H1_MeanRev_v07: buildDetailA(),
  GBPUSD_H4_Break_v03: buildDetailB(),
};

// Challenge fit — always PROXY / UNKNOWN in the current build (not wired to
// FTMOComplianceChecker; consistency threshold unset).
export function ftmoFor(strategyId: string): FTMOComplianceStatus {
  const isSynthetic = strategyId === 'GBPUSD_H4_Break_v03';
  return {
    basis: 'PROXY',
    isFullyModelled: false,
    modelledRules: ['max_daily_loss', 'max_total_drawdown', 'min_trading_days'],
    unmodelledRules: ['consistency_rule'],
    limits: {
      maxDailyLossPct: 0.05,
      maxTotalDrawdownPct: 0.1,
      profitTargetPct: val(10),
      minTradingDays: 4,
    },
    breachReasons: isSynthetic
      ? ['Total drawdown 33.2% exceeds 10% limit (proxy).']
      : [],
    drawdownHeadroom: proxy(
      isSynthetic ? -23.2 : 1.6,
      'Derived from summary drawdown, not FTMOComplianceChecker.',
    ),
    targetProgress: proxy(isSynthetic ? -3.2 : 7.4, 'Proxy from summary return.'),
    daysTraded: unknown('Per-day data not available from summary result.'),
    consistency: { evaluated: false, reason: 'NO_THRESHOLD_CONFIGURED' },
  };
}

export function challengeSimFor(strategyId: string): ChallengeSimulationResult {
  // No real simulation persisted per strategy in the current build.
  void strategyId;
  return {
    available: false,
    nSimulations: 0,
    pPass: unavailable('No challenge simulation has been run for this strategy.'),
    p95WorstDayPct: unavailable('No simulation output.'),
    drawdownDistribution: null,
    breachReasonDistribution: null,
    baselineVsStress: null,
    inputAssumptions: 'Run challenge_simulator to populate (path count, fee basis).',
  };
}

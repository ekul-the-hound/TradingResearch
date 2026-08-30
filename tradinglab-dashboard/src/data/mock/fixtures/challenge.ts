import type { ChallengeReadiness } from '../../../models/challenge';
import { proxy, unknown } from '../../../models/truth';

// DEVELOPMENT FIXTURE. Not from any database. Reflects the real modeling state:
// FTMO confirmed; daily-loss / total-drawdown / min-days modeled; consistency
// rule NOT modeled (threshold unset); everything surfaced as PROXY until wired
// to FTMOComplianceChecker against per-trade data.

export const FIXTURE_CHALLENGE_READINESS: ChallengeReadiness = {
  firm: {
    firm: 'FTMO',
    accountSize: 100_000,
    phase: 'CHALLENGE',
    profitTargetPct: 10,
    maxDailyLossPct: 5,
    maxTotalDrawdownPct: 10,
    minTradingDays: 4,
    consistencyThresholdPct: null, // not configured
  },
  ruleModel: [
    {
      rule: 'Profit target (10%)',
      modeled: true,
      basis: 'PROXY',
      note: 'Compared against summary return, not per-day equity.',
    },
    {
      rule: 'Max daily loss (5%)',
      modeled: true,
      basis: 'PROXY',
      note: 'Requires per-day equity; currently proxied from summary.',
    },
    {
      rule: 'Max total drawdown (10%)',
      modeled: true,
      basis: 'PROXY',
      note: 'Proxied from summary max drawdown.',
    },
    {
      rule: 'Min trading days (4)',
      modeled: true,
      basis: 'INCOMPLETE',
      note: 'Per-day trade counts not available from summary results.',
    },
    {
      rule: 'Consistency rule',
      modeled: false,
      basis: 'NOT_MODELED',
      note: 'Threshold not configured — cannot evaluate best-day share.',
    },
  ],
  perStrategy: [
    {
      strategyId: 'EURUSD_H1_MeanRev_v07',
      name: 'EURUSD_H1_MeanRev',
      ftmoFit: proxy('WARNING', 'Proxy: passes DD proxy but gates marginal.'),
      basis: 'PROXY',
      note: 'Not evaluated by FTMOComplianceChecker (needs trade list).',
    },
    {
      strategyId: 'GBPUSD_H4_Break_v03',
      name: 'GBPUSD_H4_Break',
      ftmoFit: proxy('FAIL', 'Proxy: 33.2% drawdown exceeds 10% limit.'),
      basis: 'PROXY',
      note: 'Also flagged synthetic-risk returns.',
    },
    {
      strategyId: 'EURUSD_H4_ADX_v01',
      name: 'EURUSD_H4_ADX',
      ftmoFit: unknown('Not yet backtested.'),
      basis: 'INCOMPLETE',
      note: 'No result to assess.',
    },
  ],
  consistency: { evaluated: false, reason: 'NO_THRESHOLD_CONFIGURED' },
  overallBasis: 'PROXY',
  readinessStatus: 'PROXY',
};

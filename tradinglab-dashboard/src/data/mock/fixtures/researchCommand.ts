import type { PipelineFunnel, PipelineRun } from '../../../models/pipeline';
import type { ResearchHealth, ResearchQueueItem } from '../../../models/queue';
import { val, unknown } from '../../../models/truth';

// DEVELOPMENT FIXTURE. Values mirror the *shape* of real backend state
// (Phase-1 style: 92 backtests, zero survivors, generic indicators) so the UI
// can be exercised honestly. Not sourced from any database.

export const FIXTURE_LATEST_RUN: PipelineRun = {
  runId: 'run_0842',
  startedAt: '2026-01-14T19:20:00Z',
  completedAt: '2026-01-14T19:42:00Z',
  durationSec: 1320,
  inputCount: val(13),
  survivorCount: val(0),
  topCandidateId: 'EURUSD_H1_MeanRev_v07',
  primaryBlocker: 'All candidates failed manual gates (min Sharpe 0.5).',
  dataFingerprint: '8f2a1c9d4b7e3a10',
  holdout: 'SEALED',
  costProfileName: 'Pessimistic Manual',
  returnsProvenance: 'REAL',
  status: 'SUCCESS',
};

export const FIXTURE_FUNNEL: PipelineFunnel = [
  {
    stage: 'DISCOVERED',
    count: val(21),
    blocked: val(0),
    rejected: val(0),
    definition: 'Ideas extracted or entered, before code validation.',
  },
  {
    stage: 'CODE_VALID',
    count: val(12),
    blocked: val(0),
    rejected: val(9),
    definition: 'Parses and passes static/prohibited-pattern checks.',
  },
  {
    stage: 'BACKTESTED',
    count: val(14),
    blocked: val(0),
    rejected: val(0),
    definition: 'Has at least one historical backtest result.',
  },
  {
    stage: 'COST_ADJUSTED',
    count: val(3),
    blocked: val(1),
    rejected: val(0),
    definition: 'Re-scored under the pessimistic cost profile.',
  },
  {
    stage: 'VALIDATED',
    count: val(0),
    blocked: val(0),
    rejected: val(14),
    definition: 'Passed manual gates + holdout/statistical validation.',
  },
  {
    stage: 'PORTFOLIO_CANDIDATE',
    count: val(0),
    blocked: val(0),
    rejected: val(0),
    definition: 'Eligible for portfolio construction.',
  },
  {
    stage: 'PAPER',
    count: val(0),
    blocked: unknown('Paper stage not yet wired.'),
    rejected: val(0),
    definition: 'Running in paper/simulated mode.',
  },
  {
    stage: 'LIVE',
    count: val(0),
    blocked: val(0),
    rejected: val(0),
    definition: 'Connected live — zero until a broker bridge exists.',
  },
];

export const FIXTURE_QUEUE: ResearchQueueItem[] = [
  {
    id: 'q1',
    severity: 'CRITICAL',
    title: 'Result has synthetic-risk returns',
    reason:
      'GBPUSD_H4_Break_v03 exposes returns not verified via require_returns().',
    affects: { kind: 'strategy', id: 'GBPUSD_H4_Break_v03' },
    sourceLabel: 'canonical_result',
    suggestedAction: 'Re-run with real trade returns or quarantine the result.',
    destination: '/strategies/GBPUSD_H4_Break_v03#provenance',
  },
  {
    id: 'q2',
    severity: 'CRITICAL',
    title: 'Compliance uses a display proxy',
    reason:
      'FTMO badge derived from summary drawdown, not FTMOComplianceChecker.',
    affects: { kind: 'strategy', id: 'EURUSD_H1_MeanRev_v07' },
    sourceLabel: 'dashboard_ftmo_panel',
    suggestedAction: 'Wire to FTMOComplianceChecker against backtest_trades.',
    destination: '/strategies/EURUSD_H1_MeanRev_v07#challenge',
  },
  {
    id: 'q3',
    severity: 'MEDIUM',
    title: 'Consistency threshold is unset',
    reason:
      'Consistency-rule headroom cannot be computed without a configured threshold.',
    affects: null,
    sourceLabel: 'consistency_rule',
    suggestedAction: 'Configure the consistency threshold in firm rules.',
    destination: '/challenge-readiness',
  },
  {
    id: 'q4',
    severity: 'MEDIUM',
    title: 'Strategy quality requires human review',
    reason:
      'Discovery pool averages 2.6/5 — generic textbook indicators dominate.',
    affects: null,
    sourceLabel: 'quality_scorer',
    suggestedAction: 'Review the discovery inbox and prune low-quality ideas.',
    destination: '/discovery-inbox',
  },
];

export const FIXTURE_HEALTH: ResearchHealth = {
  modules: [
    {
      key: 'data-coverage',
      title: 'Data Coverage',
      primary: val({ value: '2', label: 'symbols · 1H, 4H' }),
      secondary: { label: 'Window', value: '2019-01 → 2024-06' },
      buckets: [
        { label: 'EUR/USD', value: 100, tone: 'pass' },
        { label: 'GBP/USD', value: 100, tone: 'pass' },
      ],
      sourceLabel: 'HistData',
      status: 'PASS',
    },
    {
      key: 'discovery-quality',
      title: 'Discovery Quality',
      primary: val({ value: '2.6 / 5', label: 'mean quality' }),
      secondary: { label: 'Code-valid', value: '12 of 21' },
      buckets: [
        { label: '0–1', value: 2, tone: 'fail' },
        { label: '1–2', value: 6, tone: 'warn' },
        { label: '2–3', value: 9, tone: 'warn' },
        { label: '3–4', value: 3, tone: 'info' },
        { label: '4–5', value: 1, tone: 'pass' },
      ],
      sourceLabel: 'quality_scorer',
      status: 'WARNING',
    },
    {
      key: 'validation-coverage',
      title: 'Validation Coverage',
      primary: val({ value: '0', label: 'validated of 14 backtested' }),
      secondary: { label: 'Holdout', value: 'SEALED (0.20)' },
      buckets: [
        { label: 'PBO run', value: 3, tone: 'info' },
        { label: 'DSR run', value: 3, tone: 'info' },
        { label: 'Pending', value: 11, tone: 'neutral' },
      ],
      sourceLabel: 'overfitting_detector',
      status: 'INCOMPLETE',
    },
    {
      key: 'cost-realism',
      title: 'Cost Realism',
      primary: val({ value: '2p / 1p', label: 'spread / slippage' }),
      secondary: { label: 'Swaps', value: 'intraday / none' },
      buckets: null,
      sourceLabel: 'manual_cost_override',
      status: 'PASS',
    },
    {
      key: 'integrity-ledger',
      title: 'Integrity Ledger',
      primary: val({ value: '2', label: 'open warnings' }),
      secondary: { label: 'Synthetic-risk results', value: '1' },
      buckets: [
        { label: 'Synthetic risk', value: 1, tone: 'fail' },
        { label: 'Missing fingerprint', value: 1, tone: 'warn' },
      ],
      sourceLabel: 'audit_result_provenance',
      status: 'WARNING',
    },
  ],
};

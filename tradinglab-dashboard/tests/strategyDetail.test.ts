import { describe, expect, it } from 'vitest';
import {
  GATE_THRESHOLDS,
  deriveVerdict,
  gate,
} from '../src/domain/strategy-detail/verdict';
import { MockResearchRepository } from '../src/data/mock/MockResearchRepository';

describe('manual gate evaluation', () => {
  it('passes when value meets a >= threshold', () => {
    expect(gate(0.62, GATE_THRESHOLDS.minSharpe, '>=').status).toBe('PASS');
  });
  it('fails a <= threshold when exceeded', () => {
    expect(gate(33.2, GATE_THRESHOLDS.maxDrawdownPct, '<=').status).toBe('FAIL');
  });
  it('is UNKNOWN when value is null', () => {
    expect(gate(null, GATE_THRESHOLDS.minTrades, '>=').status).toBe('UNKNOWN');
  });
});

describe('promotion verdict', () => {
  const passing = {
    sharpe: gate(0.62, 0.5, '>=' as const),
    trades: gate(41, 20, '>=' as const),
    maxDrawdown: gate(18.4, 30, '<=' as const),
  };

  it('rejects on any hard gate failure', () => {
    const gates = {
      sharpe: gate(-0.1, 0.5, '>=' as const),
      trades: gate(12, 20, '>=' as const),
      maxDrawdown: gate(33, 30, '<=' as const),
    };
    expect(deriveVerdict(gates, 'unknown', null).verdict).toBe('REJECT');
  });

  it('is incomplete when a gate is unknown', () => {
    const gates = {
      sharpe: gate(null, 0.5, '>=' as const),
      trades: gate(41, 20, '>=' as const),
      maxDrawdown: gate(18, 30, '<=' as const),
    };
    expect(deriveVerdict(gates, 'value', 0.3).verdict).toBe('INCOMPLETE');
  });

  it('investigates when gates pass but PBO is missing', () => {
    expect(deriveVerdict(passing, 'unknown', null).verdict).toBe('INVESTIGATE');
  });

  it('investigates when PBO > 0.5 despite passing gates', () => {
    expect(deriveVerdict(passing, 'value', 0.71).verdict).toBe('INVESTIGATE');
  });

  it('promotes when gates pass and PBO is acceptable', () => {
    expect(deriveVerdict(passing, 'value', 0.34).verdict).toBe('PROMOTE');
  });

  it('never fabricates a composite score — reason is textual', () => {
    const r = deriveVerdict(passing, 'value', 0.34);
    expect(typeof r.reason).toBe('string');
    expect(r.reason.length).toBeGreaterThan(0);
  });
});

describe('strategy detail repository reads', () => {
  const repo = new MockResearchRepository();

  it('returns empty for an unknown strategy id', async () => {
    const r = await repo.getStrategyDetail('does_not_exist');
    expect(r.state).toBe('empty');
  });

  it('returns detail for a known strategy', async () => {
    const r = await repo.getStrategyDetail('EURUSD_H1_MeanRev_v07');
    expect(r.state).toBe('ready');
    if (r.state === 'ready') {
      expect(r.data.validation.verdict).toBe('PROMOTE');
      expect(r.data.performance.returnsProvenance).toBe('REAL');
    }
  });

  it('FTMO compliance is always PROXY, never authoritative', async () => {
    const r = await repo.getFTMOCompliance('EURUSD_H1_MeanRev_v07');
    if (r.state === 'ready') {
      expect(r.data.basis).toBe('PROXY');
      expect(r.data.isFullyModelled).toBe(false);
      expect(r.data.consistency.evaluated).toBe(false);
    }
  });

  it('challenge simulation is unavailable (none run)', async () => {
    const r = await repo.getChallengeSimulation('EURUSD_H1_MeanRev_v07');
    if (r.state === 'ready') {
      expect(r.data.available).toBe(false);
      expect(r.data.pPass.kind).toBe('unavailable');
    }
  });

  it('synthetic-risk strategy withholds equity + trades', async () => {
    const r = await repo.getStrategyDetail('GBPUSD_H4_Break_v03');
    if (r.state === 'ready') {
      expect(r.data.performance.equity.kind).toBe('unavailable');
      expect(r.data.performance.trades.kind).toBe('unknown');
      expect(r.data.provenance.datasetFingerprint).toBeNull();
    }
  });
});

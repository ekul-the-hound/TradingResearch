import { describe, expect, it } from 'vitest';
import type { PortfolioCandidate } from '../src/models/portfolio';
import {
  blockedComputation,
  evaluateCombineGate,
} from '../src/domain/portfolio-builder/combine';
import { MockResearchRepository } from '../src/data/mock/MockResearchRepository';
import { val } from '../src/models/truth';
import type { StrategyEvidence } from '../src/models/strategy';

const emptyEvidence = {} as StrategyEvidence;

function cand(
  id: string,
  eligible: boolean,
  returnsReal: boolean,
): PortfolioCandidate {
  return {
    strategyId: id,
    name: id,
    symbol: 'EUR/USD',
    timeframe: '1H',
    eligible,
    exclusionReason: eligible ? null : 'excluded',
    netSharpe: val(0.6),
    netReturnPct: val(5),
    maxDrawdownPct: val(15),
    returnsReal,
    evidence: emptyEvidence,
  };
}

describe('portfolio combine gate', () => {
  it('blocks with NO_CANDIDATES when nothing selected', () => {
    expect(evaluateCombineGate([]).status).toBe('NO_CANDIDATES');
  });

  it('blocks a single eligible real candidate (need >= 2)', () => {
    const g = evaluateCombineGate([cand('a', true, true)]);
    expect(g.canCombine).toBe(false);
  });

  it('excludes synthetic-risk candidates from combination', () => {
    const g = evaluateCombineGate([
      cand('a', true, true),
      cand('b', true, false), // synthetic
    ]);
    expect(g.canCombine).toBe(false);
    expect(g.reason).toMatch(/REAL returns/i);
  });

  it('combines two eligible REAL candidates', () => {
    const g = evaluateCombineGate([
      cand('a', true, true),
      cand('b', true, true),
    ]);
    expect(g.canCombine).toBe(true);
    expect(g.usableIds).toEqual(['a', 'b']);
  });

  it('blockedComputation fabricates no numbers', () => {
    const g = evaluateCombineGate([cand('a', true, true)]);
    const comp = blockedComputation(g);
    expect(comp.combinedSharpe.kind).toBe('unavailable');
    expect(comp.combinedMaxDrawdownPct.kind).toBe('unavailable');
    expect(comp.method).toBe('PARETO_SELECTION');
  });
});

describe('portfolio repository', () => {
  const repo = new MockResearchRepository();

  it('exposes candidates with the synthetic-risk one excluded', async () => {
    const r = await repo.getPortfolioCandidates();
    if (r.state === 'ready') {
      const synth = r.data.find((c) => c.strategyId === 'GBPUSD_H4_Break_v03');
      expect(synth?.eligible).toBe(false);
      expect(synth?.returnsReal).toBe(false);
    }
  });

  it('computePortfolio with no ids yields a NO_CANDIDATES blocked result', async () => {
    const r = await repo.computePortfolio([]);
    if (r.state === 'ready') {
      expect(r.data.status).toBe('NO_CANDIDATES');
      expect(r.data.combinedSharpe.kind).toBe('unavailable');
    }
  });

  it('always labels the method Pareto selection, never optimization', async () => {
    const r = await repo.computePortfolio(['EURUSD_H1_MeanRev_v07']);
    if (r.state === 'ready') {
      expect(r.data.method).toBe('PARETO_SELECTION');
    }
  });
});

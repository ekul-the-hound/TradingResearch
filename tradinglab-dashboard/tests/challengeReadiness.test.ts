import { describe, expect, it } from 'vitest';
import { MockResearchRepository } from '../src/data/mock/MockResearchRepository';
import { FIXTURE_CHALLENGE_READINESS } from '../src/data/mock/fixtures/challenge';

describe('challenge readiness', () => {
  const repo = new MockResearchRepository();

  it('returns readiness marked as fixture', async () => {
    const r = await repo.getChallengeReadiness();
    expect(r.state).toBe('ready');
    if (r.state === 'ready') expect(r.isFixture).toBe(true);
  });

  it('overall basis is PROXY and readiness is never an unconditional PASS', () => {
    expect(FIXTURE_CHALLENGE_READINESS.overallBasis).toBe('PROXY');
    expect(FIXTURE_CHALLENGE_READINESS.readinessStatus).not.toBe('PASS');
  });

  it('consistency rule is not evaluated (threshold unset)', () => {
    expect(FIXTURE_CHALLENGE_READINESS.consistency.evaluated).toBe(false);
    expect(FIXTURE_CHALLENGE_READINESS.firm.consistencyThresholdPct).toBeNull();
  });

  it('the consistency rule is not modeled', () => {
    const consistency = FIXTURE_CHALLENGE_READINESS.ruleModel.find((r) =>
      r.rule.toLowerCase().includes('consistency'),
    );
    expect(consistency?.modeled).toBe(false);
    expect(consistency?.basis).toBe('NOT_MODELED');
  });

  it('models the FTMO numeric limits (10/5/10/4)', () => {
    const f = FIXTURE_CHALLENGE_READINESS.firm;
    expect(f.profitTargetPct).toBe(10);
    expect(f.maxDailyLossPct).toBe(5);
    expect(f.maxTotalDrawdownPct).toBe(10);
    expect(f.minTradingDays).toBe(4);
  });

  it('no rule is on an AUTHORITATIVE basis yet (all proxy/incomplete/not-modeled)', () => {
    expect(
      FIXTURE_CHALLENGE_READINESS.ruleModel.every(
        (r) => r.basis !== 'AUTHORITATIVE',
      ),
    ).toBe(true);
  });

  it('every per-strategy fit is proxy or incomplete, never authoritative pass', () => {
    for (const p of FIXTURE_CHALLENGE_READINESS.perStrategy) {
      expect(['PROXY', 'INCOMPLETE']).toContain(p.basis);
    }
  });
});

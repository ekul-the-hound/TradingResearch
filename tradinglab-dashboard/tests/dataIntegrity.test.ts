import { describe, expect, it } from 'vitest';
import { MockResearchRepository } from '../src/data/mock/MockResearchRepository';
import { FIXTURE_INTEGRITY } from '../src/data/mock/fixtures/integrity';

describe('data integrity', () => {
  const repo = new MockResearchRepository();

  it('returns integrity status marked as fixture', async () => {
    const r = await repo.getIntegrityStatus();
    expect(r.state).toBe('ready');
    if (r.state === 'ready') expect(r.isFixture).toBe(true);
  });

  it('all dependencies default to NOT_CHECKED (never fake OK)', () => {
    for (const d of FIXTURE_INTEGRITY.dependencies) {
      expect(d.state).toBe('NOT_CHECKED');
      expect(d.lastCheckedAt).toBeNull();
    }
  });

  it('provenance coverage accounts for NULL-fingerprint rows honestly', () => {
    const c = FIXTURE_INTEGRITY.provenanceCoverage;
    expect(c.withFingerprint + c.missingFingerprint).toBe(c.totalResults);
    expect(c.missingFingerprint).toBeGreaterThan(0);
  });

  it('ALLOW_SYNTHETIC_RETURNS is false and synthetic-risk is flagged', () => {
    expect(FIXTURE_INTEGRITY.returnsLedger.allowSyntheticFlag).toBe(false);
    expect(
      FIXTURE_INTEGRITY.returnsLedger.syntheticRiskResultIds.length,
    ).toBeGreaterThan(0);
  });

  it('holdout is sealed at the configured 0.20 fraction', () => {
    expect(FIXTURE_INTEGRITY.holdout.state).toBe('SEALED');
    expect(FIXTURE_INTEGRITY.holdout.fraction).toBe(0.2);
  });

  it('a dataset with NULL fingerprint is present and marked unknown TZ', () => {
    const nullFp = FIXTURE_INTEGRITY.datasets.find((d) => d.fingerprint === null);
    expect(nullFp).toBeDefined();
    expect(nullFp?.timezoneVerified.kind).toBe('unknown');
  });

  it('config freeze records the pessimistic cost and gate thresholds', () => {
    const keys = Object.fromEntries(
      FIXTURE_INTEGRITY.configFreeze.keys.map((k) => [k.key, k.value]),
    );
    expect(keys.DEFAULT_HOLDOUT_FRACTION).toBe('0.20');
    expect(keys.MIN_SHARPE).toBe('0.5');
    expect(keys.ALLOW_SYNTHETIC_RETURNS).toBe('False');
  });
});

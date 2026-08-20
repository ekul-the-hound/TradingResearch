import { describe, expect, it } from 'vitest';
import { MockResearchRepository } from '../src/data/mock/MockResearchRepository';
import { FIXTURE_DISCOVERY } from '../src/data/mock/fixtures/discovery';

describe('discovery inbox', () => {
  const repo = new MockResearchRepository();

  it('returns inbox marked as fixture', async () => {
    const r = await repo.getDiscoveryInbox();
    expect(r.state).toBe('ready');
    if (r.state === 'ready') expect(r.isFixture).toBe(true);
  });

  it('flags a near-duplicate via semantic similarity', () => {
    const dup = FIXTURE_DISCOVERY.strategies.find((s) => s.status === 'DUPLICATE');
    expect(dup).toBeDefined();
    expect(dup?.duplicateOf).toBe('src_101');
    if (dup?.similarity.kind === 'value') {
      expect(dup.similarity.value).toBeGreaterThan(0.9);
    }
  });

  it('unscored candidate shows quality as unknown, not zero', () => {
    const unscored = FIXTURE_DISCOVERY.strategies.find((s) => s.id === 'src_105');
    expect(unscored?.qualityScore.kind).toBe('unknown');
  });

  it('mean quality reflects a low-quality pool (<3)', () => {
    if (FIXTURE_DISCOVERY.meanQuality.kind === 'value') {
      expect(FIXTURE_DISCOVERY.meanQuality.value).toBeLessThan(3);
    }
  });

  it('dedup counts are internally consistent', () => {
    const d = FIXTURE_DISCOVERY.dedup;
    expect(d.unique + d.duplicates).toBe(d.total);
    expect(d.method).toMatch(/FAISS/);
  });

  it('untestable ideas carry why-untestable and data-needed', () => {
    for (const i of FIXTURE_DISCOVERY.untestableIdeas) {
      expect(i.whyUntestable.length).toBeGreaterThan(0);
      expect(i.dataNeeded.length).toBeGreaterThan(0);
    }
  });

  it('source model is the configured discovery model', () => {
    expect(FIXTURE_DISCOVERY.sourceModels).toContain('minimax-m3:cloud');
  });
});

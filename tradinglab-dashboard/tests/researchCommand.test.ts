import { describe, expect, it } from 'vitest';
import { MockResearchRepository } from '../src/data/mock/MockResearchRepository';
import {
  FIXTURE_FUNNEL,
  FIXTURE_QUEUE,
} from '../src/data/mock/fixtures/researchCommand';

describe('research command mock reads', () => {
  const repo = new MockResearchRepository();

  it('returns a latest run marked as fixture', async () => {
    const r = await repo.getLatestRun();
    expect(r.state).toBe('ready');
    if (r.state === 'ready') {
      expect(r.isFixture).toBe(true);
      expect(r.data.runId).toBeTruthy();
    }
  });

  it('funnel LIVE stage is zero (no broker bridge)', () => {
    const live = FIXTURE_FUNNEL.find((s) => s.stage === 'LIVE');
    expect(live?.count.kind).toBe('value');
    if (live?.count.kind === 'value') expect(live.count.value).toBe(0);
  });

  it('funnel PAPER blocked is an honest UNKNOWN, not fabricated 0', () => {
    const paper = FIXTURE_FUNNEL.find((s) => s.stage === 'PAPER');
    expect(paper?.blocked.kind).toBe('unknown');
  });

  it('queue surfaces synthetic-returns risk as CRITICAL from canonical_result', () => {
    const synth = FIXTURE_QUEUE.find((q) =>
      q.title.toLowerCase().includes('synthetic'),
    );
    expect(synth).toBeDefined();
    expect(synth?.severity).toBe('CRITICAL');
    expect(synth?.sourceLabel).toBe('canonical_result');
  });

  it('queue surfaces proxy compliance as CRITICAL', () => {
    const proxy = FIXTURE_QUEUE.find((q) =>
      q.title.toLowerCase().includes('proxy'),
    );
    expect(proxy?.severity).toBe('CRITICAL');
  });

  it('every queue item has a source label and suggested action', () => {
    for (const q of FIXTURE_QUEUE) {
      expect(q.sourceLabel.length).toBeGreaterThan(0);
      expect(q.suggestedAction.length).toBeGreaterThan(0);
    }
  });

  it('health strip has exactly the five required modules', async () => {
    const r = await repo.getResearchHealth();
    if (r.state === 'ready') {
      const keys = r.data.modules.map((m) => m.key).sort();
      expect(keys).toEqual(
        [
          'cost-realism',
          'data-coverage',
          'discovery-quality',
          'integrity-ledger',
          'validation-coverage',
        ].sort(),
      );
    }
  });
});

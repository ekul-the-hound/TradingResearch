import { describe, expect, it } from 'vitest';
import { FIXTURE_STRATEGIES } from '../src/data/mock/fixtures/strategies';
import {
  SAVED_VIEWS,
  applyFilter,
  applySavedView,
  isEvidenceComplete,
} from '../src/domain/strategy-lab/filtering';
import { MockResearchRepository } from '../src/data/mock/MockResearchRepository';

const view = (id: string) => SAVED_VIEWS.find((v) => v.id === id)!;

describe('strategy lab filtering', () => {
  it('filters by stage', () => {
    const r = applyFilter(FIXTURE_STRATEGIES, { stage: 'REJECTED' });
    expect(r.every((s) => s.stage === 'REJECTED')).toBe(true);
    expect(r.length).toBe(1);
  });

  it('filters by returns provenance', () => {
    const r = applyFilter(FIXTURE_STRATEGIES, {
      returnsProvenance: 'SYNTHETIC_RISK',
    });
    expect(r).toHaveLength(1);
    expect(r[0].strategyId).toBe('GBPUSD_H4_Break_v03');
  });

  it('text filter matches name and id case-insensitively', () => {
    expect(applyFilter(FIXTURE_STRATEGIES, { text: 'meanrev' })).toHaveLength(1);
    expect(applyFilter(FIXTURE_STRATEGIES, { text: 'ZZZ' })).toHaveLength(0);
  });

  it('evidence-complete is false when any item is UNKNOWN/INCOMPLETE', () => {
    const notRun = FIXTURE_STRATEGIES.find(
      (s) => s.strategyId === 'EURUSD_H4_ADX_v01',
    )!;
    expect(isEvidenceComplete(notRun)).toBe(false);
  });

  it('Evidence Incomplete saved view excludes fully-evidenced rows', () => {
    const r = applySavedView(FIXTURE_STRATEGIES, view('evidence-incomplete'));
    expect(r.every((s) => !isEvidenceComplete(s))).toBe(true);
    expect(r.length).toBeGreaterThan(0);
  });

  it('Manual Ideas saved view returns only manual-origin strategies', () => {
    const r = applySavedView(FIXTURE_STRATEGIES, view('manual-ideas'));
    expect(r.every((s) => s.origin === 'manual')).toBe(true);
  });

  it('Data/Integrity Warnings view catches synthetic-risk and missing fingerprint', () => {
    const r = applySavedView(FIXTURE_STRATEGIES, view('integrity-warnings'));
    const ids = r.map((s) => s.strategyId);
    expect(ids).toContain('GBPUSD_H4_Break_v03'); // synthetic + no fingerprint
  });

  it('Portfolio Candidates view is empty in the current pool (honest)', () => {
    const r = applySavedView(FIXTURE_STRATEGIES, view('portfolio-candidates'));
    expect(r).toHaveLength(0);
  });
});

describe('strategy lab repository read', () => {
  it('returns fixture-marked strategies and honors a filter', async () => {
    const repo = new MockResearchRepository();
    const all = await repo.listStrategies();
    expect(all.state).toBe('ready');
    if (all.state === 'ready') expect(all.isFixture).toBe(true);

    const filtered = await repo.listStrategies({ symbol: 'GBP/USD' });
    if (filtered.state === 'ready') {
      expect(filtered.data.every((s) => s.symbol === 'GBP/USD')).toBe(true);
    }
  });
});

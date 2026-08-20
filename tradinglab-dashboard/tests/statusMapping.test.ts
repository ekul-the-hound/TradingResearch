import { describe, expect, it } from 'vitest';
import { toSegments } from '../src/app/shell/statusMapping';
import {
  FIXTURE_STATUS_COMPLETE,
  FIXTURE_STATUS_UNKNOWN,
} from '../src/data/mock/fixtures/systemStatus';
import { MockResearchRepository } from '../src/data/mock/MockResearchRepository';
import type { SystemStatus } from '../src/models/system';

function seg(status: SystemStatus, key: string) {
  const s = toSegments(status).find((x) => x.key === key);
  if (!s) throw new Error(`missing segment ${key}`);
  return s;
}

describe('status bar mapping — complete state', () => {
  const S = FIXTURE_STATUS_COMPLETE;

  it('maps operating and market mode', () => {
    expect(seg(S, 'mode').value).toBe('RESEARCH');
    expect(seg(S, 'market').value).toBe('HISTORICAL');
  });

  it('shows the actual data source', () => {
    expect(seg(S, 'source').value).toBe('HistData');
  });

  it('renders holdout SEALED with a pass tone', () => {
    const h = seg(S, 'holdout');
    expect(h.value).toBe('SEALED');
    expect(h.tone).toBe('pass');
  });

  it('exposes the full fingerprint as copyable and abbreviates the display', () => {
    const fp = seg(S, 'fingerprint');
    expect(fp.copyable).toBe('8f2a1c9d4b7e3a10');
    expect(fp.value).not.toBe('8f2a1c9d4b7e3a10'); // abbreviated
    expect(fp.value).toContain('…');
  });

  it('shows the named cost profile with its pip values', () => {
    const c = seg(S, 'cost');
    expect(c.value).toContain('Pessimistic Manual');
    expect(c.value).toContain('2p / 1p');
    expect(c.title).toContain('intraday/no swaps');
  });

  it('never shows broker as a green connected state by default', () => {
    const b = seg(S, 'broker');
    expect(b.value).toBe('OFFLINE');
    expect(b.tone).not.toBe('pass');
  });

  it('surfaces integrity warning count with a warn tone', () => {
    const w = seg(S, 'warnings');
    expect(w.value).toBe('2 warnings');
    expect(w.tone).toBe('warn');
  });
});

describe('status bar mapping — UNKNOWN / unavailable state', () => {
  const S = FIXTURE_STATUS_UNKNOWN;

  it('never invents a last-run time', () => {
    expect(seg(S, 'lastrun').value).toBe('UNKNOWN');
  });

  it('shows UNKNOWN market mode rather than defaulting to REAL-TIME', () => {
    expect(seg(S, 'market').value).toBe('UNKNOWN');
  });

  it('shows UNKNOWN data source when null', () => {
    expect(seg(S, 'source').value).toBe('UNKNOWN');
  });

  it('shows NONE (no copyable) when no fingerprint exists', () => {
    const fp = seg(S, 'fingerprint');
    expect(fp.value).toBe('NONE');
    expect(fp.copyable).toBeUndefined();
  });

  it('shows UNKNOWN cost when no profile is set', () => {
    expect(seg(S, 'cost').value).toBe('UNKNOWN');
  });

  it('shows broker NOT CONFIGURED, not connected', () => {
    const b = seg(S, 'broker');
    expect(b.value).toBe('NOT CONFIGURED');
    expect(b.tone).not.toBe('pass');
  });

  it('shows clear/pass when there are no integrity warnings', () => {
    const w = seg(S, 'warnings');
    expect(w.value).toBe('clear');
    expect(w.tone).toBe('pass');
  });
});

describe('MockResearchRepository', () => {
  it('always stamps isFixture on the repo and every read', async () => {
    const repo = new MockResearchRepository();
    expect(repo.isFixture).toBe(true);
    const r = await repo.getSystemStatus();
    expect(r.state).toBe('ready');
    if (r.state === 'ready') expect(r.isFixture).toBe(true);
  });

  it('honors an injected status scenario', async () => {
    const repo = new MockResearchRepository(FIXTURE_STATUS_UNKNOWN);
    const r = await repo.getSystemStatus();
    if (r.state === 'ready') {
      expect(r.data.broker).toBe('NOT_CONFIGURED');
      expect(r.data.lastRunAt).toBeNull();
    }
  });
});

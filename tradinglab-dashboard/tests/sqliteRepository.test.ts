import { afterEach, describe, expect, it, vi } from 'vitest';
import { SqliteResearchRepository } from '../src/data/sqlite/SqliteResearchRepository';

function stubFetch(handler: (url: string) => unknown) {
  vi.stubGlobal('fetch', async (url: string) => ({
    ok: true,
    status: 200,
    json: async () => handler(url),
  }));
}

afterEach(() => vi.unstubAllGlobals());

describe('SqliteResearchRepository', () => {
  it('is never marked as fixture', () => {
    const repo = new SqliteResearchRepository('http://localhost:8799');
    expect(repo.isFixture).toBe(false);
  });

  it('passes through a ready payload and normalizes isFixture=false', async () => {
    stubFetch(() => ({
      state: 'ready',
      data: { operatingMode: 'RESEARCH' },
      isFixture: true, // bridge could lie; adapter must force false
    }));
    const repo = new SqliteResearchRepository('http://x');
    const r = await repo.getSystemStatus();
    expect(r.state).toBe('ready');
    if (r.state === 'ready') expect(r.isFixture).toBe(false);
  });

  it('passes through an unavailable payload verbatim (missing DB)', async () => {
    stubFetch(() => ({ state: 'unavailable', reason: 'Database not found: x.db' }));
    const repo = new SqliteResearchRepository('http://x');
    const r = await repo.getIntegrityStatus();
    expect(r.state).toBe('unavailable');
    if (r.state === 'unavailable') expect(r.reason).toMatch(/not found/);
  });

  it('returns an error state when the bridge is unreachable', async () => {
    vi.stubGlobal('fetch', async () => {
      throw new Error('ECONNREFUSED');
    });
    const repo = new SqliteResearchRepository('http://127.0.0.1:1');
    const r = await repo.getSystemStatus();
    expect(r.state).toBe('error');
    if (r.state === 'error') expect(r.error).toMatch(/bridge/i);
  });

  it('applies a client-side filter to bridge strategy results', async () => {
    stubFetch(() => ({
      state: 'ready',
      isFixture: false,
      data: [
        { symbol: 'EUR/USD', strategyId: 'a', name: 'a', timeframe: '1H',
          stage: 'BACKTESTED', holdout: 'SEALED', returnsProvenance: 'REAL',
          dataFingerprint: 'x', evidence: {} },
        { symbol: 'GBP/USD', strategyId: 'b', name: 'b', timeframe: '4H',
          stage: 'BACKTESTED', holdout: 'SEALED', returnsProvenance: 'REAL',
          dataFingerprint: 'y', evidence: {} },
      ],
    }));
    const repo = new SqliteResearchRepository('http://x');
    const r = await repo.listStrategies({ symbol: 'EUR/USD' });
    if (r.state === 'ready') {
      expect(r.data).toHaveLength(1);
      expect(r.data[0].symbol).toBe('EUR/USD');
    }
  });

  it('reports not-yet-wired reads as unavailable, not fabricated', async () => {
    const repo = new SqliteResearchRepository('http://x');
    const r = await repo.getDiscoveryInbox();
    expect(r.state).toBe('unavailable');
  });
});

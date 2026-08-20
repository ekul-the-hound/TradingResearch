import { describe, expect, it } from 'vitest';
import { MockResearchRepository } from '../src/data/mock/MockResearchRepository';
import { FIXTURE_EXECUTION } from '../src/data/mock/fixtures/execution';

describe('execution', () => {
  const repo = new MockResearchRepository();

  it('returns execution status marked as fixture', async () => {
    const r = await repo.getExecutionStatus();
    expect(r.state).toBe('ready');
    if (r.state === 'ready') expect(r.isFixture).toBe(true);
  });

  it('defaults to offline / not-configured', () => {
    expect(FIXTURE_EXECUTION.mode).toBe('OFFLINE');
    expect(FIXTURE_EXECUTION.broker).toBe('NOT_CONFIGURED');
    expect(FIXTURE_EXECUTION.overallStatus).toBe('OFFLINE');
  });

  it('never exposes live P&L', () => {
    expect(FIXTURE_EXECUTION.livePnL).toBeNull();
  });

  it('has no active session', () => {
    expect(FIXTURE_EXECUTION.session.active).toBe(false);
    expect(FIXTURE_EXECUTION.session.sessionId).toBeNull();
  });

  it('no precondition is met (nothing can go live)', () => {
    expect(FIXTURE_EXECUTION.preconditions.every((p) => !p.met)).toBe(true);
  });

  it('no component is CONNECTED', () => {
    expect(
      FIXTURE_EXECUTION.components.every((c) => c.state !== 'CONNECTED'),
    ).toBe(true);
  });

  it('the MT5 bridge is scaffold-only', () => {
    const mt5 = FIXTURE_EXECUTION.components.find((c) => c.name === 'MT5 transport');
    expect(mt5?.state).toBe('SCAFFOLD');
  });
});

import type { SystemStatus } from '../../../models/system';

// DEVELOPMENT FIXTURE. None of these values come from a database. Consumed only
// through MockResearchRepository, which stamps isFixture:true on every read.

// Scenario 1: a healthy repo-backed-looking state (all fields present).
export const FIXTURE_STATUS_COMPLETE: SystemStatus = {
  operatingMode: 'RESEARCH',
  marketMode: 'HISTORICAL',
  dataSource: 'HistData',
  lastRunAt: '2026-01-14T19:42:00Z',
  datasetFingerprint: '8f2a1c9d4b7e3a10',
  holdout: 'SEALED',
  costProfile: {
    name: 'Pessimistic Manual',
    spreadPips: 2,
    slippagePips: 1,
    swaps: 'INTRADAY_NONE',
  },
  broker: 'OFFLINE',
  integrityWarningCount: 2,
};

// Scenario 2: unknown/unavailable state — no runs yet, nothing verified.
export const FIXTURE_STATUS_UNKNOWN: SystemStatus = {
  operatingMode: 'RESEARCH',
  marketMode: 'UNKNOWN',
  dataSource: null,
  lastRunAt: null,
  datasetFingerprint: null,
  holdout: 'UNKNOWN',
  costProfile: null,
  broker: 'NOT_CONFIGURED',
  integrityWarningCount: 0,
};

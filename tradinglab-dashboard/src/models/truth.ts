// Shared truth-label unions (CLAUDE.md §3). These drive uniform rendering
// across every page. Do not widen or collapse these to strings elsewhere.

export type OperatingMode = 'RESEARCH' | 'BACKTEST' | 'PAPER' | 'DEMO' | 'LIVE';
export type MarketMode = 'HISTORICAL' | 'DELAYED' | 'REAL-TIME' | 'UNKNOWN';
export type HoldoutState = 'SEALED' | 'UNSEALED' | 'UNKNOWN';
export type ReturnsProvenance =
  | 'REAL'
  | 'UNVERIFIED'
  | 'SYNTHETIC_RISK'
  | 'UNAVAILABLE';
export type ComplianceBasis =
  | 'AUTHORITATIVE'
  | 'PROXY'
  | 'INCOMPLETE'
  | 'UNKNOWN';
export type BrokerState = 'OFFLINE' | 'NOT_CONFIGURED' | 'CONNECTED';

// Canonical status semantics (CLAUDE.md §8). Never color-only.
export type Status =
  | 'PASS'
  | 'FAIL'
  | 'WARNING'
  | 'INFO'
  | 'UNKNOWN'
  | 'PROXY'
  | 'INCOMPLETE'
  | 'OFFLINE';

export type Severity = 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';

export type LifecycleStage =
  | 'DISCOVERED'
  | 'CODE_VALID'
  | 'BACKTESTED'
  | 'COST_ADJUSTED'
  | 'VALIDATED'
  | 'PORTFOLIO_CANDIDATE'
  | 'PAPER'
  | 'LIVE'
  | 'REJECTED'
  | 'RETIRED';

// A value that may not be knowable. Never collapse to 0 or "".
export type Unknowable<T> =
  | { kind: 'value'; value: T }
  | { kind: 'unavailable'; reason: string }
  | { kind: 'unknown'; reason?: string }
  | { kind: 'proxy'; value: T; note: string };

// Container-level load state for any repository read.
export type Loadable<T> =
  | { state: 'loading' }
  | { state: 'ready'; data: T; isFixture: boolean }
  | { state: 'empty' }
  | { state: 'unavailable'; reason: string }
  | { state: 'error'; error: string };

export const val = <T>(value: T): Unknowable<T> => ({ kind: 'value', value });
export const unknown = (reason?: string): Unknowable<never> => ({
  kind: 'unknown',
  reason,
});
export const unavailable = (reason: string): Unknowable<never> => ({
  kind: 'unavailable',
  reason,
});
export const proxy = <T>(value: T, note: string): Unknowable<T> => ({
  kind: 'proxy',
  value,
  note,
});

import type { Severity, Status, Unknowable } from './truth';

export interface ResearchQueueItem {
  id: string;
  severity: Severity;
  title: string;
  reason: string;
  affects: { kind: 'strategy' | 'run'; id: string } | null;
  sourceLabel: string; // e.g. "canonical_result", "FTMOComplianceChecker"
  suggestedAction: string;
  destination: string | null; // route to navigate
}

// A compact breakdown bucket for the health-strip modules.
export interface Bucket {
  label: string;
  value: number;
  tone?: 'pass' | 'fail' | 'warn' | 'info' | 'research' | 'neutral';
}

export interface HealthModule {
  key:
    | 'data-coverage'
    | 'discovery-quality'
    | 'validation-coverage'
    | 'cost-realism'
    | 'integrity-ledger';
  title: string;
  // Primary + a meaningful secondary measure (never a lone number).
  primary: Unknowable<{ value: string; label: string }>;
  secondary: { label: string; value: string } | null;
  // Optional compact distribution/breakdown when data supports it.
  buckets: Bucket[] | null;
  sourceLabel: string;
  status: Status; // INFO/PASS/WARNING/INCOMPLETE/UNKNOWN
}

export interface ResearchHealth {
  modules: HealthModule[];
}

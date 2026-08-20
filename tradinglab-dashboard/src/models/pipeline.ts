import type {
  HoldoutState,
  LifecycleStage,
  ReturnsProvenance,
  Unknowable,
} from './truth';

export interface PipelineRun {
  runId: string;
  startedAt: string | null;
  completedAt: string | null; // actual only; never inferred
  durationSec: number | null;
  inputCount: Unknowable<number>;
  survivorCount: Unknowable<number>;
  topCandidateId: string | null;
  primaryBlocker: string | null;
  dataFingerprint: string | null;
  holdout: HoldoutState;
  costProfileName: string;
  returnsProvenance: ReturnsProvenance;
  status: 'SUCCESS' | 'FAILED' | 'PARTIAL' | 'NO_RUNS';
}

export interface FunnelStage {
  stage: LifecycleStage;
  count: Unknowable<number>;
  blocked: Unknowable<number>;
  rejected: Unknowable<number>;
  definition: string;
}

export type PipelineFunnel = FunnelStage[];

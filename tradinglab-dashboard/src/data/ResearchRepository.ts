import type { Loadable } from '../models/truth';
import type { SystemStatus } from '../models/system';
import type { PipelineFunnel, PipelineRun } from '../models/pipeline';
import type { ResearchHealth, ResearchQueueItem } from '../models/queue';
import type { StrategyFilter, StrategySummary } from '../models/strategy';
import type { StrategyDetail } from '../models/detail';
import type {
  ChallengeSimulationResult,
  FTMOComplianceStatus,
} from '../models/ftmo';
import type { IntegrityStatus } from '../models/integrity';
import type {
  PortfolioCandidate,
  PortfolioComputation,
} from '../models/portfolio';
import type { ChallengeReadiness } from '../models/challenge';
import type { DiscoveryInbox } from '../models/discovery';
import type { ExecutionStatus } from '../models/execution';

// The single data-access seam every page depends on. Concrete implementations
// (mock, sqlite) are swapped behind RepositoryProvider. Pages never import a
// concrete repository (ARCHITECTURE §1/§4).
//
// This interface grows one method-group per build prompt.
export interface ResearchRepository {
  // When true, EVERY Loadable this repo returns carries isFixture:true, which
  // the UI renders as a visible DEV FIXTURE marker. Fixtures cannot masquerade
  // as production data.
  readonly isFixture: boolean;

  getSystemStatus(): Promise<Loadable<SystemStatus>>;

  // --- Research Command (Prompt 6) ---
  getLatestRun(): Promise<Loadable<PipelineRun>>;
  getFunnel(): Promise<Loadable<PipelineFunnel>>;
  getResearchQueue(): Promise<Loadable<ResearchQueueItem[]>>;
  getResearchHealth(): Promise<Loadable<ResearchHealth>>;

  // --- Strategy Lab (Prompt 7) ---
  listStrategies(filter?: StrategyFilter): Promise<Loadable<StrategySummary[]>>;

  // --- Strategy Detail (Prompt 8) ---
  getStrategyDetail(strategyId: string): Promise<Loadable<StrategyDetail>>;
  getFTMOCompliance(
    strategyId: string,
  ): Promise<Loadable<FTMOComplianceStatus>>;
  getChallengeSimulation(
    strategyId: string,
  ): Promise<Loadable<ChallengeSimulationResult>>;

  // --- Data & Integrity (Prompt 9) ---
  getIntegrityStatus(): Promise<Loadable<IntegrityStatus>>;

  // --- Portfolio Builder (Prompt 10) ---
  getPortfolioCandidates(): Promise<Loadable<PortfolioCandidate[]>>;
  computePortfolio(ids: string[]): Promise<Loadable<PortfolioComputation>>;

  // --- Challenge Readiness (Prompt 11) ---
  getChallengeReadiness(): Promise<Loadable<ChallengeReadiness>>;

  // --- Discovery Inbox (Prompt 12) ---
  getDiscoveryInbox(): Promise<Loadable<DiscoveryInbox>>;

  // --- Execution (Prompt 13) ---
  getExecutionStatus(): Promise<Loadable<ExecutionStatus>>;
}

import type { ResearchRepository } from '../ResearchRepository';
import type { Loadable } from '../../models/truth';
import type { SystemStatus } from '../../models/system';
import type { PipelineFunnel, PipelineRun } from '../../models/pipeline';
import type { ResearchHealth, ResearchQueueItem } from '../../models/queue';
import type { StrategyFilter, StrategySummary } from '../../models/strategy';
import { FIXTURE_STATUS_COMPLETE } from './fixtures/systemStatus';
import {
  FIXTURE_FUNNEL,
  FIXTURE_HEALTH,
  FIXTURE_LATEST_RUN,
  FIXTURE_QUEUE,
} from './fixtures/researchCommand';
import { FIXTURE_STRATEGIES } from './fixtures/strategies';
import {
  FIXTURE_DETAILS,
  challengeSimFor,
  ftmoFor,
} from './fixtures/detail';
import { applyFilter } from '../../domain/strategy-lab/filtering';
import type { StrategyDetail } from '../../models/detail';
import type {
  ChallengeSimulationResult,
  FTMOComplianceStatus,
} from '../../models/ftmo';
import type { IntegrityStatus } from '../../models/integrity';
import { FIXTURE_INTEGRITY } from './fixtures/integrity';
import type {
  PortfolioCandidate,
  PortfolioComputation,
} from '../../models/portfolio';
import {
  EMPTY_COMPUTATION,
  FIXTURE_CANDIDATES,
  computePortfolioFixture,
} from './fixtures/portfolio';
import type { ChallengeReadiness } from '../../models/challenge';
import { FIXTURE_CHALLENGE_READINESS } from './fixtures/challenge';
import type { DiscoveryInbox } from '../../models/discovery';
import { FIXTURE_DISCOVERY } from './fixtures/discovery';
import type { ExecutionStatus } from '../../models/execution';
import { FIXTURE_EXECUTION } from './fixtures/execution';

// DEVELOPMENT FIXTURE ADAPTER. Everything returned is marked isFixture:true so
// it can never be mistaken for production data (CLAUDE.md §7).
export class MockResearchRepository implements ResearchRepository {
  readonly isFixture = true;
  private readonly status: SystemStatus;

  constructor(status: SystemStatus = FIXTURE_STATUS_COMPLETE) {
    this.status = status;
  }

  async getSystemStatus(): Promise<Loadable<SystemStatus>> {
    return { state: 'ready', data: this.status, isFixture: true };
  }

  async getLatestRun(): Promise<Loadable<PipelineRun>> {
    return { state: 'ready', data: FIXTURE_LATEST_RUN, isFixture: true };
  }

  async getFunnel(): Promise<Loadable<PipelineFunnel>> {
    return { state: 'ready', data: FIXTURE_FUNNEL, isFixture: true };
  }

  async getResearchQueue(): Promise<Loadable<ResearchQueueItem[]>> {
    return { state: 'ready', data: FIXTURE_QUEUE, isFixture: true };
  }

  async getResearchHealth(): Promise<Loadable<ResearchHealth>> {
    return { state: 'ready', data: FIXTURE_HEALTH, isFixture: true };
  }

  async listStrategies(
    filter?: StrategyFilter,
  ): Promise<Loadable<StrategySummary[]>> {
    const data = filter
      ? applyFilter(FIXTURE_STRATEGIES, filter)
      : FIXTURE_STRATEGIES;
    return { state: 'ready', data, isFixture: true };
  }

  async getStrategyDetail(
    strategyId: string,
  ): Promise<Loadable<StrategyDetail>> {
    const known = FIXTURE_STRATEGIES.some((s) => s.strategyId === strategyId);
    if (!known) {
      return { state: 'empty' };
    }
    const detail = FIXTURE_DETAILS[strategyId];
    if (!detail) {
      return {
        state: 'unavailable',
        reason: `No detailed evidence has been persisted for ${strategyId} yet.`,
      };
    }
    return { state: 'ready', data: detail, isFixture: true };
  }

  async getFTMOCompliance(
    strategyId: string,
  ): Promise<Loadable<FTMOComplianceStatus>> {
    const known = FIXTURE_STRATEGIES.some((s) => s.strategyId === strategyId);
    if (!known) return { state: 'empty' };
    return { state: 'ready', data: ftmoFor(strategyId), isFixture: true };
  }

  async getChallengeSimulation(
    strategyId: string,
  ): Promise<Loadable<ChallengeSimulationResult>> {
    const known = FIXTURE_STRATEGIES.some((s) => s.strategyId === strategyId);
    if (!known) return { state: 'empty' };
    return { state: 'ready', data: challengeSimFor(strategyId), isFixture: true };
  }

  async getIntegrityStatus(): Promise<Loadable<IntegrityStatus>> {
    return { state: 'ready', data: FIXTURE_INTEGRITY, isFixture: true };
  }

  async getPortfolioCandidates(): Promise<Loadable<PortfolioCandidate[]>> {
    return { state: 'ready', data: FIXTURE_CANDIDATES, isFixture: true };
  }

  async computePortfolio(
    ids: string[],
  ): Promise<Loadable<PortfolioComputation>> {
    const data = ids.length === 0 ? EMPTY_COMPUTATION : computePortfolioFixture(ids);
    return { state: 'ready', data, isFixture: true };
  }

  async getChallengeReadiness(): Promise<Loadable<ChallengeReadiness>> {
    return { state: 'ready', data: FIXTURE_CHALLENGE_READINESS, isFixture: true };
  }

  async getDiscoveryInbox(): Promise<Loadable<DiscoveryInbox>> {
    return { state: 'ready', data: FIXTURE_DISCOVERY, isFixture: true };
  }

  async getExecutionStatus(): Promise<Loadable<ExecutionStatus>> {
    return { state: 'ready', data: FIXTURE_EXECUTION, isFixture: true };
  }
}

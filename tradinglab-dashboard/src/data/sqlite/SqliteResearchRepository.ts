import type { ResearchRepository } from '../ResearchRepository';
import type { Loadable } from '../../models/truth';
import type { SystemStatus } from '../../models/system';
import type { PipelineFunnel, PipelineRun } from '../../models/pipeline';
import type { ResearchHealth, ResearchQueueItem } from '../../models/queue';
import type { StrategyFilter, StrategySummary } from '../../models/strategy';
import type { StrategyDetail } from '../../models/detail';
import type {
  ChallengeSimulationResult,
  FTMOComplianceStatus,
} from '../../models/ftmo';
import type { IntegrityStatus } from '../../models/integrity';
import type {
  PortfolioCandidate,
  PortfolioComputation,
} from '../../models/portfolio';
import type { ChallengeReadiness } from '../../models/challenge';
import type { DiscoveryInbox } from '../../models/discovery';
import type { ExecutionStatus } from '../../models/execution';
import { applyFilter } from '../../domain/strategy-lab/filtering';

// Read-only adapter over the Python `sqlite_bridge.py` sidecar. Every method
// returns exactly the Loadable the bridge produces (ready/empty/unavailable/
// error) so the UI renders honest states. isFixture is false — nothing here is
// fabricated. Reads not yet wired in the bridge return an explicit `unavailable`
// rather than fake data.
export class SqliteResearchRepository implements ResearchRepository {
  readonly isFixture = false;
  private readonly baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
  }

  private async get<T>(path: string): Promise<Loadable<T>> {
    try {
      const res = await fetch(`${this.baseUrl}${path}`);
      if (!res.ok) {
        return { state: 'error', error: `Bridge ${res.status} on ${path}` };
      }
      const body = (await res.json()) as Loadable<T> | { state: 'error'; error: string };
      // Bridge returns the Loadable shape directly; normalize isFixture=false.
      if (body.state === 'ready') {
        return { state: 'ready', data: (body as { data: T }).data, isFixture: false };
      }
      return body as Loadable<T>;
    } catch (e) {
      return {
        state: 'error',
        error:
          `Cannot reach the SQLite bridge at ${this.baseUrl}. ` +
          `Start it with: python bridge/sqlite_bridge.py --root <PROJECT> (${String(
            (e as Error)?.message ?? e,
          )})`,
      };
    }
  }

  private notWired<T>(what: string): Promise<Loadable<T>> {
    return Promise.resolve({
      state: 'unavailable',
      reason: `${what} is not yet served by the read-only bridge. Extend sqlite_bridge.py to map it.`,
    });
  }

  getSystemStatus(): Promise<Loadable<SystemStatus>> {
    return this.get<SystemStatus>('/api/system-status');
  }

  async listStrategies(
    filter?: StrategyFilter,
  ): Promise<Loadable<StrategySummary[]>> {
    const r = await this.get<StrategySummary[]>('/api/strategies');
    if (r.state === 'ready' && filter) {
      return { state: 'ready', data: applyFilter(r.data, filter), isFixture: false };
    }
    return r;
  }

  getIntegrityStatus(): Promise<Loadable<IntegrityStatus>> {
    return this.get<IntegrityStatus>('/api/integrity');
  }

  getExecutionStatus(): Promise<Loadable<ExecutionStatus>> {
    return this.get<ExecutionStatus>('/api/execution');
  }

  // --- Reads that require lineage/journal joins not yet in the bridge. These
  //     honestly report 'unavailable' instead of returning fabricated data. ---
  getLatestRun(): Promise<Loadable<PipelineRun>> {
    return this.notWired('Latest pipeline run');
  }
  getFunnel(): Promise<Loadable<PipelineFunnel>> {
    return this.notWired('Strategy funnel');
  }
  getResearchQueue(): Promise<Loadable<ResearchQueueItem[]>> {
    return this.notWired('Research queue');
  }
  getResearchHealth(): Promise<Loadable<ResearchHealth>> {
    return this.notWired('Research health');
  }
  getStrategyDetail(): Promise<Loadable<StrategyDetail>> {
    return this.notWired('Strategy detail');
  }
  getFTMOCompliance(): Promise<Loadable<FTMOComplianceStatus>> {
    return this.notWired('FTMO compliance');
  }
  getChallengeSimulation(): Promise<Loadable<ChallengeSimulationResult>> {
    return this.notWired('Challenge simulation');
  }
  getPortfolioCandidates(): Promise<Loadable<PortfolioCandidate[]>> {
    return this.notWired('Portfolio candidates');
  }
  computePortfolio(): Promise<Loadable<PortfolioComputation>> {
    return this.notWired('Portfolio computation');
  }
  getChallengeReadiness(): Promise<Loadable<ChallengeReadiness>> {
    return this.notWired('Challenge readiness');
  }
  getDiscoveryInbox(): Promise<Loadable<DiscoveryInbox>> {
    return this.notWired('Discovery inbox');
  }
}

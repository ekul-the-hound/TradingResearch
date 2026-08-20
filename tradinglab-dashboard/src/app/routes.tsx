import { Navigate, Route, Routes } from 'react-router-dom';
import { Preview } from '../design/Preview';
import { PlaceholderPage } from './PlaceholderPage';
import { ResearchCommandPage } from '../domain/research-command/ResearchCommandPage';
import { StrategyLabPage } from '../domain/strategy-lab/StrategyLabPage';
import { StrategyDetailPage } from '../domain/strategy-detail/StrategyDetailPage';
import { DataIntegrityPage } from '../domain/data-integrity/DataIntegrityPage';
import { PortfolioBuilderPage } from '../domain/portfolio-builder/PortfolioBuilderPage';
import { ChallengeReadinessPage } from '../domain/challenge-readiness/ChallengeReadinessPage';
import { DiscoveryInboxPage } from '../domain/discovery-inbox/DiscoveryInboxPage';
import { ExecutionPage } from '../domain/execution/ExecutionPage';

export function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/research-command" replace />} />
      <Route path="/research-command" element={<ResearchCommandPage />} />
      <Route path="/strategy-lab" element={<StrategyLabPage />} />
      <Route path="/strategies/:strategyId" element={<StrategyDetailPage />} />
      <Route path="/portfolio-builder" element={<PortfolioBuilderPage />} />
      <Route path="/challenge-readiness" element={<ChallengeReadinessPage />} />
      <Route path="/discovery-inbox" element={<DiscoveryInboxPage />} />
      <Route path="/data-integrity" element={<DataIntegrityPage />} />
      <Route path="/execution" element={<ExecutionPage />} />
      <Route path="/preview" element={<Preview />} />
      <Route path="*" element={<PlaceholderPage title="Not found" />} />
    </Routes>
  );
}

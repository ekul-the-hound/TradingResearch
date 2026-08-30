import type { ReactNode } from 'react';
import type { ResearchRepository } from './ResearchRepository';
import { MockResearchRepository } from './mock/MockResearchRepository';
import { SqliteResearchRepository } from './sqlite/SqliteResearchRepository';
import { RepositoryContext } from './repositoryContext';

// Select the repository from build-time config:
//   VITE_BRIDGE_URL set  -> read-only SqliteResearchRepository (real data)
//   unset                -> MockResearchRepository (DEV FIXTURE)
// Pages never see which one they got; only isFixture differs.
function makeDefaultRepository(): ResearchRepository {
  const bridge = import.meta.env.VITE_BRIDGE_URL as string | undefined;
  if (bridge && bridge.trim().length > 0) {
    return new SqliteResearchRepository(bridge.trim());
  }
  return new MockResearchRepository();
}

const defaultRepository: ResearchRepository = makeDefaultRepository();

export function RepositoryProvider({
  repository = defaultRepository,
  children,
}: {
  repository?: ResearchRepository;
  children: ReactNode;
}) {
  return (
    <RepositoryContext.Provider value={repository}>
      {children}
    </RepositoryContext.Provider>
  );
}

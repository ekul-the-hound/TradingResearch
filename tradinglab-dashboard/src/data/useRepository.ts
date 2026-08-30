import { useContext } from 'react';
import type { ResearchRepository } from './ResearchRepository';
import { RepositoryContext } from './repositoryContext';

export function useRepository(): ResearchRepository {
  const repo = useContext(RepositoryContext);
  if (!repo) {
    throw new Error('useRepository must be used within a RepositoryProvider');
  }
  return repo;
}

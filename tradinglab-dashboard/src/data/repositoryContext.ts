import { createContext } from 'react';
import type { ResearchRepository } from './ResearchRepository';

export const RepositoryContext = createContext<ResearchRepository | null>(null);

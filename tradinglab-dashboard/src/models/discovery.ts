import type { Status, Unknowable } from './truth';

export type DiscoveryStatus =
  | 'NEW'
  | 'REVIEWED'
  | 'PROMOTED'
  | 'REJECTED'
  | 'DUPLICATE';

export interface DiscoveredStrategy {
  id: string;
  name: string;
  summary: string;
  sourceTitle: string;
  sourceUrl: string | null;
  sourceType: string | null; // "web", "manual", "mutation"
  sourceBias: string | null;
  modelUsed: string | null; // extraction model
  extractedAt: string | null;
  qualityScore: Unknowable<number>; // 0..5
  hasMath: boolean;
  hasCode: boolean;
  hasExplicitParams: boolean;
  hasBacktest: boolean;
  indicators: string[];
  timeframe: string | null;
  assetClass: string | null;
  status: DiscoveryStatus;
  // semantic-dedup: nearest existing neighbor
  duplicateOf: string | null;
  similarity: Unknowable<number>; // 0..1 cosine
}

export interface UntestableIdea {
  id: string;
  title: string;
  description: string;
  whyUntestable: string;
  dataNeeded: string;
  category: string | null;
  tags: string[];
  confidence: Unknowable<number>; // 0..1
  effort: string | null; // "low" | "medium" | "high"
  generatedBy: string | null;
  assetClass: string | null;
  timeframe: string | null;
  status: 'open' | 'promoted' | 'archived';
  createdAt: string | null;
}

export interface QualityBucket {
  label: string; // "0-1", "1-2"...
  count: number;
}

export interface DiscoveryInbox {
  strategies: DiscoveredStrategy[];
  untestableIdeas: UntestableIdea[];
  qualityDistribution: QualityBucket[];
  meanQuality: Unknowable<number>;
  dedup: {
    total: number;
    unique: number;
    duplicates: number;
    method: string; // "FAISS semantic dedup"
  };
  pipelineStatus: Status; // health of the discovery run
  lastDiscoveryAt: string | null;
  sourceModels: string[];
}

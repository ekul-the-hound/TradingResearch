# ==============================================================================
# semantic_dedup.py
# ==============================================================================
# Semantic Deduplication for Discovered Strategies
#
# Uses sentence-transformers + FAISS to detect near-duplicate strategies
# based on their summary embeddings.
#
# Two modes:
#   1. FAISS mode (preferred): sentence-transformers + FAISS index
#   2. Fallback mode: TF-IDF cosine similarity with numpy (no GPU needed)
#
# The fallback mode activates automatically if sentence-transformers or
# faiss-cpu aren't installed, so the pipeline never breaks.
#
# Install for full mode:
#     pip install sentence-transformers faiss-cpu
#
# Usage:
#     from semantic_dedup import SemanticDeduplicator
#
#     dedup = SemanticDeduplicator()
#     is_dup, similar_to, score = dedup.check_duplicate("RSI mean reversion strategy...")
#     dedup.add_strategy("strat_001", "RSI mean reversion strategy...")
#     dedup.save_index()
#
# ==============================================================================

import json
import logging
import numpy as np
from typing import Optional, Tuple, List, Dict
from pathlib import Path

from discovery_config import DISCOVERY_CONFIG as cfg
from research_db import ResearchDatabase

logger = logging.getLogger(__name__)

# ==============================================================================
# Try importing FAISS + sentence-transformers, fall back gracefully
# ==============================================================================
FAISS_AVAILABLE = False
SBERT_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    logger.info("faiss-cpu not installed. Using fallback dedup mode.")

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    logger.info("sentence-transformers not installed. Using fallback dedup mode.")


class SemanticDeduplicator:
    """
    Detects near-duplicate strategies using embedding similarity.

    Strategies are compared by their summary text. Two strategies with
    cosine similarity above the threshold are considered duplicates.

    The newer strategy is marked as a duplicate of the older one.
    """

    def __init__(self, db: Optional[ResearchDatabase] = None):
        self.db = db or ResearchDatabase()
        self.threshold = cfg.dedup.similarity_threshold
        self.index_path = cfg.dedup.index_path
        self.metadata_path = cfg.dedup.metadata_path
        self.embedding_dim = cfg.llm.embedding_dim

        # Strategy ID -> index position mapping
        self.metadata: Dict[int, str] = {}  # position -> strategy_id
        self.strategy_texts: Dict[str, str] = {}  # strategy_id -> summary text

        # Initialize embedding model and index
        self.use_faiss = FAISS_AVAILABLE and SBERT_AVAILABLE
        self._init_model()
        self._init_index()

        # Stats
        self.stats = {
            "checked": 0,
            "duplicates_found": 0,
            "unique_added": 0,
        }

    def _init_model(self):
        """Initialize the embedding model."""
        if self.use_faiss:
            model_name = cfg.llm.embedding_model
            logger.info(f"Loading embedding model: {model_name}")
            try:
                self.model = SentenceTransformer(model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                logger.info(f"[OK] Embedding model loaded (dim={self.embedding_dim})")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}. Using fallback.")
                self.use_faiss = False
                self.model = None
        else:
            self.model = None

        if not self.use_faiss:
            logger.info("Using TF-IDF fallback for deduplication")

    def _init_index(self):
        """Initialize or load the FAISS index."""
        if self.use_faiss:
            if self.index_path.exists() and self.metadata_path.exists():
                self._load_index()
            else:
                # Create new index (Inner Product for cosine similarity on normalized vectors)
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                self.metadata = {}
                logger.info(f"Created new FAISS index (dim={self.embedding_dim})")
        else:
            # Fallback: store raw text + simple vectors
            self._fallback_vectors: List[np.ndarray] = []
            self._fallback_ids: List[str] = []

            # Try loading fallback state
            fallback_path = Path(self.metadata_path).with_suffix('.fallback.json')
            if fallback_path.exists():
                try:
                    data = json.loads(fallback_path.read_text())
                    self._fallback_ids = data.get("ids", [])
                    self.strategy_texts = data.get("texts", {})
                    # Rebuild vectors from texts
                    for sid in self._fallback_ids:
                        text = self.strategy_texts.get(sid, "")
                        vec = self._tfidf_vector(text)
                        self._fallback_vectors.append(vec)
                    logger.info(f"Loaded fallback dedup state: {len(self._fallback_ids)} strategies")
                except Exception as e:
                    logger.warning(f"Failed to load fallback state: {e}")

    # ==========================================================================
    # CORE OPERATIONS
    # ==========================================================================

    def check_duplicate(self, summary_text: str,
                        strategy_id: str = "") -> Tuple[bool, Optional[str], float]:
        """
        Check if a strategy summary is a duplicate of an existing one.

        Args:
            summary_text: The strategy summary to check
            strategy_id: ID of the strategy being checked (for logging)

        Returns:
            (is_duplicate, duplicate_of_id, similarity_score)
        """
        self.stats["checked"] += 1

        if not summary_text or not summary_text.strip():
            return False, None, 0.0

        if self.use_faiss:
            return self._check_faiss(summary_text, strategy_id)
        else:
            return self._check_fallback(summary_text, strategy_id)

    def add_strategy(self, strategy_id: str, summary_text: str):
        """
        Add a strategy to the dedup index.

        Call this AFTER confirming the strategy is NOT a duplicate.

        Args:
            strategy_id: Unique strategy ID
            summary_text: Strategy summary text for embedding
        """
        if not summary_text or not summary_text.strip():
            return

        if self.use_faiss:
            self._add_faiss(strategy_id, summary_text)
        else:
            self._add_fallback(strategy_id, summary_text)

        self.strategy_texts[strategy_id] = summary_text
        self.stats["unique_added"] += 1

    def check_and_add(self, strategy_id: str,
                      summary_text: str) -> Tuple[bool, Optional[str], float]:
        """
        Check for duplicate and add if unique. Convenience method.

        Returns:
            (is_duplicate, duplicate_of_id, similarity_score)
        """
        is_dup, dup_of, score = self.check_duplicate(summary_text, strategy_id)

        if not is_dup:
            self.add_strategy(strategy_id, summary_text)
        else:
            self.stats["duplicates_found"] += 1
            # Log to database
            if dup_of:
                self.db.log_dedup_check(strategy_id, dup_of, score, True)

        return is_dup, dup_of, score

    # ==========================================================================
    # FAISS MODE
    # ==========================================================================

    def _embed(self, text: str) -> np.ndarray:
        """Get normalized embedding for text."""
        vec = self.model.encode([text], normalize_embeddings=True)
        return vec.astype(np.float32)

    def _check_faiss(self, summary_text: str,
                     strategy_id: str) -> Tuple[bool, Optional[str], float]:
        """Check duplicate using FAISS index."""
        if self.index.ntotal == 0:
            return False, None, 0.0

        query_vec = self._embed(summary_text)
        scores, indices = self.index.search(query_vec, min(5, self.index.ntotal))

        # Check top match
        top_score = float(scores[0][0])
        top_idx = int(indices[0][0])

        if top_score >= self.threshold:
            dup_id = self.metadata.get(top_idx, "unknown")
            logger.info(
                f"Duplicate found: {strategy_id} ≈ {dup_id} "
                f"(similarity={top_score:.3f} >= {self.threshold})"
            )
            return True, dup_id, top_score

        return False, None, top_score

    def _add_faiss(self, strategy_id: str, summary_text: str):
        """Add strategy embedding to FAISS index."""
        vec = self._embed(summary_text)
        pos = self.index.ntotal
        self.index.add(vec)
        self.metadata[pos] = strategy_id

    # ==========================================================================
    # FALLBACK MODE (TF-IDF + numpy cosine similarity)
    # ==========================================================================

    def _tfidf_vector(self, text: str, vocab_size: int = 5000) -> np.ndarray:
        """
        Create a simple TF vector for text.

        This is a basic bag-of-words approach -- not as good as sentence-transformers
        but works without any extra dependencies.
        """
        words = text.lower().split()
        # Use hash-based feature mapping to fixed-size vector
        vec = np.zeros(vocab_size, dtype=np.float32)
        for word in words:
            # Clean word
            word = ''.join(c for c in word if c.isalnum())
            if len(word) < 2:
                continue
            idx = hash(word) % vocab_size
            vec[idx] += 1.0

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b))

    def _check_fallback(self, summary_text: str,
                        strategy_id: str) -> Tuple[bool, Optional[str], float]:
        """Check duplicate using TF-IDF fallback."""
        if not self._fallback_vectors:
            return False, None, 0.0

        query_vec = self._tfidf_vector(summary_text)

        best_score = 0.0
        best_id = None

        for i, existing_vec in enumerate(self._fallback_vectors):
            score = self._cosine_similarity(query_vec, existing_vec)
            if score > best_score:
                best_score = score
                best_id = self._fallback_ids[i]

        # Fallback uses a slightly lower threshold since TF-IDF is less precise
        adjusted_threshold = self.threshold - 0.10

        if best_score >= adjusted_threshold:
            logger.info(
                f"Duplicate found (fallback): {strategy_id} ≈ {best_id} "
                f"(similarity={best_score:.3f} >= {adjusted_threshold})"
            )
            return True, best_id, best_score

        return False, None, best_score

    def _add_fallback(self, strategy_id: str, summary_text: str):
        """Add strategy to fallback index."""
        vec = self._tfidf_vector(summary_text)
        self._fallback_vectors.append(vec)
        self._fallback_ids.append(strategy_id)

    # ==========================================================================
    # PERSISTENCE
    # ==========================================================================

    def save_index(self):
        """Save the dedup index to disk."""
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)

        if self.use_faiss:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.metadata_path, 'w') as f:
                # Convert int keys to strings for JSON
                json.dump({
                    "metadata": {str(k): v for k, v in self.metadata.items()},
                    "texts": self.strategy_texts,
                }, f, indent=2)
            logger.info(
                f"Saved FAISS index ({self.index.ntotal} vectors) to {self.index_path}"
            )
        else:
            fallback_path = Path(self.metadata_path).with_suffix('.fallback.json')
            with open(fallback_path, 'w') as f:
                json.dump({
                    "ids": self._fallback_ids,
                    "texts": self.strategy_texts,
                }, f, indent=2)
            logger.info(
                f"Saved fallback index ({len(self._fallback_ids)} strategies) "
                f"to {fallback_path}"
            )

    def _load_index(self):
        """Load existing FAISS index from disk."""
        try:
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, 'r') as f:
                data = json.load(f)
                self.metadata = {int(k): v for k, v in data.get("metadata", {}).items()}
                self.strategy_texts = data.get("texts", {})
            logger.info(
                f"Loaded FAISS index ({self.index.ntotal} vectors) from {self.index_path}"
            )
        except Exception as e:
            logger.warning(f"Failed to load FAISS index: {e}. Creating new index.")
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.metadata = {}

    # ==========================================================================
    # BATCH OPERATIONS
    # ==========================================================================

    def deduplicate_batch(self, strategies: List[Dict]) -> List[Dict]:
        """
        Run deduplication on a batch of strategies.

        Updates each strategy dict in-place with dedup results,
        and updates the database.

        Args:
            strategies: List of strategy dicts with 'strategy_id' and 'summary'

        Returns:
            List of unique (non-duplicate) strategies
        """
        unique = []

        for strat in strategies:
            strat_id = strat.get("strategy_id", "unknown")
            summary = strat.get("summary", "")

            if not summary:
                # No summary to dedup against -- keep it
                unique.append(strat)
                continue

            is_dup, dup_of, score = self.check_and_add(strat_id, summary)

            if is_dup:
                strat["is_duplicate"] = True
                strat["duplicate_of"] = dup_of
                strat["similarity_score"] = score
                strat["status"] = "duplicate"

                # Update in DB
                self.db.update_strategy(strat_id, {
                    "is_duplicate": 1,
                    "duplicate_of": dup_of,
                    "similarity_score": score,
                    "status": "duplicate",
                })
                logger.info(f"  [FAIL] Duplicate: {strat_id} ≈ {dup_of} ({score:.3f})")
            else:
                strat["is_duplicate"] = False
                strat["similarity_score"] = score
                unique.append(strat)
                logger.info(f"  [OK] Unique: {strat_id} (best_match={score:.3f})")

        # Auto-save index after batch
        self.save_index()

        return unique

    # ==========================================================================
    # STATS
    # ==========================================================================

    def get_index_size(self) -> int:
        """Get number of strategies in the index."""
        if self.use_faiss:
            return self.index.ntotal
        return len(self._fallback_ids)

    def print_stats(self):
        """Print deduplication statistics."""
        s = self.stats
        mode = "FAISS" if self.use_faiss else "TF-IDF fallback"
        print("\n" + "=" * 50)
        print(f"SEMANTIC DEDUP - STATS ({mode})")
        print("=" * 50)
        print(f"Mode:              {mode}")
        print(f"Index size:        {self.get_index_size()}")
        print(f"Threshold:         {self.threshold}")
        print(f"Checked:           {s['checked']}")
        print(f"Duplicates found:  {s['duplicates_found']}")
        print(f"Unique added:      {s['unique_added']}")
        print("=" * 50)


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    dedup = SemanticDeduplicator()

    print(f"\nDedup mode: {'FAISS' if dedup.use_faiss else 'TF-IDF fallback'}")
    print(f"Threshold: {dedup.threshold}")

    # Test with sample strategies
    test_strategies = [
        ("strat_001", "RSI mean reversion strategy. Buy when RSI drops below 30, sell when RSI rises above 70. Uses 14-period RSI on daily timeframe."),
        ("strat_002", "Moving average crossover trend following. Enter long when 10-period SMA crosses above 50-period SMA. Exit on reverse crossover."),
        ("strat_003", "RSI oversold/overbought mean reversion. Purchase when RSI(14) is under 30, exit when RSI(14) exceeds 70. Daily bars."),  # Near-duplicate of strat_001
        ("strat_004", "Bollinger Band breakout strategy. Buy when price closes above upper band, sell when price drops below middle band."),
        ("strat_005", "SMA crossover momentum strategy. Go long when fast moving average (10) crosses slow moving average (50). Close on opposite cross."),  # Near-duplicate of strat_002
    ]

    print(f"\nTesting {len(test_strategies)} strategies...\n")

    for strat_id, summary in test_strategies:
        is_dup, dup_of, score = dedup.check_and_add(strat_id, summary)
        status = f"DUPLICATE of {dup_of} ({score:.3f})" if is_dup else f"UNIQUE (best={score:.3f})"
        print(f"  {strat_id}: {status}")

    dedup.print_stats()

    # Save and reload test
    dedup.save_index()
    print(f"\nIndex saved. Size: {dedup.get_index_size()}")

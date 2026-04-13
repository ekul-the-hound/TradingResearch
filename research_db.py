# ==============================================================================
# research_db.py
# ==============================================================================
# Database for the Strategy Discovery Pipeline
#
# Stores:
#   - Raw documents scraped from the web
#   - Extracted strategy summaries
#   - Generated Backtrader code
#   - Quality scores, source provenance, dedup status
#   - Lineage fields (origin_source, parent_docs, source_bias)
#
# Follows the same patterns as the existing database.py in TradingLab.
#
# Usage:
#     from research_db import ResearchDatabase
#
#     db = ResearchDatabase()
#     doc_id = db.save_document({...})
#     strat_id = db.save_strategy({...})
#     db.get_strategies_for_backtest()
#
# ==============================================================================

import sqlite3
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from discovery_config import DISCOVERY_CONFIG as cfg


class ResearchDatabase:
    """
    SQLite database for the strategy discovery pipeline.

    Tables:
        documents        - Raw scraped documents (web pages, papers, posts)
        strategies       - Extracted strategies with code and metadata
        extraction_log   - Log of all LLM extraction attempts (successes + failures)
        dedup_log        - Log of deduplication decisions
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = str(db_path or cfg.db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a connection with row_factory enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_database(self):
        """Create all tables and indices."""
        conn = self._get_conn()
        cursor = conn.cursor()

        # --- Documents table: raw scraped content ---
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                title TEXT,
                content TEXT,
                content_hash TEXT UNIQUE,
                source_type TEXT,
                source_bias TEXT,
                search_query TEXT,
                fetch_timestamp TEXT NOT NULL,
                content_length INTEGER,
                status TEXT DEFAULT 'fetched'
            )
        ''')

        # --- Strategies table: extracted + validated strategies ---
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategies (
                strategy_id TEXT PRIMARY KEY,
                doc_id TEXT,
                strategy_name TEXT NOT NULL,
                summary TEXT,
                description TEXT,
                generated_code TEXT,
                code_file_path TEXT,

                -- Provenance / Lineage
                origin_source TEXT DEFAULT 'scraped',
                source_url TEXT,
                source_type TEXT,
                source_bias TEXT,
                parent_docs TEXT,

                -- Quality scoring
                quality_score REAL,
                has_math INTEGER DEFAULT 0,
                has_backtest INTEGER DEFAULT 0,
                has_code INTEGER DEFAULT 0,
                has_explicit_params INTEGER DEFAULT 0,

                -- Extraction metadata
                extraction_model TEXT,
                extraction_timestamp TEXT,
                extraction_cost_estimate REAL DEFAULT 0.0,

                -- Validation
                code_validates INTEGER DEFAULT 0,
                validation_trades INTEGER DEFAULT 0,
                validation_error TEXT,

                -- Deduplication
                is_duplicate INTEGER DEFAULT 0,
                duplicate_of TEXT,
                similarity_score REAL,

                -- Status tracking
                status TEXT DEFAULT 'extracted',
                created_at TEXT NOT NULL,
                updated_at TEXT,

                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
        ''')

        # --- Extraction log: every LLM call attempt ---
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS extraction_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                strategy_id TEXT,
                stage TEXT,
                model_used TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                success INTEGER,
                error_message TEXT,
                duration_seconds REAL,
                timestamp TEXT NOT NULL,

                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
        ''')

        # --- Dedup log: deduplication decisions ---
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dedup_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT,
                compared_to TEXT,
                similarity_score REAL,
                is_duplicate INTEGER,
                timestamp TEXT NOT NULL
            )
        ''')

        # --- Indices ---
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_hash ON documents(content_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_status ON documents(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_strat_status ON strategies(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_strat_quality ON strategies(quality_score)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_strat_origin ON strategies(origin_source)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_strat_duplicate ON strategies(is_duplicate)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_strat_source_bias ON strategies(source_bias)')

        conn.commit()
        conn.close()
        print(f"[OK] Research database initialized at {self.db_path}")

    # ==========================================================================
    # DOCUMENT OPERATIONS
    # ==========================================================================

    @staticmethod
    def _hash_content(content: str) -> str:
        """SHA-256 hash of document content for dedup."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    @staticmethod
    def _generate_id(prefix: str) -> str:
        """Generate a unique ID with timestamp."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        short_hash = hashlib.md5(ts.encode()).hexdigest()[:8]
        return f"{prefix}_{ts}_{short_hash}"

    def document_exists(self, url: str = None, content: str = None) -> bool:
        """Check if document already exists (by URL or content hash)."""
        conn = self._get_conn()
        cursor = conn.cursor()

        if url:
            cursor.execute('SELECT 1 FROM documents WHERE url = ?', (url,))
            if cursor.fetchone():
                conn.close()
                return True

        if content:
            content_hash = self._hash_content(content)
            cursor.execute('SELECT 1 FROM documents WHERE content_hash = ?', (content_hash,))
            if cursor.fetchone():
                conn.close()
                return True

        conn.close()
        return False

    def save_document(self, doc: Dict[str, Any]) -> str:
        """
        Save a scraped document.

        Args:
            doc: Dict with keys: url, title, content, source_type, source_bias, search_query

        Returns:
            doc_id
        """
        doc_id = self._generate_id("doc")
        content = doc.get("content", "")
        content_hash = self._hash_content(content) if content else None

        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT INTO documents (
                    doc_id, url, title, content, content_hash,
                    source_type, source_bias, search_query,
                    fetch_timestamp, content_length, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                doc_id,
                doc.get("url", ""),
                doc.get("title", ""),
                content,
                content_hash,
                doc.get("source_type", "general"),
                doc.get("source_bias", "unknown"),
                doc.get("search_query", ""),
                datetime.now().isoformat(),
                len(content),
                "fetched",
            ))
            conn.commit()
        except sqlite3.IntegrityError as e:
            # Duplicate content hash - document already exists
            conn.close()
            return None
        finally:
            if conn:
                conn.close()

        return doc_id

    def get_unprocessed_documents(self, limit: int = 50) -> List[Dict]:
        """Get documents that haven't been extracted yet."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM documents
            WHERE status = 'fetched'
            ORDER BY fetch_timestamp ASC
            LIMIT ?
        ''', (limit,))
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def update_document_status(self, doc_id: str, status: str):
        """Update document processing status."""
        conn = self._get_conn()
        conn.execute('UPDATE documents SET status = ? WHERE doc_id = ?', (status, doc_id))
        conn.commit()
        conn.close()

    # ==========================================================================
    # STRATEGY OPERATIONS
    # ==========================================================================

    def save_strategy(self, strategy: Dict[str, Any]) -> str:
        """
        Save an extracted strategy.

        Args:
            strategy: Dict with keys matching the strategies table columns.

        Returns:
            strategy_id
        """
        strategy_id = strategy.get("strategy_id") or self._generate_id("strat")
        now = datetime.now().isoformat()

        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO strategies (
                strategy_id, doc_id, strategy_name, summary, description,
                generated_code, code_file_path,
                origin_source, source_url, source_type, source_bias, parent_docs,
                quality_score, has_math, has_backtest, has_code, has_explicit_params,
                extraction_model, extraction_timestamp, extraction_cost_estimate,
                code_validates, validation_trades, validation_error,
                is_duplicate, duplicate_of, similarity_score,
                status, created_at, updated_at
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?
            )
        ''', (
            strategy_id,
            strategy.get("doc_id"),
            strategy.get("strategy_name", "Unknown"),
            strategy.get("summary"),
            strategy.get("description"),
            strategy.get("generated_code"),
            strategy.get("code_file_path"),
            strategy.get("origin_source", "scraped"),
            strategy.get("source_url"),
            strategy.get("source_type"),
            strategy.get("source_bias"),
            json.dumps(strategy.get("parent_docs", [])),
            strategy.get("quality_score"),
            int(strategy.get("has_math", False)),
            int(strategy.get("has_backtest", False)),
            int(strategy.get("has_code", False)),
            int(strategy.get("has_explicit_params", False)),
            strategy.get("extraction_model"),
            strategy.get("extraction_timestamp", now),
            strategy.get("extraction_cost_estimate", 0.0),
            int(strategy.get("code_validates", False)),
            strategy.get("validation_trades", 0),
            strategy.get("validation_error"),
            int(strategy.get("is_duplicate", False)),
            strategy.get("duplicate_of"),
            strategy.get("similarity_score"),
            strategy.get("status", "extracted"),
            strategy.get("created_at", now),
            now,
        ))

        conn.commit()
        conn.close()
        return strategy_id

    def get_strategies_for_backtest(self) -> List[Dict]:
        """Get strategies that passed validation and aren't duplicates."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM strategies
            WHERE code_validates = 1
              AND is_duplicate = 0
              AND status = 'validated'
            ORDER BY quality_score DESC
        ''')
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def get_strategy_by_id(self, strategy_id: str) -> Optional[Dict]:
        """Get a single strategy by ID."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM strategies WHERE strategy_id = ?', (strategy_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_strategies_by_status(self, status: str) -> List[Dict]:
        """Get strategies by status (extracted, validated, backtested, failed)."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM strategies WHERE status = ? ORDER BY created_at DESC',
            (status,)
        )
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def update_strategy(self, strategy_id: str, updates: Dict[str, Any]):
        """Update specific fields on a strategy."""
        if not updates:
            return
        updates["updated_at"] = datetime.now().isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [strategy_id]

        conn = self._get_conn()
        conn.execute(
            f'UPDATE strategies SET {set_clause} WHERE strategy_id = ?',
            values
        )
        conn.commit()
        conn.close()

    # ==========================================================================
    # EXTRACTION LOG
    # ==========================================================================

    def log_extraction(self, log_entry: Dict[str, Any]):
        """Log an LLM extraction attempt."""
        conn = self._get_conn()
        conn.execute('''
            INSERT INTO extraction_log (
                doc_id, strategy_id, stage, model_used,
                prompt_tokens, completion_tokens, success,
                error_message, duration_seconds, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            log_entry.get("doc_id"),
            log_entry.get("strategy_id"),
            log_entry.get("stage", "unknown"),
            log_entry.get("model_used"),
            log_entry.get("prompt_tokens", 0),
            log_entry.get("completion_tokens", 0),
            int(log_entry.get("success", False)),
            log_entry.get("error_message"),
            log_entry.get("duration_seconds"),
            datetime.now().isoformat(),
        ))
        conn.commit()
        conn.close()

    # ==========================================================================
    # DEDUP LOG
    # ==========================================================================

    def log_dedup_check(self, strategy_id: str, compared_to: str,
                        similarity: float, is_dup: bool):
        """Log a deduplication comparison."""
        conn = self._get_conn()
        conn.execute('''
            INSERT INTO dedup_log (
                strategy_id, compared_to, similarity_score,
                is_duplicate, timestamp
            ) VALUES (?, ?, ?, ?, ?)
        ''', (
            strategy_id, compared_to, similarity,
            int(is_dup), datetime.now().isoformat(),
        ))
        conn.commit()
        conn.close()

    # ==========================================================================
    # STATS / REPORTING
    # ==========================================================================

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the discovery pipeline."""
        conn = self._get_conn()
        cursor = conn.cursor()

        stats = {}

        # Document counts
        cursor.execute('SELECT COUNT(*) FROM documents')
        stats["total_documents"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM documents WHERE status = 'fetched'")
        stats["unprocessed_documents"] = cursor.fetchone()[0]

        # Strategy counts
        cursor.execute('SELECT COUNT(*) FROM strategies')
        stats["total_strategies"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM strategies WHERE code_validates = 1")
        stats["validated_strategies"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM strategies WHERE is_duplicate = 1")
        stats["duplicate_strategies"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM strategies WHERE status = 'validated'")
        stats["ready_for_backtest"] = cursor.fetchone()[0]

        # Quality distribution
        cursor.execute('SELECT AVG(quality_score), MIN(quality_score), MAX(quality_score) FROM strategies WHERE quality_score IS NOT NULL')
        row = cursor.fetchone()
        stats["avg_quality"] = row[0]
        stats["min_quality"] = row[1]
        stats["max_quality"] = row[2]

        # Source bias distribution
        cursor.execute('''
            SELECT source_bias, COUNT(*) as cnt
            FROM strategies
            WHERE source_bias IS NOT NULL
            GROUP BY source_bias
            ORDER BY cnt DESC
        ''')
        stats["bias_distribution"] = {row[0]: row[1] for row in cursor.fetchall()}

        # Extraction stats
        cursor.execute('SELECT COUNT(*) FROM extraction_log WHERE success = 1')
        stats["successful_extractions"] = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM extraction_log WHERE success = 0')
        stats["failed_extractions"] = cursor.fetchone()[0]

        conn.close()
        return stats

    def print_stats(self):
        """Print pipeline statistics."""
        stats = self.get_pipeline_stats()
        print("\n" + "=" * 50)
        print("STRATEGY DISCOVERY PIPELINE - STATS")
        print("=" * 50)
        print(f"Documents scraped:       {stats['total_documents']}")
        print(f"  Unprocessed:           {stats['unprocessed_documents']}")
        print(f"Strategies extracted:    {stats['total_strategies']}")
        print(f"  Validated (code ok):   {stats['validated_strategies']}")
        print(f"  Duplicates removed:    {stats['duplicate_strategies']}")
        print(f"  Ready for backtest:    {stats['ready_for_backtest']}")
        if stats.get("avg_quality"):
            print(f"Quality score:           {stats['avg_quality']:.2f} "
                  f"(min={stats['min_quality']:.2f}, max={stats['max_quality']:.2f})")
        if stats.get("bias_distribution"):
            print(f"Source bias distribution: {stats['bias_distribution']}")
        print(f"LLM calls (ok/fail):     {stats['successful_extractions']}/{stats['failed_extractions']}")
        print("=" * 50)


if __name__ == "__main__":
    db = ResearchDatabase()
    db.print_stats()

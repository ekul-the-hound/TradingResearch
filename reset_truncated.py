"""Reset docs whose strategies failed with SyntaxError so they get reprocessed."""
from research_db import ResearchDatabase

db = ResearchDatabase()
conn = db._get_conn()
try:
    rows = conn.execute(
        "SELECT strategy_id, doc_id, strategy_name FROM strategies "
        "WHERE code_validates = 0 AND validation_error LIKE '%SyntaxError%'"
    ).fetchall()
    for sid, doc_id, name in rows:
        conn.execute("DELETE FROM strategies WHERE strategy_id = ?", (sid,))
        conn.execute(
            "UPDATE documents SET status = 'fetched' WHERE doc_id = ?", (doc_id,)
        )
        print(f"Reset: {name} (doc {doc_id})")
    conn.commit()
    print(f"\n{len(rows)} docs queued for reprocessing.")
finally:
    conn.close()
"""Print the generated code of the most recent validation_failed strategy."""
from research_db import ResearchDatabase

db = ResearchDatabase()
conn = db._get_conn()
try:
    row = conn.execute(
        "SELECT strategy_name, validation_error, generated_code FROM strategies "
        "WHERE code_validates = 0 AND validation_error IS NOT NULL "
        "ORDER BY extraction_timestamp DESC LIMIT 1"
    ).fetchone()
    if row:
        name, err, code = row
        print(f"=== {name} ===")
        print(f"ERROR: {err}\n")
        print(code)
    else:
        print("No validation_failed strategies found.")
finally:
    conn.close()
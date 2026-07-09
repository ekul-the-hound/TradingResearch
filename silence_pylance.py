# silence_pylance.py -- run once from project root
import re

SITES = {
    "lineage_tracker.py":  [r"^(\s*import mlflow(?:\.tracking)?)\s*$"],
    "run_backtests.py":    [r"^(\s*from simple_strategy import .+?)\s*$"],
    "run_pipeline.py":     [r"^(\s*from simple_strategy import .+?)\s*$"],
    "test_system.py":      [r"^(\s*from simple_strategy import .+?)\s*$"],
}
TAG = "  # pyright: ignore[reportMissingImports]"

for fname, patterns in SITES.items():
    with open(fname, encoding="utf-8", newline="") as f:
        raw = f.read()
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n") if crlf else raw
    n = 0
    for pat in patterns:
        text, k = re.subn(pat, lambda m: m.group(1) + TAG if TAG not in m.group(0) else m.group(0),
                          text, flags=re.M)
        n += k
    with open(fname, "w", encoding="utf-8", newline="") as f:
        f.write(text.replace("\n", "\r\n") if crlf else text)
    print(f"{fname}: {n} line(s) tagged")
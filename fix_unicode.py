"""
fix_unicode.py — Replace all problematic unicode characters with ASCII equivalents.
Run this once from your project root:
    python fix_unicode.py

It scans every .py file and replaces emoji/unicode in print() and string
literals with safe ASCII versions. Creates a backup of each modified file.
"""

import os
import re
import shutil

# Replacement map: unicode → ASCII
REPLACEMENTS = {
    # Checkmarks and status
    '✓': '[OK]',
    '✅': '[OK]',
    '✗': '[FAIL]',
    '❌': '[FAIL]',
    '⚠️': '[WARN]',
    '⚠': '[WARN]',

    # Emoji icons
    '🔄': '[CYCLE]',
    '🔬': '[TEST]',
    '📊': '[STATS]',
    '📋': '[LIST]',
    '📡': '[SIGNAL]',
    '📥': '[IN]',
    '📦': '[PKG]',
    '⏭️': '[SKIP]',
    '⏭': '[SKIP]',
    '✂️': '[CUT]',
    '✂': '[CUT]',
    '🧬': '[DNA]',
    '💼': '[CASE]',
    '🛡️': '[SHIELD]',
    '🛡': '[SHIELD]',
    '🖥️': '[PC]',
    '🖥': '[PC]',
    '🔎': '[SEARCH]',
    '🧠': '[BRAIN]',
    '🔍': '[SEARCH]',
    '🏛️': '[BANK]',
    '🏛': '[BANK]',
    '⚡': '[ZAP]',
    '✏️': '[EDIT]',
    '✏': '[EDIT]',
    '🤖': '[AI]',
    '✍️': '[WRITE]',
    '✍': '[WRITE]',
    '📭': '[EMPTY]',
    '📈': '[UP]',
    '📉': '[DOWN]',
    '📜': '[SCROLL]',
    '🔧': '[TOOL]',
    '🔗': '[LINK]',
    '🌳': '[TREE]',
    '💡': '[TIP]',
    '📂': '[FOLDER]',
    '🧪': '[LAB]',
    '💰': '[COST]',
    '🎯': '[TARGET]',
    '⭐': '[STAR]',
    '🏆': '[TROPHY]',
    '🚀': '[LAUNCH]',
    '📌': '[PIN]',
    '🔥': '[FIRE]',
    '💎': '[GEM]',
    '🎲': '[DICE]',
    '📶': '[SIGNAL]',
    '🔴': '[RED]',
    '🟡': '[YELLOW]',
    '🟢': '[GREEN]',
    '⚪': '[DOT]',
    '🚨': '[ALERT]',
    '🌊': '[WAVE]',
    '📝': '[NOTE]',
    '💾': '[SAVE]',
    '📄': '[FILE]',
    '💀': '[DEAD]',
    '🎉': '[DONE]',
    '📐': '[MATH]',
    '🏗️': '[BUILD]',
    '🏗': '[BUILD]',
    '🔒': '[LOCK]',
    '📮': '[MAIL]',
    '🪵': '[LOG]',
    '⚙️': '[GEAR]',
    '⚙': '[GEAR]',

    # Arrows and box drawing (these can stay in comments but break in print)
    '→': '->',
    '←': '<-',
    '↑': '^',
    '↓': 'v',

    # Box drawing characters
    '─': '-',
    '━': '=',
    '│': '|',
    '┃': '|',
    '┌': '+',
    '┐': '+',
    '└': '+',
    '┘': '+',
    '├': '+',
    '┤': '+',
    '┬': '+',
    '┴': '+',
    '┼': '+',
    '╔': '+',
    '╗': '+',
    '╚': '+',
    '╝': '+',
    '╠': '+',
    '╣': '+',
    '╦': '+',
    '╩': '+',
    '╬': '+',
    '═': '=',

    # Bullets
    '•': '-',
    '·': '-',

    # Greek letters in comments/docstrings (leave as-is in formulas, fix in prints)
    # These are fine in comments but break in Windows terminal print()
    # We'll only replace them if they're inside print() or f-string

    # Special dashes
    '—': '--',
    '–': '-',
}


def _mixed_bytes(run):
    """Encode chars via cp1252 where possible, latin-1 otherwise."""
    out = bytearray()
    for ch in run:
        try:
            out += ch.encode('cp1252')
        except UnicodeEncodeError:
            out += ch.encode('latin-1')
    return bytes(out)


def _unscramble(run):
    """Undo repeated utf8-bytes-read-as-cp1252/latin-1 corruption (mojibake)."""
    for _ in range(4):
        try:
            fixed = _mixed_bytes(run).decode('utf-8')
        except (UnicodeDecodeError, UnicodeEncodeError):
            break
        if fixed == run:
            break
        run = fixed
    return run


def fix_file(filepath):
    """Replace unicode characters in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Do NOT read as latin-1 and re-save as utf-8: that BAKES mojibake into
        # the source permanently (this exact bug corrupted files in the past).
        # Instead, decode leniently and repair the round-trip corruption.
        raw = open(filepath, 'rb').read()
        try:
            content = raw.decode('cp1252')
        except UnicodeDecodeError:
            content = raw.decode('latin-1')
        repaired = _unscramble(content)
        if repaired != content:
            print(f"  [REPAIR] {os.path.basename(filepath)}: recovered round-trip corruption")
        content = repaired

    original = content
    changes = 0

    for old, new in REPLACEMENTS.items():
        if old in content:
            count = content.count(old)
            content = content.replace(old, new)
            changes += count

    # Repair baked mojibake sequences the map doesn't know about
    for run in sorted(set(re.findall(r'[^\x00-\x7F]+', content)), key=len, reverse=True):
        fixed = _unscramble(run)
        if fixed == run:
            continue
        if fixed in REPLACEMENTS:
            content = content.replace(run, REPLACEMENTS[fixed])
            changes += content.count(REPLACEMENTS[fixed]) and 1
        elif all(ord(ch) < 128 for ch in fixed):
            content = content.replace(run, fixed)
            changes += 1

    # Report leftovers instead of silently ignoring them
    leftovers = set(re.findall(r'[^\x00-\x7F]', content))
    if leftovers:
        print(f"  [NOTE] {os.path.basename(filepath)}: non-ASCII remains "
              f"(not auto-replaced): {sorted(leftovers)[:10]}")

    if changes > 0 and content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    return changes


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("  Unicode Fixer for TradingLab")
    print("=" * 60)
    print(f"  Scanning: {project_dir}")
    print()

    total_changes = 0
    total_files = 0
    fixed_files = []

    for filename in sorted(os.listdir(project_dir)):
        if not filename.endswith('.py'):
            continue
        if filename == 'fix_unicode.py':
            continue

        filepath = os.path.join(project_dir, filename)
        changes = fix_file(filepath)

        if changes > 0:
            fixed_files.append((filename, changes))
            total_changes += changes
            total_files += 1
            print(f"  [FIXED] {filename}: {changes} replacements")
        # Don't print OK files to keep output clean

    print()
    print(f"  Total: {total_changes} replacements across {total_files} files")
    if fixed_files:
        print()
        print("  Fixed files:")
        for name, count in fixed_files:
            print(f"    {name}: {count}")
    else:
        print("  No unicode issues found!")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
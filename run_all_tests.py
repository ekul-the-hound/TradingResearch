#!/usr/bin/env python
# ==============================================================================
# run_all_tests.py
# ==============================================================================
# One command for the whole suite.
#
# Each test file runs in a SEPARATE PROCESS. That costs a second or two of
# interpreter startup and buys two things worth more:
#
#   1. A module that fails at import, or crashes the interpreter, takes down
#      its own file and nothing else. Under a shared process one bad import
#      aborts collection and the rest silently never run -- reported as a
#      smaller total, which looks like success if nobody is counting.
#   2. Module-level state cannot leak between files.
#
# SKIPS COUNT AS FAILURES. A skipped test is a test that did not run, and a
# green summary containing skips is a green summary that checked less than it
# appears to. Same rule the individual suites already apply.
#
#     python run_all_tests.py                 # everything
#     python run_all_tests.py -k portfolio    # filter by filename
#     python run_all_tests.py --list          # show what would run
#     python run_all_tests.py --quiet         # totals only
# ==============================================================================

import argparse
import glob
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

# Files that are not unittest suites despite the name, or that need resources
# unavailable in a normal run. Each needs a stated reason -- an unexplained
# exclusion is how a suite quietly stops being run.
EXCLUDE: Dict[str, str] = {
    'test_api_key.py': 'requires CLAUDE_API_KEY and network',
    'test_data_download.py': 'downloads market data over the network',
    'test_live_trading.py': 'requires a live broker connection',
}

SUMMARY_RE = re.compile(
    r'ran\s+(\d+)\s*\|\s*failures\s+(\d+)\s*\|\s*errors\s+(\d+)\s*\|'
    r'\s*skipped\s+(\d+)')
UNITTEST_RE = re.compile(r'^Ran (\d+) tests? in', re.M)
FAILED_RE = re.compile(r'^FAILED \((.*)\)', re.M)


class Outcome:
    def __init__(self, path: str):
        self.path = path
        self.ran = 0
        self.failures = 0
        self.errors = 0
        self.skipped = 0
        self.seconds = 0.0
        self.returncode = 0
        self.parsed = False
        self.output = ''

    @property
    def ok(self) -> bool:
        # Skips are failures. So is a non-zero exit we could not attribute.
        return (self.returncode == 0 and self.failures == 0
                and self.errors == 0 and self.skipped == 0)

    @property
    def status(self) -> str:
        if self.ok:
            return 'PASS'
        if not self.parsed:
            return 'CRASH'
        if self.skipped and not (self.failures or self.errors):
            return 'SKIPS'
        return 'FAIL'


def discover(pattern: Optional[str]) -> List[str]:
    files = sorted(f for f in glob.glob('test_*.py')
                   if f not in EXCLUDE)
    if pattern:
        files = [f for f in files if pattern.lower() in f.lower()]
    return files


def parse(text: str, out: Outcome) -> None:
    """
    Read a suite's own summary line, falling back to unittest's.

    Suites in this project print 'ran N | failures N | errors N | skipped N'.
    Older ones only emit unittest's default output, so both are handled --
    an unparsed file is reported as CRASH rather than assumed to be zero
    tests, because "no tests found" and "the file exploded" must not look
    the same.
    """
    m = SUMMARY_RE.search(text)
    if m:
        out.ran, out.failures, out.errors, out.skipped = (
            int(g) for g in m.groups())
        out.parsed = True
        return

    m2 = UNITTEST_RE.search(text)
    if m2:
        out.ran = int(m2.group(1))
        out.parsed = True
        f = FAILED_RE.search(text)
        if f:
            for part in f.group(1).split(','):
                part = part.strip()
                if part.startswith('failures='):
                    out.failures = int(part.split('=')[1])
                elif part.startswith('errors='):
                    out.errors = int(part.split('=')[1])
                elif part.startswith('skipped='):
                    out.skipped = int(part.split('=')[1])
        sk = re.search(r'skipped=(\d+)', text)
        if sk and not out.skipped:
            out.skipped = int(sk.group(1))


def run_one(path: str, timeout: int) -> Outcome:
    out = Outcome(path)
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, path], capture_output=True, text=True,
            timeout=timeout)
        out.output = (proc.stdout or '') + (proc.stderr or '')
        out.returncode = proc.returncode
    except subprocess.TimeoutExpired:
        out.output = f'TIMEOUT after {timeout}s'
        out.returncode = -1
    out.seconds = time.time() - t0
    parse(out.output, out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('-k', dest='pattern', default=None,
                    help='only files whose name contains this')
    ap.add_argument('--list', action='store_true',
                    help='show what would run, then exit')
    ap.add_argument('--quiet', action='store_true', help='totals only')
    ap.add_argument('--timeout', type=int, default=600)
    args = ap.parse_args()

    files = discover(args.pattern)
    if not files:
        print('No test files matched.')
        return 1

    if args.list:
        for f in files:
            print(f'  {f}')
        if EXCLUDE:
            print('\nexcluded:')
            for f, why in sorted(EXCLUDE.items()):
                print(f'  {f:34} {why}')
        return 0

    print(f'Running {len(files)} suite(s), one process each.\n')
    results: List[Outcome] = []

    for f in files:
        out = run_one(f, args.timeout)
        results.append(out)
        if not args.quiet:
            bits = f"{out.ran:>4} tests"
            if out.failures:
                bits += f", {out.failures} failed"
            if out.errors:
                bits += f", {out.errors} errors"
            if out.skipped:
                bits += f", {out.skipped} SKIPPED"
            print(f'  [{out.status:<5}] {f:<40} {bits:<34} '
                  f'{out.seconds:5.1f}s')

    total_ran = sum(o.ran for o in results)
    total_fail = sum(o.failures for o in results)
    total_err = sum(o.errors for o in results)
    total_skip = sum(o.skipped for o in results)
    bad = [o for o in results if not o.ok]

    print('\n' + '=' * 72)
    print(f'  suites {len(results)}   tests {total_ran}   '
          f'failures {total_fail}   errors {total_err}   skipped {total_skip}')
    print('=' * 72)

    if not bad:
        print('  ALL GREEN')
        return 0

    print(f'  {len(bad)} suite(s) not clean:\n')
    for o in bad:
        print(f'  --- {o.path} [{o.status}] ---')
        if o.status == 'CRASH':
            # No parseable summary means the file did not get far enough to
            # produce one; the tail of its output is the only evidence.
            tail = [ln for ln in o.output.strip().splitlines() if ln.strip()]
            for ln in tail[-12:]:
                print(f'      {ln}')
        else:
            detail = [ln for ln in o.output.splitlines()
                      if ln.startswith(('FAIL:', 'ERROR:', 'SKIPPED',
                                        '    - ', 'AssertionError'))]
            if not detail:
                # A suite can report skips in its summary without naming them
                # (verbosity=0). Echo the summary rather than printing an
                # empty block, which reads as "not clean, but no reason".
                detail = [ln for ln in o.output.splitlines()
                          if SUMMARY_RE.search(ln)] or ['(no detail emitted)']
            for ln in detail:
                print(f'      {ln.strip()}')
        print()

    if total_skip:
        print('  NOTE: skipped tests are counted as failures. A skipped test')
        print('  is a test that did not run.')
    return 1


if __name__ == '__main__':
    sys.exit(main())

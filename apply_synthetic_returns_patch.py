# ==============================================================================
# apply_synthetic_returns_patch.py
# ==============================================================================
# Phase 0, Item 5 -- synthetic returns in canonical_result.py.
#
# THE FAILURE CHAIN (three defects, not one)
# ------------------------------------------
# 1. FABRICATION -- canonical_result._compute_arrays()
#
#        elif self.total_return_pct != 0 and self.bars_tested > 0:
#            self.returns_synthetic = True
#            daily_r = (1 + total_r) ** (1 / n) - 1
#            rng = np.random.RandomState(seed)
#            self.returns = rng.normal(daily_r, max(daily_vol, 1e-8), n)
#
#    When no trade list is present, the return series is drawn from a normal
#    distribution. These are not approximate returns -- they are random numbers
#    that happen to share a mean and a rough volatility with the summary stats.
#    The seed is derived from strategy_id, so the fabricated series is stable
#    across runs and would survive a reproducibility check unchanged.
#
#    The consequence is not random noise, it is SYSTEMATIC OPTIMISM, because of
#    what consumes this array:
#      - Deflated / Probabilistic Sharpe deflate a Sharpe using the skew and
#        kurtosis of the return series. Gaussian draws have zero skew and zero
#        excess kurtosis -- the single most favourable input possible. A real
#        series with fat tails and negative skew gets penalised; this one never
#        does.
#      - CSCV / PBO measure rank consistency across in-sample and out-of-sample
#        block splits. Draws from a CONSTANT mean and variance are stationary by
#        construction, so every split agrees and PBO collapses toward zero --
#        "no overfitting detected".
#      - Bootstrap and Monte Carlo resampling of i.i.d. noise produce tight,
#        well-behaved confidence intervals.
#
#    The modules whose entire purpose is catching false positives are handed
#    the one input that cannot fail them.
#
# 2. LAUNDERING -- backtest_adapter._aggregate()
#
#        return_arrays = [r.returns for r in results if r.returns is not None ...]
#        all_returns = np.concatenate(return_arrays) if return_arrays else None
#        agg = CanonicalResult(..., returns=all_returns)
#
#    Return arrays from several results are concatenated with no synthetic
#    check, and the aggregate is built WITHOUT carrying returns_synthetic
#    forward. Mix one real result with one synthetic one and the aggregate
#    reports returns_synthetic=False. The provenance is destroyed at exactly
#    the point where it starts to matter.
#
# 3. SKIP TREATED AS PASS -- run_pipeline.py
#
#        if (cr.returns is not None and len(cr.returns) > 30
#                and not getattr(cr, "returns_synthetic", False)):
#            ... monte carlo + bootstrap ...
#            validated.append(cr)
#        else:
#            validated.append(cr)
#
#    This is the only place in the codebase that checks returns_synthetic. It
#    correctly keeps fabricated data out of the statistics -- and then appends
#    the candidate to `validated` anyway. Downstream cannot distinguish "passed
#    validation" from "was never validated".
#
# THE FIX
# -------
#   - Delete the fabrication. No trades -> returns is None, not invented data.
#   - Make provenance structural: returns_source records where the array came
#     from, and require_returns() raises rather than returning something a
#     caller might use by accident.
#   - Propagate provenance through aggregation instead of dropping it.
#   - Stop promoting unvalidated candidates as validated.
#
# An opt-in escape hatch remains for smoke tests: set
# canonical_result.ALLOW_SYNTHETIC_RETURNS = True explicitly. It is off by
# default and still marks the result.
#
# USAGE
#   python apply_synthetic_returns_patch.py --dry-run
#   python apply_synthetic_returns_patch.py
#   python apply_synthetic_returns_patch.py --revert
# ==============================================================================

import argparse
import ast
import glob
import os
import shutil
import sys
from datetime import datetime

BACKUP_SUFFIX = '.synth_bak'

PATCHES = [
    # -------------------------------------------------------------------------
    # canonical_result.py
    # -------------------------------------------------------------------------
    {
        'file': 'canonical_result.py',
        'name': 'Provenance fields + opt-in switch + error type',
        'marker': 'SYNTHETIC-RETURNS-FIX-FIELDS',
        'old': '''    # -- Computed arrays (built from trade list / equity) ------------------
    returns: Optional[np.ndarray] = None          # daily returns (decimal)
    equity_curve: Optional[np.ndarray] = None     # equity values over time
    trade_list: List[Dict] = field(default_factory=list)''',
        'new': '''    # -- Computed arrays (built from trade list / equity) ------------------
    # SYNTHETIC-RETURNS-FIX-FIELDS
    # returns is None when it could not be derived from real trades. It is
    # deliberately None rather than zeros: a caller that uses it by accident
    # should fail loudly, not silently compute a Sharpe of 0/0.
    returns: Optional[np.ndarray] = None          # daily returns (decimal)
    equity_curve: Optional[np.ndarray] = None     # equity values over time
    trade_list: List[Dict] = field(default_factory=list)

    # Where `returns` came from. Structural provenance -- checked by
    # require_returns() rather than relying on every caller remembering to
    # inspect a boolean.
    #   'trade_list'      real, derived from executed trades
    #   'none'            not available
    #   'synthetic'       fabricated (only via ALLOW_SYNTHETIC_RETURNS)
    #   'mixed'           aggregate containing at least one synthetic input
    returns_source: str = 'none'

    # Previously this was attached dynamically inside the fabrication branch,
    # so it did not exist on a normal result at all -- every consumer had to
    # reach for getattr(cr, "returns_synthetic", False) and most did not.
    # Declared here so the attribute always exists and can be relied on.
    returns_synthetic: bool = False ''',
    },
    {
        'file': 'canonical_result.py',
        'name': 'Remove the fabrication from _compute_arrays',
        'marker': 'SYNTHETIC-RETURNS-FIX-COMPUTE',
        'old': '''        elif self.total_return_pct != 0 and self.bars_tested > 0:
            self.returns_synthetic = True
            # Synthesize approximate daily returns from summary stats
            n = max(self.bars_tested, 1)
            total_r = self.total_return_pct / 100
            daily_r = (1 + total_r) ** (1 / n) - 1

            # Add noise scaled to match Sharpe if available
            if self.sharpe_ratio and self.sharpe_ratio != 0:
                daily_vol = abs(daily_r) / max(abs(self.sharpe_ratio / np.sqrt(252)), 1e-6)
            else:
                daily_vol = abs(daily_r) * 2

            import hashlib
            seed = int(hashlib.sha256(str(self.strategy_id).encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            self.returns = rng.normal(daily_r, max(daily_vol, 1e-8), n)

            # Build equity curve
            equity = self.starting_value * np.cumprod(1 + self.returns)
            self.equity_curve = np.concatenate([[self.starting_value], equity])
        else:
            self.returns = np.array([0.0])
            self.equity_curve = np.array([self.starting_value])''',
        'new': '''        elif ALLOW_SYNTHETIC_RETURNS and self.total_return_pct != 0 and self.bars_tested > 0:
            # SYNTHETIC-RETURNS-FIX-COMPUTE
            # OPT-IN ONLY. This branch fabricates a return series by drawing
            # from a normal distribution fitted to the summary statistics. It
            # is not an approximation of what happened -- it is invented data,
            # and it is systematically favourable to the overfitting detectors
            # (zero skew, zero excess kurtosis, stationary by construction).
            # Never enable this ahead of CSCV, DSR, PSR, bootstrap or Monte
            # Carlo. It exists for smoke tests that need an array of the right
            # shape and nothing more.
            self.returns_synthetic = True
            self.returns_source = 'synthetic'
            n = max(self.bars_tested, 1)
            total_r = self.total_return_pct / 100
            daily_r = (1 + total_r) ** (1 / n) - 1

            if self.sharpe_ratio and self.sharpe_ratio != 0:
                daily_vol = abs(daily_r) / max(abs(self.sharpe_ratio / np.sqrt(252)), 1e-6)
            else:
                daily_vol = abs(daily_r) * 2

            import hashlib
            seed = int(hashlib.sha256(str(self.strategy_id).encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            self.returns = rng.normal(daily_r, max(daily_vol, 1e-8), n)

            equity = self.starting_value * np.cumprod(1 + self.returns)
            self.equity_curve = np.concatenate([[self.starting_value], equity])
        else:
            # No trade list. Previously this fabricated a Gaussian series.
            # Now it reports honestly that returns are unavailable.
            self.returns = None
            self.returns_source = 'none'
            self.equity_curve = np.array([self.starting_value])''',
    },
    {
        'file': 'canonical_result.py',
        'name': 'Mark trade-derived returns as real',
        'marker': "returns_source = 'trade_list'",
        'old': '''            eq = self.equity_curve
            if len(eq) > 1:
                self.returns = np.diff(eq) / np.maximum(eq[:-1], 1e-10)
            else:
                self.returns = np.array([0.0])''',
        'new': '''            eq = self.equity_curve
            if len(eq) > 1:
                self.returns = np.diff(eq) / np.maximum(eq[:-1], 1e-10)
                self.returns_source = 'trade_list'
            else:
                self.returns = np.array([0.0])
                self.returns_source = 'trade_list' ''',
    },
    {
        'file': 'canonical_result.py',
        'name': 'require_returns() hard gate + has_real_returns',
        'marker': 'def require_returns',
        'old': '''    # ------------------------------------------------------------------
    # OUTPUT FORMATS (one per consumer)
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:''',
        'new': '''    # ------------------------------------------------------------------
    # RETURNS PROVENANCE
    # ------------------------------------------------------------------
    @property
    def has_real_returns(self) -> bool:
        """True only if `returns` was derived from executed trades."""
        return (self.returns is not None
                and len(self.returns) > 0
                and self.returns_source == 'trade_list'
                and not self.returns_synthetic)

    def require_returns(self, purpose: str = 'statistical analysis',
                        min_length: int = 0):
        """
        Return the real return series, or raise.

        Call this instead of touching `.returns` anywhere a statistic is
        computed -- CSCV, PBO, Deflated/Probabilistic Sharpe, bootstrap,
        Monte Carlo, walk-forward. The point is that fabricated or missing
        data cannot reach those modules by omission.

        Raises:
            SyntheticReturnsError: returns are missing, synthetic, or mixed.
            ValueError: series is shorter than min_length.
        """
        if self.returns is None or len(self.returns) == 0:
            raise SyntheticReturnsError(
                f"{purpose} requires a real return series, but {self.strategy_id} "
                f"has none. Re-run the backtest with trade extraction enabled so "
                f"trade_list is populated."
            )
        if self.returns_synthetic or self.returns_source in ('synthetic', 'mixed'):
            raise SyntheticReturnsError(
                f"{purpose} refused for {self.strategy_id}: return series is "
                f"'{self.returns_source}', not derived from executed trades. "
                f"Fabricated returns are Gaussian by construction, which makes "
                f"overfitting detectors report the most favourable possible "
                f"result. Re-run with trade extraction enabled."
            )
        if min_length and len(self.returns) < min_length:
            raise ValueError(
                f"{purpose} needs at least {min_length} observations for "
                f"{self.strategy_id}; only {len(self.returns)} available."
            )
        return self.returns

    # ------------------------------------------------------------------
    # OUTPUT FORMATS (one per consumer)
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:''',
    },

    # -------------------------------------------------------------------------
    # backtest_adapter.py -- stop laundering provenance through aggregation
    # -------------------------------------------------------------------------
    {
        'file': 'backtest_adapter.py',
        'name': 'Aggregation: carry provenance instead of dropping it',
        'marker': 'SYNTHETIC-RETURNS-FIX-AGGREGATE',
        'old': '''        # Concatenate return arrays
        return_arrays = [r.returns for r in results if r.returns is not None and len(r.returns) > 0]
        all_returns = np.concatenate(return_arrays) if return_arrays else None
''',
        'new': '''        # SYNTHETIC-RETURNS-FIX-AGGREGATE
        # Previously this concatenated every non-empty returns array and built
        # the aggregate without carrying returns_synthetic forward, so mixing
        # one real result with one fabricated one produced an aggregate that
        # reported itself as clean. Provenance was destroyed exactly where it
        # started to matter. Synthetic inputs are now excluded and the
        # aggregate records that it is incomplete.
        usable, n_synthetic = [], 0
        for r in results:
            if r.returns is None or len(r.returns) == 0:
                continue
            if getattr(r, 'returns_synthetic', False) or getattr(r, 'returns_source', '') == 'synthetic':
                n_synthetic += 1
                continue
            usable.append(r.returns)

        all_returns = np.concatenate(usable) if usable else None
        agg_returns_source = 'none'
        if usable:
            agg_returns_source = 'mixed' if n_synthetic else 'trade_list'
        if n_synthetic:
            print(f"[WARN] {name}: excluded {n_synthetic} synthetic return series "
                  f"from aggregation; aggregate marked '{agg_returns_source}'")
''',
    },
    {
        'file': 'backtest_adapter.py',
        'name': 'Aggregation: set returns_source on the aggregate',
        'marker': 'returns_source=agg_returns_source',
        'old': '''            profit_factor=float(np.mean(pfs)) if pfs else None,
            returns=all_returns,
        )
        return agg''',
        'new': '''            profit_factor=float(np.mean(pfs)) if pfs else None,
            returns=all_returns,
            returns_source=agg_returns_source,
        )
        return agg''',
    },

    # -------------------------------------------------------------------------
    # run_pipeline.py -- stop treating "skipped" as "validated"
    # -------------------------------------------------------------------------
    {
        'file': 'run_pipeline.py',
        'name': 'Do not promote unvalidated candidates as validated',
        'marker': 'SYNTHETIC-RETURNS-FIX-PIPELINE',
        'old': '''                    cr.strategy_params["mc_mean_sharpe"] = getattr(mc, "sharpe_ratio_mean", 0)
                    cr.strategy_params["bootstrap_ci_low"] = getattr(bs, "ci_lower", 0)
                    validated.append(cr)
                else:
                    validated.append(cr)''',
        'new': '''                    cr.strategy_params["mc_mean_sharpe"] = getattr(mc, "sharpe_ratio_mean", 0)
                    cr.strategy_params["bootstrap_ci_low"] = getattr(bs, "ci_lower", 0)
                    cr.strategy_params["validation_status"] = "validated"
                    validated.append(cr)
                else:
                    # SYNTHETIC-RETURNS-FIX-PIPELINE
                    # This branch used to append the candidate unchanged, so a
                    # strategy that was never validated became indistinguishable
                    # from one that passed. Tag it instead.
                    if cr.returns is None or len(cr.returns) == 0:
                        why = "no return series (trade extraction disabled?)"
                    elif getattr(cr, "returns_synthetic", False) or getattr(cr, "returns_source", "") in ("synthetic", "mixed"):
                        why = f"return series is '{getattr(cr, 'returns_source', 'synthetic')}', not trade-derived"
                    else:
                        why = f"only {len(cr.returns)} observations, need > 30"
                    cr.strategy_params["validation_status"] = "skipped"
                    cr.strategy_params["validation_skipped_reason"] = why
                    self._log(f"  [WARN] {cr.strategy_id} NOT validated: {why}")
                    validated.append(cr)''',
    },

    # -------------------------------------------------------------------------
    # test_integration.py -- CR.05 pinned the behaviour we just removed
    # -------------------------------------------------------------------------
    {
        'file': 'test_integration.py',
        'name': 'CR.05: pin the absence of fabrication, not its presence',
        'marker': 'SYNTHETIC-RETURNS-FIX-CR05',
        'old': '''def test_cr_05_returns_synthetic():
    raw = {"strategy_name": "test", "total_return_pct": 20, "sharpe_ratio": 1.5,
           "max_drawdown_pct": 10, "total_trades": 30, "bars_tested": 252,
           "starting_value": 10000}
    cr = CanonicalResult.from_backtest(raw, strategy_id="synth")
    assert cr.returns is not None
    assert len(cr.returns) == 252''',
        'new': '''def test_cr_05_returns_synthetic():
    # SYNTHETIC-RETURNS-FIX-CR05
    # This test used to assert that a result with NO trade list still produced
    # a 252-element return series. That series came from rng.normal() -- data
    # the overfitting detectors cannot fail, because Gaussian draws have no
    # skew, no excess kurtosis, and are stationary by construction. The pin is
    # inverted: summary statistics alone must NOT yield a return series.
    raw = {"strategy_name": "test", "total_return_pct": 20, "sharpe_ratio": 1.5,
           "max_drawdown_pct": 10, "total_trades": 30, "bars_tested": 252,
           "starting_value": 10000}
    cr = CanonicalResult.from_backtest(raw, strategy_id="synth")
    assert cr.returns is None, "no trade list must not fabricate returns"
    assert cr.returns_source == "none"
    assert not cr.has_real_returns''',
    },
]

# Module-level switch, inserted near the top of canonical_result.py.
SWITCH_PATCH = {
    'file': 'canonical_result.py',
    'name': 'ALLOW_SYNTHETIC_RETURNS switch + SyntheticReturnsError',
    'marker': 'ALLOW_SYNTHETIC_RETURNS',
    'old': '''@dataclass
class CanonicalResult:''',
    'new': '''# ==============================================================================
# SYNTHETIC RETURNS SWITCH
# ==============================================================================
# Off by default. When False, a result with no trade list reports
# returns=None instead of fabricating a Gaussian series from summary stats.
#
# Enable ONLY for smoke tests that need an array of the right shape:
#     import canonical_result
#     canonical_result.ALLOW_SYNTHETIC_RETURNS = True
#
# Never enable it upstream of CSCV, PBO, Deflated Sharpe, Probabilistic
# Sharpe, bootstrap, Monte Carlo or walk-forward. Fabricated returns are
# Gaussian by construction, so those tests return the most favourable
# possible answer regardless of whether the strategy has any edge.
ALLOW_SYNTHETIC_RETURNS = False


class SyntheticReturnsError(RuntimeError):
    """Raised when a statistic is requested on missing or fabricated returns."""


@dataclass
class CanonicalResult:''',
}

POST_CONDITIONS = [
    ('canonical_result.py', 'ALLOW_SYNTHETIC_RETURNS = False', 'switch not added or not defaulted off'),
    ('canonical_result.py', 'class SyntheticReturnsError', 'error type not defined'),
    ('canonical_result.py', 'def require_returns', 'hard gate not added'),
    ('canonical_result.py', 'def has_real_returns', 'provenance property not added'),
    ('canonical_result.py', 'returns_source: str', 'provenance field not added'),
    ('backtest_adapter.py', 'returns_source=agg_returns_source', 'aggregate does not carry provenance'),
    ('run_pipeline.py', 'validation_skipped_reason', 'pipeline still conflates skipped with validated'),
]

ABSENT_CONDITIONS = [
    ('canonical_result.py', 'elif self.total_return_pct != 0 and self.bars_tested > 0:',
     'fabrication branch still runs unconditionally'),
]


def read_text(path):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        raw = f.read()
    return raw.replace('\r\n', '\n'), ('\r\n' in raw)


def write_text(path, text, crlf):
    out = text.replace('\n', '\r\n') if crlf else text
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(out)


def verify_syntax(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        return True, None
    except SyntaxError as e:
        return False, f"line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


def apply_patches(project_dir, dry_run=False):
    all_patches = [SWITCH_PATCH] + PATCHES
    by_file = {}
    for p in all_patches:
        by_file.setdefault(p['file'], []).append(p)

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backups = {}
    failed_any = False

    for filename, patches in by_file.items():
        path = os.path.join(project_dir, filename)
        print(f"\n{'=' * 70}")
        print(f"FILE: {filename}")
        print('=' * 70)

        if not os.path.exists(path):
            print(f"  [FAIL] Not found: {path}")
            failed_any = True
            continue

        text, crlf = read_text(path)
        applied, skipped, failed = [], [], []

        for p in patches:
            if p['marker'] in text:
                skipped.append(p['name'])
                continue
            count = text.count(p['old'])
            if count == 0:
                failed.append((p['name'], 'anchor not found - file differs from the snapshot'))
                continue
            if count > 1:
                failed.append((p['name'], f'anchor matched {count} times - ambiguous, refusing'))
                continue
            text = text.replace(p['old'], p['new'], 1)
            applied.append(p['name'])

        for n in applied:
            print(f"  [APPLY] {n}")
        for n in skipped:
            print(f"  [SKIP]  {n} (already patched)")
        for n, why in failed:
            print(f"  [FAIL]  {n}\n          {why}")

        if failed:
            print(f"\n  Refusing to partially patch {filename}. File unchanged.")
            failed_any = True
            continue
        if not applied:
            print("  Nothing to write.")
            continue
        if dry_run:
            print(f"  [DRY-RUN] Would write {len(applied)} change(s). No file modified.")
            continue

        backup = f"{path}{BACKUP_SUFFIX}.{stamp}"
        shutil.copy2(path, backup)
        backups[filename] = backup
        print(f"  [BACKUP] {os.path.basename(backup)}")

        write_text(path, text, crlf)

        ok, err = verify_syntax(path)
        if ok:
            print("  [VERIFY] Syntax OK")
        else:
            print(f"  [VERIFY] SYNTAX ERROR - {err}")
            print("  [ROLLBACK] Restoring from backup")
            shutil.copy2(backup, path)
            failed_any = True

    if dry_run or failed_any:
        return not failed_any

    # Structural post-conditions across all touched files.
    problems = []
    for filename, needle, msg in POST_CONDITIONS:
        path = os.path.join(project_dir, filename)
        if not os.path.exists(path):
            continue
        txt, _ = read_text(path)
        if needle not in txt:
            problems.append(f"{filename}: {msg}")
    for filename, needle, msg in ABSENT_CONDITIONS:
        path = os.path.join(project_dir, filename)
        if not os.path.exists(path):
            continue
        txt, _ = read_text(path)
        if needle in txt:
            problems.append(f"{filename}: {msg}")

    print(f"\n{'=' * 70}")
    if problems:
        print("  [VERIFY] POST-CONDITIONS FAILED:")
        for p in problems:
            print(f"           - {p}")
        for filename, backup in backups.items():
            shutil.copy2(backup, os.path.join(project_dir, filename))
            print(f"  [ROLLBACK] {filename}")
        return False

    print(f"  [VERIFY] Post-conditions OK ({len(POST_CONDITIONS)} checked)")
    return True


def revert(project_dir):
    print("\nREVERT")
    print("=" * 70)
    done = False
    for filename in sorted({p['file'] for p in [SWITCH_PATCH] + PATCHES}):
        path = os.path.join(project_dir, filename)
        bks = sorted(glob.glob(f"{path}{BACKUP_SUFFIX}.*"))
        if not bks:
            print(f"  [SKIP] No backup for {filename}")
            continue
        shutil.copy2(bks[-1], path)
        print(f"  [OK] {filename}  <-  {os.path.basename(bks[-1])}")
        done = True
    if not done:
        print("\n  Nothing to revert.")
    return done


def main():
    ap = argparse.ArgumentParser(description="Remove synthetic-returns fabrication")
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--revert', action='store_true')
    ap.add_argument('--dir', default='.')
    args = ap.parse_args()

    project_dir = os.path.abspath(args.dir)

    print("=" * 70)
    print("SYNTHETIC RETURNS FIX - PATCHER")
    print("=" * 70)
    print(f"Project: {project_dir}")
    if args.dry_run:
        print("Mode:    DRY RUN (no files will be modified)")

    if args.revert:
        revert(project_dir)
        return 0

    ok = apply_patches(project_dir, dry_run=args.dry_run)

    print(f"\n{'=' * 70}")
    if args.dry_run:
        print("DRY RUN COMPLETE - re-run without --dry-run to apply")
    elif ok:
        print("PATCH COMPLETE")
        print("=" * 70)
        print("\nNEXT:")
        print("  python test_synthetic_returns.py")
        print("  python test_integration.py     (CR.05 now pins the OPPOSITE behaviour)")
        print("  python test_system.py")
        print("\nEXPECT: strategies backtested without trade extraction now report")
        print("returns=None and get skipped by validation with a logged reason,")
        print("instead of quietly passing on fabricated data. If a lot of them")
        print("appear, that is the real state of the pipeline becoming visible.")
    else:
        print("PATCH INCOMPLETE - see failures above.")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

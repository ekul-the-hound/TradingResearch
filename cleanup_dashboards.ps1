<#
    cleanup_dashboards.ps1  --  TradingResearch repo cleanup
    ------------------------------------------------------------------
    Removes the superseded Python dashboards and dead scratch files that
    the new React dashboard replaces. Verified safe: none of the TIER 1/2
    files below are imported by any core pipeline module (backtester,
    run_pipeline, canonical_result, ftmo_compliance, etc.).

    USAGE (from the repo root, PowerShell):
        .\cleanup_dashboards.ps1              # DRY RUN — lists what would go, deletes nothing
        .\cleanup_dashboards.ps1 -Execute     # actually delete (via git rm so it's revertable)

    Everything is removed with `git rm`, so `git checkout -- <file>` or a
    branch reset brings anything back. Run it on a branch to be safe:
        git checkout -b cleanup/remove-old-dashboards
#>

param([switch]$Execute)

$ErrorActionPreference = 'Stop'

# ---- TIER 1: superseded Python dashboards (0 real importers) -------------
$dashboards = @(
    'dashboard_react.py',
    'react_dashboard2.py',
    'dashboard_for_now.py',
    'dashboard_vizro.py',
    'dashboard_compare_page.py',
    'dashboard_ftmo_panel.py',
    'dashboard_portfolio_panel.py',
    'page_sources.py',                 # source-extraction page for react_dashboard2 only
    'rxconfig.py',
    'setup_reflex_dashboard.bat',
    'requirements_dashboard.txt'
)

# ---- TIER 1b: tests that only covered the deleted dashboards -------------
$dashboardTests = @(
    'test_dashboard_ftmo_panel.py',
    'test_dashboard_portfolio_panel.py'
)

# ---- TIER 2: one-shot patchers already applied + dead scratch files ------
#      (kept separate so you can opt out; these are NOT imported anywhere)
$scratch = @(
    'patch_asset_type.py',
    'patch_mutation_indicators.py',
    'patch_mutation_source.py',
    'patch_step2_sysmodules.py',
    'fix_broken_variants.py',
    'fix_unicode.py',
    'silence_pylance.py',
    'reset_truncated.py',
    'dumb_failed.py',
    'hboard_ftmo_patch.py__',
    'failed_strategy.txt',
    'TradingLab_log.txt',
    'discovery_log.txt'
)

# NOTE: page_sources.py is Tier 1 only if you confirm you don't want the
# source-extraction UI. If unsure, comment it out above.

function Remove-Group($title, $files) {
    Write-Host ""
    Write-Host "== $title ==" -ForegroundColor Cyan
    foreach ($f in $files) {
        if (Test-Path $f) {
            if ($Execute) {
                git rm --quiet -- $f
                Write-Host "  removed  $f" -ForegroundColor Yellow
            } else {
                Write-Host "  would remove  $f"
            }
        } else {
            Write-Host "  (absent)      $f" -ForegroundColor DarkGray
        }
    }
}

if (-not (Test-Path '.git')) {
    Write-Host "Not a git repo root. cd to the repo root first." -ForegroundColor Red
    exit 1
}

Remove-Group "TIER 1  superseded dashboards"          $dashboards
Remove-Group "TIER 1b superseded dashboard tests"     $dashboardTests
Remove-Group "TIER 2  one-shot patchers + scratch"    $scratch

Write-Host ""
if ($Execute) {
    Write-Host "Done. Review with:  git status" -ForegroundColor Green
    Write-Host "Undo anything with: git checkout -- <file>   (before commit)" -ForegroundColor Green
    Write-Host "Commit when happy:  git commit -m 'Remove superseded Python dashboards and scratch files'"
} else {
    Write-Host "DRY RUN. Re-run with -Execute to delete via git rm." -ForegroundColor Green
}

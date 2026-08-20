# TradingLab Dashboard — Visual Critique Pass

> Prompt 16 deliverable. Structured self-critique against `CLAUDE.md` §5–§6
> (visual system + anti-slop) and the design system, based on an audit of the
> rendered markup and CSS. Clear wins were implemented in this pass; remaining
> items are logged with a recommendation and severity.

## Method

No live screenshots were available in this environment, so the critique is
grounded in the actual DOM structure, the CSS token usage, and the component
composition rather than pixels. Where a judgment needs a real render (fine
spacing rhythm, optical alignment), it is flagged as **needs-eyes** rather than
asserted.

---

## 1. Anti-slop compliance (CLAUDE.md §6) — PASS

Audited the full CSS/TSX for each forbidden pattern:

| Forbidden pattern                     | Found? | Notes |
| ------------------------------------- | ------ | ----- |
| Purple→blue gradients                 | No     | Only gradient is the loading-skeleton shimmer (legitimate). |
| Glassmorphism / backdrop blur         | No     | None. |
| "Welcome back" hero copy              | No     | Pages open with title + question subtitle. |
| AI-confidence gauges                  | No     | None. |
| Crypto-exchange visual language       | No     | None. |
| Fake real-time tickers                | No     | Execution is explicitly offline; no tickers anywhere. |
| Decorative 3D/blobs/stock art         | No     | Text glyphs only for nav/status. |
| Uniform rounded-card grid             | No     | Panels are differentiated (funnel, run panel, tables, strips). |
| Chart-only panels without context     | No     | Every `ChartFrame` carries title, unit, time basis, and a status. |
| Isolated metrics without basis        | No     | `MetricValue` always pairs a value with a label; missing → UNKNOWN/UNAVAILABLE. |
| Green/red-only signaling              | No     | Every status carries a glyph + text (`StatusChip`). |
| Excessive animation / glow / pills    | No     | One shimmer; reduced-motion honored. |

No radius exceeds 6px; no shadows; no hardcoded hex in components (all tokens).

---

## 2. Visual system adherence (CLAUDE.md §5) — PASS

- **Layout:** fixed 232px nav, 44px status bar, 12-col grid, 8px spacing scale,
  6px max radius, 1px borders, minimal shadows — all present in `shell.css` and
  the token file.
- **Color:** exact tokens from the spec are used via CSS variables; status
  colors map to pass/fail/warn/info/research consistently.
- **Numerals:** tabular numerals applied globally and on every metric/table.
- **Density:** compact table rows, tight strips, no oversized hero content.

The interface reads as an institutional research terminal, not a retail
dashboard — the stated goal.

---

## 3. Wins implemented in this pass

1. **Micro-label legibility.** The mini-histogram axis labels were 8px — below a
   comfortable floor even for dense UI. Bumped to 9px in the health strip and
   discovery quality bars.
2. **Chart / distribution accessibility.** The mini-bar histograms and the
   quality distribution were `aria-hidden` with no text alternative; screen
   readers got nothing. Added `sr-only` textual summaries of each distribution.
3. **Chart accessible name.** `LineChart` now announces the series it plots
   (e.g. "Line chart of Net and Gross") instead of a generic label.

(These are in addition to the Prompt-15 pass: skip link, roving tab focus,
keyboard rows, reduced-motion, contrast bump, route-change focus.)

---

## 4. Remaining items (logged, not blocking)

| # | Item | Severity | Recommendation |
| - | ---- | -------- | -------------- |
| 1 | Spacing rhythm between stacked panels uses ad-hoc `height` spacers in a few pages rather than a single vertical-stack utility. | Low | Introduce a `.tl-stack` gap utility and replace the spacer divs. Cosmetic; **needs-eyes** to tune. |
| 2 | The correlation heatmap color steps are threshold-based, not a continuous scale. | Low | Fine for 2–5 strategies; revisit if portfolios grow. |
| 3 | Long dense tables rely on horizontal scroll below ~1100px. | Low | Acceptable for a desktop-first terminal; a column-priority hide could be added later. |
| 4 | `LineChart` has no per-point focus/tooltip. | Medium (future) | Add hover crosshair + value readout when real series are wired; not needed for fixtures. |
| 5 | Empty/loading/unavailable states are consistent but not yet visually A/B-checked against real data density. | Low | **needs-eyes** once the SQLite bridge feeds real rows. |
| 6 | Some pages (Data & Integrity, Challenge Readiness) are long single-column scrolls. | Low | Consider anchored sub-nav if they grow; fine at current length. |

None of these compromise the truth-telling requirements or the anti-slop rules.

---

## 5. Truth-telling review (the product's core) — PASS

Spot-checked that honesty holds visually across states:

- Missing metrics render **UNKNOWN / N/A / ≈ PROXY**, never blank or zero.
- FTMO fit shows **PROXY**, never an unconditional green pass.
- Consistency headroom shows **"Configuration required"**, no fabricated number.
- Execution shows a prominent **EXECUTION OFFLINE** state with no live P&L.
- Dependencies show **NOT CHECKED**, never a fake green OK.
- Synthetic-risk returns are flagged red at the row, ledger, and detail levels.
- DEV FIXTURE badges appear on every data panel in mock mode and disappear when
  the read-only SQLite bridge is connected.

---

## 6. Verdict

The dashboard meets the CLAUDE.md visual system and anti-slop rules, and — more
importantly for this product — it never trades honesty for polish. The clear
wins from this pass were the micro-label size and the chart/distribution
accessibility gaps. Remaining items are low-severity refinements best tuned
against a live render with real data, and are logged above rather than
speculatively "fixed."

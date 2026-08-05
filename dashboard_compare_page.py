# ==============================================================================
# dashboard_compare_page.py
# ==============================================================================
# The "Compare & Portfolio" page.
#
# WHY A FACTORY INSTEAD OF A MODULE-LEVEL COMPONENT
# -------------------------------------------------
# The page needs react_dashboard2's theme dict and UI helpers (_card, _tbl,
# _badge...). Importing them from here while react_dashboard2 imports this
# module is a cycle. So the page is built by make_page(), which takes those
# helpers as arguments. react_dashboard2 calls it once after its helpers are
# defined:
#
#     import dashboard_compare_page
#     PgCompare = dashboard_compare_page.make_page(
#         component=component, html=html, hooks=hooks, T=T,
#         card=_card, tbl=_tbl, badge=_badge, metric=_metric,
#         title=_title, empty=_empty, grid=_grid, col=_col,
#         db_results=DB_RESULTS,
#     )
#
# Four more lines register it in NAV / TITLES / PAGES. Nothing else in the
# 1792-line file changes.
#
# All data logic lives in dashboard_portfolio_panel.py. This file only draws.
# ==============================================================================

from __future__ import annotations

import dashboard_portfolio_panel as PANEL

try:
    import firm_rules
    _RULES_OK = True
except Exception:                                          # pragma: no cover
    _RULES_OK = False


def make_page(component, html, hooks, T, card, tbl, badge, metric,
              title, empty, grid, col, db_results: str):
    """Build the PgCompare component with the host dashboard's helpers."""

    def _style(**kw):
        return kw

    # ------------------------------------------------------------------
    # small local widgets
    # ------------------------------------------------------------------

    def _label(text, color=None, size='11px'):
        return html.span({"style": {"color": color or T["dim"],
                                    "fontSize": size}}, text)

    def _field_row(f, on_change):
        border = T["red"] if f.error else T["border"]
        return html.div({"style": {"marginBottom": "10px"}},
            html.div({"style": {"display": "flex", "justifyContent": "space-between",
                                "alignItems": "baseline", "marginBottom": "3px"}},
                _label(f.label, T["muted"], "12px"),
                _label(f.kind, T["faint"], "10px")),
            html.input({
                "value": "" if f.value is None else str(f.value),
                "onChange": lambda e, n=f.name: on_change(n, e["target"]["value"]),
                "style": {"width": "100%", "padding": "6px 9px",
                          "backgroundColor": T["elevated"], "color": T["text"],
                          "border": f"1px solid {border}", "borderRadius": "6px",
                          "fontSize": "12px", "boxSizing": "border-box"}}),
            html.p({"style": {"margin": "3px 0 0", "fontSize": "10px",
                              "color": T["red"] if f.error else T["faint"]}},
                   f.error or f.help))

    def _capability_row(t, on_toggle):
        """
        Locked capabilities render disabled WITH their reason.

        A greyed control that does not say why reads as a bug. Saying why
        turns it into documentation of what the engine actually models.
        """
        return html.div({"style": {
                "display": "flex", "alignItems": "flex-start", "gap": "8px",
                "padding": "6px 8px", "marginBottom": "3px", "borderRadius": "6px",
                "backgroundColor": T["elevated"] if not t.locked else "transparent",
                "opacity": "0.55" if t.locked else "1"}},
            html.input({"type": "checkbox", "checked": t.enabled,
                        "disabled": t.locked,
                        "onChange": (lambda e, c=t.capability: on_toggle(c))
                                    if not t.locked else (lambda e: None),
                        "style": {"marginTop": "2px",
                                  "cursor": "not-allowed" if t.locked else "pointer"}}),
            html.div({"style": {"flex": "1"}},
                html.div({"style": {"display": "flex", "alignItems": "center", "gap": "6px"}},
                    html.span({"style": {"fontSize": "12px",
                                         "color": T["dim"] if t.locked else T["text"]}}, t.label),
                    badge("NOT MODELLED", T["amber"]) if t.locked else ""),
                html.p({"style": {"margin": "2px 0 0", "fontSize": "10px",
                                  "color": T["faint"], "lineHeight": "1.35"}},
                       t.reason) if t.locked else ""))

    def _delta_cell(d):
        if d is None or d.raw is None:
            return html.span({"style": {"color": T["faint"]}}, "--")
        color = T["dim"] if d.better is None else (T["green"] if d.better else T["red"])
        arrow = "" if d.better is None else ("^" if d.better else "v")
        return html.span({"style": {"color": color, "fontWeight": "600"}},
                         f"{d.text} {arrow}".strip())

    # ------------------------------------------------------------------
    # the page
    # ------------------------------------------------------------------

    @component
    def PgCompare():
        selected, set_selected = hooks.use_state(())
        profile_key, set_profile_key = hooks.use_state("ftmo")
        form_vals, set_form_vals = hooks.use_state({})
        caps, set_caps = hooks.use_state(None)
        account, set_account = hooks.use_state(100_000.0)
        overlap, set_overlap = hooks.use_state("intersection")

        candidates = PANEL.list_candidates(db_results)

        # -- resolve the active firm profile ---------------------------
        if not _RULES_OK:
            return empty("firm_rules.py could not be imported.", "[ERR]")

        base = firm_rules.load_profile(profile_key)
        if form_vals or caps is not None:
            rules, fields = PANEL.apply_firm_form(
                form_vals, capabilities=caps, base=base)
            if rules is None:
                rules = base
        else:
            rules = base
            fields, _ = PANEL.build_firm_form(base)
        _, toggles = PANEL.build_firm_form(rules)
        status = PANEL.firm_status_line(rules)

        def on_field(name, value):
            set_form_vals({**form_vals, name: value})

        def on_toggle(capability):
            current = list(caps if caps is not None
                           else [c.value for c in rules.required_capabilities])
            if capability in current:
                current.remove(capability)
            else:
                current.append(capability)
            set_caps(current)

        def on_pick(bid):
            s = set(selected)
            s.discard(bid) if bid in s else s.add(bid)
            set_selected(tuple(sorted(s)))

        # -- load + merge ----------------------------------------------
        results, failures = PANEL.load_selection(db_results, selected)
        merged = None
        merge_info = None
        if len(results) >= 2:
            merge_info = PANEL.try_merge(
                results, rules, account_size=float(account), overlap=overlap)
            if merge_info.get("ok"):
                merged = merge_info["canonical"]

        table = PANEL.build_comparison(results, portfolio=merged)
        for bid, why in failures:
            table.columns.append(PANEL.unavailable_column(f"bt{bid}", why))

        # ==============================================================
        # LEFT: selector + firm rules
        # ==============================================================
        selector = card(
            title("Strategies", "[PICK]"),
            html.p({"style": {"color": T["faint"], "fontSize": "11px",
                              "margin": "-8px 0 12px"}},
                   "Merging needs persisted trades. Backtests without them "
                   "cannot be selected."),
            *([empty("No backtests found.", "[EMPTY]")] if not candidates else
              [html.div({"style": {"maxHeight": "260px", "overflowY": "auto"}},
                *[html.div({"style": {
                        "display": "flex", "alignItems": "center", "gap": "8px",
                        "padding": "6px 8px", "borderRadius": "6px",
                        "marginBottom": "2px",
                        "backgroundColor": T["elevated"] if c["id"] in selected else "transparent",
                        "opacity": "0.45" if not c["has_trades"] else "1"}},
                    html.input({"type": "checkbox",
                                "checked": c["id"] in selected,
                                "disabled": not c["has_trades"],
                                "onChange": (lambda e, b=c["id"]: on_pick(b))
                                            if c["has_trades"] else (lambda e: None)}),
                    html.div({"style": {"flex": "1", "minWidth": "0"}},
                        html.div({"style": {"fontSize": "12px", "color": T["text"]}},
                                 c["label"]),
                        _label(c["blocked_reason"] or
                               f"{c['symbol']} {c['timeframe']} - "
                               f"{c['n_trades_persisted']} trades", T["faint"], "10px")),
                ) for c in candidates])]))

        firm_card = card(
            html.div({"style": {"display": "flex", "justifyContent": "space-between",
                                "alignItems": "center", "marginBottom": "12px"}},
                title(f"Firm Rules", "[RULE]"),
                badge("ALL RULES MODELLED" if status["complete"]
                      else f"{status['n_unchecked']} NOT CHECKED",
                      T["green"] if status["complete"] else T["amber"])),
            html.div({"style": {"display": "flex", "gap": "6px", "marginBottom": "14px"}},
                *[html.button({
                    "onClick": lambda e, k=k: (set_profile_key(k),
                                               set_form_vals({}), set_caps(None)),
                    "style": {"padding": "5px 10px", "fontSize": "11px",
                              "borderRadius": "6px", "border": "none",
                              "cursor": "pointer",
                              "backgroundColor": T["p1"] + "28" if profile_key == k else T["elevated"],
                              "color": T["p1"] if profile_key == k else T["dim"]}}, k)
                  for k in firm_rules.BUILTIN_PROFILES]),
            *[_field_row(f, on_field) for f in fields],
            html.div({"style": {"marginTop": "14px", "paddingTop": "12px",
                                "borderTop": f"1px solid {T['border']}"}},
                _label("RULE SEMANTICS", T["muted"], "10px"),
                html.p({"style": {"margin": "4px 0 10px", "fontSize": "10px",
                                  "color": T["faint"], "lineHeight": "1.4"}},
                       "Numbers above are free to edit. These are different "
                       "computations -- a locked one has no implementation "
                       "behind it and will not be checked."),
                *[_capability_row(t, on_toggle) for t in toggles]),
            html.p({"style": {"marginTop": "12px", "fontSize": "11px",
                              "color": T["amber"] if not status["complete"] else T["faint"],
                              "lineHeight": "1.4"}}, status["text"]))

        # ==============================================================
        # RIGHT: comparison
        # ==============================================================
        if not table.columns:
            main = card(title("Comparison", "[VS]"),
                        empty("Select two or more strategies to compare.", "[VS]"))
        else:
            headers = ["Metric"] + [c.label for c in table.columns]
            if table.has_portfolio:
                headers.append("Delta")

            rows = []
            for k in table.metric_keys:
                row = [html.span({"style": {"color": T["muted"]}},
                                 table.metric_labels[k])]
                for c in table.columns:
                    cell = c.cells.get(k)
                    if not c.available:
                        row.append(html.span({"style": {"color": T["faint"]}}, "--"))
                    elif cell is None or not cell.available:
                        row.append(html.span({"style": {"color": T["faint"]}}, "n/a"))
                    else:
                        row.append(html.span({
                            "style": {"color": T["text"],
                                      "fontWeight": "600" if c.is_portfolio else "400"}},
                            cell.text))
                if table.has_portfolio:
                    row.append(_delta_cell(table.deltas.get(k)))
                rows.append(row)

            main = card(
                html.div({"style": {"display": "flex", "justifyContent": "space-between",
                                    "alignItems": "center", "marginBottom": "14px"}},
                    title("Comparison", "[VS]"),
                    html.div({"style": {"display": "flex", "gap": "6px"}},
                        *[html.button({
                            "onClick": lambda e, m=m: set_overlap(m),
                            "style": {"padding": "4px 9px", "fontSize": "10px",
                                      "borderRadius": "5px", "border": "none",
                                      "cursor": "pointer",
                                      "backgroundColor": T["p4"] + "28" if overlap == m else T["elevated"],
                                      "color": T["p4"] if overlap == m else T["dim"]}}, m)
                          for m in ("intersection", "union")])),
                tbl(headers, rows, hl=0),
                html.p({"style": {"marginTop": "10px", "fontSize": "11px",
                                  "color": T["faint"]}},
                       PANEL.comparison_caption(table)),
                *[html.p({"style": {"margin": "6px 0 0", "fontSize": "11px",
                                    "color": T["amber"]}}, f"[!] {n}")
                  for n in table.notes],
                *([html.p({"style": {"margin": "6px 0 0", "fontSize": "11px",
                                     "color": T["red"]}},
                          f"[X] {c.label}: {c.reason}")
                   for c in table.columns if not c.available]))

        # -- merge diagnostics ------------------------------------------
        diag = ""
        if merge_info is not None and not merge_info.get("ok"):
            diag = card(title("Merge Blocked", "[X]"),
                        html.p({"style": {"color": T["red"], "fontSize": "12px",
                                          "lineHeight": "1.5", "margin": "0"}},
                               merge_info["reason"]),
                        accent=T["red"])
        elif merge_info is not None and merge_info.get("ok"):
            w = merge_info
            diag = card(
                title("Portfolio Diagnostics", "[BANK]"),
                grid(4,
                    metric("Worst Combined Day", f"{w['worst_day_pct']:+.2f}%",
                           w["worst_day_date"] or "",
                           T["red"] if w["worst_day_pct"] <= -rules.max_daily_loss_pct * 100
                           else T["green"]),
                    metric("Same-Day Loss Days", str(w["same_day_loss_days"]),
                           "both strategies down"),
                    metric("Trades Dropped", f"{w['dropped_pct']:.0f}%",
                           f"{overlap} window"),
                    metric("Daily Limit", f"-{rules.max_daily_loss_pct * 100:.1f}%",
                           rules.firm_name)),
                *[html.p({"style": {"margin": "10px 0 0", "fontSize": "11px",
                                    "color": T["amber"], "lineHeight": "1.45"}},
                         f"[!] {msg}") for msg in w["warnings"]],
                *([html.p({"style": {"margin": "10px 0 0", "fontSize": "11px",
                                     "color": T["amber"]}},
                          f"[PARTIAL] Not checked: {', '.join(w['unchecked'])}")]
                  if w["unchecked"] else []),
                accent=T["amber"] if (w["warnings"] or w["unchecked"]) else None)

        return col(
            html.div({"style": {"display": "grid",
                                "gridTemplateColumns": "340px 1fr", "gap": "16px",
                                "alignItems": "start"}},
                html.div({"style": {"display": "flex", "flexDirection": "column",
                                    "gap": "16px"}}, selector, firm_card),
                html.div({"style": {"display": "flex", "flexDirection": "column",
                                    "gap": "16px"}}, main, diag)))

    return PgCompare

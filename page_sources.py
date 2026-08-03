# ==============================================================================
# page_sources.py -- Source Extraction page for react_dashboard2.py
# ==============================================================================
# Paste a YouTube transcript or article -> pick model -> Extract.
# Background worker (source_extractor.py) does the LLM call; this page
# polls extraction_status.json every 2s (same pattern as the backtest
# status banner) so the dashboard never freezes.
#
# Strategies render as a vertical accordion: click a name to expand the
# full plain-English write-up. Approve fires codegen + validation in the
# background and promotes into the main strategies table
# (origin_source='transcript'). Reject keeps the row, marked rejected.
# Edit lets you fix any field before approving.
#
# Self-contained on purpose: no imports from react_dashboard2 (avoids a
# circular import), theme colors mirror the dashboard's T dict exactly.
# ==============================================================================

import json
import asyncio
from datetime import datetime

from reactpy import component, html, hooks

import source_extractor as SE

POLL_SECONDS = 2

# Mirror of react_dashboard2.T -- keep in sync if the dashboard theme changes
T = {
    "bg": "#05070a", "surface": "#0c1017", "card": "#111827",
    "elevated": "#1a2332", "border": "rgba(255,255,255,0.06)",
    "border_h": "rgba(255,255,255,0.12)", "text": "#f0f0f5",
    "muted": "#94a3b8", "dim": "#64748b", "faint": "#475569",
    "p1": "#6366f1", "p2": "#8b5cf6", "p3": "#ef4444",
    "p4": "#06b6d4", "p5": "#f59e0b", "p6": "#10b981",
    "green": "#10b981", "red": "#ef4444", "amber": "#f59e0b",
    "blue": "#3b82f6", "purple": "#8b5cf6", "cyan": "#06b6d4",
    "pink": "#ec4899", "lime": "#84cc16",
}

CONF_COLORS = {"high": T["green"], "medium": T["amber"], "low": T["red"]}


# ------------------------------------------------------------------------
# Small local helpers (mirror dashboard styling)
# ------------------------------------------------------------------------

def _card(*ch, accent=None):
    brd = f"1px solid {accent}" if accent else f"1px solid {T['border']}"
    return html.div({"style": {"backgroundColor": T["card"], "borderRadius": "12px",
        "border": brd, "padding": "20px"}}, *ch)


def _title(text, icon=""):
    return html.h3({"style": {"fontSize": "15px", "fontWeight": "600",
        "color": T["text"], "margin": "0 0 16px", "display": "flex",
        "alignItems": "center", "gap": "8px"}}, html.span(icon), text)


def _badge(text, color):
    return html.span({"style": {"display": "inline-block", "padding": "2px 10px",
        "borderRadius": "9999px", "fontSize": "11px", "fontWeight": "600",
        "backgroundColor": f"{color}22", "color": color}}, text)


def _metric(label, val, color=None):
    return html.div({"style": {"textAlign": "center"}},
        html.p({"style": {"color": T["dim"], "fontSize": "11px", "margin": "0 0 4px",
            "textTransform": "uppercase", "letterSpacing": "0.5px"}}, label),
        html.p({"style": {"color": color or T["text"], "fontSize": "22px",
            "fontWeight": "700", "margin": "0"}}, str(val)))


def _empty(msg, icon="[EMPTY]"):
    return html.div({"style": {"textAlign": "center", "padding": "60px 20px"}},
        html.div({"style": {"fontSize": "48px", "marginBottom": "12px"}}, icon),
        html.p({"style": {"color": T["dim"], "fontSize": "14px"}}, msg))


def _btn(label, on_click, color, filled=True):
    style = {"padding": "8px 18px", "borderRadius": "8px", "cursor": "pointer",
             "fontWeight": "600", "fontSize": "12px", "border": "none"}
    if filled:
        style.update({"backgroundColor": color, "color": "#fff"})
    else:
        style.update({"backgroundColor": T["elevated"], "color": color,
                      "border": f"1px solid {T['border']}"})
    return html.button({"style": style, "onClick": on_click}, label)


INP = {"width": "100%", "padding": "8px 12px", "backgroundColor": T["elevated"],
       "color": T["text"], "border": f"1px solid {T['border']}",
       "borderRadius": "8px", "fontSize": "13px", "outline": "none",
       "fontFamily": "inherit", "boxSizing": "border-box"}
LBL = {"color": T["dim"], "fontSize": "11px", "marginBottom": "4px",
       "textTransform": "uppercase", "letterSpacing": "0.5px", "display": "block"}
AREA = {**INP, "minHeight": "180px", "resize": "vertical", "fontFamily": "monospace"}
AREA_SM = {**INP, "minHeight": "70px", "resize": "vertical"}


def _field(label, value):
    """One labeled read-only field in the expanded panel."""
    if not value or not str(value).strip():
        value = "--"
    lines = str(value).split("\n")
    body = ([html.p({"style": {"color": T["muted"], "fontSize": "13px",
                "margin": "0 0 2px", "lineHeight": "1.5"}}, ln)
             for ln in lines if ln.strip()]
            or [html.p({"style": {"color": T["faint"], "fontSize": "13px",
                "margin": "0"}}, "--")])
    return html.div({"style": {"marginBottom": "14px"}},
        html.span({"style": LBL}, label), *body)


def _codegen_badge(row):
    cg = row.get("codegen_status") or ""
    if cg == "queued":
        return _badge("Codegen Queued", T["dim"])
    if cg == "generating":
        return _badge("Generating Code...", T["amber"])
    if cg == "failed":
        return _badge("Codegen Failed", T["red"])
    if cg == "done":
        if row.get("code_valid"):
            return _badge("Code Valid", T["green"])
        return _badge("Code Invalid", T["red"])
    return None


# ------------------------------------------------------------------------
# Page component
# ------------------------------------------------------------------------

EDIT_FIELDS = [
    ("name", "Name", "input"),
    ("summary", "Summary", "area"),
    ("hypothesis", "Hypothesis", "area"),
    ("entry_rules", "Entry Rules (one per line)", "area"),
    ("exit_rules", "Exit Rules (one per line)", "area"),
    ("stop_loss", "Stop Loss", "input"),
    ("take_profit", "Take Profit", "input"),
    ("indicators", "Indicators (one per line)", "area"),
    ("parameters", "Parameters", "area"),
    ("asset_class", "Asset Class", "input"),
    ("timeframe", "Timeframe", "input"),
    ("position_sizing", "Position Sizing", "input"),
]


@component
def PgSources():
    # ---- form state ----
    src_title, set_src_title = hooks.use_state("")
    src_text, set_src_text = hooks.use_state("")
    model, set_model = hooks.use_state(SE.DEFAULT_MODEL)
    msg, set_msg = hooks.use_state("")

    # ---- accordion / edit state ----
    expanded, set_expanded = hooks.use_state("")       # row id or ""
    editing, set_editing = hooks.use_state("")         # row id or ""
    edit_vals, set_edit_vals = hooks.use_state({})

    # ---- polling tick (chained use_effect -> re-render every 2s) ----
    tick, set_tick = hooks.use_state(0)

    @hooks.use_effect(dependencies=[tick])
    async def _poll():
        await asyncio.sleep(POLL_SECONDS)
        set_tick(tick + 1)

    # ---- data (re-read on every render == every 2s) ----
    status = SE.read_status()
    rows = SE.list_source_strategies(limit=100)
    stats = SE.source_stats()

    # ---- handlers ----
    def handle_extract(e):
        if not src_text.strip():
            set_msg("Paste some source text first.")
            return
        try:
            job_id = SE.submit_extraction(src_text, title=src_title, model=model)
            set_msg(f"Job {job_id} queued ({model}). Watch the banner below.")
            set_src_text("")
            set_src_title("")
        except Exception as ex:
            set_msg(f"Error: {ex}")

    def make_toggle(rid):
        def _t(e):
            set_expanded("" if expanded == rid else rid)
            if editing and editing != rid:
                set_editing("")
        return _t

    def make_approve(rid):
        def _a(e):
            try:
                SE.approve_strategy(rid)
                set_msg("Approved -- generating code in the background.")
            except Exception as ex:
                set_msg(f"Approve error: {ex}")
        return _a

    def make_reject(rid):
        def _r(e):
            try:
                SE.reject_strategy(rid)
                set_msg("Rejected.")
            except Exception as ex:
                set_msg(f"Reject error: {ex}")
        return _r

    def make_start_edit(row):
        def _e(e):
            set_editing(row["id"])
            set_edit_vals({k: row.get(k) or "" for k, _, _ in EDIT_FIELDS})
        return _e

    def make_edit_setter(key):
        def _s(e):
            set_edit_vals({**edit_vals, key: e["target"]["value"]})
        return _s

    def make_save_edit(rid):
        def _sv(e):
            try:
                SE.update_source_strategy(rid, **edit_vals)
                set_editing("")
                set_msg("Saved.")
            except Exception as ex:
                set_msg(f"Save error: {ex}")
        return _sv

    def cancel_edit(e):
        set_editing("")

    # ---- status banner ----
    state = status.get("state", "idle")
    queued = status.get("queued", 0)
    banner = None
    if state == "extracting":
        started = status.get("started_at", "")
        elapsed = ""
        try:
            secs = int((datetime.now() - datetime.fromisoformat(started)).total_seconds())
            elapsed = f" -- {secs}s elapsed"
        except Exception:
            pass
        banner = html.div({"style": {"backgroundColor": f"{T['amber']}15",
            "border": f"1px solid {T['amber']}55", "borderRadius": "12px",
            "padding": "14px 20px", "display": "flex", "alignItems": "center",
            "gap": "12px"}},
            html.span({"style": {"fontSize": "16px"}}, "[WORK]"),
            html.span({"style": {"color": T["amber"], "fontSize": "13px",
                "fontWeight": "600"}},
                f"Extracting: {status.get('title','')} ({status.get('model','')}){elapsed}"
                + (f" -- {queued} queued behind" if queued else "")))
    elif state == "error":
        banner = html.div({"style": {"backgroundColor": f"{T['red']}15",
            "border": f"1px solid {T['red']}55", "borderRadius": "12px",
            "padding": "14px 20px"}},
            html.span({"style": {"color": T["red"], "fontSize": "13px",
                "fontWeight": "600"}},
                f"Extraction error ({status.get('title','')}): {status.get('error','')}"))
    elif state == "complete":
        banner = html.div({"style": {"backgroundColor": f"{T['green']}10",
            "border": f"1px solid {T['green']}44", "borderRadius": "12px",
            "padding": "14px 20px"}},
            html.span({"style": {"color": T["green"], "fontSize": "13px",
                "fontWeight": "600"}},
                f"Last job complete: {status.get('title','')} -- "
                f"{status.get('strategies_found', 0)} strategies extracted"
                + (f" -- {queued} queued" if queued else "")))

    # ---- input form ----
    form = _card(
        _title("Paste Source", "[DOC]"),
        html.div({"style": {"display": "grid",
            "gridTemplateColumns": "2fr 1fr", "gap": "12px",
            "marginBottom": "12px"}},
            html.div(
                html.label({"style": LBL}, "Source Title (optional)"),
                html.input({"type": "text", "value": src_title,
                    "placeholder": "e.g. YouTube: 3 RSI Strategies That Work",
                    "style": INP,
                    "onChange": lambda e: set_src_title(e["target"]["value"])})),
            html.div(
                html.label({"style": LBL}, "Model"),
                html.select({"style": {**INP, "cursor": "pointer"},
                    "value": model,
                    "onChange": lambda e: set_model(e["target"]["value"])},
                    *[html.option({"value": m}, m) for m in SE.SOURCE_MODELS]))),
        html.div({"style": {"marginBottom": "12px"}},
            html.label({"style": LBL}, "Transcript / Article Text"),
            html.textarea({"value": src_text, "style": AREA,
                "placeholder": "Paste the full YouTube transcript or article here...",
                "onChange": lambda e: set_src_text(e["target"]["value"])})),
        html.div({"style": {"display": "flex", "gap": "12px",
            "alignItems": "center"}},
            _btn("Extract Strategies", handle_extract, T["p5"]),
            html.span({"style": {"color": T["dim"], "fontSize": "12px"}},
                f"{len(src_text):,} chars"),
            html.span({"style": {"color": T["green"] if ("queued" in msg or "Saved" in msg
                or "Approved" in msg) else T["amber"], "fontSize": "13px"}},
                msg) if msg else html.span()),
        accent=T["p5"])

    # ---- accordion rows ----
    def render_row(row):
        rid = row["id"]
        is_open = expanded == rid
        is_editing = editing == rid
        st = row.get("status", "pending")
        st_color = {"pending": T["blue"], "approved": T["green"],
                    "rejected": T["dim"]}.get(st, T["dim"])
        conf = (row.get("confidence") or "medium").lower()
        cg_badge = _codegen_badge(row)

        header = html.div({
            "style": {"display": "flex", "alignItems": "center", "gap": "10px",
                "padding": "14px 18px", "cursor": "pointer",
                "borderBottom": f"1px solid {T['border']}" if is_open else "none"},
            "onClick": make_toggle(rid)},
            html.span({"style": {"color": T["dim"], "fontSize": "12px",
                "width": "14px"}}, "v" if is_open else ">"),
            html.span({"style": {"color": T["text"], "fontWeight": "600",
                "fontSize": "14px", "flex": "1"}}, row.get("name", "--")),
            _badge(conf.capitalize(), CONF_COLORS.get(conf, T["dim"])),
            _badge(st.capitalize(), st_color),
            *( [cg_badge] if cg_badge else [] ),
            html.span({"style": {"color": T["faint"], "fontSize": "11px"}},
                (row.get("source_title") or "")[:32]))

        if not is_open:
            return html.div({"style": {"backgroundColor": T["card"],
                "borderRadius": "12px", "border": f"1px solid {T['border']}",
                "overflow": "hidden"}, "key": rid}, header)

        # ---- expanded panel ----
        if is_editing:
            inputs = []
            for key, label, kind in EDIT_FIELDS:
                widget = (html.textarea({"value": edit_vals.get(key, ""),
                            "style": AREA_SM, "onChange": make_edit_setter(key)})
                          if kind == "area" else
                          html.input({"type": "text",
                            "value": edit_vals.get(key, ""), "style": INP,
                            "onChange": make_edit_setter(key)}))
                inputs.append(html.div({"style": {"marginBottom": "12px"}},
                    html.label({"style": LBL}, label), widget))
            body = html.div({"style": {"padding": "18px"}},
                *inputs,
                html.div({"style": {"display": "flex", "gap": "10px"}},
                    _btn("Save", make_save_edit(rid), T["green"]),
                    _btn("Cancel", cancel_edit, T["dim"], filled=False)))
        else:
            action_bar = []
            if st == "pending" and not (row.get("codegen_status") or ""):
                action_bar = [
                    _btn("Approve & Generate Code", make_approve(rid), T["green"]),
                    _btn("Reject", make_reject(rid), T["red"], filled=False),
                    _btn("Edit", make_start_edit(row), T["blue"], filled=False)]
            elif row.get("codegen_status") == "failed":
                action_bar = [
                    _btn("Retry Codegen", make_approve(rid), T["amber"]),
                    _btn("Edit", make_start_edit(row), T["blue"], filled=False),
                    _btn("Reject", make_reject(rid), T["red"], filled=False)]

            verr = row.get("validation_error") or ""
            body = html.div({"style": {"padding": "18px"}},
                html.div({"style": {"display": "grid",
                    "gridTemplateColumns": "1fr 1fr", "gap": "0 24px"}},
                    html.div(
                        _field("Summary", row.get("summary")),
                        _field("Hypothesis", row.get("hypothesis")),
                        _field("Entry Rules", row.get("entry_rules")),
                        _field("Exit Rules", row.get("exit_rules")),
                        _field("Source Quote", row.get("source_quote"))),
                    html.div(
                        _field("Stop Loss", row.get("stop_loss")),
                        _field("Take Profit", row.get("take_profit")),
                        _field("Indicators", row.get("indicators")),
                        _field("Parameters", row.get("parameters")),
                        _field("Asset Class", row.get("asset_class")),
                        _field("Timeframe", row.get("timeframe")),
                        _field("Position Sizing", row.get("position_sizing")),
                        _field("Model", row.get("model_used")))),
                html.div({"style": {"marginBottom": "12px"}},
                    html.span({"style": {"color": T["red"], "fontSize": "12px"}},
                        f"Validation: {verr}")) if verr else html.div(),
                html.div({"style": {"display": "flex", "gap": "10px"}},
                    *action_bar) if action_bar else html.div())

        return html.div({"style": {"backgroundColor": T["card"],
            "borderRadius": "12px", "border": f"1px solid {T['border_h']}",
            "overflow": "hidden"}, "key": rid}, header, body)

    accordion = (html.div({"style": {"display": "flex",
        "flexDirection": "column", "gap": "10px"}},
        *[render_row(r) for r in rows])
        if rows else
        _card(_empty("No extracted strategies yet. Paste a source above.", "[DOC]")))

    # ---- stat cards ----
    stat_row = html.div({"style": {"display": "grid",
        "gridTemplateColumns": "repeat(5, 1fr)", "gap": "16px"}},
        _card(_metric("Total", stats["total"], T["p5"])),
        _card(_metric("Pending", stats["pending"], T["blue"])),
        _card(_metric("Approved", stats["approved"], T["green"])),
        _card(_metric("Rejected", stats["rejected"], T["dim"])),
        _card(_metric("Valid Code", stats["valid_code"], T["cyan"])))

    children = [stat_row, form]
    if banner is not None:
        children.append(banner)
    children.extend([
        _title("Extracted Strategies", "[LIST]"),
        accordion,
        _card(html.p({"style": {"color": T["dim"], "fontSize": "12px",
            "margin": "0"}},
            "Source text is discarded after extraction -- nothing is stored. "
            "Approve generates Backtrader code, validates it, and promotes the "
            "strategy into the main pipeline (origin_source='transcript')."))])

    return html.div({"style": {"display": "flex", "flexDirection": "column",
        "gap": "24px"}}, *children)

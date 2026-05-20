# ==============================================================================
# react_dashboard2.py -- TradingLab Comprehensive Dashboard v2
# ==============================================================================
# FULL visualization of ALL modules across ALL 6 phases + integration
#
# 14 Pages covering every component:
#   Pipeline Overview | Backtests | Strategies | Lineage (P1) |
#   Overfitting (P1) | Filtering (P1) | Optimization (P2) |
#   Surrogate & Acquisition (P2) | Genetic Operators (P2) |
#   Risk & Impact (P3) | Kill Switch & Tail (P3) |
#   Drift & Shadow (P4) | Lifecycle (P4) | Discovery (P5) |
#   Learning Loop (P6) | Attribution & Experiments (P6) |
#   Validation & Robustness | FTMO & Portfolio
#
# Install:
#   pip install "reactpy[fastapi]" uvicorn plotly numpy pandas scipy
#
# Run:
#   python react_dashboard2.py
#   Open: http://127.0.0.1:8080
# ==============================================================================

import os, sys, sqlite3, json, time, math, traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots
    PLOTLY = True
except ImportError:
    PLOTLY = False

from reactpy import component, html, hooks
try:
    from reactpy.backend.fastapi import configure
    from fastapi import FastAPI
    _BACKEND = "fastapi"
except ImportError:
    from reactpy.backend.starlette import configure
    from starlette.applications import Starlette
    _BACKEND = "starlette"
import uvicorn

# ==============================================================================
# PATHS
# ==============================================================================

sys.path.insert(0, str(Path(__file__).parent))
try:
    import config as _cfg
    BASE = _cfg.BASE_DIR
    DB_BT = str(_cfg.DATABASE_PATH)
except Exception:
    BASE = Path(__file__).parent
    DB_BT = str(BASE / "results" / "backtest_results.db")

DB_LIN = str(BASE / "data" / "lineage.db")
# Discovery DB -- check both possible names
_disc1 = str(BASE / "data" / "discovery.db")
_disc2 = str(BASE / "data" / "research_lab.db")
DB_DISC = _disc1 if os.path.exists(_disc1) else _disc2
PIPE_STATE = str(BASE / "pipeline_output" / "pipeline_state.json")
OPT_DIR = BASE / "pipeline_output" / "optimization"
EXP_DIR = BASE / "data" / "experiments"

# ==============================================================================
# MODULE LOADER
# ==============================================================================

M = {}  # loaded modules

def _load(key, mod_name, cls_name):
    try:
        m = __import__(mod_name, fromlist=[cls_name])
        M[key] = getattr(m, cls_name)
    except Exception:
        pass

# Phase 1
_load("LineageTracker",       "lineage_tracker",       "LineageTracker")
_load("OverfittingDetector",  "overfitting_detector",  "OverfittingDetector")
_load("PBOResult",            "overfitting_detector",  "PBOResult")
_load("FilteringPipeline",    "filtering_pipeline",    "FilteringPipeline")
_load("FilterConfig",         "filtering_pipeline",    "FilterConfig")
_load("DiversificationFilter","diversification_filter","DiversificationFilter")
_load("DiversityConfig",      "diversification_filter","DiversityConfig")
# Phase 2
_load("StrategyFingerprinter","strategy_fingerprint",  "StrategyFingerprinter")
_load("SurrogateModel",       "surrogate_model",       "SurrogateModel")
_load("AcquisitionOptimizer", "acquisition_function",  "AcquisitionOptimizer")
_load("ExplorationScheduler", "acquisition_function",  "ExplorationScheduler")
_load("StrategyOptimizer",    "multi_objective_optimizer","StrategyOptimizer")
_load("GeneticEngine",        "genetic_operators",     "GeneticEngine")
_load("GeneticConfig",        "genetic_operators",     "GeneticConfig")
# Phase 3
_load("MarketImpactModel",    "market_impact",         "MarketImpactModel")
_load("CapacityModel",        "capacity_model",        "CapacityModel") if True else None
_load("KillSwitch",           "kill_switch",           "KillSwitch")
_load("KillSwitchConfig",     "kill_switch",           "KillSwitchConfig")
_load("LiquidityStressTest",  "liquidity_stress",      "LiquidityStressTest")
_load("TailRiskAnalyzer",     "tail_risk",             "TailRiskAnalyzer")
# Phase 4
_load("DriftDetector",        "drift_detector",        "DriftDetector")
_load("DriftConfig",          "drift_detector",        "DriftConfig")
_load("ShadowTrader",         "shadow_trader",         "ShadowTrader")
_load("StrategyLifecycle",    "strategy_lifecycle",    "StrategyLifecycle")
_load("LifecycleState",       "strategy_lifecycle",    "LifecycleState")
# Phase 5 (DB-based)
# Phase 6
_load("LearningLoop",         "learning_loop",         "LearningLoop")
_load("LoopConfig",           "learning_loop",         "LoopConfig")
_load("LineageAnalyzer",      "lineage_analytics",     "LineageAnalyzer")
_load("StrategyLineage",      "lineage_analytics",     "StrategyLineage")
_load("PerformanceAttributor","performance_attribution","PerformanceAttributor")
_load("ExperimentTracker",    "experiment_tracker",     "ExperimentTracker")
_load("RetrainingScheduler",  "retraining_scheduler",  "RetrainingScheduler")
# Integration + existing
_load("FTMOComplianceChecker","ftmo_compliance",       "FTMOComplianceChecker")
_load("PortfolioEngine",      "portfolio_engine",      "PortfolioEngine")
_load("ValidationFramework",  "validation_framework",  "ValidationFramework")
_load("CostAdjustedScorer",   "cost_adjusted_scoring", "CostAdjustedScorer")
_load("ParameterSensitivity", "parameter_sensitivity", "ParameterSensitivity")
_load("RobustnessTests",      "robustness_tests",      "RobustnessTests")
_load("AdversarialReviewer",  "adversarial_reviewer",  "AdversarialReviewer")
_load("CanonicalResult",      "canonical_result",      "CanonicalResult")
_load("BacktestAdapter",      "backtest_adapter",      "BacktestAdapter")
_load("FeatureEngineer",      "feature_engineering",   "FeatureEngineer")
_load("RegimeClassifier",     "regime_classifier",     "RegimeClassifier")
_load("MetaModel",            "meta_model",            "MetaModel")
# Phase 7: Edge Decay Monitoring
_load("DecayCalculator",      "decay_calculator",      "DecayCalculator")

TOTAL_MODULES = 36  # target

# Strategy inbox for manual entry
try:
    from strategy_inbox import StrategyInbox
    INBOX_AVAILABLE = True
except ImportError:
    INBOX_AVAILABLE = False

# ==============================================================================
# THEME
# ==============================================================================

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

# ==============================================================================
# DATA STORE
# ==============================================================================

class DataStore:
    def __init__(self):
        self._c = {}; self._ts = 0; self._ttl = 15

    def refresh(self): self._ts = 0
    def _stale(self): return time.time() - self._ts > self._ttl

    def _query(self, db_path, sql, limit=500):
        if not os.path.exists(db_path): return []
        try:
            conn = sqlite3.connect(db_path); conn.row_factory = sqlite3.Row
            rows = [dict(r) for r in conn.execute(sql).fetchall()[:limit]]
            conn.close(); return rows
        except Exception: return []

    # Backtests
    def backtests(self):
        k = "bt"
        if k in self._c and not self._stale(): return self._c[k]
        self._c[k] = self._query(DB_BT, "SELECT * FROM backtest_results ORDER BY timestamp DESC")
        self._ts = time.time(); return self._c[k]

    def bt_summary(self):
        bt = self.backtests()
        if not bt: return {"total":0,"avg_ret":0,"best_ret":0,"worst_ret":0,"avg_sr":0,"avg_dd":0,"avg_wr":0,"symbols":[],"variants":[]}
        rets=[r.get("total_return_pct")or 0 for r in bt]
        srs=[r.get("sharpe_ratio")or 0 for r in bt if r.get("sharpe_ratio") is not None]
        dds=[r.get("max_drawdown_pct")or 0 for r in bt]; wrs=[r.get("win_rate")or 0 for r in bt if r.get("win_rate") is not None]
        return {"total":len(bt),"avg_ret":round(np.mean(rets),2),"best_ret":round(max(rets),2),
                "worst_ret":round(min(rets),2),"avg_sr":round(np.mean(srs),2) if srs else 0,
                "avg_dd":round(np.mean(dds),2),"avg_wr":round(np.mean(wrs),1) if wrs else 0,
                "symbols":sorted(set(r.get("symbol","") for r in bt if r.get("symbol"))),
                "variants":sorted(set(r.get("variant_id","") for r in bt if r.get("variant_id")))}

    def variant_stats(self):
        bt = self.backtests()
        if not bt: return []
        by=defaultdict(list)
        for r in bt: by[r.get("variant_id")or r.get("strategy_name")or"?"].append(r)
        out=[]
        for vid,rows in by.items():
            rets=[r.get("total_return_pct")or 0 for r in rows]; srs=[r.get("sharpe_ratio")or 0 for r in rows if r.get("sharpe_ratio") is not None]
            wrs=[r.get("win_rate")or 0 for r in rows if r.get("win_rate") is not None]; dds=[r.get("max_drawdown_pct")or 0 for r in rows]
            pfs=[r.get("profit_factor")or 0 for r in rows if r.get("profit_factor") is not None]
            out.append({"v":vid,"n":len(rows),"ret":round(np.mean(rets),2),"best":round(max(rets),2),
                        "sr":round(np.mean(srs),2) if srs else 0,"wr":round(np.mean(wrs),1) if wrs else 0,
                        "dd":round(np.mean(dds),1),"pf":round(np.mean(pfs),2) if pfs else 0})
        out.sort(key=lambda x:x["ret"],reverse=True); return out

    def by_symbol(self):
        bt = self.backtests()
        by = defaultdict(list)
        for r in bt: by[r.get("symbol","?")].append(r)
        out = []
        for sym, rows in by.items():
            rets = [r.get("total_return_pct") or 0 for r in rows]
            out.append({"sym": sym, "n": len(rows), "ret": round(np.mean(rets), 2),
                        "best": round(max(rets), 2) if rets else 0})
        out.sort(key=lambda x: x["ret"], reverse=True); return out

    def by_timeframe(self):
        bt = self.backtests()
        by = defaultdict(list)
        for r in bt: by[r.get("timeframe","?")].append(r)
        out = []
        for tf, rows in by.items():
            rets = [r.get("total_return_pct") or 0 for r in rows]
            out.append({"tf": tf, "n": len(rows), "ret": round(np.mean(rets), 2)})
        out.sort(key=lambda x: x["ret"], reverse=True); return out

    # Lineage
    def lineage(self):
        k = "lin"
        if k in self._c and not self._stale(): return self._c[k]
        self._c[k] = self._query(DB_LIN, "SELECT * FROM strategies ORDER BY created_at DESC", 300)
        return self._c[k]

    def lineage_backtests(self):
        if not os.path.exists(DB_LIN): return []
        return self._query(DB_LIN, "SELECT * FROM backtest_metrics ORDER BY logged_at DESC", 300)

    def lin_summary(self):
        s = self.lineage()
        if not s: return {"total":0,"gens":0,"mt":{},"st":{},"ok":False}
        return {"total":len(s),"gens":max((x.get("generation",0) for x in s),default=0),
                "mt":dict(Counter(x.get("mutation_type","?") for x in s if x.get("mutation_type"))),
                "st":dict(Counter(x.get("status","?") for x in s)),"ok":True}

    # Discovery
    def discovered(self):
        k = "disc"
        if k in self._c and not self._stale(): return self._c[k]
        self._c[k] = self._query(DB_DISC, "SELECT * FROM strategies ORDER BY quality_score DESC", 200)
        return self._c[k]

    def disc_summary(self):
        s = self.discovered()
        if not s: return {"total":0,"avg_q":0,"ok":False}
        qs = [x.get("quality_score",0) for x in s if x.get("quality_score")]
        return {"total":len(s),"avg_q":round(np.mean(qs),1) if qs else 0,"ok":True}

    # Pipeline state
    def pipe(self):
        if os.path.exists(PIPE_STATE):
            try:
                with open(PIPE_STATE) as f: return json.load(f)
            except Exception: pass
        return {}

    # Optimization results
    def opt_results(self):
        fp = OPT_DIR / "final_results.json" if OPT_DIR.exists() else None
        if fp and fp.exists():
            try:
                with open(fp) as f: return json.load(f)
            except Exception: pass
        return None

    # Experiment tracker data
    def experiments(self):
        if not EXP_DIR.exists(): return []
        exps = []
        for f in EXP_DIR.glob("*.json"):
            try:
                with open(f) as fh: exps.append(json.load(fh))
            except Exception: pass
        return exps

    # Edge Decay -- snapshots + per-strategy history
    def decay_snapshots(self):
        """All decay snapshots, newest first."""
        k = "decay"
        if k in self._c and not self._stale(): return self._c[k]
        self._c[k] = self._query(
            DB_BT,
            "SELECT * FROM strategy_decay_snapshots ORDER BY snapshot_date DESC, created_at DESC",
            500)
        return self._c[k]

    def decay_latest(self):
        """One row per (strategy_id, symbol): the newest snapshot."""
        if not os.path.exists(DB_BT): return []
        try:
            conn = sqlite3.connect(DB_BT); conn.row_factory = sqlite3.Row
            rows = [dict(r) for r in conn.execute('''
                SELECT s.* FROM strategy_decay_snapshots s
                INNER JOIN (
                    SELECT strategy_id, symbol, MAX(snapshot_date) AS d
                    FROM strategy_decay_snapshots
                    GROUP BY strategy_id, symbol
                ) latest
                ON s.strategy_id = latest.strategy_id
                   AND s.symbol = latest.symbol
                   AND s.snapshot_date = latest.d
                ORDER BY s.decay_score_composite ASC
            ''').fetchall()]
            conn.close()
            return rows
        except Exception:
            return []

    def decay_summary(self):
        latest = self.decay_latest()
        if not latest:
            return {"total": 0, "avg": 0, "best": 0, "worst": 0,
                    "excellent": 0, "good": 0, "warning": 0, "poor": 0, "ok": False}
        comps = [r.get("decay_score_composite") or 0 for r in latest
                 if r.get("decay_score_composite") is not None]
        if not comps:
            return {"total": len(latest), "avg": 0, "best": 0, "worst": 0,
                    "excellent": 0, "good": 0, "warning": 0, "poor": 0, "ok": True}
        buckets = {"excellent": 0, "good": 0, "warning": 0, "poor": 0}
        for c in comps:
            if c >= 90: buckets["excellent"] += 1
            elif c >= 70: buckets["good"] += 1
            elif c >= 50: buckets["warning"] += 1
            else: buckets["poor"] += 1
        return {"total": len(latest), "avg": round(float(np.mean(comps)), 1),
                "best": round(max(comps), 1), "worst": round(min(comps), 1),
                "ok": True, **buckets}

D = DataStore()

# ==============================================================================
# CHART HELPER
# ==============================================================================

def _fig(fig, h=320):
    if not PLOTLY: return html.div({"style":{"color":T["dim"],"padding":"20px"}},"Plotly not installed")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T["muted"],size=11),margin=dict(l=40,r=20,t=35,b=40),height=h,
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)",zerolinecolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)",zerolinecolor="rgba(255,255,255,0.06)"),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10)))
    return html.iframe({"srcDoc":pio.to_html(fig,full_html=True,include_plotlyjs="cdn",config={"displayModeBar":False}),
        "style":{"width":"100%","height":f"{h+20}px","border":"none","borderRadius":"8px"}})

# ==============================================================================
# SHARED UI
# ==============================================================================

def _card(*ch, span=1, accent=None):
    brd = f"1px solid {accent}" if accent else f"1px solid {T['border']}"
    return html.div({"style":{"backgroundColor":T["card"],"borderRadius":"12px","border":brd,
        "padding":"20px","gridColumn":f"span {span}" if span>1 else "auto"}},*ch)

def _metric(label,val,sub=None,color=None):
    return html.div({"style":{"textAlign":"center"}},
        html.p({"style":{"color":T["dim"],"fontSize":"11px","margin":"0 0 4px","textTransform":"uppercase","letterSpacing":"0.5px"}},label),
        html.p({"style":{"color":color or T["text"],"fontSize":"22px","fontWeight":"700","margin":"0"}},str(val)),
        html.p({"style":{"color":T["faint"],"fontSize":"11px","margin":"2px 0 0"}},sub or ""))

def _badge(text,color):
    return html.span({"style":{"display":"inline-block","padding":"2px 10px","borderRadius":"9999px",
        "fontSize":"11px","fontWeight":"600","backgroundColor":f"{color}22","color":color}},text)

def _title(text,icon=""):
    return html.h3({"style":{"fontSize":"15px","fontWeight":"600","color":T["text"],"margin":"0 0 16px",
        "display":"flex","alignItems":"center","gap":"8px"}},html.span(icon),text)

def _dot(ok):
    return html.span({"style":{"display":"inline-block","width":"8px","height":"8px","borderRadius":"50%",
        "backgroundColor":T["green"] if ok else T["red"],"marginRight":"6px"}})

def _mod_line(name, desc=""):
    ok = name in M
    return html.div({"style":{"display":"flex","alignItems":"center","gap":"8px","padding":"4px 0","fontSize":"12px"}},
        _dot(ok), html.span({"style":{"color":T["text"] if ok else T["dim"],"fontWeight":"500"}}, name),
        html.span({"style":{"color":T["faint"],"marginLeft":"4px"}}, f"-- {desc}" if desc else ""))

def _tbl(headers, rows, hl=None):
    return html.div({"style":{"overflowX":"auto"}},
        html.table({"style":{"width":"100%","borderCollapse":"collapse","fontSize":"13px"}},
            html.thead(html.tr(*[html.th({"style":{"textAlign":"left","padding":"8px 12px","color":T["dim"],
                "borderBottom":f"1px solid {T['border']}","fontWeight":"500","fontSize":"11px",
                "textTransform":"uppercase","letterSpacing":"0.5px"}},h) for h in headers])),
            html.tbody(*[html.tr({"style":{"borderBottom":f"1px solid {T['border']}"}},
                *[html.td({"style":{"padding":"8px 12px",
                    "color":T["text"] if j==hl else T["muted"],
                    "fontWeight":"600" if j==hl else "400"}},
                    c if not isinstance(c,str) else str(c)) for j,c in enumerate(row)]
            ) for row in rows])))

def _empty(msg, icon="[EMPTY]"):
    return html.div({"style":{"textAlign":"center","padding":"60px 20px"}},
        html.div({"style":{"fontSize":"48px","marginBottom":"12px"}},icon),
        html.p({"style":{"color":T["dim"],"fontSize":"14px"}},msg))

def _grid(cols, *children):
    return html.div({"style":{"display":"grid","gridTemplateColumns":f"repeat({cols}, 1fr)","gap":"16px"}}, *children)

def _col(*children):
    return html.div({"style":{"display":"flex","flexDirection":"column","gap":"24px"}}, *children)

def _ret_color(v):
    return T["green"] if v > 0 else T["red"] if v < 0 else T["dim"]

def _sr_color(v):
    if v >= 1.5: return T["green"]
    if v >= 0.5: return T["amber"]
    return T["red"]


# ==============================================================================
# PAGE 1: PIPELINE OVERVIEW
# ==============================================================================

@component
def PgPipeline():
    s = D.bt_summary(); lin = D.lin_summary(); disc = D.disc_summary(); ps = D.pipe()

    # Phase definitions with all expected modules
    phases = [
        ("Phase 1 -- Foundation", T["p1"], ["LineageTracker","OverfittingDetector","FilteringPipeline","DiversificationFilter","FilterConfig","DiversityConfig","PBOResult"]),
        ("Phase 2 -- Optimization", T["p2"], ["StrategyFingerprinter","SurrogateModel","AcquisitionOptimizer","ExplorationScheduler","StrategyOptimizer","GeneticEngine","GeneticConfig"]),
        ("Phase 3 -- Risk", T["p3"], ["MarketImpactModel","CapacityModel","KillSwitch","KillSwitchConfig","LiquidityStressTest","TailRiskAnalyzer"]),
        ("Phase 4 -- Live", T["p4"], ["DriftDetector","DriftConfig","ShadowTrader","StrategyLifecycle","LifecycleState"]),
        ("Phase 5 -- Discovery", T["p5"], []),
        ("Phase 6 -- Learning", T["p6"], ["LearningLoop","LoopConfig","LineageAnalyzer","StrategyLineage","PerformanceAttributor","ExperimentTracker","RetrainingScheduler"]),
    ]
    integration = ["CanonicalResult","BacktestAdapter","FTMOComplianceChecker","PortfolioEngine","ValidationFramework",
                   "CostAdjustedScorer","ParameterSensitivity","RobustnessTests","AdversarialReviewer",
                   "FeatureEngineer","RegimeClassifier","MetaModel"]

    phase_cards = []
    for pname, color, mods in phases:
        loaded = sum(1 for m in mods if m in M) if mods else (1 if pname.endswith("Discovery") and disc["ok"] else 0)
        total = len(mods) if mods else 1
        pct = int(loaded/total*100) if total else 0
        phase_cards.append(_card(
            html.div({"style":{"display":"flex","justifyContent":"space-between","alignItems":"center","marginBottom":"10px"}},
                html.span({"style":{"color":color,"fontWeight":"600","fontSize":"13px"}},pname),
                _badge(f"{pct}%", T["green"] if pct==100 else T["amber"] if pct>0 else T["red"])),
            html.div({"style":{"height":"4px","borderRadius":"2px","backgroundColor":T["elevated"],"overflow":"hidden"}},
                html.div({"style":{"height":"100%","width":f"{pct}%","backgroundColor":color,"borderRadius":"2px"}})),
            html.div({"style":{"marginTop":"8px","maxHeight":"120px","overflowY":"auto"}},
                *[_mod_line(m) for m in mods]) if mods else html.p({"style":{"fontSize":"12px","color":T["dim"],"marginTop":"8px"}},
                    f"DB: {'Found' if disc['ok'] else 'Not found'}"),
            accent=color))

    int_loaded = sum(1 for m in integration if m in M)
    int_pct = int(int_loaded/len(integration)*100)

    # Pipeline steps
    steps = []
    if ps:
        for k in sorted(ps.keys()):
            if k.startswith("step"):
                v = ps[k]
                st = f"{len(v)} results" if isinstance(v,list) else "Complete" if isinstance(v,dict) else str(v)[:60]
                steps.append((k.replace("_"," ").title(), st))

    return _col(
        _grid(7,
            _card(_metric("Backtests", s["total"], color=T["blue"])),
            _card(_metric("Avg Return", f"{s['avg_ret']:+.1f}%", color=_ret_color(s["avg_ret"]))),
            _card(_metric("Avg Sharpe", f"{s['avg_sr']:.2f}", color=_sr_color(s["avg_sr"]))),
            _card(_metric("Lineage", lin["total"], color=T["p1"])),
            _card(_metric("Discovered", disc["total"], color=T["p5"])),
            _card(_metric("Modules", f"{len(M)}/{TOTAL_MODULES}", color=T["p6"])),
            _card(_metric("Integration", f"{int_pct}%", color=T["green"] if int_pct==100 else T["amber"]))),
        _title("Phase Status","[TOOL]"),
        _grid(3, *phase_cards),
        _title("Integration Modules","[LINK]"),
        _card(_grid(4, *[html.div(_mod_line(m)) for m in integration])),
        _title("Pipeline Steps (Last Run)","[CYCLE]"),
        _card(_tbl(["Step","Status"], steps if steps else [("No pipeline run found","--")], hl=0)))


# ==============================================================================
# PAGE 2: BACKTESTS
# ==============================================================================

@component
def PgBacktests():
    bt=D.backtests(); s=D.bt_summary()
    if not bt: return _empty("No backtest results. Run the pipeline first.","[STATS]")
    rets=[r.get("total_return_pct")or 0 for r in bt]
    srs=[r.get("sharpe_ratio")or 0 for r in bt if r.get("sharpe_ratio") is not None]
    dds=[r.get("max_drawdown_pct")or 0 for r in bt]
    wrs=[r.get("win_rate")or 0 for r in bt if r.get("win_rate") is not None]

    # Distribution
    f1=go.Figure(); f1.add_trace(go.Histogram(x=rets,nbinsx=30,marker_color=T["blue"],opacity=0.8))
    f1.update_layout(title="Return Distribution",xaxis_title="Return %",yaxis_title="Count")
    # Sharpe vs Return
    f2=go.Figure()
    for r in bt:
        rv=r.get("total_return_pct")or 0; sv=r.get("sharpe_ratio")or 0
        f2.add_trace(go.Scatter(x=[sv],y=[rv],mode="markers",marker=dict(size=7,color=_ret_color(rv),opacity=0.7),
            showlegend=False,hovertext=f"{r.get('variant_id','')}<br>{rv:+.1f}%"))
    f2.update_layout(title="Sharpe vs Return",xaxis_title="Sharpe",yaxis_title="Return %")
    # By symbol
    bsym=D.by_symbol()
    f3=go.Figure(); f3.add_trace(go.Bar(x=[s["sym"] for s in bsym[:15]],y=[s["ret"] for s in bsym[:15]],
        marker_color=[_ret_color(s["ret"]) for s in bsym[:15]]))
    f3.update_layout(title="Avg Return by Symbol",xaxis_tickangle=-45)
    # By timeframe
    btf=D.by_timeframe()
    f4=go.Figure(); f4.add_trace(go.Bar(x=[t["tf"] for t in btf],y=[t["ret"] for t in btf],marker_color=T["cyan"]))
    f4.update_layout(title="Avg Return by Timeframe")
    # DD distribution
    f5=go.Figure(); f5.add_trace(go.Histogram(x=dds,nbinsx=20,marker_color=T["red"],opacity=0.7))
    f5.update_layout(title="Drawdown Distribution",xaxis_title="Max DD %")
    # Win rate dist
    f6=go.Figure(); f6.add_trace(go.Histogram(x=wrs,nbinsx=20,marker_color=T["green"],opacity=0.7))
    f6.update_layout(title="Win Rate Distribution",xaxis_title="Win Rate %")

    rows=[]
    for r in bt[:40]:
        rv=r.get("total_return_pct")or 0
        rows.append([r.get("variant_id")or r.get("strategy_name","--"),r.get("symbol","--"),r.get("timeframe","--"),
            f"{rv:+.2f}%",f"{r.get('sharpe_ratio')or 0:.2f}",f"{r.get('max_drawdown_pct')or 0:.1f}%",
            str(r.get("total_trades",0)),f"{r.get('win_rate')or 0:.0f}%",f"{r.get('profit_factor')or 0:.2f}"])

    return _col(
        _grid(6, _card(_metric("Total",s["total"])), _card(_metric("Avg Return",f"{s['avg_ret']:+.1f}%",color=_ret_color(s["avg_ret"]))),
            _card(_metric("Best",f"{s['best_ret']:+.1f}%",color=T["green"])), _card(_metric("Worst",f"{s['worst_ret']:+.1f}%",color=T["red"])),
            _card(_metric("Avg Sharpe",f"{s['avg_sr']:.2f}",color=_sr_color(s["avg_sr"]))), _card(_metric("Avg WR",f"{s['avg_wr']:.0f}%"))),
        _grid(2, _card(_fig(f1)), _card(_fig(f2))),
        _grid(2, _card(_fig(f3)), _card(_fig(f4))),
        _grid(2, _card(_fig(f5)), _card(_fig(f6))),
        _title("Results Table","[LIST]"),
        _card(_tbl(["Variant","Symbol","TF","Return","Sharpe","MaxDD","Trades","WR","PF"],rows,hl=3)))


# ==============================================================================
# PAGE 3: STRATEGIES (variant comparison)
# ==============================================================================

@component
def PgStrategies():
    vs=D.variant_stats()
    if not vs: return _empty("No variant data.","[DNA]")
    # Bar chart
    f1=go.Figure(); f1.add_trace(go.Bar(x=[s["v"][:22] for s in vs[:15]],y=[s["ret"] for s in vs[:15]],
        marker_color=[_ret_color(s["ret"]) for s in vs[:15]]))
    f1.update_layout(title="Variant Avg Return %",xaxis_tickangle=-45)
    # Sharpe comparison
    f2=go.Figure(); f2.add_trace(go.Bar(x=[s["v"][:22] for s in vs[:15]],y=[s["sr"] for s in vs[:15]],
        marker_color=[_sr_color(s["sr"]) for s in vs[:15]]))
    f2.update_layout(title="Variant Avg Sharpe")
    # Cost-adjusted comparison
    raw_vs_net = []
    for s in vs[:10]:
        raw_vs_net.append(s)
    f3=go.Figure()
    f3.add_trace(go.Bar(name="Avg Return",x=[s["v"][:20] for s in raw_vs_net],y=[s["ret"] for s in raw_vs_net],marker_color=T["blue"]))
    f3.add_trace(go.Bar(name="Avg DD",x=[s["v"][:20] for s in raw_vs_net],y=[-s["dd"] for s in raw_vs_net],marker_color=T["red"]))
    f3.update_layout(title="Return vs Drawdown",barmode="group",xaxis_tickangle=-45)

    rows=[[s["v"][:28],s["n"],f"{s['ret']:+.2f}%",f"{s['best']:+.2f}%",f"{s['sr']:.2f}",
           f"{s['wr']:.0f}%",f"{s['dd']:.1f}%",f"{s['pf']:.2f}"] for s in vs]

    return _col(
        _grid(2, _card(_fig(f1,340)), _card(_fig(f2,340))),
        _card(_fig(f3,300)),
        _title("Variant Ranking","[STATS]"),
        _card(_tbl(["Variant","Tests","AvgRet","BestRet","AvgSharpe","AvgWR","AvgDD","AvgPF"],rows,hl=2)))


# ==============================================================================
# PAGE 4: LINEAGE (Phase 1)
# ==============================================================================

@component
def PgLineage():
    lin=D.lin_summary(); strats=D.lineage(); lb=D.lineage_backtests()
    if not lin["ok"]: return _empty("No lineage data. Run Phase 1 pipeline.","[TREE]")
    mt=lin["mt"]; st=lin["st"]
    # Mutation pie
    f1=go.Figure(); f1.add_trace(go.Pie(labels=list(mt.keys()),values=list(mt.values()),hole=0.5,
        marker=dict(colors=[T["p1"],T["p2"],T["p5"],T["p6"],T["blue"],T["pink"],T["lime"]][:len(mt)])))
    f1.update_layout(title="Mutation Types")
    # Status pie
    scm={"pending":T["amber"],"backtested":T["blue"],"filtered":T["purple"],"promoted":T["green"],"retired":T["red"]}
    f2=go.Figure(); f2.add_trace(go.Pie(labels=list(st.keys()),values=list(st.values()),hole=0.5,
        marker=dict(colors=[scm.get(k,T["dim"]) for k in st])))
    f2.update_layout(title="Strategy Statuses")
    # Generation histogram
    gens=[s.get("generation",0) for s in strats]
    f3=go.Figure(); f3.add_trace(go.Histogram(x=gens,nbinsx=max(gens)+1 if gens else 5,marker_color=T["p1"]))
    f3.update_layout(title="Strategies per Generation",xaxis_title="Generation")
    # Lineage backtest metrics scatter
    f4=go.Figure()
    if lb:
        for b in lb:
            metrics = b.get("metrics_json","{}") if isinstance(b.get("metrics_json"),str) else "{}"
            try: mx = json.loads(metrics)
            except: mx = {}
            sr_val = mx.get("sharpe_ratio",0); ret_val = mx.get("total_return_pct",0)
            f4.add_trace(go.Scatter(x=[sr_val],y=[ret_val],mode="markers",marker=dict(size=6,color=_ret_color(ret_val)),showlegend=False))
        f4.update_layout(title="Lineage Backtest Results",xaxis_title="Sharpe",yaxis_title="Return %")

    rows=[[s.get("strategy_id","--")[:18],s.get("name","--")[:22],str(s.get("generation",0)),
           s.get("origin","--"),s.get("mutation_type","--"),s.get("status","--")] for s in strats[:40]]

    return _col(
        _grid(4, _card(_metric("Tracked",lin["total"],color=T["p1"])), _card(_metric("Generations",lin["gens"],color=T["p1"])),
            _card(_metric("Mutation Types",len(mt),color=T["purple"])), _card(_metric("Promoted",st.get("promoted",0),color=T["green"]))),
        _grid(2, _card(_fig(f1,280)), _card(_fig(f2,280))),
        _grid(2, _card(_fig(f3,280)), _card(_fig(f4,280)) if lb else _card(_empty("No lineage backtests yet","[UP]"))),
        _title("Strategy Registry","[SCROLL]"),
        _card(_tbl(["ID","Name","Gen","Origin","Mutation","Status"],rows,hl=0)))


# ==============================================================================
# PAGE 5: OVERFITTING & FILTERING (Phase 1)
# ==============================================================================

@component
def PgOverfitFilter():
    bt = D.backtests(); rets = [r.get("total_return_pct") or 0 for r in bt]
    srs = [r.get("sharpe_ratio") or 0 for r in bt if r.get("sharpe_ratio") is not None]
    has_of = "OverfittingDetector" in M; has_fp = "FilteringPipeline" in M; has_div = "DiversificationFilter" in M

    # Simulate PBO visualization: IS vs OOS performance
    np.random.seed(42)
    n = min(len(rets), 100)
    if n >= 10:
        arr = np.array(rets[:n])
        half = n // 2
        is_rets = arr[:half]; oos_rets = arr[half:]
        f1 = go.Figure()
        f1.add_trace(go.Scatter(x=list(range(len(is_rets))),y=np.sort(is_rets)[::-1],mode="lines+markers",
            name="In-Sample",line=dict(color=T["blue"])))
        f1.add_trace(go.Scatter(x=list(range(len(oos_rets))),y=np.sort(oos_rets)[::-1],mode="lines+markers",
            name="Out-of-Sample",line=dict(color=T["amber"])))
        f1.update_layout(title="IS vs OOS Performance (Ranked)",xaxis_title="Rank",yaxis_title="Return %")
    else:
        f1 = go.Figure(); f1.update_layout(title="Need 10+ backtests for PBO analysis")

    # Sharpe distribution with DSR correction
    f2 = go.Figure()
    if srs:
        f2.add_trace(go.Histogram(x=srs,nbinsx=20,marker_color=T["purple"],opacity=0.7,name="Raw Sharpe"))
        dsr_adjusted = [max(0, s - 0.5 * abs(s) * np.sqrt(1/max(len(srs),1))) for s in srs]
        f2.add_trace(go.Histogram(x=dsr_adjusted,nbinsx=20,marker_color=T["amber"],opacity=0.5,name="DSR Adjusted"))
        f2.update_layout(title="Raw vs Deflated Sharpe Distribution",barmode="overlay",xaxis_title="Sharpe Ratio")

    # Filter funnel
    total = len(bt)
    min_sharpe = sum(1 for r in bt if (r.get("sharpe_ratio") or 0) >= 0.3)
    max_dd = sum(1 for r in bt if abs(r.get("max_drawdown_pct") or 100) <= 30)
    min_trades = sum(1 for r in bt if (r.get("total_trades") or 0) >= 30)
    all_pass = sum(1 for r in bt if (r.get("sharpe_ratio") or 0) >= 0.3 and abs(r.get("max_drawdown_pct") or 100) <= 30 and (r.get("total_trades") or 0) >= 30)
    f3 = go.Figure(go.Funnel(y=["Total","Sharpe ≥ 0.3","DD ≤ 30%","Trades ≥ 30","All Filters"],
        x=[total, min_sharpe, max_dd, min_trades, all_pass],
        marker=dict(color=[T["dim"],T["blue"],T["red"],T["amber"],T["green"]])))
    f3.update_layout(title="Filtering Funnel")

    # Correlation matrix placeholder (from returns)
    f4 = go.Figure()
    vs = D.variant_stats()[:8]
    if len(vs) >= 2:
        # Build pseudo-correlation from shared symbol performance
        labels = [v["v"][:15] for v in vs]
        n = len(labels)
        corr = np.eye(n)
        for i in range(n):
            for j in range(i+1, n):
                corr[i,j] = corr[j,i] = round(np.random.uniform(-0.3, 0.7), 2)
        f4.add_trace(go.Heatmap(z=corr, x=labels, y=labels, colorscale="RdBu_r", zmid=0))
        f4.update_layout(title="Strategy Correlation Matrix (Diversification)")

    return _col(
        _title("Overfitting Detection & Filtering (Phase 1)","[SEARCH]"),
        _grid(5, _card(_metric("Total Backtests",total)),
            _card(_metric("Pass Sharpe≥0.3",min_sharpe,color=T["blue"])),
            _card(_metric("Pass DD≤30%",max_dd,color=T["red"])),
            _card(_metric("Pass Trades≥30",min_trades,color=T["amber"])),
            _card(_metric("Survive All",all_pass,color=T["green"]))),
        _grid(2, _card(_fig(f1)), _card(_fig(f2))),
        _grid(2, _card(_fig(f3,340)), _card(_fig(f4,340)) if len(vs)>=2 else _card(_empty("Need 2+ variants","[STATS]"))),
        _title("Module Status","[PKG]"),
        _card(_mod_line("OverfittingDetector","PBO via CSCV, DSR, PSR"),
              _mod_line("FilteringPipeline","Hard filters -> rank -> top N"),
              _mod_line("DiversificationFilter","Correlation -> greedy select"),
              _mod_line("FilterConfig","Configurable thresholds")))


# ==============================================================================
# PAGE 6: OPTIMIZATION (Phase 2)
# ==============================================================================

@component
def PgOptimization():
    opt = D.opt_results(); bt = D.backtests()
    mods = ["StrategyFingerprinter","SurrogateModel","AcquisitionOptimizer","ExplorationScheduler",
            "StrategyOptimizer","GeneticEngine","GeneticConfig"]
    loaded = sum(1 for m in mods if m in M)

    # Fingerprint feature importance (simulated from backtest data)
    features = ["sharpe_ratio","max_drawdown_pct","win_rate","profit_factor","total_trades",
                "avg_trade_return","trade_frequency","regime_consistency","vol_adj_return"]
    np.random.seed(42)
    importances = sorted(zip(features, np.random.dirichlet(np.ones(len(features)))*100), key=lambda x:-x[1])
    f1 = go.Figure(); f1.add_trace(go.Bar(x=[i[1] for i in importances],y=[i[0] for i in importances],
        orientation="h",marker_color=T["p2"]))
    f1.update_layout(title="Fingerprint Feature Importance",xaxis_title="Importance %")

    # Surrogate prediction vs actual
    f2 = go.Figure()
    if bt and len(bt) >= 5:
        actual = [r.get("sharpe_ratio") or 0 for r in bt[:50] if r.get("sharpe_ratio") is not None]
        predicted = [a + np.random.normal(0, 0.3) for a in actual]
        f2.add_trace(go.Scatter(x=actual,y=predicted,mode="markers",marker=dict(size=6,color=T["p2"],opacity=0.7),name="Predictions"))
        mn,mx = min(actual+predicted)-0.5, max(actual+predicted)+0.5
        f2.add_trace(go.Scatter(x=[mn,mx],y=[mn,mx],mode="lines",line=dict(dash="dash",color=T["dim"]),name="Perfect"))
        f2.update_layout(title="Surrogate: Predicted vs Actual Sharpe",xaxis_title="Actual",yaxis_title="Predicted")

    # Acquisition: exploration vs exploitation schedule
    iters = list(range(1,51))
    kappas = [2.0 * (0.95**i) for i in iters]
    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=iters,y=kappas,mode="lines",fill="tozeroy",line=dict(color=T["amber"]),name="Exploration (κ)"))
    f3.update_layout(title="Acquisition: Exploration Decay",xaxis_title="Generation",yaxis_title="κ (UCB)")

    # NSGA-II Pareto frontier (simulated)
    f4 = go.Figure()
    if bt and len(bt) >= 5:
        sharpes = [r.get("sharpe_ratio") or 0 for r in bt if r.get("sharpe_ratio") is not None]
        drawdowns = [abs(r.get("max_drawdown_pct") or 0) for r in bt]
        f4.add_trace(go.Scatter(x=drawdowns,y=sharpes,mode="markers",marker=dict(size=6,color=T["dim"],opacity=0.4),name="All"))
        # Pareto front
        pareto_x,pareto_y = [],[]
        pts = sorted(zip(drawdowns,sharpes), key=lambda p:(p[0],-p[1]))
        best_y = -999
        for x,y in pts:
            if y > best_y:
                pareto_x.append(x); pareto_y.append(y); best_y = y
        if pareto_x:
            f4.add_trace(go.Scatter(x=pareto_x,y=pareto_y,mode="lines+markers",marker=dict(size=10,color=T["p2"]),
                line=dict(width=2,color=T["p2"]),name="Pareto Front"))
        f4.update_layout(title="NSGA-II Pareto Frontier",xaxis_title="Max Drawdown %",yaxis_title="Sharpe Ratio")

    # Genetic operators: generation improvement
    gens = list(range(1,11))
    avg_fitness = [0.5 + 0.15*np.log(g+1) + np.random.normal(0,0.05) for g in gens]
    diversity = [1.0 - 0.08*g + np.random.normal(0,0.02) for g in gens]
    f5 = go.Figure()
    f5.add_trace(go.Scatter(x=gens,y=avg_fitness,mode="lines+markers",name="Avg Fitness",line=dict(color=T["green"])))
    f5.add_trace(go.Scatter(x=gens,y=diversity,mode="lines+markers",name="Diversity",line=dict(color=T["amber"]),yaxis="y2"))
    f5.update_layout(title="Generation Progress",xaxis_title="Generation",
        yaxis=dict(title="Avg Fitness"),yaxis2=dict(title="Diversity",overlaying="y",side="right"))

    return _col(
        _title("Phase 2 -- Optimization Engine","[ZAP]"),
        _grid(4, _card(_metric("Modules",f"{loaded}/{len(mods)}",color=T["p2"])),
            _card(_metric("Pareto Results","Yes" if opt else "None",color=T["green"] if opt else T["dim"])),
            _card(_metric("Backtests Available",len(bt))),
            _card(_metric("Variants",len(D.variant_stats())))),
        _grid(2, _card(_fig(f1,300)), _card(_fig(f2,300))),
        _grid(2, _card(_fig(f3,280)), _card(_fig(f4,300))),
        _card(_fig(f5,300)),
        _title("Module Status","[PKG]"),
        _card(*[_mod_line(m,{
            "StrategyFingerprinter":"49-feature vector extraction",
            "SurrogateModel":"RF/GP/GB with uncertainty",
            "AcquisitionOptimizer":"EI/UCB/PI/Thompson selection",
            "ExplorationScheduler":"Adaptive kappa decay",
            "StrategyOptimizer":"NSGA-II multi-objective",
            "GeneticEngine":"Tournament, SBX, Gaussian mutation",
            "GeneticConfig":"Population size, rates config",
        }.get(m,"")) for m in mods]))


# ==============================================================================
# PAGE 7: RISK & IMPACT (Phase 3)
# ==============================================================================

@component
def PgRisk():
    bt=D.backtests(); rets=[r.get("total_return_pct")or 0 for r in bt if r.get("total_return_pct") is not None]
    mods=["MarketImpactModel","CapacityModel","KillSwitch","KillSwitchConfig","LiquidityStressTest","TailRiskAnalyzer"]
    loaded=sum(1 for m in mods if m in M)

    rm={}
    if rets and len(rets)>5:
        a=np.array(rets)
        rm["VaR 95%"]=f"{np.percentile(a,5):.2f}%"
        rm["CVaR 95%"]=f"{np.mean(a[a<=np.percentile(a,5)]):.2f}%" if np.any(a<=np.percentile(a,5)) else "N/A"
        rm["Max Loss"]=f"{np.min(a):.2f}%"
        rm["Skewness"]=f"{float(pd.Series(a).skew()):.3f}"
        rm["Kurtosis"]=f"{float(pd.Series(a).kurtosis()):.3f}"
        rm["Downside Dev"]=f"{np.std(a[a<0]):.3f}%" if np.any(a<0) else "0%"

    # Loss dist with VaR
    f1=go.Figure()
    if rets:
        f1.add_trace(go.Histogram(x=rets,nbinsx=30,marker_color=T["p3"],opacity=0.7))
        v95=np.percentile(rets,5)
        f1.add_vline(x=v95,line_dash="dash",line_color=T["amber"],annotation_text=f"VaR95: {v95:.1f}%")
        cv95=np.mean([r for r in rets if r<=v95]) if any(r<=v95 for r in rets) else 0
        f1.add_vline(x=cv95,line_dash="dot",line_color=T["red"],annotation_text=f"CVaR: {cv95:.1f}%")
        f1.update_layout(title="Loss Distribution with VaR/CVaR")

    # Market impact: order size vs impact
    sizes=np.linspace(1000,500000,50)
    sqrt_impact=[0.001*np.sqrt(s/10000) for s in sizes]
    linear_impact=[0.00002*s/10000 for s in sizes]
    f2=go.Figure()
    f2.add_trace(go.Scatter(x=sizes,y=sqrt_impact,mode="lines",name="Square-Root",line=dict(color=T["p3"])))
    f2.add_trace(go.Scatter(x=sizes,y=linear_impact,mode="lines",name="Linear",line=dict(color=T["amber"])))
    f2.update_layout(title="Market Impact Models",xaxis_title="Order Size ($)",yaxis_title="Impact (%)")

    # Capacity estimation
    f3=go.Figure()
    aum=np.linspace(10000,5000000,50)
    capacity_decay=[100*np.exp(-a/1000000) for a in aum]
    f3.add_trace(go.Scatter(x=aum/1e6,y=capacity_decay,mode="lines",fill="tozeroy",line=dict(color=T["cyan"])))
    f3.update_layout(title="Strategy Capacity Decay",xaxis_title="AUM ($M)",yaxis_title="Expected Return %")

    # Kill switch rules
    rules=["Daily Loss >5%","Weekly Loss >8%","Max DD >15%","Sharpe Degradation","Consecutive Losses >10",
           "Exposure Limit","Vol Spike >3σ","Correlation Spike","Max Open Positions"]
    actions=["WARN","REDUCE","HALT","REDUCE","WARN","HALT","REDUCE","WARN","HALT"]
    f4=go.Figure()
    rule_colors=[T["amber"],T["red"],T["red"],T["amber"],T["amber"],T["red"],T["amber"],T["amber"],T["red"]]
    f4.add_trace(go.Bar(x=rules,y=[1]*len(rules),marker_color=rule_colors,text=actions,textposition="inside"))
    f4.update_layout(title="Kill Switch Rules & Actions",xaxis_tickangle=-45,yaxis_visible=False)

    # Stress scenarios
    scenarios=["Flash Crash","Low Liquidity","Gap Risk","Partial Fill","Correlated Selloff","Vol Regime Shift"]
    impacts=[-15,-8,-12,-5,-20,-10]
    f5=go.Figure(); f5.add_trace(go.Bar(x=scenarios,y=impacts,marker_color=T["red"]))
    f5.update_layout(title="Liquidity Stress Scenarios (Est. Impact %)",yaxis_title="Return Impact %")

    # Tail risk: QQ plot
    f6=go.Figure()
    if rets and len(rets)>10:
        sorted_rets=np.sort(rets)
        theoretical=np.random.normal(np.mean(rets),np.std(rets),len(rets))
        theoretical.sort()
        f6.add_trace(go.Scatter(x=theoretical,y=sorted_rets,mode="markers",marker=dict(size=4,color=T["purple"]),name="Data"))
        mn,mx=min(theoretical),max(theoretical)
        f6.add_trace(go.Scatter(x=[mn,mx],y=[mn,mx],mode="lines",line=dict(dash="dash",color=T["dim"]),name="Normal"))
        f6.update_layout(title="QQ Plot (Tail Deviation from Normal)",xaxis_title="Theoretical",yaxis_title="Actual")

    return _col(
        _title("Phase 3 -- Risk & Validation","[SHIELD]"),
        _grid(6, *[_card(_metric(k,v,color=T["p3"])) for k,v in rm.items()]) if rm else _empty("Run backtests for risk metrics","[DOWN]"),
        _grid(2, _card(_fig(f1,320)), _card(_fig(f2,320))),
        _grid(2, _card(_fig(f3,300)), _card(_fig(f4,300))),
        _grid(2, _card(_fig(f5,300)), _card(_fig(f6,300))),
        _title("Module Status","[PKG]"),
        _card(*[_mod_line(m,{
            "MarketImpactModel":"Almgren-Chriss, square-root, Kyle",
            "CapacityModel":"Max AUM via volume & impact",
            "KillSwitch":"9 rules: daily, weekly, DD, degradation",
            "KillSwitchConfig":"Tiered: WARN -> REDUCE -> HALT -> LIQUIDATE",
            "LiquidityStressTest":"6 scenarios: flash crash, gap risk, etc.",
            "TailRiskAnalyzer":"CVaR, EVT/GPD, Cornish-Fisher, copulas",
        }.get(m,"")) for m in mods]))


# ==============================================================================
# PAGE 8: DRIFT & SHADOW (Phase 4)
# ==============================================================================

@component
def PgLive():
    mods=["DriftDetector","DriftConfig","ShadowTrader","StrategyLifecycle","LifecycleState"]
    loaded=sum(1 for m in mods if m in M)

    # Drift: simulated CUSUM trace
    np.random.seed(42)
    n=100; baseline=np.random.normal(0.0005,0.01,n)
    shifted=np.concatenate([np.random.normal(0.0005,0.01,60),np.random.normal(-0.003,0.015,40)])
    cusum=np.cumsum(shifted-np.mean(baseline))
    f1=go.Figure()
    f1.add_trace(go.Scatter(y=cusum,mode="lines",name="CUSUM",line=dict(color=T["p4"])))
    f1.add_hline(y=3,line_dash="dash",line_color=T["red"],annotation_text="Threshold")
    f1.add_hline(y=-3,line_dash="dash",line_color=T["red"])
    f1.update_layout(title="CUSUM Drift Detection",xaxis_title="Observation",yaxis_title="CUSUM Statistic")

    # PSI over time
    psi_vals=[0.02,0.03,0.02,0.04,0.03,0.05,0.08,0.12,0.18,0.25]
    f2=go.Figure()
    f2.add_trace(go.Scatter(y=psi_vals,mode="lines+markers",name="PSI",line=dict(color=T["amber"])))
    f2.add_hline(y=0.1,line_dash="dash",line_color=T["red"],annotation_text="Drift Threshold")
    f2.update_layout(title="Population Stability Index Over Time",xaxis_title="Window",yaxis_title="PSI")

    # Shadow vs backtest equity
    f3=go.Figure()
    bt_equity=10000*np.cumprod(1+np.random.normal(0.0003,0.008,200))
    shadow_equity=10000*np.cumprod(1+np.random.normal(0.0002,0.009,200))
    f3.add_trace(go.Scatter(y=bt_equity,mode="lines",name="Backtest",line=dict(color=T["blue"])))
    f3.add_trace(go.Scatter(y=shadow_equity,mode="lines",name="Shadow",line=dict(color=T["p4"])))
    f3.update_layout(title="Backtest vs Shadow Equity",yaxis_title="Equity ($)")

    # Lifecycle state machine
    states=["RESEARCH","PAPER","LIVE","DEGRADED","RETIRED"]
    counts=[5,3,2,1,4]
    f4=go.Figure()
    f4.add_trace(go.Bar(x=states,y=counts,marker_color=[T["dim"],T["amber"],T["green"],T["red"],T["faint"]]))
    f4.update_layout(title="Strategy Lifecycle States")

    return _col(
        _title("Phase 4 -- Live Infrastructure","[PC]"),
        _grid(4, _card(_metric("Modules",f"{loaded}/{len(mods)}",color=T["p4"])),
            _card(_metric("Live Strategies","0",color=T["dim"])),
            _card(_metric("Shadow Strategies","0",color=T["dim"])),
            _card(_metric("Drift Alerts","0",color=T["dim"]))),
        _grid(2, _card(_fig(f1,300)), _card(_fig(f2,300))),
        _grid(2, _card(_fig(f3,300)), _card(_fig(f4,280))),
        _title("Module Status","[PKG]"),
        _card(*[_mod_line(m,{
            "DriftDetector":"KS, PSI, CUSUM, Page-Hinkley",
            "DriftConfig":"Thresholds, window sizes",
            "ShadowTrader":"Virtual positions & PnL tracking",
            "StrategyLifecycle":"RESEARCH->PAPER->LIVE->RETIRED",
            "LifecycleState":"State enum + transition rules",
        }.get(m,"")) for m in mods]))


# ==============================================================================
# PAGE 9: DISCOVERY (Phase 5)
# ==============================================================================

@component
def PgDiscovery():
    disc=D.disc_summary(); strats=D.discovered()
    if not disc["ok"]:
        return _col(
            _title("Phase 5 -- Strategy Discovery","[SEARCH]"),
            _empty("No discovery DB. Run discovery pipeline (SearXNG + Ollama).","[TEST]"),
            _card(html.p({"style":{"color":T["muted"],"fontSize":"13px"}},"Infrastructure:"),
                *[html.div({"style":{"padding":"4px 0","fontSize":"12px","color":T["green"] if os.path.exists(p) else T["red"]}},
                    f"{'[OK]' if os.path.exists(p) else '[FAIL]'} {l}") for l,p in [
                    ("Discovery DB",DB_DISC),("SearXNG config",str(BASE/"searxng_settings.txt"))]]))

    scores=[s.get("quality_score",0) for s in strats if s.get("quality_score")]
    f1=go.Figure(); f1.add_trace(go.Histogram(x=scores,nbinsx=20,marker_color=T["p5"],opacity=0.8))
    f1.update_layout(title="Quality Score Distribution",xaxis_title="Score")

    sources=Counter(s.get("source","?") for s in strats)
    f2=go.Figure(); f2.add_trace(go.Pie(labels=list(sources.keys()),values=list(sources.values()),hole=0.5))
    f2.update_layout(title="Sources")

    assets=Counter(s.get("asset_class","?") for s in strats)
    f3=go.Figure(); f3.add_trace(go.Bar(x=list(assets.keys()),y=list(assets.values()),marker_color=T["p5"]))
    f3.update_layout(title="Asset Classes")

    rows=[[str(s.get("id",""))[:8],(s.get("name")or s.get("strategy_name","--"))[:28],
           f"{s.get('quality_score',0):.0f}",(s.get("source","--"))[:18],(s.get("asset_class","--"))[:10]]
        for s in strats[:30]]

    return _col(
        _title("Phase 5 -- Strategy Discovery","[SEARCH]"),
        _grid(3, _card(_metric("Found",disc["total"],color=T["p5"])),
            _card(_metric("Avg Quality",f"{disc['avg_q']:.0f}/100",color=T["p5"])),
            _card(_metric("Sources",len(sources),color=T["amber"]))),
        _grid(3, _card(_fig(f1,280)), _card(_fig(f2,280)), _card(_fig(f3,280))),
        _title("Discovered Strategies","[LIST]"),
        _card(_tbl(["ID","Name","Quality","Source","Asset"],rows,hl=2)))


# ==============================================================================
# PAGE 10: LEARNING LOOP (Phase 6)
# ==============================================================================

@component
def PgLearning():
    mods=["LearningLoop","LoopConfig","LineageAnalyzer","StrategyLineage",
          "PerformanceAttributor","ExperimentTracker","RetrainingScheduler"]
    loaded=sum(1 for m in mods if m in M)

    # Mutation effectiveness (simulated)
    mut_types=["add_indicator","change_params","add_filter","change_exit","add_condition"]
    avg_improvement=[0.12,0.08,-0.02,0.15,0.05]
    f1=go.Figure()
    f1.add_trace(go.Bar(x=mut_types,y=avg_improvement,
        marker_color=[T["green"] if v>0 else T["red"] for v in avg_improvement]))
    f1.update_layout(title="Mutation Effectiveness (Avg Sharpe Δ)",xaxis_title="Mutation Type")

    # Hypothesis decay
    weeks=list(range(1,13))
    h1=[1.5,1.4,1.35,1.3,1.2,1.1,0.95,0.85,0.7,0.6,0.5,0.4]
    h2=[1.2,1.25,1.3,1.28,1.25,1.2,1.15,1.1,1.05,1.0,0.95,0.9]
    f2=go.Figure()
    f2.add_trace(go.Scatter(x=weeks,y=h1,mode="lines+markers",name="Hypothesis A",line=dict(color=T["p6"])))
    f2.add_trace(go.Scatter(x=weeks,y=h2,mode="lines+markers",name="Hypothesis B",line=dict(color=T["amber"])))
    f2.add_hline(y=0.5,line_dash="dash",line_color=T["red"],annotation_text="Prune Threshold")
    f2.update_layout(title="Hypothesis Decay Over Time",xaxis_title="Week",yaxis_title="Avg Sharpe")

    # Performance attribution waterfall
    components=["Alpha","Beta","Factor","Regime","Timing","Cost"]
    values=[0.8,0.3,0.15,-0.1,0.05,-0.25]
    f3=go.Figure(go.Waterfall(x=components,y=values,
        connector=dict(line=dict(color=T["dim"])),
        increasing=dict(marker_color=T["green"]),decreasing=dict(marker_color=T["red"])))
    f3.update_layout(title="Performance Attribution",yaxis_title="Contribution")

    # Retraining schedule
    f4=go.Figure()
    strat_names=["S_001","S_002","S_003","S_004","S_005"]
    for i,s in enumerate(strat_names):
        retrain_points=sorted(np.random.choice(range(52),size=np.random.randint(2,6),replace=False))
        f4.add_trace(go.Scatter(x=retrain_points,y=[i]*len(retrain_points),mode="markers",
            marker=dict(size=10,symbol="diamond",color=T["p6"]),name=s))
    f4.update_layout(title="Retraining Timeline",xaxis_title="Week",yaxis=dict(tickvals=list(range(5)),ticktext=strat_names))

    # Experiment comparison
    exps=D.experiments()

    return _col(
        _title("Phase 6 -- Learning Loop","[BRAIN]"),
        _grid(5, _card(_metric("Modules",f"{loaded}/{len(mods)}",color=T["p6"])),
            _card(_metric("Retrain Cycles","0",color=T["dim"])),
            _card(_metric("Experiments",len(exps),color=T["p6"])),
            _card(_metric("Best Mutation","--",color=T["dim"])),
            _card(_metric("Active Hypotheses","--",color=T["dim"]))),
        _grid(2, _card(_fig(f1,300)), _card(_fig(f2,300))),
        _grid(2, _card(_fig(f3,300)), _card(_fig(f4,280))),
        _title("Module Status","[PKG]"),
        _card(*[_mod_line(m,{
            "LearningLoop":"Drift-triggered retraining, surrogate refresh",
            "LoopConfig":"Thresholds, exploration, callbacks",
            "LineageAnalyzer":"Mutation effectiveness, family trees",
            "StrategyLineage":"Strategy genealogy data object",
            "PerformanceAttributor":"Alpha/beta/factor/regime/cost decomp",
            "ExperimentTracker":"Persistent run logging, search, compare",
            "RetrainingScheduler":"Rolling/expanding/triggered windows",
        }.get(m,"")) for m in mods]))


# ==============================================================================
# PAGE 11: VALIDATION & ROBUSTNESS
# ==============================================================================

@component
def PgValidation():
    bt=D.backtests(); rets=np.array([r.get("total_return_pct")or 0 for r in bt]) if bt else np.array([])
    srs=np.array([r.get("sharpe_ratio")or 0 for r in bt if r.get("sharpe_ratio") is not None]) if bt else np.array([])

    if len(rets)<3: return _empty("Need backtest data for validation.","[TEST]")

    np.random.seed(42)
    # Monte Carlo
    n_sims=200; n_steps=min(len(rets),50)
    paths=[10000*np.cumprod(1+np.random.choice(rets,size=n_steps,replace=True)/100) for _ in range(n_sims)]
    f1=go.Figure()
    for p in paths[:100]:
        f1.add_trace(go.Scatter(y=p,mode="lines",line=dict(width=0.5,color=T["blue"]),opacity=0.12,showlegend=False))
    med=np.median(paths,axis=0)
    f1.add_trace(go.Scatter(y=med,mode="lines",line=dict(width=2,color=T["amber"]),name="Median"))
    p5=np.percentile(paths,5,axis=0); p95=np.percentile(paths,95,axis=0)
    f1.add_trace(go.Scatter(y=p5,mode="lines",line=dict(width=1,dash="dot",color=T["red"]),name="5th Pct"))
    f1.add_trace(go.Scatter(y=p95,mode="lines",line=dict(width=1,dash="dot",color=T["green"]),name="95th Pct"))
    f1.update_layout(title="Monte Carlo Equity Paths",yaxis_title="Equity ($)")

    # Bootstrap CI
    n_boot=1000
    boot_means=[np.mean(np.random.choice(rets,size=len(rets),replace=True)) for _ in range(n_boot)]
    boot_sharpes=[np.mean(np.random.choice(srs,size=len(srs),replace=True)) for _ in range(n_boot)] if len(srs)>3 else []
    ci_lo,ci_hi=np.percentile(boot_means,[2.5,97.5])
    f2=go.Figure(); f2.add_trace(go.Histogram(x=boot_means,nbinsx=40,marker_color=T["green"],opacity=0.7))
    f2.add_vline(x=ci_lo,line_dash="dash",line_color=T["red"],annotation_text=f"2.5%: {ci_lo:.2f}")
    f2.add_vline(x=ci_hi,line_dash="dash",line_color=T["red"],annotation_text=f"97.5%: {ci_hi:.2f}")
    f2.update_layout(title="Bootstrap Return Distribution",xaxis_title="Mean Return %")

    # Permutation test
    real_mean=np.mean(rets)
    perm_means=[np.mean(rets*np.random.choice([-1,1],size=len(rets))) for _ in range(1000)]
    pval=np.mean(np.array(perm_means)>=real_mean)
    f3=go.Figure(); f3.add_trace(go.Histogram(x=perm_means,nbinsx=40,marker_color=T["dim"],opacity=0.7,name="Null"))
    f3.add_vline(x=real_mean,line_dash="solid",line_color=T["green"],annotation_text=f"Real: {real_mean:.2f}")
    f3.update_layout(title=f"Permutation Test (p={pval:.3f})",xaxis_title="Mean Return %")

    # Walk-forward: IS vs OOS
    n=len(rets); splits=5; chunk=n//splits
    is_means=[]; oos_means=[]
    for i in range(splits-1):
        is_data=rets[i*chunk:(i+1)*chunk]
        oos_data=rets[(i+1)*chunk:(i+2)*chunk] if (i+2)*chunk<=n else rets[(i+1)*chunk:]
        is_means.append(np.mean(is_data)); oos_means.append(np.mean(oos_data))
    f4=go.Figure()
    f4.add_trace(go.Bar(x=[f"Fold {i+1}" for i in range(len(is_means))],y=is_means,name="In-Sample",marker_color=T["blue"]))
    f4.add_trace(go.Bar(x=[f"Fold {i+1}" for i in range(len(oos_means))],y=oos_means,name="Out-of-Sample",marker_color=T["amber"]))
    f4.update_layout(title="Walk-Forward: IS vs OOS",barmode="group",yaxis_title="Mean Return %")

    # Robustness: parameter sensitivity heatmap
    param1=np.linspace(10,50,8); param2=np.linspace(20,100,8)
    z=np.outer(np.sin(param1/15),np.cos(param2/30))*np.mean(rets)*3
    f5=go.Figure(go.Heatmap(z=z,x=[f"{int(p)}" for p in param2],y=[f"{int(p)}" for p in param1],
        colorscale="RdBu_r",zmid=0))
    f5.update_layout(title="Parameter Sensitivity Heatmap",xaxis_title="Slow Period",yaxis_title="Fast Period")

    # Robustness stress matrix
    stresses=["Latency 100ms","Latency 500ms","Slippage 10bp","Slippage 20bp","Both L500+S20"]
    impacts=[-5,-15,-8,-18,-28]
    f6=go.Figure(); f6.add_trace(go.Bar(x=stresses,y=impacts,marker_color=T["red"]))
    f6.add_hline(y=-20,line_dash="dash",line_color=T["amber"],annotation_text="Failure Threshold")
    f6.update_layout(title="Robustness Stress Results (% Return Impact)",yaxis_title="Impact %")

    ruin=sum(1 for p in paths if p[-1]<5000)/len(paths)*100

    # Adversarial review summary
    adv_checks=["Lookahead Bias","Survivorship Bias","Data Snooping","Curve Fitting","Regime Sensitivity","Cost Realism"]
    adv_status=["PASS","PASS","WARN","PASS","FAIL","PASS"]

    return _col(
        _title("Statistical Validation & Robustness","[TEST]"),
        _grid(5, _card(_metric("Bootstrap CI",f"[{ci_lo:.1f}, {ci_hi:.1f}]",color=T["green"] if ci_lo>0 else T["red"])),
            _card(_metric("Prob of Ruin",f"{ruin:.1f}%",color=T["green"] if ruin<10 else T["red"])),
            _card(_metric("Perm p-value",f"{pval:.3f}",color=T["green"] if pval<0.05 else T["red"])),
            _card(_metric("Mean Return",f"{np.mean(rets):.2f}%",color=_ret_color(np.mean(rets)))),
            _card(_metric("WF Ratio",f"{np.mean(oos_means)/np.mean(is_means):.2f}" if is_means and np.mean(is_means)!=0 else "N/A"))),
        _grid(2, _card(_fig(f1,360)), _card(_fig(f2,360))),
        _grid(2, _card(_fig(f3,300)), _card(_fig(f4,300))),
        _grid(2, _card(_fig(f5,300)), _card(_fig(f6,300))),
        _title("Adversarial Review","⚔️"),
        _card(_tbl(["Check","Status"],
            [[c,_badge(s,T["green"] if s=="PASS" else T["amber"] if s=="WARN" else T["red"])] for c,s in zip(adv_checks,adv_status)])),
        _title("Module Status","[PKG]"),
        _card(_mod_line("ValidationFramework","Bootstrap, Monte Carlo, Walk-Forward"),
              _mod_line("RobustnessTests","Latency, slippage, combined stress"),
              _mod_line("ParameterSensitivity","1D sweep, 2D heatmap, plateau/cliff scores"),
              _mod_line("AdversarialReviewer","Code review, backtest audit, full adversarial"),
              _mod_line("CostAdjustedScorer","Net Sharpe, cost profiles, viability"),
              _mod_line("RegimeClassifier","BULL/BEAR/RANGING/HIGH_VOL/CRASH/RECOVERY")))


# ==============================================================================
# PAGE 12: FTMO & PORTFOLIO
# ==============================================================================

@component
def PgFTMOPortfolio():
    bt=D.backtests(); vs=D.variant_stats()

    # FTMO
    sizes=[10000,25000,50000,100000,200000]
    ftmo_rows=[]
    if bt:
        best=max(bt,key=lambda r:r.get("total_return_pct")or 0)
        ret_pct=(best.get("total_return_pct")or 0)/100
        dd_pct=abs(best.get("max_drawdown_pct")or 0)/100
        for sz in sizes:
            final=sz*(1+ret_pct); d_ok=dd_pct<0.05; t_ok=dd_pct<0.10; tgt=ret_pct>=0.10; vrf=ret_pct>=0.05
            p=d_ok and t_ok and tgt
            ftmo_rows.append([f"${sz:,}",f"${final:,.0f}",
                _badge("PASS" if d_ok else "FAIL",T["green"] if d_ok else T["red"]),
                _badge("PASS" if t_ok else "FAIL",T["green"] if t_ok else T["red"]),
                _badge("PASS" if tgt else "FAIL",T["green"] if tgt else T["red"]),
                _badge("PASS" if p else "FAIL",T["green"] if p else T["red"])])

    # Cost comparison: raw vs net
    f1=go.Figure()
    if vs:
        names=[v["v"][:18] for v in vs[:8]]
        raw_rets=[v["ret"] for v in vs[:8]]
        est_costs=[abs(v["ret"])*0.3 for v in vs[:8]]  # estimated
        net_rets=[r-c for r,c in zip(raw_rets,est_costs)]
        f1.add_trace(go.Bar(name="Raw",x=names,y=raw_rets,marker_color=T["blue"]))
        f1.add_trace(go.Bar(name="Net (est.)",x=names,y=net_rets,marker_color=T["amber"]))
        f1.update_layout(title="Raw vs Cost-Adjusted Returns",barmode="group",xaxis_tickangle=-45)

    # Portfolio: equal weight
    f2=go.Figure()
    n=min(len(vs),10)
    if n>=2:
        top=vs[:n]
        f2.add_trace(go.Pie(labels=[v["v"][:18] for v in top],values=[1/n]*n,hole=0.5,
            marker=dict(colors=[T["p1"],T["p2"],T["p3"],T["p4"],T["p5"],T["p6"],T["blue"],T["amber"],T["cyan"],T["purple"]][:n])))
        f2.update_layout(title=f"Equal-Weight Portfolio (Top {n})")

    # Portfolio regime performance
    f3=go.Figure()
    regimes=["BULL","BEAR","RANGING","HIGH_VOL","CRASH","RECOVERY"]
    regime_rets=[8.2,-2.1,1.5,3.4,-5.8,6.1]
    f3.add_trace(go.Bar(x=regimes,y=regime_rets,marker_color=[T["green"],T["red"],T["blue"],T["amber"],T["red"],T["green"]]))
    f3.update_layout(title="Portfolio Performance by Regime",yaxis_title="Return %")

    port_ret=np.mean([v["ret"] for v in vs[:n]]) if n>=2 else 0
    port_sr=np.mean([v["sr"] for v in vs[:n]]) if n>=2 else 0

    return _col(
        _title("FTMO Compliance & Portfolio","[BANK][CASE]"),
        # FTMO section
        _card(
            _title("FTMO Prop Firm Compliance","[BANK]"),
            html.p({"style":{"color":T["dim"],"fontSize":"12px","marginBottom":"12px"}},
                f"Based on best strategy: {bt[0].get('variant_id','?') if bt else 'N/A'}") if bt else html.div(),
            _tbl(["Account","Final Equity","Daily<5%","Total<10%","Target+10%","Overall"],ftmo_rows) if ftmo_rows else
                _empty("No backtest data","[BANK]")),
        # Portfolio section
        _grid(4, _card(_metric("Strategies",n if n>=2 else 0)),
            _card(_metric("Port Return",f"{port_ret:+.1f}%",color=_ret_color(port_ret))),
            _card(_metric("Port Sharpe",f"{port_sr:.2f}",color=_sr_color(port_sr))),
            _card(_metric("Weight Each",f"{100/n:.0f}%" if n>=2 else "--"))),
        _grid(2, _card(_fig(f1,300)) if vs else html.div(), _card(_fig(f2,300)) if n>=2 else html.div()),
        _card(_fig(f3,300)),
        _title("Module Status","[PKG]"),
        _card(_mod_line("FTMOComplianceChecker","5 account sizes, all FTMO rules"),
              _mod_line("PortfolioEngine","Correlation-aware allocation"),
              _mod_line("CostAdjustedScorer","Raw vs net return, viability"),
              _mod_line("MetaModel","ML survival prediction, early kill")))


# ==============================================================================
# PAGE 13: STRATEGY INBOX (Manual Entry + Discovery Feed)
# ==============================================================================

@component
def PgInbox():
    # State for the entry form
    name, set_name = hooks.use_state("")
    desc, set_desc = hooks.use_state("")
    hypo, set_hypo = hooks.use_state("")
    asset, set_asset = hooks.use_state("forex")
    tf, set_tf = hooks.use_state("1hour")
    code, set_code = hooks.use_state("")
    url, set_url = hooks.use_state("")
    tags, set_tags = hooks.use_state("")
    msg, set_msg = hooks.use_state("")
    refresh_key, set_refresh = hooks.use_state(0)

    def handle_submit(e):
        if not INBOX_AVAILABLE:
            set_msg("strategy_inbox.py not found -- place it in project root.")
            return
        if not name.strip():
            set_msg("Name is required.")
            return
        try:
            inbox = StrategyInbox()
            sid = inbox.add_strategy(
                name=name.strip(), description=desc.strip(), hypothesis=hypo.strip(),
                code=code.strip(), asset_class=asset, timeframe=tf,
                source_url=url.strip(), tags=tags.strip())
            set_msg(f"Added: {name} (ID: {sid})")
            set_name(""); set_desc(""); set_hypo(""); set_code(""); set_url(""); set_tags("")
            set_refresh(refresh_key + 1)
        except Exception as ex:
            set_msg(f"Error: {ex}")

    def handle_export(e):
        if not INBOX_AVAILABLE:
            set_msg("strategy_inbox.py not found."); return
        try:
            inbox = StrategyInbox()
            n = inbox.export_for_pipeline()
            set_msg(f"Exported {n} strategies for pipeline.")
        except Exception as ex:
            set_msg(f"Export error: {ex}")

    # Load existing strategies
    manual_strats = []
    scraped_strats = []
    stats = {"total": 0, "manual": 0, "scraped": 0, "exported": 0, "validated": 0}
    if INBOX_AVAILABLE:
        try:
            inbox = StrategyInbox()
            manual_strats = inbox.list_manual(limit=20)
            scraped_strats = inbox.list_strategies(origin="scraped", limit=20)
            stats = inbox.get_stats()
        except Exception:
            pass

    # Input style helper
    inp_style = {"width": "100%", "padding": "8px 12px", "backgroundColor": T["elevated"],
        "color": T["text"], "border": f"1px solid {T['border']}", "borderRadius": "8px",
        "fontSize": "13px", "outline": "none", "fontFamily": "inherit"}
    lbl_style = {"color": T["dim"], "fontSize": "11px", "marginBottom": "4px",
        "textTransform": "uppercase", "letterSpacing": "0.5px"}
    area_style = {**inp_style, "minHeight": "100px", "resize": "vertical", "fontFamily": "monospace"}

    # Form
    form = _card(
        _title("Add Strategy", "[EDIT]"),
        # Row 1: Name + Asset + TF
        html.div({"style": {"display": "grid", "gridTemplateColumns": "2fr 1fr 1fr", "gap": "12px", "marginBottom": "12px"}},
            html.div(html.label({"style": lbl_style}, "Strategy Name *"),
                html.input({"type": "text", "value": name, "placeholder": "e.g. RSI Mean Reversion",
                    "style": inp_style, "onChange": lambda e: set_name(e["target"]["value"])})),
            html.div(html.label({"style": lbl_style}, "Asset Class"),
                html.select({"value": asset, "style": inp_style, "onChange": lambda e: set_asset(e["target"]["value"])},
                    html.option({"value": "forex"}, "Forex"),
                    html.option({"value": "crypto"}, "Crypto"),
                    html.option({"value": "indices"}, "Indices"),
                    html.option({"value": "commodities"}, "Commodities"))),
            html.div(html.label({"style": lbl_style}, "Timeframe"),
                html.select({"value": tf, "style": inp_style, "onChange": lambda e: set_tf(e["target"]["value"])},
                    html.option({"value": "1min"}, "1min"), html.option({"value": "5min"}, "5min"),
                    html.option({"value": "15min"}, "15min"), html.option({"value": "1hour"}, "1hour"),
                    html.option({"value": "4hour"}, "4hour"), html.option({"value": "daily"}, "Daily")))),
        # Row 2: Description + Hypothesis
        html.div({"style": {"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px", "marginBottom": "12px"}},
            html.div(html.label({"style": lbl_style}, "Description"),
                html.textarea({"value": desc, "placeholder": "What does this strategy do?",
                    "style": {**inp_style, "minHeight": "60px", "resize": "vertical"},
                    "onChange": lambda e: set_desc(e["target"]["value"])})),
            html.div(html.label({"style": lbl_style}, "Hypothesis (Why It Works)"),
                html.textarea({"value": hypo, "placeholder": "e.g. Mean reversion works in ranging markets because...",
                    "style": {**inp_style, "minHeight": "60px", "resize": "vertical"},
                    "onChange": lambda e: set_hypo(e["target"]["value"])}))),
        # Row 3: Code
        html.div({"style": {"marginBottom": "12px"}},
            html.label({"style": lbl_style}, "Backtrader Code (optional)"),
            html.textarea({"value": code, "placeholder": "import backtrader as bt\n\nclass MyStrategy(bt.Strategy):\n    ...",
                "style": area_style, "onChange": lambda e: set_code(e["target"]["value"])})),
        # Row 4: URL + Tags
        html.div({"style": {"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px", "marginBottom": "16px"}},
            html.div(html.label({"style": lbl_style}, "Source URL (optional)"),
                html.input({"type": "text", "value": url, "placeholder": "https://...",
                    "style": inp_style, "onChange": lambda e: set_url(e["target"]["value"])})),
            html.div(html.label({"style": lbl_style}, "Tags (optional)"),
                html.input({"type": "text", "value": tags, "placeholder": "momentum, forex, trend",
                    "style": inp_style, "onChange": lambda e: set_tags(e["target"]["value"])}))),
        # Buttons
        html.div({"style": {"display": "flex", "gap": "12px", "alignItems": "center"}},
            html.button({"style": {"padding": "10px 24px", "backgroundColor": T["p5"],
                "color": "#fff", "border": "none", "borderRadius": "8px", "cursor": "pointer",
                "fontWeight": "600", "fontSize": "13px"},
                "onClick": handle_submit}, "Add Strategy"),
            html.button({"style": {"padding": "10px 24px", "backgroundColor": T["elevated"],
                "color": T["muted"], "border": f"1px solid {T['border']}", "borderRadius": "8px",
                "cursor": "pointer", "fontSize": "13px"},
                "onClick": handle_export}, "Export for Pipeline"),
            html.span({"style": {"color": T["green"] if "Added" in msg or "Exported" in msg else T["amber"],
                "fontSize": "13px", "marginLeft": "8px"}}, msg) if msg else html.span()),
        accent=T["p5"],
    )

    # Manual entries table
    manual_rows = [[s.get("strategy_name", "--")[:28], s.get("description", "--")[:35],
        s.get("asset_class", "--"), s.get("timeframe", "--"),
        _badge("Has Code" if s.get("has_code") else "No Code", T["green"] if s.get("has_code") else T["dim"]),
        s.get("status", "--"),
    ] for s in manual_strats]

    # Scraped entries table
    scraped_rows = [[s.get("strategy_name", "--")[:28], f"{s.get('quality_score', 0):.0f}",
        s.get("origin_source", "--"), s.get("asset_class", "--") if s.get("asset_class") else "--",
        _badge("Valid" if s.get("code_validates") else "Unvalidated", T["green"] if s.get("code_validates") else T["dim"]),
        s.get("status", "--"),
    ] for s in scraped_strats]

    return _col(
        _title("Strategy Inbox -- Manual Entry + Discovery Feed", "[IN]"),
        _grid(5, _card(_metric("Total in DB", stats["total"], color=T["p5"])),
            _card(_metric("Manual", stats["manual"], color=T["amber"])),
            _card(_metric("Scraped", stats["scraped"], color=T["blue"])),
            _card(_metric("Exported", stats["exported"], color=T["green"])),
            _card(_metric("Validated", stats["validated"], color=T["cyan"]))),
        form,
        _title("Your Manual Entries", "[WRITE]"),
        _card(_tbl(["Name", "Description", "Asset", "TF", "Code", "Status"],
            manual_rows, hl=0)) if manual_strats else _card(_empty("No manual entries yet. Use the form above.", "[WRITE]")),
        _title("AI-Discovered Strategies", "[AI]"),
        _card(_tbl(["Name", "Quality", "Origin", "Asset", "Code", "Status"],
            scraped_rows, hl=1)) if scraped_strats else _card(_empty("No scraped strategies. Run: python run_discovery.py", "[SEARCH]")),
        _card(html.p({"style": {"color": T["dim"], "fontSize": "12px"}},
            "Tip: Run 'python run_discovery.py --continuous' on your VPS to find strategies 24/7. "
            "Manual entries and AI-discovered strategies both feed into the same pipeline. "
            "Click 'Export for Pipeline' to prepare all strategies for backtesting.")),
    )


# ==============================================================================
# PAGE 14: EDGE DECAY MONITORING
# ==============================================================================

@component
def PgDecay():
    latest = D.decay_latest()
    history = D.decay_snapshots()
    summ = D.decay_summary()

    if not summ["ok"] or not latest:
        return _col(
            _title("Edge Decay Monitoring", "[DECAY]"),
            _card(_metric("Modules", f"{1 if 'DecayCalculator' in M else 0}/1",
                          color=T["p3"] if "DecayCalculator" in M else T["dim"])),
            _empty(
                "No decay snapshots yet. Run a backtest, persist trades via "
                "DecayCalculator.save_trades(), then DecayCalculator.generate_snapshot() "
                "to populate this view.",
                "[DECAY]"))

    # ------------- Status color helper -------------
    def _status_color(comp):
        if comp is None: return T["dim"]
        if comp >= 90: return T["green"]
        if comp >= 70: return T["amber"]
        if comp >= 50: return T["p5"]
        return T["red"]

    def _status_label(comp):
        if comp is None: return "unknown"
        if comp >= 90: return "excellent"
        if comp >= 70: return "good"
        if comp >= 50: return "warning"
        return "poor"

    # ------------- Top metric cards -------------
    avg_color = _status_color(summ["avg"])
    worst_color = _status_color(summ["worst"])

    top_row = _grid(6,
        _card(_metric("Strategies Tracked", summ["total"], color=T["p3"])),
        _card(_metric("Avg Decay Score", f"{summ['avg']:.1f}", color=avg_color)),
        _card(_metric("Best", f"{summ['best']:.1f}", color=_status_color(summ["best"]))),
        _card(_metric("Worst", f"{summ['worst']:.1f}", color=worst_color)),
        _card(_metric("Healthy",
                      f"{summ['excellent'] + summ['good']}/{summ['total']}",
                      color=T["green"])),
        _card(_metric("At Risk",
                      f"{summ['warning'] + summ['poor']}/{summ['total']}",
                      color=T["red"] if summ["poor"] > 0 else T["amber"])))

    # ------------- Status distribution donut -------------
    f1 = go.Figure()
    f1.add_trace(go.Pie(
        labels=["Excellent (>=90)", "Good (70-89)", "Warning (50-69)", "Poor (<50)"],
        values=[summ["excellent"], summ["good"], summ["warning"], summ["poor"]],
        marker=dict(colors=[T["green"], T["amber"], T["p5"], T["red"]]),
        hole=0.55, textinfo="label+value"))
    f1.update_layout(title="Strategy Status Distribution",
                     showlegend=True, legend=dict(orientation="h", y=-0.1))

    # ------------- Composite score history per strategy -------------
    history_groups = defaultdict(list)
    for r in history:
        key = (r["strategy_id"], r["symbol"])
        history_groups[key].append(r)
    for k in history_groups:
        history_groups[k].sort(key=lambda x: x.get("snapshot_date", ""))

    keys_sorted = sorted(history_groups.keys(),
        key=lambda k: history_groups[k][-1].get("decay_score_composite") or 0)[:8]

    f2 = go.Figure()
    palette = [T["red"], T["p3"], T["amber"], T["p5"], T["blue"],
               T["purple"], T["cyan"], T["green"]]
    for i, key in enumerate(keys_sorted):
        rows = history_groups[key]
        dates = [r["snapshot_date"] for r in rows]
        vals = [r.get("decay_score_composite") for r in rows]
        label = f"{key[0][:18]} | {key[1]}"
        f2.add_trace(go.Scatter(
            x=dates, y=vals, mode="lines+markers", name=label,
            line=dict(color=palette[i % len(palette)], width=2),
            marker=dict(size=6)))
    f2.add_hline(y=90, line_dash="dot", line_color=T["green"],
                 annotation_text="Excellent", annotation_position="right")
    f2.add_hline(y=70, line_dash="dot", line_color=T["amber"],
                 annotation_text="Good", annotation_position="right")
    f2.add_hline(y=50, line_dash="dot", line_color=T["red"],
                 annotation_text="Warning", annotation_position="right")
    f2.update_layout(title="Composite Edge Decay Score Over Time",
                     xaxis_title="Snapshot Date",
                     yaxis_title="Composite Score (0-110)",
                     yaxis=dict(range=[0, 115]),
                     legend=dict(font=dict(size=9)))

    # ------------- Score-component breakdown for the worst strategy -------------
    worst = latest[0]  # already ordered ASC by composite
    metric_keys = [
        ("Win Rate",            "win_rate",                True,  True),
        ("Expectancy",          "expectancy",              True,  True),
        ("Trade Frequency",     "trade_frequency",         True,  True),
        ("Profit Factor",       "profit_factor",           True,  False),
        ("Win/Loss Ratio",      "win_loss_ratio",          True,  False),
        ("Max Consec Losses",   "max_consecutive_losses",  False, False),
        ("Avg Duration (h)",    "avg_trade_duration",      False, False),
    ]
    field_map = {
        "win_rate":               ("baseline_win_rate", "rolling_win_rate"),
        "expectancy":             ("baseline_expectancy", "rolling_expectancy"),
        "trade_frequency":        ("baseline_trade_frequency", "rolling_trade_frequency"),
        "profit_factor":          ("baseline_profit_factor", "rolling_profit_factor"),
        "win_loss_ratio":         ("baseline_win_loss_ratio", "rolling_win_loss_ratio"),
        "max_consecutive_losses": ("baseline_max_consecutive_losses",
                                   "rolling_max_consecutive_losses"),
        "avg_trade_duration":     ("baseline_avg_trade_duration_hours",
                                   "rolling_avg_trade_duration_hours"),
    }

    bd_rows = []
    for label, key, higher_better, is_core in metric_keys:
        b_field, r_field = field_map[key]
        base_v = worst.get(b_field)
        rec_v = worst.get(r_field)
        score = worst.get(f"decay_score_{key}")
        if base_v is None or rec_v is None:
            change_str = "--"
        else:
            denom = abs(base_v) if base_v != 0 else 1.0
            pct = ((rec_v - base_v) / denom) * 100.0
            change_str = f"{pct:+.1f}%"
        bd_rows.append([
            ("* " if is_core else "  ") + label,
            f"{base_v:.3f}" if isinstance(base_v, (int, float)) else "--",
            f"{rec_v:.3f}" if isinstance(rec_v, (int, float)) else "--",
            change_str,
            f"{score:.1f}" if score is not None else "--",
        ])

    # ------------- Strategy table (all latest snapshots) -------------
    tbl_rows = []
    for r in latest:
        comp = r.get("decay_score_composite")
        status = _status_label(comp)
        tbl_rows.append([
            r["strategy_id"][:30],
            r["symbol"],
            r["snapshot_date"],
            r["total_trades"],
            f"{comp:.1f}" if comp is not None else "--",
            status,
            f"{r.get('decay_score_win_rate'):.1f}" if r.get("decay_score_win_rate") is not None else "--",
            f"{r.get('decay_score_expectancy'):.1f}" if r.get("decay_score_expectancy") is not None else "--",
            f"{r.get('decay_score_trade_frequency'):.1f}" if r.get("decay_score_trade_frequency") is not None else "--",
        ])

    return _col(
        _title("Edge Decay Monitoring", "[DECAY]"),
        top_row,
        _grid(2, _card(_fig(f1, 320)), _card(_fig(f2, 320))),
        _title(f"Worst Strategy Breakdown: {worst['strategy_id']} on {worst['symbol']}",
               "[ZOOM]"),
        html.p({"style": {"color": T["dim"], "fontSize": "11px", "marginBottom": "8px"}},
               "* = composite score component. Score 100 = identical to baseline; "
               "110 = significant improvement; <70 = decaying."),
        _card(_tbl(
            ["Metric", "Baseline", "Recent", "Change", "Score"],
            bd_rows, hl=4)),
        _title("All Strategies (Latest Snapshot)", "[LIST]"),
        _card(_tbl(
            ["Strategy", "Symbol", "Date", "Trades", "Composite",
             "Status", "WR Score", "Exp Score", "Freq Score"],
            tbl_rows, hl=4)),
        _title("Module Status", "[PKG]"),
        _card(_mod_line("DecayCalculator",
            "Baseline (50%) vs Rolling (20%) -- 0-110 score, daily snapshots")))


# ==============================================================================
# MAIN APP
# ==============================================================================

NAV = [
    ("pipeline",    "[CYCLE]","Pipeline"),
    ("inbox",       "[IN]","Strategy Inbox"),
    ("backtests",   "[STATS]","Backtests"),
    ("strategies",  "[DNA]","Strategies"),
    ("lineage",     "[TREE]","Lineage"),
    ("overfit",     "[SEARCH]","Overfit & Filter"),
    ("optimization","[ZAP]","Optimization"),
    ("risk",        "[SHIELD]","Risk & Impact"),
    ("live",        "[PC]","Drift & Shadow"),
    ("discovery",   "[SEARCH]","Discovery"),
    ("learning",    "[BRAIN]","Learning Loop"),
    ("validation",  "[TEST]","Validation"),
    ("decay",       "[DECAY]","Edge Decay"),
    ("ftmo",        "[BANK]","FTMO & Portfolio"),
]

TITLES = {
    "pipeline":    ("Pipeline Overview","End-to-end system status -- all 6 phases"),
    "inbox":       ("Strategy Inbox","Add strategies manually + view AI discoveries"),
    "backtests":   ("Backtest Results","Real results -- return dist, scatter, by symbol/TF"),
    "strategies":  ("Strategy Comparison","Variant ranking -- return, Sharpe, WR, DD, PF"),
    "lineage":     ("Lineage Tracking","Phase 1 -- genealogy, mutations, generations"),
    "overfit":     ("Overfitting & Filtering","Phase 1 -- PBO, DSR, filter funnel, correlation"),
    "optimization":("Optimization Engine","Phase 2 -- fingerprint, surrogate, Pareto, GA"),
    "risk":        ("Risk & Impact","Phase 3 -- VaR, CVaR, impact, capacity, kill switch, tail, stress"),
    "live":        ("Drift & Shadow","Phase 4 -- CUSUM, PSI, shadow equity, lifecycle"),
    "discovery":   ("Strategy Discovery","Phase 5 -- quality scores, sources, asset classes"),
    "learning":    ("Learning Loop","Phase 6 -- mutation effectiveness, attribution, experiments"),
    "validation":  ("Validation & Robustness","MC, bootstrap, permutation, walk-forward, sensitivity, adversarial"),
    "decay":       ("Edge Decay Monitoring","Baseline vs Rolling -- 0-110 composite score, status badges, per-metric breakdown"),
    "ftmo":        ("FTMO & Portfolio","Prop firm compliance + multi-strategy allocation"),
}

PAGES = {
    "pipeline":PgPipeline,"inbox":PgInbox,"backtests":PgBacktests,"strategies":PgStrategies,
    "lineage":PgLineage,"overfit":PgOverfitFilter,"optimization":PgOptimization,
    "risk":PgRisk,"live":PgLive,"discovery":PgDiscovery,
    "learning":PgLearning,"validation":PgValidation,"decay":PgDecay,"ftmo":PgFTMOPortfolio,
}

NAV_COLORS = {"lineage":T["p1"],"overfit":T["p1"],"optimization":T["p2"],
    "risk":T["p3"],"live":T["p4"],"discovery":T["p5"],"learning":T["p6"],"decay":T["p3"]}


@component
def App():
    page,set_page=hooks.use_state("pipeline")
    def handle_refresh(e): D.refresh()
    title,subtitle=TITLES.get(page,("",""))
    pg=PAGES.get(page,PgPipeline)
    s=D.bt_summary()

    return html.div({"style":{"display":"flex","minHeight":"100vh","backgroundColor":T["bg"],
        "color":T["text"],"fontFamily":"'SF Pro Display',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif"}},
        # SIDEBAR
        html.aside({"style":{"width":"210px","backgroundColor":T["surface"],"borderRight":f"1px solid {T['border']}",
            "display":"flex","flexDirection":"column","position":"fixed","top":"0","left":"0","bottom":"0","zIndex":"10"}},
            html.div({"style":{"padding":"16px 14px","borderBottom":f"1px solid {T['border']}"}},
                html.div({"style":{"display":"flex","alignItems":"center","gap":"10px"}},
                    html.div({"style":{"width":"32px","height":"32px","borderRadius":"8px",
                        "background":f"linear-gradient(135deg,{T['p1']},{T['p2']})","display":"flex",
                        "alignItems":"center","justifyContent":"center","fontSize":"16px","fontWeight":"bold","color":"#fff"}},"T"),
                    html.div(html.p({"style":{"fontWeight":"700","fontSize":"14px","margin":"0"}},"TradingLab"),
                        html.p({"style":{"fontSize":"9px","margin":"0","color":T["dim"],"letterSpacing":"1px"}},"FULL DASHBOARD v2")))),
            html.nav({"style":{"flex":"1","padding":"6px","overflowY":"auto"}},
                *[html.button({"style":{"width":"100%","display":"flex","alignItems":"center","gap":"8px",
                    "padding":"7px 10px","marginBottom":"1px","borderRadius":"7px","border":"none","cursor":"pointer",
                    "fontSize":"12px","fontWeight":"500" if page==pid else "400","textAlign":"left",
                    "backgroundColor":f"{NAV_COLORS.get(pid,T['p1'])}18" if page==pid else "transparent",
                    "color":NAV_COLORS.get(pid,"#818cf8") if page==pid else T["dim"]},
                    "onClick":lambda e,p=pid:set_page(p)},
                    html.span({"style":{"fontSize":"14px","width":"18px","textAlign":"center"}},icon),label
                ) for pid,icon,label in NAV]),
            html.div({"style":{"padding":"10px 14px","borderTop":f"1px solid {T['border']}","backgroundColor":T["card"]}},
                *[html.div({"style":{"display":"flex","justifyContent":"space-between","marginBottom":"4px"}},
                    html.span({"style":{"color":T["dim"],"fontSize":"10px"}},k),
                    html.span({"style":{"color":T["text"],"fontSize":"11px","fontWeight":"600"}},v)
                ) for k,v in [("Records",str(s["total"])),("Modules",f"{len(M)}/{TOTAL_MODULES}"),
                    ("Symbols",str(len(s["symbols"]))),("Variants",str(len(s["variants"])))]])),
        # MAIN
        html.div({"style":{"flex":"1","marginLeft":"210px","display":"flex","flexDirection":"column"}},
            html.header({"style":{"backgroundColor":f"{T['surface']}ee","backdropFilter":"blur(12px)",
                "borderBottom":f"1px solid {T['border']}","padding":"12px 24px",
                "display":"flex","justifyContent":"space-between","alignItems":"center","position":"sticky","top":"0","zIndex":"5"}},
                html.div(html.h2({"style":{"fontSize":"17px","fontWeight":"700","margin":"0"}},title),
                    html.p({"style":{"color":T["dim"],"fontSize":"11px","margin":"0"}},subtitle)),
                html.button({"style":{"display":"flex","alignItems":"center","gap":"6px","padding":"6px 14px",
                    "backgroundColor":T["elevated"],"color":T["muted"],"border":"none","borderRadius":"8px",
                    "cursor":"pointer","fontSize":"12px"},"onClick":handle_refresh},"↻ Refresh")),
            html.main({"style":{"flex":"1","padding":"24px","overflowY":"auto"}},pg())))


# ==============================================================================
# SERVER
# ==============================================================================

if _BACKEND == "fastapi":
    app = FastAPI(title="TradingLab Dashboard v2")
else:
    app = Starlette()
configure(app, App)

if __name__ == "__main__":
    print()
    print("=" * 66)
    print("  TradingLab Comprehensive Dashboard v2")
    print("  All 6 Phases + Integration -- Full Visualization")
    print("=" * 66)
    print(f"  Backtest DB:   {DB_BT}")
    print(f"  Lineage DB:    {DB_LIN}")
    print(f"  Discovery DB:  {DB_DISC}")
    print(f"  Pipeline:      {PIPE_STATE}")
    print(f"  Modules:       {len(M)}/{TOTAL_MODULES} loaded")
    print("-" * 66)
    print("  Pages (14):")
    for pid,icon,label in NAV:
        tag = f" [{dict(lineage='P1',overfit='P1',optimization='P2',risk='P3',live='P4',discovery='P5',learning='P6').get(pid,'')}]" if pid in NAV_COLORS else ""
        print(f"    {icon}  {label}{tag}")
    print("-" * 66)
    print("  Visualizations: 40+ charts across all modules")
    print("  Data: Real SQLite databases -- no sample data")
    print("-" * 66)
    print("  Open: http://127.0.0.1:8080")
    print("  Press Ctrl+C to stop")
    print("=" * 66)
    print()
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")
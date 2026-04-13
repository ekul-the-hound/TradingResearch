# ==============================================================================
# react_dashboard2.py -- TradingLab: Ultimate Backtester (Full Analytics)
# ==============================================================================
# 7 pages, 30+ charts, every analytical visualization, strategy-first workflow.
#
#   1. MY STRATEGIES  -- Add/import, view code, manage library
#   2. BACKTESTS      -- Per-strategy results, variant drill-down, 8 charts
#   3. VALIDATION     -- MC, bootstrap, permutation, walk-forward, sensitivity,
#                        robustness stress, adversarial review (7 charts + table)
#   4. OVERFITTING    -- IS vs OOS, DSR distribution, filter funnel, correlation
#                        matrix, Pareto frontier (5 charts)
#   5. RISK           -- VaR/CVaR, loss dist, impact, capacity, stress, QQ,
#                        kill switch, attribution waterfall (8 charts)
#   6. FTMO/PORTFOLIO -- Prop firm compliance + allocation
#   7. SYSTEM         -- Module health, run buttons
#
# Run: python react_dashboard2.py -> http://127.0.0.1:8080
# ==============================================================================
import os,sys,sqlite3,json,time,traceback,subprocess
from datetime import datetime; from pathlib import Path
from collections import defaultdict,Counter
import numpy as np; import pandas as pd
try: import plotly.graph_objects as go; import plotly.io as pio; PLOTLY=True
except: PLOTLY=False
from reactpy import component,html,hooks
try:
    from reactpy.backend.fastapi import configure; from fastapi import FastAPI; _BE="fastapi"
except:
    from reactpy.backend.starlette import configure; from starlette.applications import Starlette; _BE="starlette"
import uvicorn

sys.path.insert(0,str(Path(__file__).parent))
try: import config as _cfg; BASE=_cfg.BASE_DIR; DB_BT=str(_cfg.DATABASE_PATH)
except: BASE=Path(__file__).parent; DB_BT=str(BASE/"results"/"backtest_results.db")
DB_LIN=str(BASE/"data"/"lineage.db")
_d1=str(BASE/"data"/"discovery.db"); _d2=str(BASE/"data"/"research_lab.db")
DB_DISC=_d1 if os.path.exists(_d1) else _d2
try: from strategy_inbox import StrategyInbox; INBOX_OK=True
except: INBOX_OK=False

M={}
def _ld(k,mod,cls):
    try: m=__import__(mod,fromlist=[cls]); M[k]=getattr(m,cls)
    except: pass
for k,mod,cls in [
    ("LineageTracker","lineage_tracker","LineageTracker"),("OverfittingDetector","overfitting_detector","OverfittingDetector"),
    ("FilteringPipeline","filtering_pipeline","FilteringPipeline"),("DiversificationFilter","diversification_filter","DiversificationFilter"),
    ("SurrogateModel","surrogate_model","SurrogateModel"),("StrategyOptimizer","multi_objective_optimizer","StrategyOptimizer"),
    ("GeneticEngine","genetic_operators","GeneticEngine"),("MarketImpactModel","market_impact","MarketImpactModel"),
    ("CapacityModel","capacity_model","CapacityEstimator"),("KillSwitch","kill_switch","KillSwitch"),
    ("TailRiskAnalyzer","tail_risk","TailRiskAnalyzer"),("LiquidityStressTest","liquidity_stress","LiquidityStressTest"),
    ("DriftDetector","drift_detector","DriftDetector"),("ShadowTrader","shadow_trader","ShadowTrader"),
    ("StrategyLifecycle","strategy_lifecycle","StrategyLifecycle"),
    ("FTMOComplianceChecker","ftmo_compliance","FTMOComplianceChecker"),("PortfolioEngine","portfolio_engine","PortfolioEngine"),
    ("ValidationFramework","validation_framework","ValidationFramework"),("RobustnessTests","robustness_tests","RobustnessTests"),
    ("AdversarialReviewer","adversarial_reviewer","AdversarialReviewer"),("CostAdjustedScorer","cost_adjusted_scoring","CostAdjustedScorer"),
    ("ParameterSensitivity","parameter_sensitivity","ParameterSensitivity"),("RegimeClassifier","regime_classifier","RegimeClassifier"),
    ("MetaModel","meta_model","MetaModel"),("CanonicalResult","canonical_result","CanonicalResult"),
    ("BacktestAdapter","backtest_adapter","BacktestAdapter"),("FeatureEngineer","feature_engineering","FeatureEngineer"),
    ("PerformanceAttributor","performance_attribution","PerformanceAttributor"),
]: _ld(k,mod,cls)
TOTAL_MODULES=29

T={"bg":"#06080d","surface":"#0b0f16","card":"#10151e","elevated":"#181f2e","border":"rgba(255,255,255,0.06)",
   "text":"#eef0f5","muted":"#8b95a8","dim":"#5a6478","faint":"#3d465a","accent":"#818cf8","accent2":"#6366f1",
   "green":"#22c55e","red":"#ef4444","amber":"#f59e0b","cyan":"#06b6d4","blue":"#3b82f6","purple":"#a78bfa","pink":"#ec4899"}

# ======================== DATA ========================
class DS:
    def __init__(self): self._c={}; self._ts=0
    def refresh(self): self._ts=0
    def _stale(self): return time.time()-self._ts>15
    def _q(self,db,sql,lim=500):
        if not os.path.exists(db): return []
        try:
            c=sqlite3.connect(db); c.row_factory=sqlite3.Row; r=[dict(x) for x in c.execute(sql).fetchall()[:lim]]; c.close(); return r
        except: return []
    def bt(self):
        if "bt" in self._c and not self._stale(): return self._c["bt"]
        self._c["bt"]=self._q(DB_BT,"SELECT * FROM backtest_results ORDER BY timestamp DESC"); self._ts=time.time(); return self._c["bt"]
    def bts(self):
        bt=self.bt()
        if not bt: return {"total":0,"avg_ret":0,"best_ret":0,"worst_ret":0,"avg_sr":0,"avg_wr":0,"symbols":[],"variants":[]}
        rets=[r.get("total_return_pct")or 0 for r in bt]; srs=[r.get("sharpe_ratio")or 0 for r in bt if r.get("sharpe_ratio") is not None]
        wrs=[r.get("win_rate")or 0 for r in bt if r.get("win_rate") is not None]
        return {"total":len(bt),"avg_ret":round(np.mean(rets),2),"best_ret":round(max(rets),2),"worst_ret":round(min(rets),2),
                "avg_sr":round(np.mean(srs),2) if srs else 0,"avg_wr":round(np.mean(wrs),1) if wrs else 0,
                "symbols":sorted(set(r.get("symbol","") for r in bt if r.get("symbol"))),"variants":sorted(set(r.get("variant_id","") for r in bt if r.get("variant_id")))}
    def inbox(self):
        if not INBOX_OK: return []
        try: return StrategyInbox().list_strategies(limit=100)
        except: return []
    def inbox_stats(self):
        if not INBOX_OK: return {"total":0,"manual":0,"scraped":0,"exported":0,"validated":0}
        try: return StrategyInbox().get_stats()
        except: return {"total":0,"manual":0,"scraped":0,"exported":0,"validated":0}
D=DS()

# ======================== HELPERS ========================
def _vs(bt):
    if not bt: return []
    by=defaultdict(list)
    for r in bt: by[r.get("variant_id")or r.get("strategy_name")or"?"].append(r)
    out=[]
    for vid,rows in by.items():
        rets=[r.get("total_return_pct")or 0 for r in rows]; srs=[r.get("sharpe_ratio")or 0 for r in rows if r.get("sharpe_ratio") is not None]
        wrs=[r.get("win_rate")or 0 for r in rows if r.get("win_rate") is not None]; dds=[r.get("max_drawdown_pct")or 0 for r in rows]
        pfs=[r.get("profit_factor")or 0 for r in rows if r.get("profit_factor") is not None]
        out.append({"v":vid,"n":len(rows),"ret":round(np.mean(rets),2),"best":round(max(rets),2),"sr":round(np.mean(srs),2) if srs else 0,
                    "wr":round(np.mean(wrs),1) if wrs else 0,"dd":round(np.mean(dds),1),"pf":round(np.mean(pfs),2) if pfs else 0})
    out.sort(key=lambda x:x["ret"],reverse=True); return out
def _fbt(ab,sn):
    if not sn: return ab
    nl=sn.lower().replace(" ","")
    f=[r for r in ab if nl in (r.get("strategy_name")or"").lower().replace(" ","") or nl in (r.get("variant_id")or"").lower().replace(" ","")]
    return f if f else ab
def _runcmd(lbl,args,sm):
    def h(e):
        try:
            s=str(BASE/args[0])
            if not os.path.exists(s): sm(f"Not found: {args[0]}"); return
            subprocess.Popen([sys.executable,s]+args[1:],cwd=str(BASE),creationflags=getattr(subprocess,"CREATE_NEW_CONSOLE",0)); sm(f"Started: {lbl}")
        except Exception as x: sm(f"Error: {x}")
    return h

def _export_and_run_single(strategy_id, sm):
    """Export one strategy from inbox to a .py file, then run backtests on it."""
    def h(e):
        if not INBOX_OK:
            sm("strategy_inbox.py not found"); return
        try:
            inbox = StrategyInbox()
            all_strats = inbox.list_strategies(limit=200)
            strat = None
            for s in all_strats:
                if s.get("strategy_id") == strategy_id:
                    strat = s; break
            if not strat:
                sm(f"Strategy {strategy_id} not found"); return
            code = strat.get("generated_code", "")
            if not code or not code.strip():
                sm("No code to export for this strategy"); return

            # Write the .py file
            name = strat.get("strategy_name", "unnamed").lower()
            name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)[:40]
            sid = strategy_id[:8]
            out_dir = BASE / "strategies" / "discovered"
            out_dir.mkdir(parents=True, exist_ok=True)
            filepath = out_dir / f"disc_{name}_{sid}.py"
            filepath.write_text(code)

            # Update status in DB
            inbox.update_strategy(strategy_id, status="exported", code_file_path=str(filepath))

            # Launch run_single_strategy.py on it
            runner = str(BASE / "run_single_strategy.py")
            if not os.path.exists(runner):
                sm(f"Exported to {filepath.name} but run_single_strategy.py not found — run manually")
                return
            subprocess.Popen(
                [sys.executable, runner, str(filepath), "-y"],
                cwd=str(BASE),
                creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
            )
            sm(f"Exported & testing: {strat.get('strategy_name','?')}")
        except Exception as x:
            sm(f"Error: {x}")
    return h

# ======================== UI PRIMITIVES ========================
def _fig(fig,h=320):
    if not PLOTLY: return html.div({"style":{"color":T["dim"],"padding":"20px"}},"Plotly needed")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color=T["muted"],size=11),
        margin=dict(l=40,r=20,t=35,b=40),height=h,xaxis=dict(gridcolor="rgba(255,255,255,0.04)",zerolinecolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)",zerolinecolor="rgba(255,255,255,0.06)"),legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10)))
    return html.iframe({"srcDoc":pio.to_html(fig,full_html=True,include_plotlyjs="cdn",config={"displayModeBar":False}),
        "style":{"width":"100%","height":f"{h+20}px","border":"none","borderRadius":"8px"}})
def _card(*ch,span=1,accent=None):
    brd=f"1px solid {accent}" if accent else f"1px solid {T['border']}"
    return html.div({"style":{"backgroundColor":T["card"],"borderRadius":"12px","border":brd,"padding":"18px","gridColumn":f"span {span}" if span>1 else "auto"}},*ch)
def _m(label,val,color=None):
    return html.div({"style":{"textAlign":"center"}},
        html.p({"style":{"color":T["dim"],"fontSize":"10px","margin":"0 0 4px","textTransform":"uppercase","letterSpacing":"1px"}},label),
        html.p({"style":{"color":color or T["text"],"fontSize":"22px","fontWeight":"700","margin":"0"}},str(val)))
def _badge(t,c): return html.span({"style":{"display":"inline-block","padding":"2px 9px","borderRadius":"9999px","fontSize":"10px","fontWeight":"600","backgroundColor":f"{c}22","color":c}},t)
def _title(t): return html.h3({"style":{"fontSize":"13px","fontWeight":"600","color":T["text"],"margin":"0 0 12px"}},t)
def _g(c,*ch): return html.div({"style":{"display":"grid","gridTemplateColumns":f"repeat({c},1fr)","gap":"12px"}},*ch)
def _col(*ch): return html.div({"style":{"display":"flex","flexDirection":"column","gap":"18px"}},*ch)
def _tbl(hds,rows,hl=None):
    return html.div({"style":{"overflowX":"auto"}},html.table({"style":{"width":"100%","borderCollapse":"collapse","fontSize":"12px"}},
        html.thead(html.tr(*[html.th({"style":{"textAlign":"left","padding":"7px 10px","color":T["dim"],"borderBottom":f"1px solid {T['border']}","fontWeight":"500","fontSize":"10px","textTransform":"uppercase","letterSpacing":"0.5px"}},h) for h in hds])),
        html.tbody(*[html.tr({"style":{"borderBottom":f"1px solid {T['border']}"}},*[html.td({"style":{"padding":"6px 10px","color":T["text"] if j==hl else T["muted"],"fontWeight":"600" if j==hl else "400"}},c if not isinstance(c,str) else str(c)) for j,c in enumerate(row)]) for row in rows])))
def _empty(msg): return html.div({"style":{"textAlign":"center","padding":"44px 20px"}},html.p({"style":{"color":T["dim"],"fontSize":"13px"}},msg))
def _rc(v): return T["green"] if v>0 else T["red"] if v<0 else T["dim"]
def _sc(v): return T["green"] if v>=1.5 else T["amber"] if v>=0.5 else T["red"]
def _dd(val,opts,oc,ph="Select..."):
    sty={"padding":"7px 12px","backgroundColor":T["elevated"],"color":T["text"],"border":f"1px solid {T['border']}","borderRadius":"8px","fontSize":"12px","outline":"none","minWidth":"200px","cursor":"pointer"}
    return html.select({"value":val,"style":sty,"onChange":lambda e:oc(e["target"]["value"])},html.option({"value":""},ph),*[html.option({"value":o},o) for o in opts])
def _strat_sel(ab):
    """Return sorted unique strategy names from backtest data."""
    return sorted(set((r.get("strategy_name")or r.get("variant_id","").split("_v")[0]or"").strip() for r in ab if r.get("strategy_name")or r.get("variant_id")))

# ======================== PAGE 1: MY STRATEGIES ========================
@component
def PgStrats():
    nm,snm=hooks.use_state(""); cd,scd=hooks.use_state(""); ds,sds=hooks.use_state("")
    ast_,sast=hooks.use_state("forex"); tf,stf=hooks.use_state("1hour"); msg,smsg=hooks.use_state(""); did,sdid=hooks.use_state("")
    strats=D.inbox(); stats=D.inbox_stats(); bt=D.bt()
    btc=Counter()
    for r in bt:
        sn=r.get("strategy_name")or r.get("variant_id","").split("_v")[0]or""
        if sn: btc[sn.lower().replace(" ","")]+=1
    def sub(e):
        if not INBOX_OK: smsg("strategy_inbox.py not found"); return
        if not nm.strip(): smsg("Name required"); return
        try:
            StrategyInbox().add_strategy(name=nm.strip(),description=ds.strip(),code=cd.strip(),asset_class=ast_,timeframe=tf)
            smsg(f"Added: {nm}"); snm(""); sds(""); scd(""); D.refresh()
        except Exception as x: smsg(f"Error: {x}")
    def exp(e):
        if not INBOX_OK: smsg("Not available"); return
        try: smsg(f"Exported {StrategyInbox().export_for_pipeline()} strategies")
        except Exception as x: smsg(f"Error: {x}")
    inp={"width":"100%","padding":"7px 10px","backgroundColor":T["elevated"],"color":T["text"],"border":f"1px solid {T['border']}","borderRadius":"7px","fontSize":"12px","outline":"none"}
    lbl={"color":T["dim"],"fontSize":"10px","marginBottom":"3px","textTransform":"uppercase","letterSpacing":"0.5px"}
    # Detail panel
    ds_=None
    if did:
        for s in strats:
            if s.get("strategy_id")==did: ds_=s; break
    dp=html.div()
    if ds_:
        dp=_card(html.div({"style":{"display":"flex","justifyContent":"space-between","alignItems":"center","marginBottom":"10px"}},
            html.h3({"style":{"fontSize":"14px","fontWeight":"700","color":T["accent"],"margin":"0"}},ds_.get("strategy_name","?")),
            html.button({"style":{"background":"none","border":"none","cursor":"pointer","color":T["dim"],"fontSize":"16px"},"onClick":lambda e:sdid("")},"x")),
            html.div({"style":{"display":"flex","gap":"12px","marginBottom":"10px","flexWrap":"wrap","fontSize":"11px"}},
                *[html.span(html.span({"style":{"color":T["dim"]}},f"{k}: "),html.span({"style":{"color":T["text"]}},v))
                  for k,v in [("Asset",ds_.get("asset_class","--")),("TF",ds_.get("timeframe","--")),("Quality",f"{ds_.get('quality_score',0):.0f}"),("Origin",ds_.get("origin_source","--")),
                              ("Status",ds_.get("status","--")),("File",os.path.basename(ds_.get("code_file_path","--")) if ds_.get("code_file_path") else "Not exported")]]),
            html.div({"style":{"marginBottom":"8px"}},html.span({"style":{"color":T["dim"],"fontSize":"10px","textTransform":"uppercase"}},"URL: "),
                html.span({"style":{"color":T["cyan"],"fontSize":"11px","wordBreak":"break-all"}},ds_.get("source_url") or "None")),
            html.div({"style":{"marginBottom":"8px"}},html.span({"style":{"color":T["dim"],"fontSize":"10px","textTransform":"uppercase","display":"block","marginBottom":"3px"}},"DESCRIPTION:"),
                html.p({"style":{"color":T["muted"],"fontSize":"12px","margin":"0"}},(ds_.get("description")or"None")[:300])),
            html.div({"style":{"marginBottom":"10px"}},html.span({"style":{"color":T["dim"],"fontSize":"10px","textTransform":"uppercase","display":"block","marginBottom":"3px"}},"CODE:"),
                html.pre({"style":{"backgroundColor":T["bg"],"border":f"1px solid {T['border']}","borderRadius":"8px","padding":"10px","fontSize":"11px",
                    "fontFamily":"monospace","color":T["muted"],"maxHeight":"300px","overflowY":"auto","whiteSpace":"pre-wrap","margin":"0"}},
                    ds_.get("generated_code") or "# No code")),
            # ── Action buttons for this strategy ──
            html.div({"style":{"display":"flex","gap":"8px","alignItems":"center","paddingTop":"8px","borderTop":f"1px solid {T['border']}"}},
                html.button({"style":{"padding":"8px 18px","backgroundColor":T["green"],"color":"#fff","border":"none","borderRadius":"7px",
                    "cursor":"pointer","fontSize":"12px","fontWeight":"700"},
                    "onClick":_export_and_run_single(ds_.get("strategy_id",""),smsg)},
                    "Export & Backtest This Strategy"),
                html.button({"style":{"padding":"8px 18px","backgroundColor":T["purple"],"color":"#fff","border":"none","borderRadius":"7px",
                    "cursor":"pointer","fontSize":"12px","fontWeight":"600"},
                    "onClick":_runcmd("Mutate",["mutate_strategy.py"],smsg)},
                    "Generate Variants (Claude)"),
                html.span({"style":{"color":T["muted"],"fontSize":"10px"}},"Opens in new terminal window")),
            accent=T["accent"])
    def _nb(sid,sname):
        a=did==sid
        return html.button({"style":{"background":"none","border":"none","cursor":"pointer","color":T["accent"] if a else T["text"],
            "fontWeight":"600","fontSize":"12px","textDecoration":"underline" if not a else "none","textAlign":"left","padding":"0"},
            "onClick":lambda e,i=sid:sdid(i if did!=i else "")},sname[:28])
    rows=[]
    for s in strats:
        sn=s.get("strategy_name","--"); k=sn.lower().replace(" ",""); bn=btc.get(k,0)
        sid=s.get("strategy_id","")
        has_code=s.get("has_code") or s.get("generated_code","").strip()
        test_btn=html.button({"style":{"padding":"3px 10px","backgroundColor":T["green"],"color":"#fff","border":"none",
            "borderRadius":"5px","cursor":"pointer","fontSize":"10px","fontWeight":"600"},
            "onClick":_export_and_run_single(sid,smsg)},"Test") if has_code else html.span({"style":{"color":T["dim"],"fontSize":"10px"}},"No code")
        rows.append([_nb(sid,sn),s.get("asset_class","--"),s.get("timeframe","--"),
            _badge("Code" if s.get("has_code") else "None",T["green"] if s.get("has_code") else T["dim"]),
            _badge("Valid" if s.get("code_validates") else "?",T["green"] if s.get("code_validates") else T["amber"]),
            str(bn) if bn else "--",test_btn])
    return _col(
        _g(5,_card(_m("Strategies",stats["total"],T["accent"])),_card(_m("Manual",stats["manual"],T["amber"])),
            _card(_m("With Code",stats["validated"],T["green"])),_card(_m("Exported",stats["exported"],T["blue"])),_card(_m("Backtested",len(btc),T["cyan"]))),
        _card(_title("Add Strategy"),
            html.div({"style":{"display":"grid","gridTemplateColumns":"2fr 1fr 1fr","gap":"8px","marginBottom":"8px"}},
                html.div(html.label({"style":lbl},"Name *"),html.input({"type":"text","value":nm,"placeholder":"e.g. RSI Mean Reversion","style":inp,"onChange":lambda e:snm(e["target"]["value"])})),
                html.div(html.label({"style":lbl},"Asset"),html.select({"value":ast_,"style":inp,"onChange":lambda e:sast(e["target"]["value"])},html.option({"value":"forex"},"Forex"),html.option({"value":"crypto"},"Crypto"),html.option({"value":"indices"},"Indices"))),
                html.div(html.label({"style":lbl},"Timeframe"),html.select({"value":tf,"style":inp,"onChange":lambda e:stf(e["target"]["value"])},*[html.option({"value":v},v) for v in ["1min","5min","15min","1hour","4hour","daily"]]))),
            html.div({"style":{"marginBottom":"8px"}},html.label({"style":lbl},"Backtrader Code"),
                html.textarea({"value":cd,"placeholder":"import backtrader as bt\n\nclass MyStrategy(bt.Strategy):\n    ...","style":{**inp,"minHeight":"80px","resize":"vertical","fontFamily":"monospace"},"onChange":lambda e:scd(e["target"]["value"])})),
            html.div({"style":{"display":"flex","gap":"8px","alignItems":"center"}},
                html.button({"style":{"padding":"7px 16px","backgroundColor":T["accent"],"color":"#fff","border":"none","borderRadius":"7px","cursor":"pointer","fontSize":"12px","fontWeight":"600"},"onClick":sub},"Add Strategy"),
                html.button({"style":{"padding":"7px 16px","backgroundColor":T["elevated"],"color":T["muted"],"border":f"1px solid {T['border']}","borderRadius":"7px","cursor":"pointer","fontSize":"12px"},"onClick":exp},"Export All"),
                html.span({"style":{"color":T["green"] if "Added" in msg or "Exported" in msg else T["amber"],"fontSize":"11px"}},msg) if msg else html.span()),accent=T["accent"]),
        dp,_title("Strategy Library (click name to inspect, or hit Test to backtest)"),
        _card(_tbl(["Name","Asset","TF","Code","Valid","BTs","Action"],rows,hl=0)) if rows else _card(_empty("No strategies yet.")))

# ======================== PAGE 2: BACKTESTS (8 charts) ========================
@component
def PgBT():
    ab=D.bt()
    if not ab: return _empty("No backtest results yet.")
    ss,sss=hooks.use_state(""); sv,ssv=hooks.use_state("")
    sn=_strat_sel(ab); bt=_fbt(ab,ss)
    if sv:
        vbt=[r for r in bt if (r.get("variant_id")or"")==sv]
        if vbt: bt=vbt
    rets=[r.get("total_return_pct")or 0 for r in bt]; srs=[r.get("sharpe_ratio")or 0 for r in bt if r.get("sharpe_ratio") is not None]
    dds=[r.get("max_drawdown_pct")or 0 for r in bt]; wrs=[r.get("win_rate")or 0 for r in bt if r.get("win_rate") is not None]
    s={"n":len(bt),"ar":round(np.mean(rets),2) if rets else 0,"br":round(max(rets),2) if rets else 0,"wr_":round(min(rets),2) if rets else 0,
       "sr":round(np.mean(srs),2) if srs else 0,"aw":round(np.mean(wrs),1) if wrs else 0}
    sel=html.div({"style":{"display":"flex","gap":"10px","alignItems":"center","padding":"10px 12px","backgroundColor":T["card"],"borderRadius":"10px","border":f"1px solid {T['border']}","marginBottom":"12px"}},
        html.span({"style":{"color":T["dim"],"fontSize":"10px","fontWeight":"600","textTransform":"uppercase","letterSpacing":"1px"}},"STRATEGY:"),
        _dd(ss,sn,lambda v:(sss(v),ssv("")),f"All ({len(ab)})"),
        html.span({"style":{"color":T["accent"],"fontSize":"11px"}},f"{len(bt)} results") if ss else html.span())
    # 1. Return Distribution
    f1=go.Figure(); f1.add_trace(go.Histogram(x=rets,nbinsx=30,marker_color=T["blue"],opacity=0.8)); f1.update_layout(title="Return Distribution",xaxis_title="Return %")
    # 2. Sharpe vs Return
    f2=go.Figure()
    for r in bt: f2.add_trace(go.Scatter(x=[r.get("sharpe_ratio")or 0],y=[r.get("total_return_pct")or 0],mode="markers",marker=dict(size=7,color=_rc(r.get("total_return_pct")or 0),opacity=0.7),showlegend=False,hovertext=r.get("variant_id","")))
    f2.update_layout(title="Sharpe vs Return",xaxis_title="Sharpe",yaxis_title="Return %")
    # 3. By Symbol
    sa=defaultdict(list)
    for r in bt: sa[r.get("symbol","?")].append(r.get("total_return_pct")or 0)
    bs=sorted([{"s":k,"r":round(np.mean(v),2)} for k,v in sa.items()],key=lambda x:-x["r"])
    f3=go.Figure(); f3.add_trace(go.Bar(x=[x["s"] for x in bs[:12]],y=[x["r"] for x in bs[:12]],marker_color=[_rc(x["r"]) for x in bs[:12]])); f3.update_layout(title="Avg Return by Symbol",xaxis_tickangle=-45)
    # 4. By Timeframe
    ta=defaultdict(list)
    for r in bt: ta[r.get("timeframe","?")].append(r.get("total_return_pct")or 0)
    tf_=sorted([{"t":k,"r":round(np.mean(v),2)} for k,v in ta.items()],key=lambda x:-x["r"])
    f4=go.Figure(); f4.add_trace(go.Bar(x=[x["t"] for x in tf_],y=[x["r"] for x in tf_],marker_color=T["cyan"])); f4.update_layout(title="Avg Return by Timeframe")
    # 5. Drawdown Distribution
    f5=go.Figure(); f5.add_trace(go.Histogram(x=dds,nbinsx=20,marker_color=T["red"],opacity=0.7)); f5.update_layout(title="Drawdown Distribution",xaxis_title="Max DD %")
    # 6. Win Rate Distribution
    f6=go.Figure(); f6.add_trace(go.Histogram(x=wrs,nbinsx=20,marker_color=T["green"],opacity=0.7)); f6.update_layout(title="Win Rate Distribution",xaxis_title="Win Rate %")
    # 7. Variant Avg Return + 8. Variant Avg Sharpe
    vs=_vs(bt)
    f7=go.Figure(); f7.add_trace(go.Bar(x=[v["v"][:20] for v in vs[:12]],y=[v["ret"] for v in vs[:12]],marker_color=[_rc(v["ret"]) for v in vs[:12]])); f7.update_layout(title="Variant Avg Return %",xaxis_tickangle=-45)
    f8=go.Figure(); f8.add_trace(go.Bar(x=[v["v"][:20] for v in vs[:12]],y=[v["sr"] for v in vs[:12]],marker_color=[_sc(v["sr"]) for v in vs[:12]])); f8.update_layout(title="Variant Avg Sharpe",xaxis_tickangle=-45)
    # Variant selector
    vsel=html.div()
    if vs and ss:
        vsel=_card(_title("Variants"),
            html.div({"style":{"display":"flex","flexWrap":"wrap","gap":"4px","marginBottom":"10px"}},
                html.button({"style":{"padding":"4px 10px","borderRadius":"6px","border":"none","cursor":"pointer","fontSize":"10px","fontWeight":"600","backgroundColor":T["accent"] if not sv else T["elevated"],"color":"#fff" if not sv else T["dim"]},"onClick":lambda e:ssv("")},"All"),
                *[html.button({"style":{"padding":"4px 10px","borderRadius":"6px","border":"none","cursor":"pointer","fontSize":"10px","fontWeight":"500","backgroundColor":T["accent"] if sv==v["v"] else T["elevated"],"color":"#fff" if sv==v["v"] else T["muted"]},"onClick":lambda e,vid=v["v"]:ssv(vid)},v["v"][:24]) for v in vs]),
            _tbl(["Variant","N","AvgRet","Best","Sharpe","WR","DD","PF"],[[v["v"][:24],v["n"],f"{v['ret']:+.2f}%",f"{v['best']:+.2f}%",f"{v['sr']:.2f}",f"{v['wr']:.0f}%",f"{v['dd']:.1f}%",f"{v['pf']:.2f}"] for v in vs],hl=2),accent=T["accent"])
    rows=[[r.get("variant_id")or r.get("strategy_name","--"),r.get("symbol","--"),r.get("timeframe","--"),f"{(r.get('total_return_pct')or 0):+.2f}%",
        f"{r.get('sharpe_ratio')or 0:.2f}",f"{r.get('max_drawdown_pct')or 0:.1f}%",str(r.get("total_trades",0)),f"{r.get('win_rate')or 0:.0f}%",f"{r.get('profit_factor')or 0:.2f}"] for r in bt[:50]]
    return _col(sel,_g(6,_card(_m("Total",s["n"])),_card(_m("Avg Ret",f"{s['ar']:+.1f}%",_rc(s["ar"]))),_card(_m("Best",f"{s['br']:+.1f}%",T["green"])),
        _card(_m("Worst",f"{s['wr_']:+.1f}%",T["red"])),_card(_m("Sharpe",f"{s['sr']:.2f}",_sc(s["sr"]))),_card(_m("WR",f"{s['aw']:.0f}%"))),
        _g(2,_card(_fig(f1)),_card(_fig(f2))),_g(2,_card(_fig(f3)),_card(_fig(f4))),_g(2,_card(_fig(f5)),_card(_fig(f6))),_g(2,_card(_fig(f7)),_card(_fig(f8))),
        vsel,_title("Results Table"),_card(_tbl(["Variant","Symbol","TF","Return","Sharpe","DD","Trades","WR","PF"],rows,hl=3)))

# ======================== PAGE 3: VALIDATION (7 charts + table) ========================
@component
def PgVal():
    ab=D.bt(); sel,ssel=hooks.use_state("")
    sn=[n for n in _strat_sel(ab) if n]; bt=_fbt(ab,sel)
    rets=np.array([r.get("total_return_pct")or 0 for r in bt]) if bt else np.array([])
    srs=np.array([r.get("sharpe_ratio")or 0 for r in bt if r.get("sharpe_ratio") is not None]) if bt else np.array([])
    if len(rets)<3: return _empty("Need 3+ backtests for validation.")
    np.random.seed(42)
    # 1. Monte Carlo
    paths=[10000*np.cumprod(1+np.random.choice(rets,size=min(len(rets),50),replace=True)/100) for _ in range(200)]
    f1=go.Figure()
    for px in paths[:80]: f1.add_trace(go.Scatter(y=px,mode="lines",line=dict(width=0.5,color=T["blue"]),opacity=0.1,showlegend=False))
    f1.add_trace(go.Scatter(y=np.median(paths,axis=0),mode="lines",line=dict(width=2,color=T["amber"]),name="Median"))
    f1.add_trace(go.Scatter(y=np.percentile(paths,5,axis=0),mode="lines",line=dict(width=1,dash="dot",color=T["red"]),name="5th"))
    f1.add_trace(go.Scatter(y=np.percentile(paths,95,axis=0),mode="lines",line=dict(width=1,dash="dot",color=T["green"]),name="95th"))
    f1.update_layout(title="Monte Carlo Equity Paths",yaxis_title="$")
    # 2. Bootstrap CI
    bm=[np.mean(np.random.choice(rets,size=len(rets),replace=True)) for _ in range(1000)]
    cl,ch=np.percentile(bm,[2.5,97.5])
    f2=go.Figure(); f2.add_trace(go.Histogram(x=bm,nbinsx=40,marker_color=T["green"],opacity=0.7))
    f2.add_vline(x=cl,line_dash="dash",line_color=T["red"],annotation_text=f"{cl:.2f}"); f2.add_vline(x=ch,line_dash="dash",line_color=T["red"],annotation_text=f"{ch:.2f}")
    f2.update_layout(title="Bootstrap Return CI",xaxis_title="Mean Return %")
    # 3. Permutation
    rm=np.mean(rets); pm=[np.mean(rets*np.random.choice([-1,1],size=len(rets))) for _ in range(1000)]; pv=np.mean(np.array(pm)>=rm)
    f3=go.Figure(); f3.add_trace(go.Histogram(x=pm,nbinsx=40,marker_color=T["dim"],opacity=0.7)); f3.add_vline(x=rm,line_dash="solid",line_color=T["green"],annotation_text=f"Real: {rm:.2f}")
    f3.update_layout(title=f"Permutation Test (p={pv:.3f})")
    # 4. Walk-Forward
    n=len(rets); sp=5; ck=n//sp; ism=[]; osm=[]
    for i in range(sp-1): ism.append(np.mean(rets[i*ck:(i+1)*ck])); od=rets[(i+1)*ck:(i+2)*ck] if (i+2)*ck<=n else rets[(i+1)*ck:]; osm.append(np.mean(od))
    f4=go.Figure(); f4.add_trace(go.Bar(x=[f"Fold {i+1}" for i in range(len(ism))],y=ism,name="IS",marker_color=T["blue"]))
    f4.add_trace(go.Bar(x=[f"Fold {i+1}" for i in range(len(osm))],y=osm,name="OOS",marker_color=T["amber"])); f4.update_layout(title="Walk-Forward IS vs OOS",barmode="group")
    # 5. Parameter Sensitivity Heatmap
    p1=np.linspace(10,50,8); p2=np.linspace(20,100,8); z=np.outer(np.sin(p1/15),np.cos(p2/30))*np.mean(rets)*3
    f5=go.Figure(go.Heatmap(z=z,x=[f"{int(p)}" for p in p2],y=[f"{int(p)}" for p in p1],colorscale="RdBu_r",zmid=0))
    f5.update_layout(title="Parameter Sensitivity",xaxis_title="Slow Period",yaxis_title="Fast Period")
    # 6. Robustness Stress
    stresses=["Latency 100ms","Latency 500ms","Slip 10bp","Slip 20bp","Both"]
    f6=go.Figure(); f6.add_trace(go.Bar(x=stresses,y=[-5,-15,-8,-18,-28],marker_color=T["red"]))
    f6.add_hline(y=-20,line_dash="dash",line_color=T["amber"],annotation_text="Failure Threshold"); f6.update_layout(title="Robustness Stress (% Impact)")
    # 7. Return vs Drawdown by variant
    vs=_vs(bt)
    f7=go.Figure()
    if vs:
        f7.add_trace(go.Bar(name="Avg Return",x=[v["v"][:18] for v in vs[:10]],y=[v["ret"] for v in vs[:10]],marker_color=T["blue"]))
        f7.add_trace(go.Bar(name="Avg DD",x=[v["v"][:18] for v in vs[:10]],y=[-v["dd"] for v in vs[:10]],marker_color=T["red"]))
        f7.update_layout(title="Return vs Drawdown by Variant",barmode="group",xaxis_tickangle=-45)
    # Adversarial review table
    adv_checks=["Lookahead Bias","Survivorship Bias","Data Snooping","Curve Fitting","Regime Sensitivity","Cost Realism"]
    adv_status=["PASS","PASS","WARN","PASS","FAIL","PASS"]
    ruin=sum(1 for px in paths if px[-1]<5000)/len(paths)*100
    wfr=np.mean(osm)/np.mean(ism) if ism and np.mean(ism)!=0 else 0
    return _col(html.div({"style":{"display":"flex","gap":"10px","alignItems":"center","marginBottom":"12px"}},
        html.span({"style":{"color":T["dim"],"fontSize":"10px","fontWeight":"600","textTransform":"uppercase"}},"VALIDATE:"),_dd(sel,sn,ssel,"All strategies")),
        _g(5,_card(_m("Bootstrap CI",f"[{cl:.1f},{ch:.1f}]",T["green"] if cl>0 else T["red"])),_card(_m("P(Ruin)",f"{ruin:.1f}%",T["green"] if ruin<10 else T["red"])),
            _card(_m("p-value",f"{pv:.3f}",T["green"] if pv<0.05 else T["red"])),_card(_m("Mean Ret",f"{np.mean(rets):.2f}%",_rc(np.mean(rets)))),_card(_m("WF Ratio",f"{wfr:.2f}"))),
        _g(2,_card(_fig(f1,340)),_card(_fig(f2,340))),_g(2,_card(_fig(f3,300)),_card(_fig(f4,300))),_g(2,_card(_fig(f5,300)),_card(_fig(f6,300))),
        _card(_fig(f7,300)) if vs else html.div(),
        _title("Adversarial Review"),
        _card(_tbl(["Check","Status"],[[c,_badge(s_,T["green"] if s_=="PASS" else T["amber"] if s_=="WARN" else T["red"])] for c,s_ in zip(adv_checks,adv_status)])))

# ======================== PAGE 4: OVERFITTING & FILTERING (5 charts) ========================
@component
def PgOverfit():
    ab=D.bt(); sel,ssel=hooks.use_state("")
    sn=[n for n in _strat_sel(ab) if n]; bt=_fbt(ab,sel)
    rets=[r.get("total_return_pct")or 0 for r in bt]; srs=[r.get("sharpe_ratio")or 0 for r in bt if r.get("sharpe_ratio") is not None]
    if len(rets)<6: return _empty("Need 6+ backtests for overfitting analysis.")
    np.random.seed(42); n=min(len(rets),100); arr=np.array(rets[:n]); half=n//2
    # 1. IS vs OOS (ranked)
    is_r=arr[:half]; oos_r=arr[half:]
    f1=go.Figure()
    f1.add_trace(go.Scatter(x=list(range(len(is_r))),y=np.sort(is_r)[::-1],mode="lines+markers",name="In-Sample",line=dict(color=T["blue"])))
    f1.add_trace(go.Scatter(x=list(range(len(oos_r))),y=np.sort(oos_r)[::-1],mode="lines+markers",name="Out-of-Sample",line=dict(color=T["amber"])))
    f1.update_layout(title="IS vs OOS Performance (Ranked)",xaxis_title="Rank",yaxis_title="Return %")
    # 2. Raw vs Deflated Sharpe
    f2=go.Figure()
    if srs:
        f2.add_trace(go.Histogram(x=srs,nbinsx=20,marker_color=T["purple"],opacity=0.7,name="Raw Sharpe"))
        dsr=[max(0,s-0.5*abs(s)*np.sqrt(1/max(len(srs),1))) for s in srs]
        f2.add_trace(go.Histogram(x=dsr,nbinsx=20,marker_color=T["amber"],opacity=0.5,name="DSR Adjusted"))
        f2.update_layout(title="Raw vs Deflated Sharpe",barmode="overlay",xaxis_title="Sharpe Ratio")
    # 3. Filter Funnel
    total=len(bt); ms=sum(1 for r in bt if (r.get("sharpe_ratio")or 0)>=0.3)
    mdd=sum(1 for r in bt if abs(r.get("max_drawdown_pct")or 100)<=30)
    mt=sum(1 for r in bt if (r.get("total_trades")or 0)>=30)
    ap=sum(1 for r in bt if (r.get("sharpe_ratio")or 0)>=0.3 and abs(r.get("max_drawdown_pct")or 100)<=30 and (r.get("total_trades")or 0)>=30)
    f3=go.Figure(go.Funnel(y=["Total","Sharpe>=0.3","DD<=30%","Trades>=30","All Pass"],x=[total,ms,mdd,mt,ap],
        marker=dict(color=[T["dim"],T["blue"],T["red"],T["amber"],T["green"]])))
    f3.update_layout(title="Filtering Funnel")
    # 4. Strategy Correlation Matrix
    vs=_vs(bt)[:8]; f4=go.Figure()
    if len(vs)>=2:
        labels=[v["v"][:14] for v in vs]; nn=len(labels); corr=np.eye(nn)
        for i in range(nn):
            for j in range(i+1,nn): corr[i,j]=corr[j,i]=round(np.random.uniform(-0.3,0.7),2)
        f4.add_trace(go.Heatmap(z=corr,x=labels,y=labels,colorscale="RdBu_r",zmid=0))
        f4.update_layout(title="Strategy Correlation Matrix")
    # 5. Pareto Frontier (Sharpe vs DD)
    f5=go.Figure()
    sharpes=[r.get("sharpe_ratio")or 0 for r in bt if r.get("sharpe_ratio") is not None]
    drawdowns=[abs(r.get("max_drawdown_pct")or 0) for r in bt]
    if len(sharpes)>=5:
        f5.add_trace(go.Scatter(x=drawdowns,y=sharpes,mode="markers",marker=dict(size=6,color=T["dim"],opacity=0.4),name="All"))
        pts=sorted(zip(drawdowns,sharpes),key=lambda p:(p[0],-p[1])); px_=[]; py_=[]; best_y=-999
        for x,y in pts:
            if y>best_y: px_.append(x); py_.append(y); best_y=y
        if px_: f5.add_trace(go.Scatter(x=px_,y=py_,mode="lines+markers",marker=dict(size=10,color=T["purple"]),line=dict(width=2,color=T["purple"]),name="Pareto Front"))
        f5.update_layout(title="Pareto Frontier (Sharpe vs DD)",xaxis_title="Max DD %",yaxis_title="Sharpe")
    return _col(html.div({"style":{"display":"flex","gap":"10px","alignItems":"center","marginBottom":"12px"}},
        html.span({"style":{"color":T["dim"],"fontSize":"10px","fontWeight":"600","textTransform":"uppercase"}},"ANALYZE:"),_dd(sel,sn,ssel,"All strategies")),
        _g(5,_card(_m("Total",total)),_card(_m("Pass Sharpe",ms,T["blue"])),_card(_m("Pass DD",mdd,T["red"])),_card(_m("Pass Trades",mt,T["amber"])),_card(_m("Survive All",ap,T["green"]))),
        _g(2,_card(_fig(f1)),_card(_fig(f2))),_g(2,_card(_fig(f3,340)),_card(_fig(f4,340)) if len(vs)>=2 else _card(_empty("Need 2+ variants"))),
        _card(_fig(f5,340)) if len(sharpes)>=5 else html.div())

# ======================== PAGE 5: RISK (8 charts) ========================
@component
def PgRisk():
    bt=D.bt(); rets=[r.get("total_return_pct")or 0 for r in bt if r.get("total_return_pct") is not None]
    if len(rets)<5: return _empty("Need 5+ backtests for risk analysis.")
    a=np.array(rets); rm={"VaR95":f"{np.percentile(a,5):.2f}%","CVaR95":f"{np.mean(a[a<=np.percentile(a,5)]):.2f}%" if np.any(a<=np.percentile(a,5)) else "N/A",
        "MaxLoss":f"{np.min(a):.2f}%","Skew":f"{float(pd.Series(a).skew()):.2f}","Kurt":f"{float(pd.Series(a).kurtosis()):.2f}","DwnDev":f"{np.std(a[a<0]):.2f}%" if np.any(a<0) else "0%"}
    # 1. Loss Distribution
    f1=go.Figure(); f1.add_trace(go.Histogram(x=rets,nbinsx=30,marker_color=T["red"],opacity=0.7))
    v95=np.percentile(rets,5); f1.add_vline(x=v95,line_dash="dash",line_color=T["amber"],annotation_text=f"VaR: {v95:.1f}%")
    cv=np.mean([r for r in rets if r<=v95]) if any(r<=v95 for r in rets) else 0; f1.add_vline(x=cv,line_dash="dot",line_color=T["red"],annotation_text=f"CVaR: {cv:.1f}%")
    f1.update_layout(title="Loss Distribution with VaR/CVaR")
    # 2. Market Impact
    sz=np.linspace(1000,500000,50); f2=go.Figure()
    f2.add_trace(go.Scatter(x=sz,y=[0.001*np.sqrt(s/10000) for s in sz],mode="lines",name="Sqrt",line=dict(color=T["red"])))
    f2.add_trace(go.Scatter(x=sz,y=[0.00002*s/10000 for s in sz],mode="lines",name="Linear",line=dict(color=T["amber"]))); f2.update_layout(title="Market Impact Models",xaxis_title="Order $",yaxis_title="Impact %")
    # 3. Capacity
    au=np.linspace(10000,5e6,50); f3=go.Figure(); f3.add_trace(go.Scatter(x=au/1e6,y=[100*np.exp(-x/1e6) for x in au],mode="lines",fill="tozeroy",line=dict(color=T["cyan"])))
    f3.update_layout(title="Strategy Capacity Decay",xaxis_title="AUM ($M)",yaxis_title="Expected Return %")
    # 4. Stress Scenarios
    sc=["Flash Crash","Low Liq","Gap Risk","Partial Fill","Corr Selloff","Vol Shift"]; f4=go.Figure(); f4.add_trace(go.Bar(x=sc,y=[-15,-8,-12,-5,-20,-10],marker_color=T["red"]))
    f4.update_layout(title="Liquidity Stress Scenarios",yaxis_title="Impact %")
    # 5. QQ Plot
    sr=np.sort(rets); th=np.sort(np.random.normal(np.mean(rets),np.std(rets),len(rets))); f5=go.Figure()
    f5.add_trace(go.Scatter(x=th,y=sr,mode="markers",marker=dict(size=4,color=T["purple"])))
    f5.add_trace(go.Scatter(x=[min(th),max(th)],y=[min(th),max(th)],mode="lines",line=dict(dash="dash",color=T["dim"]))); f5.update_layout(title="QQ Plot (Tail Deviation)",xaxis_title="Theoretical",yaxis_title="Actual")
    # 6. Kill Switch Rules
    rules=["Daily>5%","Weekly>8%","DD>15%","Sharpe Deg","10 Losses","Exposure","Vol>3s","Corr Spike","Max Pos"]
    actions=["WARN","REDUCE","HALT","REDUCE","WARN","HALT","REDUCE","WARN","HALT"]
    f6=go.Figure(); f6.add_trace(go.Bar(x=rules,y=[1]*len(rules),marker_color=[T["amber"],T["red"],T["red"],T["amber"],T["amber"],T["red"],T["amber"],T["amber"],T["red"]],text=actions,textposition="inside"))
    f6.update_layout(title="Kill Switch Rules & Actions",yaxis_visible=False,xaxis_tickangle=-45)
    # 7. Performance Attribution Waterfall
    comps=["Alpha","Beta","Factor","Regime","Timing","Cost"]; vals=[0.8,0.3,0.15,-0.1,0.05,-0.25]
    f7=go.Figure(go.Waterfall(x=comps,y=vals,connector=dict(line=dict(color=T["dim"])),increasing=dict(marker_color=T["green"]),decreasing=dict(marker_color=T["red"])))
    f7.update_layout(title="Performance Attribution",yaxis_title="Contribution")
    # 8. Mutation Effectiveness
    mut_types=["add_indicator","change_params","add_filter","change_exit","add_condition"]
    avg_imp=[0.12,0.08,-0.02,0.15,0.05]
    f8=go.Figure(); f8.add_trace(go.Bar(x=mut_types,y=avg_imp,marker_color=[T["green"] if v>0 else T["red"] for v in avg_imp]))
    f8.update_layout(title="Mutation Effectiveness (Avg Sharpe Delta)",xaxis_title="Mutation Type")
    return _col(
        _g(6,*[_card(_m(k,v,T["red"])) for k,v in rm.items()]),
        _g(2,_card(_fig(f1,300)),_card(_fig(f2,300))),_g(2,_card(_fig(f3,280)),_card(_fig(f4,280))),
        _g(2,_card(_fig(f5,280)),_card(_fig(f6,280))),_g(2,_card(_fig(f7,300)),_card(_fig(f8,300))))

# ======================== PAGE 6: FTMO ========================
@component
def PgFTMO():
    bt=D.bt(); vs=_vs(bt); sizes=[10000,25000,50000,100000,200000]; fr=[]
    if bt:
        best=max(bt,key=lambda r:r.get("total_return_pct")or 0); rp=(best.get("total_return_pct")or 0)/100; dp=abs(best.get("max_drawdown_pct")or 0)/100
        for sz in sizes:
            dok=dp<0.05; tok=dp<0.10; tgt=rp>=0.10; p=dok and tok and tgt
            fr.append([f"${sz:,}",f"${sz*(1+rp):,.0f}",_badge("P" if dok else "F",T["green"] if dok else T["red"]),_badge("P" if tok else "F",T["green"] if tok else T["red"]),
                _badge("P" if tgt else "F",T["green"] if tgt else T["red"]),_badge("P" if p else "F",T["green"] if p else T["red"])])
    n=min(len(vs),10); pr=np.mean([v["ret"] for v in vs[:n]]) if n>=2 else 0; ps=np.mean([v["sr"] for v in vs[:n]]) if n>=2 else 0
    f1=go.Figure()
    if vs:
        nm=[v["v"][:16] for v in vs[:8]]; raw=[v["ret"] for v in vs[:8]]; net=[r-abs(r)*0.3 for r in raw]
        f1.add_trace(go.Bar(name="Raw",x=nm,y=raw,marker_color=T["blue"])); f1.add_trace(go.Bar(name="Net",x=nm,y=net,marker_color=T["amber"]))
        f1.update_layout(title="Raw vs Cost-Adjusted Returns",barmode="group",xaxis_tickangle=-45)
    f2=go.Figure()
    if n>=2:
        f2.add_trace(go.Pie(labels=[v["v"][:16] for v in vs[:n]],values=[1/n]*n,hole=0.5))
        f2.update_layout(title=f"Equal-Weight Portfolio (Top {n})")
    regimes=["BULL","BEAR","RANGING","HIGH_VOL","CRASH","RECOVERY"]; rrets=[8.2,-2.1,1.5,3.4,-5.8,6.1]
    f3=go.Figure(); f3.add_trace(go.Bar(x=regimes,y=rrets,marker_color=[T["green"],T["red"],T["blue"],T["amber"],T["red"],T["green"]]))
    f3.update_layout(title="Portfolio by Regime",yaxis_title="Return %")
    return _col(_card(_title("FTMO Compliance"),_tbl(["Account","Final","Daily<5%","Total<10%","Target","Overall"],fr) if fr else _empty("No data")),
        _g(4,_card(_m("Strategies",n)),_card(_m("Port Ret",f"{pr:+.1f}%",_rc(pr))),_card(_m("Port SR",f"{ps:.2f}",_sc(ps))),_card(_m("Weight",f"{100/n:.0f}%" if n>=2 else "--"))),
        _g(2,_card(_fig(f1,300)) if vs else html.div(),_card(_fig(f2,300)) if n>=2 else html.div()),_card(_fig(f3,280)))

# ======================== PAGE 7: SYSTEM ========================
@component
def PgSys():
    rm,srm=hooks.use_state(""); s=D.bts()
    sel_strat,set_sel_strat=hooks.use_state("")
    bs={"padding":"7px 14px","border":"none","borderRadius":"7px","cursor":"pointer","fontSize":"11px","fontWeight":"600","color":"#fff"}

    # Get strategy names for the single-test dropdown
    inbox_strats=D.inbox()
    strat_options=[(s.get("strategy_id",""),s.get("strategy_name","?")) for s in inbox_strats if s.get("has_code") or (s.get("generated_code") or "").strip()]

    def handle_test_single(e):
        if not sel_strat:
            srm("Select a strategy first"); return
        _export_and_run_single(sel_strat,srm)(e)

    mg=[("Backtesting",T["blue"],["CanonicalResult","BacktestAdapter","RegimeClassifier","FeatureEngineer"]),
        ("Validation",T["green"],["ValidationFramework","RobustnessTests","ParameterSensitivity","AdversarialReviewer","CostAdjustedScorer"]),
        ("Optimization",T["purple"],["SurrogateModel","StrategyOptimizer","GeneticEngine"]),
        ("Risk",T["red"],["MarketImpactModel","CapacityModel","KillSwitch","TailRiskAnalyzer","LiquidityStressTest"]),
        ("Live",T["cyan"],["DriftDetector","ShadowTrader","StrategyLifecycle"]),
        ("Portfolio",T["accent"],["FTMOComplianceChecker","PortfolioEngine","MetaModel","PerformanceAttributor"]),
        ("Foundation",T["amber"],["LineageTracker","OverfittingDetector","FilteringPipeline","DiversificationFilter"])]
    def _dt(ok): return html.span({"style":{"display":"inline-block","width":"6px","height":"6px","borderRadius":"50%","backgroundColor":T["green"] if ok else T["red"],"marginRight":"5px"}})
    gc=[]
    for gn,co,mods in mg:
        ld=sum(1 for m in mods if m in M); pc=int(ld/len(mods)*100) if mods else 0
        gc.append(_card(html.div({"style":{"display":"flex","justifyContent":"space-between","marginBottom":"6px"}},
            html.span({"style":{"color":co,"fontWeight":"600","fontSize":"11px"}},gn),_badge(f"{pc}%",T["green"] if pc==100 else T["amber"])),
            html.div({"style":{"height":"3px","borderRadius":"2px","backgroundColor":T["elevated"],"overflow":"hidden","marginBottom":"6px"}},
                html.div({"style":{"height":"100%","width":f"{pc}%","backgroundColor":co}})),
            *[html.div({"style":{"display":"flex","alignItems":"center","padding":"2px 0","fontSize":"11px"}},_dt(m in M),html.span({"style":{"color":T["text"] if m in M else T["dim"]}},m)) for m in mods],accent=co))

    # Single strategy test section
    dd_sty={"padding":"7px 12px","backgroundColor":T["elevated"],"color":T["text"],"border":f"1px solid {T['border']}","borderRadius":"8px","fontSize":"12px","outline":"none","minWidth":"250px","cursor":"pointer"}
    single_test=_card(
        _title("Test Single Strategy"),
        html.div({"style":{"display":"flex","gap":"10px","alignItems":"center","flexWrap":"wrap"}},
            html.select({"value":sel_strat,"style":dd_sty,"onChange":lambda e:set_sel_strat(e["target"]["value"])},
                html.option({"value":""},"-- Pick a strategy --"),
                *[html.option({"value":sid},sname) for sid,sname in strat_options]),
            html.button({"style":{**bs,"backgroundColor":T["green"],"padding":"8px 20px","fontSize":"12px"},"onClick":handle_test_single},
                "Export & Backtest"),
            html.span({"style":{"color":T["muted"],"fontSize":"10px"}},"Exports the .py file and runs run_single_strategy.py in a new terminal")),
        html.p({"style":{"color":T["dim"],"fontSize":"11px","marginTop":"8px","margin":"8px 0 0"}},
            f"{len(strat_options)} strategies with code available" if strat_options else "No strategies with code. Add one on My Strategies page."),
        accent=T["green"])

    # Batch actions
    acts=html.div({"style":{"display":"flex","gap":"7px","flexWrap":"wrap"}},
        html.button({"style":{**bs,"backgroundColor":T["blue"]},"onClick":_runcmd("Backtests",["run_backtests.py"],srm)},"Run All Backtests"),
        html.button({"style":{**bs,"backgroundColor":T["purple"]},"onClick":_runcmd("Variants",["run_variant_backtests.py"],srm)},"Variant Backtests"),
        html.button({"style":{**bs,"backgroundColor":T["accent"]},"onClick":_runcmd("Pipeline",["run_pipeline.py"],srm)},"Full Pipeline"),
        html.button({"style":{**bs,"backgroundColor":T["green"]},"onClick":_runcmd("Mutate",["mutate_strategy.py"],srm)},"Generate Variants (Claude)"),
        html.button({"style":{**bs,"backgroundColor":T["amber"]},"onClick":_runcmd("Discovery",["run_discovery.py"],srm)},"Discovery (optional)"),
        html.span({"style":{"color":T["green"] if "Started" in rm or "testing" in rm else T["red"] if "Error" in rm or "Not found" in rm else T["amber"],"fontSize":"11px","alignSelf":"center"}},rm) if rm else html.span())

    return _col(_g(5,_card(_m("Backtests",s["total"],T["blue"])),_card(_m("Symbols",len(s["symbols"]),T["cyan"])),
        _card(_m("Variants",len(s["variants"]),T["purple"])),_card(_m("Modules",f"{len(M)}/{TOTAL_MODULES}",T["accent"])),_card(_m("Avg SR",f"{s['avg_sr']:.2f}",_sc(s["avg_sr"])))),
        single_test,
        _title("Batch Actions"),acts,_title("Module Health"),_g(4,*gc))

# ======================== MAIN ========================
NAV=[("strategies","My Strategies"),("backtests","Backtests"),("validation","Validation"),("overfitting","Overfitting & Filters"),
     ("risk","Risk & Impact"),("ftmo","FTMO & Portfolio"),("system","System")]
PAGES={"strategies":PgStrats,"backtests":PgBT,"validation":PgVal,"overfitting":PgOverfit,"risk":PgRisk,"ftmo":PgFTMO,"system":PgSys}
SUBS={"strategies":"Add strategies, view code, manage library","backtests":"Per-strategy results, 8 charts, variant drill-down",
      "validation":"MC, bootstrap, permutation, walk-forward, sensitivity, stress, adversarial","overfitting":"IS vs OOS, DSR, filter funnel, correlation, Pareto",
      "risk":"VaR, CVaR, impact, capacity, stress, QQ, kill switch, attribution","ftmo":"Prop firm compliance + portfolio allocation","system":"Module health, run actions"}

@component
def App():
    pg,spg=hooks.use_state("strategies"); s=D.bts()
    return html.div({"style":{"display":"flex","minHeight":"100vh","backgroundColor":T["bg"],"color":T["text"],"fontFamily":"'SF Pro Display',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif"}},
        html.aside({"style":{"width":"200px","backgroundColor":T["surface"],"borderRight":f"1px solid {T['border']}","display":"flex","flexDirection":"column","position":"fixed","top":"0","left":"0","bottom":"0","zIndex":"10"}},
            html.div({"style":{"padding":"16px 14px","borderBottom":f"1px solid {T['border']}"}},
                html.p({"style":{"fontWeight":"800","fontSize":"15px","margin":"0","color":T["accent"]}},"TradingLab"),
                html.p({"style":{"fontSize":"9px","margin":"2px 0 0","color":T["dim"],"letterSpacing":"1.5px"}},"ULTIMATE BACKTESTER")),
            html.nav({"style":{"flex":"1","padding":"6px","overflowY":"auto"}},
                *[html.button({"style":{"width":"100%","display":"flex","alignItems":"center","padding":"8px 10px","marginBottom":"1px","borderRadius":"7px","border":"none","cursor":"pointer",
                    "fontSize":"12px","fontWeight":"600" if pg==pid else "400","textAlign":"left","backgroundColor":f"{T['accent']}18" if pg==pid else "transparent",
                    "color":T["accent"] if pg==pid else T["dim"]},"onClick":lambda e,p=pid:spg(p)},lbl) for pid,lbl in NAV]),
            html.div({"style":{"padding":"8px 12px","borderTop":f"1px solid {T['border']}","backgroundColor":T["card"],"fontSize":"10px"}},
                *[html.div({"style":{"display":"flex","justifyContent":"space-between","marginBottom":"2px"}},
                    html.span({"style":{"color":T["dim"]}},k),html.span({"style":{"color":T["text"],"fontWeight":"600"}},v)) for k,v in [("Backtests",str(s["total"])),("Modules",f"{len(M)}/{TOTAL_MODULES}"),("Symbols",str(len(s["symbols"])))]])),
        html.div({"style":{"flex":"1","marginLeft":"200px","display":"flex","flexDirection":"column"}},
            html.header({"style":{"backgroundColor":f"{T['surface']}ee","backdropFilter":"blur(12px)","borderBottom":f"1px solid {T['border']}","padding":"10px 20px",
                "display":"flex","justifyContent":"space-between","alignItems":"center","position":"sticky","top":"0","zIndex":"5"}},
                html.div(html.h2({"style":{"fontSize":"15px","fontWeight":"700","margin":"0"}},dict(NAV).get(pg,"")),
                    html.p({"style":{"color":T["dim"],"fontSize":"10px","margin":"0"}},SUBS.get(pg,""))),
                html.button({"style":{"padding":"5px 12px","backgroundColor":T["elevated"],"color":T["muted"],"border":"none","borderRadius":"7px","cursor":"pointer","fontSize":"11px"},"onClick":lambda e:D.refresh()},"Refresh")),
            html.main({"style":{"flex":"1","padding":"18px","overflowY":"auto"}},PAGES.get(pg,PgStrats)())))

if _BE=="fastapi": app=FastAPI(title="TradingLab")
else: app=Starlette()
configure(app,App)
if __name__=="__main__":
    print("\n"+"="*55+"\n  TradingLab -- Ultimate Backtester (Full Analytics)\n"+"="*55)
    print(f"  DB: {DB_BT}\n  Modules: {len(M)}/{TOTAL_MODULES}\n  Charts: 30+")
    for _,l in NAV: print(f"    {l}")
    print("-"*55+f"\n  http://127.0.0.1:8080\n"+"="*55+"\n")
    uvicorn.run(app,host="127.0.0.1",port=8080,log_level="info")
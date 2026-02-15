# ==============================================================================
# dashboard_vizro.py
# ==============================================================================
# Trading Research Dashboard using Vizro (by McKinsey)
#
# Install: pip install vizro pandas numpy
# Run: python dashboard_vizro.py
# Opens at: http://127.0.0.1:8050
# ==============================================================================

import vizro.models as vm
import vizro.plotly.express as px
from vizro import Vizro
from vizro.tables import dash_ag_grid
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==============================================================================
# SAMPLE DATA
# ==============================================================================

np.random.seed(42)

# Strategy data
strategies_df = pd.DataFrame({
    'Strategy': ['variant_04_atr_stop', 'variant_01_rsi_filter', 'variant_02_volume', 
                 'variant_05_momentum', 'simple_strategy', 'variant_03_breakout'],
    'Description': ['ATR Trailing Stop', 'RSI Entry Filter', 'Volume Confirmation',
                    'ADX Momentum Filter', 'Base SMA Crossover', 'Range Breakout'],
    'Gross Return (%)': [15.7, 12.4, 8.2, 5.3, 3.1, -2.1],
    'Net Return (%)': [11.2, 8.1, 4.9, 1.8, -0.8, -6.4],
    'Sharpe': [2.1, 1.8, 1.2, 0.9, 0.5, -0.3],
    'Max DD (%)': [-6.2, -8.1, -11.3, -9.8, -12.4, -15.2],
    'Trades': [67, 145, 89, 156, 112, 234],
    'Win Rate (%)': [64, 58, 52, 49, 47, 41],
    'Status': ['promising', 'testing', 'testing', 'review', 'baseline', 'failed']
})

# Equity curve data
equity_data = []
equity = 10000
for i in range(200):
    change = (np.sin(i / 15) * 50 + (np.random.randn() - 0.4) * 100) * (1 + i / 500)
    equity = max(equity + change, equity * 0.85)
    equity_data.append({'Bar': i, 'Equity': equity, 'Type': 'Strategy'})
    equity_data.append({'Bar': i, 'Equity': 10000 + i * 25, 'Type': 'Benchmark'})
equity_df = pd.DataFrame(equity_data)

# Walk-forward data
wf_data = []
for i in range(8):
    wf_data.append({'Fold': f'Fold {i+1}', 'Return': 12 + np.random.randn() * 4, 'Type': 'In-Sample'})
    wf_data.append({'Fold': f'Fold {i+1}', 'Return': 8 + np.random.randn() * 5, 'Type': 'Out-of-Sample'})
wf_df = pd.DataFrame(wf_data)

# Regime data
regime_df = pd.DataFrame({
    'Regime': ['BULL', 'RECOVERY', 'HIGH_VOL', 'BEAR', 'RANGING', 'CRASH'],
    'Return (%)': [18.2, 12.3, 8.7, 4.1, -2.3, -5.4],
    'Trades': [89, 28, 32, 45, 156, 12]
})

# Monte Carlo final equities
mc_finals = []
for i in range(500):
    equity = 10000
    for _ in range(100):
        equity += (np.random.randn() * 0.02 + 0.001) * equity
    mc_finals.append({'Simulation': i, 'Final Equity': max(equity, 0)})
mc_df = pd.DataFrame(mc_finals)

# Latency sensitivity
latency_df = pd.DataFrame({
    'Delay (ms)': [0, 100, 250, 500, 1000, 2000],
    'Return (%)': [15.7, 14.2, 12.8, 10.1, 6.4, 2.1]
})

# Slippage sensitivity
slippage_df = pd.DataFrame({
    'Slippage (bps)': [0, 5, 10, 20, 50, 100],
    'Return (%)': [15.7, 14.1, 12.5, 9.3, 1.2, -12.4]
})

# Performance by asset
asset_df = pd.DataFrame({
    'Asset': ['EUR-USD', 'GBP-USD', 'USD-JPY', 'AUD-USD', 'BTC-USD'],
    'Return (%)': [12.3, 8.7, -2.1, 5.4, 18.9],
    'Trades': [234, 189, 156, 178, 45]
})

# ==============================================================================
# DASHBOARD PAGES
# ==============================================================================

overview_page = vm.Page(
    title="Overview",
    components=[
        vm.Card(
            text="""
            # Trading Research Dashboard
            
            **Best Strategy:** variant_04_atr_stop  
            **Best Return:** +15.7% (gross) / +11.2% (net)  
            **Best Sharpe:** 2.1  
            **Total Strategies:** 15 (3 promising)
            
            ---
            
            ### Quick Validation Summary
            - ✅ Permutation Test: p=0.023
            - ✅ Walk-Forward: Ratio 0.82
            - ✅ Bootstrap CI: [4.2%, 18.1%]
            - ✅ Monte Carlo VaR: -8.2%
            - ⚠️ Slippage Test: -12% @ 20bps
            """
        ),
        vm.Graph(
            figure=px.line(
                equity_df, x='Bar', y='Equity', color='Type',
                title='Equity Curve',
                color_discrete_map={'Strategy': '#6366f1', 'Benchmark': '#6b7280'}
            )
        ),
        vm.Graph(
            figure=px.bar(
                strategies_df, x='Strategy', y='Gross Return (%)',
                color='Gross Return (%)',
                color_continuous_scale=['#ef4444', '#f59e0b', '#10b981'],
                title='Strategy Returns'
            )
        ),
    ],
    layout=vm.Layout(grid=[[0, 1], [0, 2]])
)

strategies_page = vm.Page(
    title="Strategies",
    components=[
        vm.Card(
            text="""
            # Strategy Variants
            Compare and analyze all strategy variants ranked by net return.
            """
        ),
        vm.AgGrid(
            figure=dash_ag_grid(strategies_df)
        ),
        vm.Graph(
            figure=px.scatter(
                strategies_df, x='Sharpe', y='Gross Return (%)',
                size='Trades', color='Status',
                hover_name='Strategy',
                title='Return vs Sharpe (bubble size = trades)',
                color_discrete_map={
                    'promising': '#10b981', 'testing': '#3b82f6',
                    'review': '#f59e0b', 'baseline': '#6b7280', 'failed': '#ef4444'
                }
            )
        ),
    ],
    layout=vm.Layout(grid=[[0], [1], [2]])
)

validation_page = vm.Page(
    title="Validation",
    components=[
        vm.Card(
            text="""
            # Statistical Validation
            
            Comprehensive statistical tests to validate strategy performance.
            
            | Test | Result | Status |
            |------|--------|--------|
            | Permutation Test | p=0.023 | ✅ Pass |
            | Walk-Forward | Ratio 0.82 | ✅ Pass |
            | Bootstrap CI | [4.2%, 18.1%] | ✅ Pass |
            | Monte Carlo | 2.3% ruin prob | ✅ Pass |
            """
        ),
        vm.Graph(
            figure=px.histogram(
                mc_df, x='Final Equity', nbins=50,
                title='Monte Carlo: Final Equity Distribution',
                color_discrete_sequence=['#6366f1']
            )
        ),
        vm.Graph(
            figure=px.bar(
                wf_df, x='Fold', y='Return', color='Type', barmode='group',
                title='Walk-Forward Analysis',
                color_discrete_map={'In-Sample': '#6366f1', 'Out-of-Sample': '#10b981'}
            )
        ),
    ],
    layout=vm.Layout(grid=[[0, 1], [0, 2]])
)

robustness_page = vm.Page(
    title="Robustness",
    components=[
        vm.Card(
            text="""
            # Robustness Testing
            
            Stress tests and sensitivity analysis.
            
            ### Summary
            - ✅ Latency 500ms: -36% degradation
            - ⚠️ Slippage 20bps: -41% degradation
            - ✅ Parameter Stability: ±3%
            - ⚠️ RANGING regime shows negative returns
            """
        ),
        vm.Graph(
            figure=px.line(
                latency_df, x='Delay (ms)', y='Return (%)',
                title='Latency Sensitivity',
                markers=True, color_discrete_sequence=['#6366f1']
            )
        ),
        vm.Graph(
            figure=px.line(
                slippage_df, x='Slippage (bps)', y='Return (%)',
                title='Slippage Stress Test',
                markers=True, color_discrete_sequence=['#10b981']
            )
        ),
        vm.Graph(
            figure=px.bar(
                regime_df, x='Return (%)', y='Regime', orientation='h',
                title='Performance by Regime',
                color='Return (%)',
                color_continuous_scale=['#ef4444', '#f59e0b', '#10b981']
            )
        ),
    ],
    layout=vm.Layout(grid=[[0, 1], [0, 2], [3, 3]])
)

analysis_page = vm.Page(
    title="Analysis",
    components=[
        vm.Card(
            text="""
            # Detailed Analysis
            
            ### Performance Metrics (Best Strategy)
            
            | Metric | Value |
            |--------|-------|
            | Total Return | +15.7% |
            | Net Return | +11.2% |
            | Sharpe Ratio | 2.1 |
            | Sortino Ratio | 3.2 |
            | Max Drawdown | -6.2% |
            | Win Rate | 64% |
            | Profit Factor | 2.1 |
            | Total Trades | 67 |
            """
        ),
        vm.Graph(
            figure=px.bar(
                asset_df, x='Asset', y='Return (%)',
                color='Return (%)',
                color_continuous_scale=['#ef4444', '#f59e0b', '#10b981'],
                title='Performance by Asset'
            )
        ),
        vm.Graph(
            figure=px.pie(
                regime_df[regime_df['Return (%)'] > 0],
                values='Trades', names='Regime',
                title='Trade Distribution (Profitable Regimes)',
                color_discrete_sequence=['#10b981', '#22c55e', '#6366f1', '#8b5cf6']
            )
        ),
    ],
    layout=vm.Layout(grid=[[0, 1], [0, 2]])
)

# ==============================================================================
# BUILD DASHBOARD
# ==============================================================================

dashboard = vm.Dashboard(
    title="Trading Research",
    pages=[overview_page, strategies_page, validation_page, robustness_page, analysis_page],
    theme="vizro_dark"
)

# ==============================================================================
# RUN
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Trading Research Dashboard (Vizro)")
    print("="*50)
    print("  Open: http://127.0.0.1:8050")
    print("  Press Ctrl+C to stop")
    print("="*50 + "\n")
    Vizro().build(dashboard).run()
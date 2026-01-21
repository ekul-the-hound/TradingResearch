# ==============================================================================
# dashboard.py
# ==============================================================================
# VISUAL DASHBOARD (Streamlit)
# 
# Run with: streamlit run dashboard.py
# 
# Features:
# - Interactive charts
# - Variant comparison
# - Filter by asset, timeframe, variant
# - Equity curves
# - Performance metrics
# - Export functionality
#
# Cost: FREE
# ==============================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from pathlib import Path
from datetime import datetime

import config

# ==============================================================================
# PAGE CONFIG
# ==============================================================================

st.set_page_config(
    page_title="Trading Research Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# DATA LOADING
# ==============================================================================

@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_results():
    """Load all backtest results from database"""
    
    try:
        conn = sqlite3.connect(config.DATABASE_PATH)
        df = pd.read_sql_query("SELECT * FROM backtest_results", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return pd.DataFrame()


def get_unique_values(df, column):
    """Get unique values for a column, handling None"""
    if column in df.columns:
        return sorted([v for v in df[column].unique() if v is not None])
    return []


# ==============================================================================
# SIDEBAR
# ==============================================================================

def render_sidebar(df):
    """Render sidebar filters"""
    
    st.sidebar.title("📊 Filters")
    
    # Strategy/Variant filter
    strategies = get_unique_values(df, 'strategy_name')
    variant_ids = get_unique_values(df, 'variant_id')
    
    all_variants = list(set(strategies + variant_ids))
    
    selected_variants = st.sidebar.multiselect(
        "Strategy/Variant",
        options=all_variants,
        default=all_variants[:5] if len(all_variants) > 5 else all_variants
    )
    
    # Asset filter
    assets = get_unique_values(df, 'symbol')
    selected_assets = st.sidebar.multiselect(
        "Assets",
        options=assets,
        default=assets
    )
    
    # Timeframe filter
    timeframes = get_unique_values(df, 'timeframe')
    selected_timeframes = st.sidebar.multiselect(
        "Timeframes",
        options=timeframes,
        default=timeframes
    )
    
    # Metrics to display
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Metrics")
    
    show_sharpe = st.sidebar.checkbox("Show Sharpe Ratio", value=True)
    show_drawdown = st.sidebar.checkbox("Show Drawdown", value=True)
    show_win_rate = st.sidebar.checkbox("Show Win Rate", value=True)
    
    return {
        'variants': selected_variants,
        'assets': selected_assets,
        'timeframes': selected_timeframes,
        'show_sharpe': show_sharpe,
        'show_drawdown': show_drawdown,
        'show_win_rate': show_win_rate
    }


def filter_dataframe(df, filters):
    """Apply filters to dataframe"""
    
    filtered = df.copy()
    
    if filters['variants']:
        filtered = filtered[
            (filtered['strategy_name'].isin(filters['variants'])) |
            (filtered['variant_id'].isin(filters['variants']))
        ]
    
    if filters['assets']:
        filtered = filtered[filtered['symbol'].isin(filters['assets'])]
    
    if filters['timeframes']:
        filtered = filtered[filtered['timeframe'].isin(filters['timeframes'])]
    
    return filtered


# ==============================================================================
# CHARTS
# ==============================================================================

def create_return_comparison_chart(df):
    """Bar chart comparing average returns by variant"""
    
    # Group by strategy/variant
    df['variant'] = df['variant_id'].fillna(df['strategy_name'])
    
    grouped = df.groupby('variant').agg({
        'total_return_pct': 'mean',
        'sharpe_ratio': 'mean',
        'win_rate': 'mean',
        'total_trades': 'sum'
    }).reset_index()
    
    grouped = grouped.sort_values('total_return_pct', ascending=True)
    
    # Create color scale (red for negative, green for positive)
    colors = ['#ef4444' if x < 0 else '#22c55e' for x in grouped['total_return_pct']]
    
    fig = go.Figure(go.Bar(
        x=grouped['total_return_pct'],
        y=grouped['variant'],
        orientation='h',
        marker_color=colors,
        text=[f"{x:+.2f}%" for x in grouped['total_return_pct']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Average Return by Variant",
        xaxis_title="Average Return (%)",
        yaxis_title="",
        height=max(400, len(grouped) * 30),
        showlegend=False
    )
    
    return fig


def create_heatmap(df):
    """Heatmap of returns by asset and timeframe"""
    
    pivot = df.pivot_table(
        values='total_return_pct',
        index='symbol',
        columns='timeframe',
        aggfunc='mean'
    )
    
    # Reorder columns by timeframe
    tf_order = ['1min', '5min', '15min', '30min', '1hour', '4hour', '1day', '1week', '1month']
    cols = [c for c in tf_order if c in pivot.columns]
    pivot = pivot[cols]
    
    fig = px.imshow(
        pivot,
        labels=dict(x="Timeframe", y="Asset", color="Return %"),
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0,
        aspect='auto'
    )
    
    fig.update_layout(
        title="Returns Heatmap (Asset × Timeframe)",
        height=400
    )
    
    return fig


def create_scatter_plot(df):
    """Scatter plot of Return vs Sharpe Ratio"""
    
    # Filter out None values
    plot_df = df.dropna(subset=['total_return_pct', 'sharpe_ratio'])
    
    if plot_df.empty:
        return None
    
    # Cap extreme sharpe values for better visualization
    plot_df['sharpe_capped'] = plot_df['sharpe_ratio'].clip(-10, 10)
    
    fig = px.scatter(
        plot_df,
        x='sharpe_capped',
        y='total_return_pct',
        color='symbol',
        hover_data=['strategy_name', 'timeframe', 'total_trades'],
        labels={
            'sharpe_capped': 'Sharpe Ratio',
            'total_return_pct': 'Return (%)',
            'symbol': 'Asset'
        }
    )
    
    fig.update_layout(
        title="Return vs Sharpe Ratio",
        height=500
    )
    
    # Add quadrant lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    return fig


def create_trades_distribution(df):
    """Histogram of trade counts"""
    
    fig = px.histogram(
        df,
        x='total_trades',
        nbins=30,
        color='symbol',
        labels={'total_trades': 'Number of Trades'}
    )
    
    fig.update_layout(
        title="Trade Count Distribution",
        height=400
    )
    
    return fig


def create_timeframe_comparison(df):
    """Box plot comparing returns across timeframes"""
    
    # Order timeframes
    tf_order = ['1min', '5min', '15min', '30min', '1hour', '4hour', '1day', '1week', '1month']
    df['tf_order'] = df['timeframe'].apply(lambda x: tf_order.index(x) if x in tf_order else 99)
    df = df.sort_values('tf_order')
    
    fig = px.box(
        df,
        x='timeframe',
        y='total_return_pct',
        color='timeframe',
        labels={
            'timeframe': 'Timeframe',
            'total_return_pct': 'Return (%)'
        }
    )
    
    fig.update_layout(
        title="Returns by Timeframe",
        height=400,
        showlegend=False
    )
    
    return fig


# ==============================================================================
# METRICS CARDS
# ==============================================================================

def render_metrics(df):
    """Render key metric cards"""
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Backtests",
            f"{len(df):,}"
        )
    
    with col2:
        avg_return = df['total_return_pct'].mean()
        st.metric(
            "Avg Return",
            f"{avg_return:+.2f}%",
            delta=f"{avg_return:+.2f}%"
        )
    
    with col3:
        positive_pct = (df['total_return_pct'] > 0).mean() * 100
        st.metric(
            "% Profitable",
            f"{positive_pct:.1f}%"
        )
    
    with col4:
        best = df['total_return_pct'].max()
        st.metric(
            "Best Return",
            f"{best:+.2f}%"
        )
    
    with col5:
        worst = df['total_return_pct'].min()
        st.metric(
            "Worst Return",
            f"{worst:+.2f}%"
        )


# ==============================================================================
# RESULTS TABLE
# ==============================================================================

def render_results_table(df):
    """Render sortable results table"""
    
    # Select columns to display
    display_cols = [
        'strategy_name', 'variant_id', 'symbol', 'timeframe',
        'total_return_pct', 'sharpe_ratio', 'win_rate',
        'max_drawdown_pct', 'total_trades'
    ]
    
    available_cols = [c for c in display_cols if c in df.columns]
    display_df = df[available_cols].copy()
    
    # Format numeric columns
    if 'total_return_pct' in display_df.columns:
        display_df['total_return_pct'] = display_df['total_return_pct'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
    if 'sharpe_ratio' in display_df.columns:
        display_df['sharpe_ratio'] = display_df['sharpe_ratio'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    if 'win_rate' in display_df.columns:
        display_df['win_rate'] = display_df['win_rate'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
    if 'max_drawdown_pct' in display_df.columns:
        display_df['max_drawdown_pct'] = display_df['max_drawdown_pct'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
    
    # Rename columns for display
    display_df.columns = [c.replace('_', ' ').title() for c in display_df.columns]
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )


# ==============================================================================
# EXPORT
# ==============================================================================

def render_export_section(df):
    """Render export options"""
    
    st.subheader("📥 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"backtest_results_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Summary stats
        summary = {
            'Total Backtests': len(df),
            'Avg Return': f"{df['total_return_pct'].mean():.2f}%",
            'Best Return': f"{df['total_return_pct'].max():.2f}%",
            'Worst Return': f"{df['total_return_pct'].min():.2f}%",
            'Unique Strategies': df['strategy_name'].nunique(),
            'Unique Assets': df['symbol'].nunique()
        }
        
        summary_text = "\n".join([f"{k}: {v}" for k, v in summary.items()])
        
        st.download_button(
            label="Download Summary",
            data=summary_text,
            file_name=f"backtest_summary_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )


# ==============================================================================
# MAIN APP
# ==============================================================================

def main():
    """Main dashboard app"""
    
    # Header
    st.title("📊 Trading Research Dashboard")
    st.markdown("---")
    
    # Load data
    df = load_results()
    
    if df.empty:
        st.warning("⚠️ No backtest results found in database.")
        st.info("Run `python run_backtests.py` to generate results.")
        return
    
    # Sidebar filters
    filters = render_sidebar(df)
    
    # Apply filters
    filtered_df = filter_dataframe(df, filters)
    
    if filtered_df.empty:
        st.warning("No results match the selected filters.")
        return
    
    # Metrics row
    render_metrics(filtered_df)
    
    st.markdown("---")
    
    # Charts - Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            create_return_comparison_chart(filtered_df),
            use_container_width=True
        )
    
    with col2:
        st.plotly_chart(
            create_heatmap(filtered_df),
            use_container_width=True
        )
    
    # Charts - Row 2
    col3, col4 = st.columns(2)
    
    with col3:
        scatter_fig = create_scatter_plot(filtered_df)
        if scatter_fig:
            st.plotly_chart(scatter_fig, use_container_width=True)
        else:
            st.info("Not enough data for scatter plot")
    
    with col4:
        st.plotly_chart(
            create_timeframe_comparison(filtered_df),
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Results table
    st.subheader("📋 All Results")
    render_results_table(filtered_df)
    
    st.markdown("---")
    
    # Export
    render_export_section(filtered_df)
    
    # Footer
    st.markdown("---")
    st.caption(f"Data from: {config.DATABASE_PATH}")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

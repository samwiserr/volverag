"""
Streamlit dashboard for RAG system performance metrics.

Run with: streamlit run src/monitoring/dashboard.py
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
import os

from .performance_monitor import PerformanceMonitor

st.set_page_config(page_title="RAG Performance Dashboard", layout="wide")

# Initialize monitor
metrics_path = os.getenv("RAG_METRICS_LOG_PATH", "./data/monitoring/metrics.jsonl")
monitor = PerformanceMonitor(metrics_path)


def load_metrics(limit: int = 1000) -> List[Dict[str, Any]]:
    """Load metrics from log file."""
    return monitor.get_recent_metrics(limit)


def filter_metrics_by_timeframe(metrics: List[Dict[str, Any]], hours: int = 24) -> List[Dict[str, Any]]:
    """Filter metrics by timeframe."""
    cutoff = datetime.now() - timedelta(hours=hours)
    filtered = []
    for m in metrics:
        try:
            timestamp = datetime.fromisoformat(m.get('timestamp', ''))
            if timestamp >= cutoff:
                filtered.append(m)
        except Exception:
            continue
    return filtered


def aggregate_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics by metric name."""
    aggregated = {}
    for m in metrics:
        metric_name = m.get('metric', 'unknown')
        value = m.get('value', 0)
        
        if metric_name not in aggregated:
            aggregated[metric_name] = []
        aggregated[metric_name].append(value)
    
    # Calculate statistics
    stats = {}
    for metric_name, values in aggregated.items():
        if values:
            stats[metric_name] = {
                'count': len(values),
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'sum': sum(values)
            }
    
    return stats


# Main dashboard
st.title("RAG System Performance Dashboard")

# Sidebar filters
st.sidebar.header("Filters")
timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["Last 1 hour", "Last 6 hours", "Last 24 hours", "Last 7 days", "All time"],
    index=2
)

hours_map = {
    "Last 1 hour": 1,
    "Last 6 hours": 6,
    "Last 24 hours": 24,
    "Last 7 days": 168,
    "All time": None
}
hours = hours_map[timeframe]

# Load metrics
all_metrics = load_metrics(limit=10000)
if hours:
    metrics = filter_metrics_by_timeframe(all_metrics, hours)
else:
    metrics = all_metrics

st.sidebar.metric("Total Metrics", len(metrics))

# Overview metrics
st.header("Overview")
col1, col2, col3, col4 = st.columns(4)

# Query latency
latency_stats = monitor.get_statistics("query.total_time", limit=1000)
col1.metric("Avg Query Time", f"{latency_stats.get('mean', 0):.2f}s")
col2.metric("Min Query Time", f"{latency_stats.get('min', 0):.2f}s")
col3.metric("Max Query Time", f"{latency_stats.get('max', 0):.2f}s")
col4.metric("Total Queries", latency_stats.get('count', 0))

# Token usage
st.header("Token Usage")
token_stats = monitor.get_statistics("query.tokens.total", limit=1000)
if token_stats.get('count', 0) > 0:
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Tokens/Query", f"{token_stats.get('mean', 0):.0f}")
    col2.metric("Total Tokens", f"{token_stats.get('sum', 0):.0f}")
    col3.metric("Est. Cost (gpt-4o)", f"${token_stats.get('sum', 0) * 0.0025 / 1000:.2f}")

# Retrieval metrics
st.header("Retrieval Performance")
retrieval_stats = monitor.get_statistics("query.retrieval.num_docs", limit=1000)
if retrieval_stats.get('count', 0) > 0:
    col1, col2 = st.columns(2)
    col1.metric("Avg Docs Retrieved", f"{retrieval_stats.get('mean', 0):.1f}")
    col2.metric("Retrieval Queries", retrieval_stats.get('count', 0))

# Cache performance
cache_stats = monitor.get_statistics("query.cache.hit", limit=1000)
if cache_stats.get('count', 0) > 0:
    cache_hit_rate = cache_stats.get('mean', 0) * 100
    st.metric("Cache Hit Rate", f"{cache_hit_rate:.1f}%")

# Time series charts
st.header("Time Series")

# Prepare data for charts
if metrics:
    df_data = []
    for m in metrics:
        try:
            timestamp = datetime.fromisoformat(m.get('timestamp', ''))
            df_data.append({
                'timestamp': timestamp,
                'metric': m.get('metric', ''),
                'value': m.get('value', 0)
            })
        except Exception:
            continue
    
    if df_data:
        df = pd.DataFrame(df_data)
        
        # Query latency over time
        latency_df = df[df['metric'] == 'query.total_time'].copy()
        if not latency_df.empty:
            latency_df = latency_df.set_index('timestamp').resample('5T')['value'].mean().reset_index()
            st.subheader("Query Latency Over Time")
            st.line_chart(latency_df.set_index('timestamp'))
        
        # Token usage over time
        token_df = df[df['metric'] == 'query.tokens.total'].copy()
        if not token_df.empty:
            token_df = token_df.set_index('timestamp').resample('5T')['value'].sum().reset_index()
            st.subheader("Token Usage Over Time")
            st.line_chart(token_df.set_index('timestamp'))

# Metric breakdown
st.header("Metric Breakdown")
aggregated = aggregate_metrics(metrics)
if aggregated:
    metric_names = sorted(aggregated.keys())
    selected_metric = st.selectbox("Select Metric", metric_names)
    
    if selected_metric:
        stats = aggregated[selected_metric]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Count", stats['count'])
        col2.metric("Mean", f"{stats['mean']:.4f}")
        col3.metric("Min", f"{stats['min']:.4f}")
        col4.metric("Max", f"{stats['max']:.4f}")

# Recent metrics table
st.header("Recent Metrics")
if metrics:
    recent_df = pd.DataFrame(metrics[-50:])  # Last 50 metrics
    if not recent_df.empty:
        st.dataframe(recent_df[['timestamp', 'metric', 'value']], use_container_width=True)





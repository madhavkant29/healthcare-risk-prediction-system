import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))
from api_client import get_history, get_stats

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

if not st.session_state.get("token"):
    st.warning("Please log in first.")
    st.stop()

st.title("Prediction Dashboard")
st.caption("Summary of all predictions made in this session.")

# ── Load data ──────────────────────────────────────────────
with st.spinner("Loading..."):
    try:
        stats = get_stats()
        history = get_history(limit=200)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

if stats["total"] == 0:
    st.info("No predictions yet. Go to **Single Prediction** or **Batch Upload** to get started.")
    st.stop()

# ── Metric cards ───────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
by_class = stats.get("by_class", {})

with col1:
    st.metric("Total predictions", stats["total"])
with col2:
    st.metric("No readmission", by_class.get("NO", 0))
with col3:
    st.metric("Readmitted <30 days", by_class.get("<30", 0), delta_color="inverse")
with col4:
    st.metric("Readmitted >30 days", by_class.get(">30", 0), delta_color="inverse")

st.divider()

# ── Charts ─────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Class distribution")
    if by_class:
        COLORS = {"NO": "#1D9E75", "<30": "#E24B4A", ">30": "#BA7517"}
        fig = go.Figure(go.Pie(
            labels=list(by_class.keys()),
            values=list(by_class.values()),
            marker_colors=[COLORS.get(k, "#888") for k in by_class.keys()],
            hole=0.45,
            textinfo="label+percent",
        ))
        fig.update_layout(
            showlegend=False,
            height=300,
            margin=dict(l=0, r=0, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Confidence distribution")
    if history:
        df = pd.DataFrame(history)
        fig = px.histogram(
            df,
            x="confidence",
            color="predicted_class",
            nbins=20,
            color_discrete_map={"NO": "#1D9E75", "<30": "#E24B4A", ">30": "#BA7517"},
            labels={"confidence": "Confidence", "predicted_class": "Class"},
        )
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Prediction over time ────────────────────────────────────
if history:
    st.subheader("Predictions over time")
    df = pd.DataFrame(history)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df_time = df.groupby([df["created_at"].dt.date, "predicted_class"]).size().reset_index(name="count")
    fig = px.line(
        df_time,
        x="created_at",
        y="count",
        color="predicted_class",
        markers=True,
        color_discrete_map={"NO": "#1D9E75", "<30": "#E24B4A", ">30": "#BA7517"},
        labels={"created_at": "Date", "count": "Predictions", "predicted_class": "Class"},
    )
    fig.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── History table ──────────────────────────────────────────
st.subheader("Recent predictions")
if history:
    df_display = pd.DataFrame(history)[
        ["id", "predicted_class", "confidence", "created_at", "patient_ref"]
    ].copy()
    df_display["confidence"] = (df_display["confidence"] * 100).round(1).astype(str) + "%"
    df_display.columns = ["ID", "Predicted class", "Confidence", "Created at", "Patient ref"]
    st.dataframe(df_display, use_container_width=True, hide_index=True)
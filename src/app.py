import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import seaborn as sns

from functions import (
    reactor_full_visualization_interactive,
    calculate_imbalance,
    plot_imbalance
)

st.set_page_config(page_title="Shunt Reactor Dashboard", layout="wide")
st.title("⚡ Shunt Reactor Monitoring Dashboard")


@st.cache_data(show_spinner=False)
def load_data():
    base_dir = Path(__file__).resolve().parent  
    data_path = base_dir / "data" / "Park1_SR_data.csv"

    df = pd.read_csv(
        data_path,
        sep=";",
        decimal=",",
        encoding="utf-8",
        engine="python"
    )

    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce", format="mixed")

    for col in df.columns:
        if col != "TimeStamp":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("TimeStamp").reset_index(drop=True)

    df = calculate_imbalance(df)

    return df, str(data_path)


df, data_path_used = load_data()

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

st.sidebar.header("Controls")
selected_col = st.sidebar.selectbox(
    "Select variable",
    [c for c in numeric_cols if c not in ["U_line_imbalance_%", "U_phase_imbalance_%", "I_imbalance_%"]],
    index=0
)

show_debug = st.sidebar.checkbox("Show debug info", value=False)

if show_debug:
    with st.expander("🛠 Debug", expanded=True):
        st.write("CSV path used:", data_path_used)
        st.write("Shape:", df.shape)
        st.write("Numeric columns:", numeric_cols)

        base_numeric = [c for c in numeric_cols if not c.endswith("_%")]
        full_zero = (df[base_numeric] == 0).all(axis=1)
        st.write("FULL ZERO rows:", int(full_zero.sum()))

tab1, tab2 = st.tabs(["📈 Variable Monitoring", "⚖️ Imbalance Monitoring"])

with tab1:
    fig = reactor_full_visualization_interactive(df, selected_col)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    imb_fig = plot_imbalance(df)
    st.plotly_chart(imb_fig, use_container_width=True)

    st.markdown("#### Quick stats")
    st.write({
        "Max line-voltage imbalance (%)": float(df["U_line_imbalance_%"].max()),
        "Max phase-voltage imbalance (%)": float(df["U_phase_imbalance_%"].max()),
        "Max current imbalance (%)": float(df["I_imbalance_%"].max()),
    })
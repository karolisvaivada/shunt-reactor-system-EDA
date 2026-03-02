import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import seaborn as sns

# Allow import from src
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))

from functions import reactor_full_visualization_interactive
from functions import (
    calculate_imbalance,
    plot_imbalance
)

st.set_page_config(layout="wide")

st.title("⚡ Shunt Reactor Monitoring Dashboard")

# =====================================================
# LOAD SAME DATA AS NOTEBOOK
# =====================================================

@st.cache_data
def load_data():
    data_path = BASE_DIR / "data" / "Park1_SR_data.csv"

    df = pd.read_csv(data_path, sep=";", decimal=",")
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], format="mixed", errors="coerce")
    df = df.sort_values("TimeStamp").reset_index(drop=True)

    return df

df = load_data()
df = calculate_imbalance(df)

numeric_cols = df.select_dtypes(include=np.number).columns

# =====================================================
# SELECT VARIABLE
# =====================================================

selected_col = st.selectbox("Select Variable", numeric_cols)

# =====================================================
# GENERATE SAME FIGURE AS NOTEBOOK
# =====================================================

fig = reactor_full_visualization_interactive(df, selected_col)

st.plotly_chart(fig, use_container_width=True)

st.markdown("### Phase & Line Imbalance Analysis")

imbalance_fig = plot_imbalance(df)

st.plotly_chart(imbalance_fig, use_container_width=True)
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
import plotly.graph_objects as go


sns.set_theme(context="notebook", style="whitegrid")
sns.set_context("notebook")
plt.style.use("seaborn-v0_8-whitegrid")


def df_check(df: pd.DataFrame, time_col: str = "TimeStamp"):

    df = df.copy()

    print("\n" + "="*70)
    print("DATAFRAME BASIC INFO")
    print("="*70)
    print(f"Shape: {df.shape}")
    print("\nDtypes:")
    print(df.dtypes)

    # ------------------------------------------------------------
    # TIMESTAMP CHECK
    # ------------------------------------------------------------
    print("\n" + "="*70)
    print("TIMESTAMP CHECK")
    print("="*70)

    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

        corrupted = df[time_col].isna().sum()
        duplicate_ts = df[time_col].duplicated().sum()

        print(f"Corrupted timestamps: {corrupted}")
        print(f"Duplicate timestamps: {duplicate_ts}")

        df_sorted = df.sort_values(time_col)
        diff = df_sorted[time_col].diff()

        if not diff.mode().empty:
            expected_freq = diff.mode()[0]
            gaps = (diff > expected_freq).sum()
            print(f"Detected sampling interval: {expected_freq}")
            print(f"Timestamp gaps: {gaps}")
        else:
            print("Could not determine sampling interval.")

    # ------------------------------------------------------------
    # DUPLICATE ROWS
    # ------------------------------------------------------------
    print("\n" + "="*70)
    print("DUPLICATE ROWS")
    print("="*70)
    print(f"Fully duplicated rows: {df.duplicated().sum()}")

    # ------------------------------------------------------------
    # MISSING VALUES
    # ------------------------------------------------------------
    print("\n" + "="*70)
    print("MISSING VALUES")
    print("="*70)

    missing_per_col = df.isna().sum()
    print("Total missing values:", missing_per_col.sum())
    print("\nMissing per column:")
    print(missing_per_col[missing_per_col > 0])

    # ------------------------------------------------------------
    # ZERO VALUES
    # ------------------------------------------------------------
    print("\n" + "="*70)
    print("ZERO VALUES")
    print("="*70)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    zero_per_col = (df[numeric_cols] == 0).sum()

    print("Total zero values:", zero_per_col.sum())
    print("\nZero values per column:")
    print(zero_per_col[zero_per_col > 0])

    # ------------------------------------------------------------
    # NEGATIVE VALUES
    # ------------------------------------------------------------
    print("\n" + "="*70)
    print("NEGATIVE VALUES")
    print("="*70)

    negative_per_col = (df[numeric_cols] < 0).sum()

    print("Total negative values:", negative_per_col.sum())
    print("\nNegative values per column:")
    print(negative_per_col[negative_per_col > 0])

    print("\n" + "="*70)
    print("END OF DF CHECK")
    print("="*70)

def _get_true_blocks(mask: pd.Series):
    """
    Return list of (start_idx, end_idx) for consecutive True runs in mask.
    Works correctly for single-row blocks and edge blocks (start at 0 / end at last row).
    """
    mask = mask.fillna(False).astype(bool).to_numpy()
    n = len(mask)
    if n == 0:
        return []

    # Find boundaries where mask changes value
    changes = np.diff(mask.astype(int))
    starts = list(np.where(changes == 1)[0] + 1)
    ends = list(np.where(changes == -1)[0])

    # If the mask starts True, add start at 0
    if mask[0]:
        starts = [0] + starts

    # If the mask ends True, add end at last index
    if mask[-1]:
        ends = ends + [n - 1]

    return list(zip(starts, ends))


def _print_blocks(df: pd.DataFrame, blocks, time_col: str, label: str):
    print("\n" + "=" * 100)
    print(label)
    print("=" * 100)

    if not blocks:
        print("No blocks detected.")
        return

    total_rows = 0
    total_duration = pd.Timedelta(0)

    for s, e in blocks:
        start_time = df.loc[s, time_col]
        end_time = df.loc[e, time_col]
        row_count = e - s + 1

        total_rows += row_count
        total_duration += (end_time - start_time)

        print(f"Rows {s} → {e} | {start_time} → {end_time} | Count: {row_count}")

    print("\nTOTAL BLOCKS:", len(blocks))
    print("TOTAL ROWS IN BLOCKS:", total_rows)
    print("TOTAL DURATION:", total_duration)


def df_event_report(df: pd.DataFrame, time_col: str = "TimeStamp", top_n: int = 10) -> None:
    """
    Event-focused report:
    - Timestamp gaps (with row indices)
    - Total zero blocks (data loss)
    - Reactor OFF blocks (I phases = 0, voltage present; excluding full-zero rows)
    - Design limit violation blocks (excluding full-zero & scaling)
    - IQR outlier summary for ALL numeric variables + top N examples per variable
    """

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values(time_col).reset_index(drop=True)

    numeric_cols = df.select_dtypes(include="number").columns

    # =====================================================
    # TIMESTAMP GAPS
    # =====================================================
    print("\n" + "=" * 100)
    print("TIMESTAMP GAPS")
    print("=" * 100)

    diff = df[time_col].diff()
    expected_freq = diff.mode()[0] if not diff.mode().empty else None

    if expected_freq is None:
        print("Sampling interval could not be determined.")
    else:
        gap_mask = diff > expected_freq
        gap_indices = df.index[gap_mask]
        print(f"Gaps detected: {len(gap_indices)}")

        for idx in gap_indices:
            prev_time = df.loc[idx - 1, time_col]
            curr_time = df.loc[idx, time_col]
            missing = curr_time - prev_time - expected_freq
            print(f"Rows {idx-1} → {idx} | {prev_time} → {curr_time} | Missing: {missing}")

    # =====================================================
    # TOTAL ZERO (DATA LOSS)
    # =====================================================
    total_zero = (df[numeric_cols] == 0).all(axis=1)
    _print_blocks(df, _get_true_blocks(total_zero), time_col, "TOTAL ZERO BLOCKS (DATA LOSS)")

    # =====================================================
    # REACTOR OFF (I = 0, Voltage present; exclude full-zero)
    # =====================================================
    reactor_off = (
        (df[["SR_I1_A", "SR_I2_A", "SR_I3_A"]] == 0).all(axis=1)
        & (df["SR_U12_kV"] > 0)
        & (~total_zero)
    )
    _print_blocks(df, _get_true_blocks(reactor_off), time_col, "REACTOR OFF BLOCKS")

    # =====================================================
    # DESIGN LIMIT VIOLATIONS (exclude full-zero & scaling)
    # =====================================================
    print("\n" + "=" * 100)
    print("DESIGN LIMIT VIOLATIONS")
    print("=" * 100)

    valid_mask = (~total_zero) & (df["SR_U12_kV"] < 100)

    voltage_cols = ["SR_U12_kV", "SR_U23_kV", "SR_U31_kV"]
    voltage_violation = valid_mask & (
        (df[voltage_cols] < 27).any(axis=1) | (df[voltage_cols] > 33).any(axis=1)
    )
    _print_blocks(
        df,
        _get_true_blocks(voltage_violation),
        time_col,
        "VOLTAGE LIMIT VIOLATION BLOCKS (27–33 kV)"
    )

    current_cols = ["SR_I1_A", "SR_I2_A", "SR_I3_A"]
    current_violation = valid_mask & (
        (df[current_cols] < 112.6).any(axis=1) | (df[current_cols] > 137.6).any(axis=1)
    )
    _print_blocks(
        df,
        _get_true_blocks(current_violation),
        time_col,
        "CURRENT LIMIT VIOLATION BLOCKS (112.6–137.6 A)"
    )

    # =====================================================
    # IQR OUTLIERS (exclude ONLY full-zero rows)
    # =====================================================
    print("\n" + "=" * 100)
    print("IQR OUTLIER DETECTION (Exclude Only FULL ZERO Rows)")
    print("=" * 100)

    # ---- 1) Summary table for ALL numeric columns ----
    summary_rows = []
    outlier_info = {}  # store computed (lower, upper, outliers_df) for drill-down

    for col in numeric_cols:
        series = df.loc[~total_zero, col].dropna()

        if series.shape[0] < 10:
            summary_rows.append([col, np.nan, np.nan, 0, "skip(<10 valid)"])
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0 or pd.isna(iqr):
            summary_rows.append([col, float(q1), float(q3), 0, "skip(IQR=0)"])
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (~total_zero) & ((df[col] < lower) | (df[col] > upper))
        outliers = df.loc[mask, [time_col, col]].copy()

        summary_rows.append([col, float(lower), float(upper), int(mask.sum()), "ok"])
        outlier_info[col] = (lower, upper, outliers)

    summary = pd.DataFrame(
        summary_rows,
        columns=["variable", "iqr_lower", "iqr_upper", "outliers_count", "status"]
    ).sort_values("outliers_count", ascending=False)

    print("\nIQR OUTLIER SUMMARY (all numeric variables):")
    print(summary.to_string(index=False))

    # ---- 2) Detailed examples ONLY for variables that have outliers ----
    cols_with_outliers = summary.loc[summary["outliers_count"] > 0, "variable"].tolist()

    print("\n" + "-" * 100)
    print(f"IQR OUTLIER EXAMPLES (top {top_n} rows per variable; only variables with outliers)")
    print("-" * 100)

    if not cols_with_outliers:
        print("No IQR outliers detected in any numeric column (after excluding full-zero rows).")
    else:
        for col in cols_with_outliers:
            lower, upper, outliers = outlier_info[col]

            print(f"\nVariable: {col}")
            print(f"Lower bound: {lower:.3f} | Upper bound: {upper:.3f}")
            print(f"Outliers detected: {len(outliers)}")

            # show earliest outliers; change to .sort_values(col) if you want extremes
            outliers_sorted = outliers.sort_values(time_col)
            print(outliers_sorted.head(top_n).to_string(index=True))

    print("\n" + "=" * 100)
    print(f"END OF REPORT | Processed {len(numeric_cols)} numeric columns: {list(numeric_cols)}")
    print("=" * 100)


def reactor_full_visualization_interactive(
    df: pd.DataFrame,
    selected_col: str,
    time_col: str = "TimeStamp"
):

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values(time_col).reset_index(drop=True)

    numeric_cols = df.select_dtypes(include=np.number).columns

    # =====================================================
    # TIMESTAMP GAPS
    # =====================================================

    diff = df[time_col].diff()
    expected_freq = diff.mode()[0] if not diff.mode().empty else None

    gap_mask = pd.Series(False, index=df.index)
    gap_times = []

    if expected_freq is not None:
        gap_mask = diff > expected_freq
        gap_times = df.loc[gap_mask, time_col]

    df_plot = df.copy()
    df_plot.loc[gap_mask, selected_col] = np.nan

    # =====================================================
    # FULL ZERO (DATA LOSS)
    # =====================================================

    total_zero = (df[numeric_cols] == 0).all(axis=1)

    # =====================================================
    # REACTOR OFF
    # =====================================================

    reactor_off = (
        (df[["SR_I1_A", "SR_I2_A", "SR_I3_A"]] == 0).all(axis=1)
        & (df["SR_U12_kV"] > 0)
        & (~total_zero)
    )

    # =====================================================
    # IQR OUTLIERS
    # =====================================================

    valid = df.loc[~total_zero, selected_col]

    Q1 = valid.quantile(0.25)
    Q3 = valid.quantile(0.75)
    IQR = Q3 - Q1

    lower_iqr = Q1 - 1.5 * IQR
    upper_iqr = Q3 + 1.5 * IQR

    iqr_mask = (
        ~total_zero
        & ((df[selected_col] < lower_iqr) | (df[selected_col] > upper_iqr))
    )

    # =====================================================
    # DESIGN LIMIT VIOLATIONS
    # =====================================================

    limit_mask = pd.Series(False, index=df.index)

    if "U" in selected_col and selected_col not in ["SR_U1_kV", "SR_U2_kV", "SR_U3_kV"]:
        limit_mask = (df[selected_col] < 27) | (df[selected_col] > 33)

    if "I" in selected_col:
        limit_mask = (df[selected_col] < 112.6) | (df[selected_col] > 137.6)

    # =====================================================
    # CREATE FIGURE
    # =====================================================

    fig = go.Figure()

    # MAIN SIGNAL
    fig.add_trace(go.Scatter(
        x=df_plot[time_col],
        y=df_plot[selected_col],
        mode="lines",
        line=dict(color="black", width=1.8),
        name="Signal"
    ))

    # IQR OUTLIERS (BRIGHT RED)
    fig.add_trace(go.Scatter(
        x=df.loc[iqr_mask, time_col],
        y=df.loc[iqr_mask, selected_col],
        mode="markers",
        marker=dict(color="red", size=5, line=dict(color="black", width=0.5)),
        name="IQR Outlier"
    ))

    # LIMIT VIOLATIONS (BRIGHT BLUE)
    fig.add_trace(go.Scatter(
        x=df.loc[limit_mask, time_col],
        y=df.loc[limit_mask, selected_col],
        mode="markers",
        marker=dict(color="blue", size=5, line=dict(color="black", width=0.5)),
        name="Design Limit Exceeded"
    ))

    # =====================================================
    # LIMIT LINES
    # =====================================================

    if "U" in selected_col and selected_col not in ["SR_U1_kV", "SR_U2_kV", "SR_U3_kV"]:
        fig.add_hline(y=27, line_dash="dash", line_color="green")
        fig.add_hline(y=33, line_dash="dash", line_color="green")

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color="green", dash="dash"),
            name="Voltage Limits"
        ))

    if "I" in selected_col:
        fig.add_hline(y=112.6, line_dash="dash", line_color="orange")
        fig.add_hline(y=137.6, line_dash="dash", line_color="orange")

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color="orange", dash="dash"),
            name="Current Limits"
        ))

    # =====================================================
    # REACTOR OFF (BRIGHT GREEN)
    # =====================================================

    change = reactor_off.astype(int).diff()
    starts = df.loc[change == 1, time_col]
    ends = df.loc[change == -1, time_col]

    for start, end in zip(starts, ends):
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="rgba(0,255,0,0.6)",
            line_color="green",
            line_width=1
        )

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(color="lime", size=12),
        name="Reactor OFF"
    ))

    # =====================================================
    # FULL DATA LOSS (BRIGHT MAGENTA)
    # =====================================================

    change = total_zero.astype(int).diff()
    starts = df.loc[change == 1, time_col]
    ends = df.loc[change == -1, time_col]

    for start, end in zip(starts, ends):
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="rgba(255,0,150,0.5)",
            line_color="magenta",
            line_width=1
        )

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(color="magenta", size=12),
        name="Full Data Loss"
    ))

    # =====================================================
    # TIMESTAMP GAPS (THICK BLACK DOTTED)
    # =====================================================

    for gap_time in gap_times:
        fig.add_vline(
            x=gap_time,
            line_dash="dot",
            line_color="black",
            line_width=2
        )

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="lines",
        line=dict(color="black", dash="dot", width=2),
        name="Timestamp Gap"
    ))

    # =====================================================
    # LAYOUT
    # =====================================================

    fig.update_layout(
        title=f"{selected_col} – Interactive Reactor Monitoring",
        height=750,
        template="plotly_white",
        legend=dict(
            orientation="h",
            y=1.05,
            x=0
        )
    )

    return fig

def calculate_imbalance(df: pd.DataFrame):

    df = df.copy()

    # -------- LINE VOLTAGE IMBALANCE ----------
    line_cols = ["SR_U12_kV", "SR_U23_kV", "SR_U31_kV"]

    df["U_line_imbalance_%"] = (
        (df[line_cols].max(axis=1) - df[line_cols].min(axis=1))
        / df[line_cols].mean(axis=1)
    ) * 100

    # -------- PHASE VOLTAGE IMBALANCE ----------
    phase_cols = ["SR_U1_kV", "SR_U2_kV", "SR_U3_kV"]

    df["U_phase_imbalance_%"] = (
        (df[phase_cols].max(axis=1) - df[phase_cols].min(axis=1))
        / df[phase_cols].mean(axis=1)
    ) * 100

    # -------- CURRENT IMBALANCE ----------
    current_cols = ["SR_I1_A", "SR_I2_A", "SR_I3_A"]

    df["I_imbalance_%"] = (
        (df[current_cols].max(axis=1) - df[current_cols].min(axis=1))
        / df[current_cols].mean(axis=1)
    ) * 100

    return df

def plot_imbalance(df: pd.DataFrame, time_col="TimeStamp"):

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    fig = go.Figure()

    # -------- LINE VOLTAGE ----------
    fig.add_trace(go.Scatter(
        x=df[time_col],
        y=df["U_line_imbalance_%"],
        mode="lines",
        line=dict(color="blue", width=2),
        name="Line Voltage Imbalance %"
    ))

    # -------- PHASE VOLTAGE ----------
    fig.add_trace(go.Scatter(
        x=df[time_col],
        y=df["U_phase_imbalance_%"],
        mode="lines",
        line=dict(color="purple", width=2),
        name="Phase Voltage Imbalance %"
    ))

    # -------- CURRENT ----------
    fig.add_trace(go.Scatter(
        x=df[time_col],
        y=df["I_imbalance_%"],
        mode="lines",
        line=dict(color="red", width=2),
        name="Current Imbalance %"
    ))

    # -------- Threshold Lines ----------
    fig.add_hline(y=2, line_dash="dash", line_color="green")
    fig.add_hline(y=3, line_dash="dash", line_color="orange")
    fig.add_hline(y=5, line_dash="dash", line_color="red")

    fig.update_layout(
        title="Voltage & Current Imbalance Monitoring",
        height=750,
        template="plotly_white",
        legend=dict(orientation="h", y=1.05, x=0),
        yaxis_title="Imbalance (%)"
    )

    return fig
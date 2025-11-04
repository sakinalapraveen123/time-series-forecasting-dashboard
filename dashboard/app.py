"""Time Series Forecasting Dashboard (Streamlit)
Self-contained. Uses dynamic_import for optional dependencies.
"""

import os
import sys
import subprocess
import importlib
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np

# ---------- utilities ----------
def install_package(package_name: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

def dynamic_import(module_name: str, pip_name: Optional[str] = None):
    try:
        return importlib.import_module(module_name)
    except Exception:
        pkg = pip_name or module_name.split(".")[0]
        install_package(pkg)
        return importlib.import_module(module_name)

def try_import_gdown() -> bool:
    try:
        importlib.import_module("gdown")
        return True
    except Exception:
        return False

def download_from_gdrive(file_id: str, out_path: str) -> bool:
    try:
        gdown = importlib.import_module("gdown")
    except Exception:
        install_package("gdown")
        gdown = importlib.import_module("gdown")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, out_path, quiet=True)
    return os.path.exists(out_path)

def find_datetime_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if c.lower() in ("date", "ds", "timestamp", "time")]
    if candidates:
        return candidates[0]
    for c in df.columns:
        try:
            _ = pd.to_datetime(df[c], errors="coerce")
            if _.notna().sum() > 0.6 * len(df):
                return c
        except Exception:
            continue
    return None

def find_value_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if c.lower() in ("y", "value", "close", "target", "sales", "consumption")]
    if candidates:
        return candidates[0]
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 5:
            return c
    return None

def load_any_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_excel(path)
        except Exception as e:
            st.error(f"Could not read file {path}: {e}")
            return None

# ---------- config ----------
DATA_DIR = os.path.join(os.getcwd(), "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

# --- Always visible drag-and-drop uploader ---
uploaded = st.file_uploader(
    "🗂️ Upload a timeseries file (CSV/Excel/Parquet)",
    type=["csv", "xlsx", "xls", "parquet"],
    key="file_uploader"
)
if uploaded:
    save_path = os.path.join(DATA_DIR, uploaded.name)
    with open(save_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success(f"Saved to {save_path}. Refresh the app to see it listed in the selector.")
    st.stop()

DRIVE_FILES = {
    "file_01.csv": "1aVcpVsMID2ZQ3mdyHBUz-rNKYOCeYiiO",
}

# ---------- sidebar ----------
st.sidebar.title("⚙️ Setup & Settings")

st.sidebar.header("1. Install Dependencies")
if st.sidebar.button("📦 Install Required Packages"):
    with st.spinner("Installing dependencies..."):
        packages = [
            ("plotly", "Plotting"),
            ("scikit-learn", "Metrics"),
            ("statsmodels", "ARIMA & ACF"),
            ("gdown", "Drive downloads"),
            ("openpyxl", "Excel support"),
            ("pyarrow", "Parquet support")
        ]
        success = []
        failed = []
        for pkg, purpose in packages:
            try:
                with st.spinner(f"Installing {pkg} ({purpose})..."):
                    install_package(pkg)
                success.append(pkg)
                st.sidebar.success(f"✅ {pkg}")
            except Exception as e:
                failed.append(pkg)
                st.sidebar.error(f"❌ {pkg}: {e}")
    if success and not failed:
        st.sidebar.success("✨ All packages installed! Restart the app to use them.")
    elif success:
        st.sidebar.warning(f"✓ Installed: {', '.join(success)}\n✗ Failed: {', '.join(failed)}")
    else:
        st.sidebar.error("Failed to install packages. Try manual installation.")

st.sidebar.header("2. Data Files")
if st.sidebar.button("Attempt to download shared Drive files (if missing)", key="download_drive"):
    st.sidebar.info("Will try to download missing files (requires internet).")
    if not try_import_gdown():
        with st.spinner("Installing gdown..."):
            install_package("gdown")
    downloaded = 0
    for fname, fid in DRIVE_FILES.items():
        out_path = os.path.join(DATA_DIR, fname)
        if os.path.exists(out_path):
            st.sidebar.write(f"✅ {fname} already present")
            downloaded += 1
            continue
        with st.spinner(f"Downloading {fname} ..."):
            ok = download_from_gdrive(fid, out_path)
        if ok:
            st.sidebar.write(f"✅ Downloaded {fname}")
            downloaded += 1
        else:
            st.sidebar.write(f"❌ Failed to download {fname} — place it in data/raw/ manually.")
    st.sidebar.success(f"Download attempts finished. {downloaded}/{len(DRIVE_FILES)} present.")
st.sidebar.markdown("---")
st.sidebar.write("Or place your files manually in `data/raw/` and refresh.")

# ---------- file selection ----------
st.title("📈 Time Series Forecasting Dashboard (Streamlit)")
st.write("Load your dataset (from data/raw/) or click the sidebar button to attempt automatic download from the shared Drive links.")
available_files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.csv', '.xlsx', '.xls', '.pkl', '.parquet'))])
if available_files:
    st.success(f"Found {len(available_files)} file(s) in data/raw/")
else:
    st.warning("No files found in data/raw/. Please upload a file.")

st.sidebar.header("3. Choose Dataset")
chosen = st.sidebar.selectbox("Select data file", ["-- select --"] + available_files, key="dataset_selector")
if chosen == "-- select --":
    st.info("Select a dataset from the sidebar to begin.")
    st.stop()
file_path = os.path.join(DATA_DIR, chosen)
st.sidebar.write(f"Selected: `{chosen}`")

# ---------- load file ----------
if chosen.lower().endswith((".pkl", ".parquet")):
    try:
        if chosen.lower().endswith(".pkl"):
            df = pd.read_pickle(file_path)
        else:
            df = pd.read_parquet(file_path)
    except Exception as e:
        st.error(f"Failed to read {chosen}: {e}")
        st.stop()
else:
    df = load_any_csv(file_path)
    if df is None:
        st.stop()
st.write(f"### Dataset: `{chosen}`")
st.write(f"Rows: **{len(df)}** — Columns: **{len(df.columns)}**")
with st.expander("Show raw data"):
    st.dataframe(df.head(200))

# ---------- detect date & value columns ----------
date_col = find_datetime_col(df)
value_col = find_value_col(df)
st.sidebar.write("Detected columns (auto)")
st.sidebar.write(f"Date column: **{date_col}**")
st.sidebar.write(f"Value column: **{value_col}**")
st.sidebar.markdown("---")
user_date = st.sidebar.selectbox("Override Date column (if wrong)", [None] + list(df.columns), key="override_date")
user_value = st.sidebar.selectbox("Override Value column (if wrong)", [None] + list(df.columns), key="override_value")
if user_date:
    date_col = user_date
if user_value:
    value_col = user_value
if date_col is None or value_col is None:
    st.error("Could not detect a date or numeric column automatically. Choose them from the sidebar.")
    st.stop()

# ---------- preprocess ----------
try:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
except Exception:
    st.error("Failed to parse the chosen date column to datetime. Check your dataset.")
    st.stop()
df = df.sort_values(by=date_col).reset_index(drop=True)
df = df[[date_col] + [c for c in df.columns if c != date_col]]
if not pd.api.types.is_numeric_dtype(df[value_col]):
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
df = df.dropna(subset=[date_col]).reset_index(drop=True)

# ---------- main plots ----------
st.header("Time Series Overview")
col1, col2 = st.columns([3, 1])
with col1:
    try:
        px = dynamic_import("plotly.express", "plotly")
    except Exception:
        px = None
    if px is not None:
        fig = px.line(df, x=date_col, y=value_col, title=f"{value_col} over time")
        fig.update_traces(line_color="#0047AB", line_width=2)
        fig.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor="white",
            font=dict(size=14),
            xaxis=dict(showgrid=True, gridcolor="lightgray"),
            yaxis=dict(showgrid=True, gridcolor="lightgray"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Plotly not available — cannot render time series chart.")
with col2:
    st.write("Summary Stats")
    st.metric("Start", str(df[date_col].min().date()))
    st.metric("End", str(df[date_col].max().date()))
    st.metric("Observations", len(df))
    st.write(df[value_col].describe())

# ---------- resampling & aggregation ----------
st.sidebar.header("4. Analysis Options")
freq = st.sidebar.selectbox("Resample frequency (for smoothing/aggregates)", ["D", "W", "M", "Q", "Y"], index=2, key="resample_freq")
agg_method = st.sidebar.selectbox("Aggregation", ["sum", "mean"], index=1, key="agg_method")
if st.sidebar.button("Resample & plot", key="resample_btn"):
    df_resampled = df.set_index(date_col).resample(freq).agg({value_col: agg_method}).reset_index()
    st.subheader(f"Resampled ({freq}) - {agg_method}")
    try:
        px = dynamic_import("plotly.express", "plotly")
    except Exception:
        px = None
    if px is not None:
        fig2 = px.line(df_resampled, x=date_col, y=value_col, title=f"Resampled {value_col} ({freq})")
        fig2.update_traces(line_color="#532A8F", line_width=3)
        fig2.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor="white",
            font=dict(size=14),
            xaxis=dict(showgrid=True, gridcolor="lightgray"),
            yaxis=dict(showgrid=True, gridcolor="lightgray"))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.write("Plotly not available — cannot render resampled chart.")

# ---------- forecast detection & ARIMA ----------
st.header("Forecast / Model results (auto-detect)")
forecast_cols = [c for c in df.columns if any(k in c.lower() for k in ("yhat", "forecast", "pred", "predicted", "y_pred", "fcst"))]
has_forecast_cols = len(forecast_cols) > 0

if has_forecast_cols:
    st.success(f"Detected forecast columns: {forecast_cols}")
    pred_col = st.selectbox("Choose prediction/forecast column", forecast_cols, key="forecast_col_select")
    lower_candidates = [c for c in df.columns if "lower" in c.lower() or "lwr" in c.lower()]
    upper_candidates = [c for c in df.columns if "upper" in c.lower() or "upr" in c.lower()]
    lower_col = st.selectbox("Lower CI (optional)", [None] + lower_candidates, key="lower_ci_select")
    upper_col = st.selectbox("Upper CI (optional)", [None] + upper_candidates, key="upper_ci_select")
    try:
        go = dynamic_import("plotly.graph_objects", "plotly")
    except Exception:
        go = None
    if go is not None:
        figf = go.Figure()
        figf.add_trace(go.Scatter(
            x=df[date_col], y=df[value_col],
            mode="lines", name="Actual", line=dict(color="#0047AB", width=3)))
        figf.add_trace(go.Scatter(
            x=df[date_col], y=df[pred_col],
            mode="lines", name="Predicted", line=dict(color="#F44336", width=3, dash="dash")))
        if lower_col and upper_col:
            figf.add_trace(go.Scatter(
                x=pd.concat([df[date_col], df[date_col][::-1]]),
                y=pd.concat([df[upper_col], df[lower_col][::-1]]),
                fill="toself",
                fillcolor="rgba(0,100,80,0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="Confidence Interval"
            ))
        figf.update_layout(title="Actual vs Predicted", xaxis_title="Date", yaxis_title=value_col,
                          plot_bgcolor="white", font=dict(size=14),
                          margin=dict(l=10, r=10, t=40, b=10),
                          legend=dict(font=dict(size=13)))
        st.plotly_chart(figf, use_container_width=True)
    else:
        st.write("Plotly not available — cannot render forecast chart.")
    if pd.api.types.is_numeric_dtype(df[pred_col]):
        st.subheader("Forecast Metrics (on data file)")
        mask = df[value_col].notna() & df[pred_col].notna()
        if mask.sum() > 0:
            metrics_mod = dynamic_import("sklearn.metrics", "scikit-learn")
            mae = metrics_mod.mean_absolute_error(df.loc[mask, value_col], df.loc[mask, pred_col])
            rmse = metrics_mod.mean_squared_error(df.loc[mask, value_col], df.loc[mask, pred_col], squared=False)
            mape = (np.mean(np.abs((df.loc[mask, value_col] - df.loc[mask, pred_col]) / df.loc[mask, value_col].replace(0, np.nan))) * 100)
            st.write(pd.DataFrame({
                "metric": ["MAE", "RMSE", "MAPE(%)"],
                "value": [mae, rmse, round(mape, 2)]
            }).set_index("metric"))
        else:
            st.info("No overlapping actual & forecast values to compute metrics.")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download shown data (CSV)", csv, file_name=f"{chosen.replace('.','_')}_with_pred.csv", mime="text/csv")
else:
    st.subheader("Automatic ARIMA Forecasting (Line Plot)")
    forecast_horizon = st.slider("Forecast horizon (future periods to predict)", min_value=1, max_value=60, value=14, key="arima_horizon")
    if st.button("Run ARIMA Forecast", key="arima_btn"):
        try:
            statsmodels_arima = dynamic_import("statsmodels.tsa.arima.model", "statsmodels")
            import plotly.graph_objects as go
        except Exception:
            st.error("statsmodels or plotly required for ARIMA. Please install them first.")
            statsmodels_arima = None
        if statsmodels_arima is not None:
            try:
                series = df[value_col].dropna()
                arima_model = statsmodels_arima.ARIMA(series, order=(1,1,1))
                arima_fit = arima_model.fit()
                forecast = arima_fit.forecast(steps=forecast_horizon)
                last_date = df[date_col].max()
                freq_guess = pd.infer_freq(df[date_col])
                if freq_guess is None:
                    freq_guess = 'D'
                future_dates = pd.date_range(start=last_date, periods=forecast_horizon+1, freq=freq_guess)[1:]
                pred_df = pd.DataFrame({date_col: future_dates, 'arima_predicted': forecast})
                plot_df = pd.concat([df[[date_col, value_col]], pred_df], ignore_index=True, sort=False)
                actual_start = plot_df.loc[plot_df[value_col].notnull(), date_col].min()
                actual_end = plot_df.loc[plot_df[value_col].notnull(), date_col].max()
                pred_start = plot_df.loc[plot_df['arima_predicted'].notnull(), date_col].min()
                pred_end = plot_df.loc[plot_df['arima_predicted'].notnull(), date_col].max()
                st.write(f"**Actual data:** {actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}")
                st.write(f"**Predicted (ARIMA):** {pred_start.strftime('%Y-%m-%d')} to {pred_end.strftime('%Y-%m-%d')}")
                fig_arima = go.Figure()
                fig_arima.add_trace(go.Scatter(
                    x=plot_df.loc[plot_df[value_col].notnull(), date_col],
                    y=plot_df.loc[plot_df[value_col].notnull(), value_col],
                    mode="lines",
                    name="Actual",
                    line=dict(color="#0047AB", width=3)
                ))
                fig_arima.add_trace(go.Scatter(
                    x=plot_df.loc[plot_df['arima_predicted'].notnull(), date_col],
                    y=plot_df.loc[plot_df['arima_predicted'].notnull(), 'arima_predicted'],
                    mode="lines",
                    name="Predicted (ARIMA)",
                    line=dict(color="#F44336", width=3, dash="dash")
                ))
                fig_arima.add_vrect(
                    x0=pred_start, x1=pred_end,
                    fillcolor="rgba(244,67,54,0.08)", opacity=0.2,
                    layer="below", line_width=0
                )
                fig_arima.update_layout(
                    title="Actual vs ARIMA Forecast (Line Plot)",
                    xaxis_title="Date", yaxis_title=value_col,
                    plot_bgcolor="white",
                    margin=dict(l=10, r=10, t=40, b=10),
                    font=dict(size=14),
                    legend=dict(font=dict(size=13)),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig_arima, use_container_width=True)
                st.write(pred_df.rename(columns={'arima_predicted':'Predicted (ARIMA)'}))
                pred_csv = pred_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download forecasted values (CSV)", pred_csv, file_name=f"{chosen.replace('.','_')}_arima_forecast.csv", mime="text/csv")
            except Exception as e:
                st.error(f"ARIMA forecasting failed: {e}")

# ---------- Model Comparison / Metrics (improved) ----------
metrics_files = [f for f in os.listdir(DATA_DIR) if "metric" in f.lower() or "score" in f.lower()]
if metrics_files:
    st.header("Optional: Model Comparison / Metrics")
    mf = st.selectbox("Choose metrics file", metrics_files, key="metrics_file_select")
    if mf:
        mdf = load_any_csv(os.path.join(DATA_DIR, mf))
        if mdf is not None:
            st.write("Loaded metrics:")
            st.dataframe(mdf)

# ---------- Improved Quick EDA ----------
import matplotlib.pyplot as plt

def quick_eda(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in df.columns if df[col].dtype == "object" and df[col].nunique() < 30]

    st.subheader("Quick EDA")
    eda_success = False

    # Missing values
    mv = df.isna().sum()
    if mv.sum() > 0:
        st.write("**Missing values per column:**")
        st.dataframe(mv[mv > 0])
        eda_success = True

    # Column types & unique counts
    st.write("**Column types & unique counts:**")
    col_summary = pd.DataFrame({
        "dtype": [df[c].dtype for c in df.columns],
        "unique": [df[c].nunique() for c in df.columns]
    }, index=df.columns)
    st.dataframe(col_summary)
    eda_success = True

    # Summary stats
    if numeric_cols:
        st.write("**Summary statistics for numeric columns:**")
        st.dataframe(df[numeric_cols].describe().T)
        eda_success = True

    # Histograms
    if len(numeric_cols) > 0:
        st.write("**Histograms:**")
        for col in numeric_cols[:2]:
            fig, ax = plt.subplots()
            ax.hist(df[col].dropna(), bins=30, color='cornflowerblue', alpha=0.7)
            ax.set_title(f"{col} distribution")
            st.pyplot(fig)
        eda_success = True

    # Top unique values for categoricals
    if categorical_cols:
        st.write("**Top unique values for categorical columns:**")
        for col in categorical_cols[:2]:
            vals = df[col].value_counts().head(10)
            st.write(f"`{col}`: {dict(vals)}")
        eda_success = True

    # Autocorrelation: first numeric column
    if len(numeric_cols) > 0:
        ac_col = numeric_cols[0]
        try:
            sm = dynamic_import("statsmodels.api", "statsmodels")
            acf_vals = sm.tsa.stattools.acf(df[ac_col].dropna(), nlags=10)
            acf_df = pd.DataFrame({"lag": range(len(acf_vals)), "acf": acf_vals})
            st.write(f"**Autocorrelation for `{ac_col}` (first 10 lags):**")
            st.bar_chart(acf_df.set_index("lag"))
            eda_success = True
        except Exception:
            pass

    return eda_success

if not quick_eda(df):
    st.info("Quick EDA not available for this dataset (no numeric/categorical variability or all columns missing).")

# ---------- footer / instructions ----------
st.markdown("---")
st.markdown("**How to use**: \n\n"
    "1. Upload any number of `.csv`, `.xlsx`, or `.parquet` files with the uploader above, or put files in `data/raw/`.\n"
    "2. Select the file from the sidebar and choose the right date & value columns (sidebar).\n"
    "3. Use 'Resample & plot' to explore aggregated trends.\n"
    "4. If your file contains forecast columns (`yhat`, `predicted`, `forecast`, etc.), the dashboard will plot them automatically. If not, use the ARIMA forecast feature above."
)

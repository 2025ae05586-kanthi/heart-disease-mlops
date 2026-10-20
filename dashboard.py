
import json
from pathlib import Path

import pandas as pd
import streamlit as st


# Monitoring log shared with the local Docker API.
LOG_FILE = Path(__file__).resolve().parent / "monitoring" / "requests.jsonl"

st.set_page_config(
    page_title="Heart Disease API Monitoring",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Heart Disease API Monitoring Dashboard")
st.caption("Local API monitoring · FastAPI · Docker · Streamlit")

# Sidebar controls
st.sidebar.header("Dashboard Controls")
st.sidebar.caption("Data source: monitoring/requests.jsonl")

if st.sidebar.button("🔄 Refresh dashboard", use_container_width=True):
    st.rerun()

if st.sidebar.button("🗑️ Clear monitoring logs", use_container_width=True):
    st.session_state["confirm_clear"] = True

if st.session_state.get("confirm_clear", False):
    st.sidebar.warning("This permanently clears the local monitoring log.")
    col_yes, col_no = st.sidebar.columns(2)

    if col_yes.button("Confirm"):
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOG_FILE.write_text("", encoding="utf-8")
        st.session_state["confirm_clear"] = False
        st.rerun()

    if col_no.button("Cancel"):
        st.session_state["confirm_clear"] = False
        st.rerun()


def load_request_logs():
    """Read valid JSON records from the shared log file."""
    if not LOG_FILE.exists():
        return pd.DataFrame()

    records = []

    with LOG_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
                records.append(record)
            except (json.JSONDecodeError, TypeError):
                continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    required_columns = [
        "timestamp",
        "method",
        "path",
        "status_code",
        "duration_ms",
        "success",
    ]

    for column in required_columns:
        if column not in df.columns:
            df[column] = None

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce", utc=True
    )
    df["status_code"] = pd.to_numeric(
        df["status_code"], errors="coerce"
    )
    df["duration_ms"] = pd.to_numeric(
        df["duration_ms"], errors="coerce"
    )
    df["success"] = df["success"].fillna(False).astype(bool)

    return df.dropna(subset=["timestamp", "status_code"])


df = load_request_logs()

if df.empty:
    st.info(
        "No monitoring records found yet. Send a request to "
        "http://localhost:8000/health and refresh this dashboard."
    )
    st.stop()

# Summary metrics
total_requests = len(df)
successful_requests = int(df["success"].sum())
failed_requests = total_requests - successful_requests
success_rate = successful_requests / total_requests * 100
avg_latency = df["duration_ms"].mean()
max_latency = df["duration_ms"].max()

st.subheader("Key API Metrics")

metric1, metric2, metric3, metric4, metric5 = st.columns(5)

metric1.metric("Total Requests", f"{total_requests:,}")
metric2.metric("Successful", f"{successful_requests:,}")
metric3.metric("Failed", f"{failed_requests:,}")
metric4.metric("Success Rate", f"{success_rate:.1f}%")
metric5.metric(
    "Average Latency",
    f"{avg_latency:.2f} ms" if pd.notna(avg_latency) else "N/A",
)

st.caption(f"Maximum recorded latency: {max_latency:.2f} ms")

# Request filtering
st.sidebar.subheader("Filters")

endpoints = sorted(df["path"].dropna().unique().tolist())
selected_endpoints = st.sidebar.multiselect(
    "Endpoints",
    options=endpoints,
    default=endpoints,
)

filtered_df = df[df["path"].isin(selected_endpoints)].copy()

if filtered_df.empty:
    st.warning("No requests match the selected filters.")
    st.stop()

# Charts
left, right = st.columns(2)

with left:
    st.subheader("HTTP Status Code Distribution")
    status_counts = (
        filtered_df["status_code"]
        .astype(int)
        .value_counts()
        .sort_index()
    )
    status_counts.index = status_counts.index.astype(str)
    st.bar_chart(status_counts)

with right:
    st.subheader("Requests by Endpoint")
    endpoint_counts = filtered_df["path"].value_counts()
    st.bar_chart(endpoint_counts)

st.subheader("API Response Time Over Time")

latency_df = (
    filtered_df.dropna(subset=["duration_ms"])
    .sort_values("timestamp")
    .set_index("timestamp")[["duration_ms"]]
)
latency_df.index = latency_df.index.tz_convert(None)

if latency_df.empty:
    st.info("No response-time data is available.")
else:
    st.line_chart(latency_df, y="duration_ms")

st.subheader("Recent Request Logs")

display_df = filtered_df.sort_values(
    "timestamp", ascending=False
).head(100).copy()

display_df["timestamp"] = display_df["timestamp"].dt.strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)
display_df["success"] = display_df["success"].map(
    {True: "Success", False: "Failure"}
)

display_df = display_df[
    [
        "timestamp",
        "method",
        "path",
        "status_code",
        "duration_ms",
        "success",
    ]
]

display_df.columns = [
    "Timestamp (UTC)",
    "Method",
    "Endpoint",
    "HTTP Status",
    "Latency (ms)",
    "Result",
]

st.dataframe(display_df, use_container_width=True, hide_index=True)

# Export the filtered monitoring records.
csv_data = display_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download logs as CSV",
    data=csv_data,
    file_name="heart_disease_api_logs.csv",
    mime="text/csv",
)

st.caption(
    "Educational ML monitoring demonstration. "
    "Metrics reflect requests recorded by this local API instance."
)

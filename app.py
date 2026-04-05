from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import streamlit as st

from src.workflow_runner import get_latest_result_files, run_workflow

OUTPUT_DIR = Path("data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"


def is_valid_email(value: str) -> bool:
    return bool(re.match(EMAIL_PATTERN, value.strip()))


st.set_page_config(page_title="Agentic AI Trading Workflow", layout="wide")

st.title("Agentic AI Trading Workflow")
st.caption("Run the full AWS-backed pipeline and send results to an email address")

if "run_result" not in st.session_state:
    st.session_state.run_result = None

if "run_logs" not in st.session_state:
    st.session_state.run_logs = ""

with st.form("full_pipeline_form"):
    recipient_email = st.text_input(
        "Recipient email address",
        placeholder="examiner@example.com",
    )
    submitted = st.form_submit_button("Run Full Pipeline")

if submitted:
    email_value = recipient_email.strip()

    if not email_value:
        st.error("Please enter a recipient email address.")
    elif not is_valid_email(email_value):
        st.error("Please enter a valid email address.")
    else:
        try:
            with st.spinner("Running full workflow... this may take several minutes."):
                result, logs = run_workflow(email_value)

            st.session_state.run_result = result
            st.session_state.run_logs = logs
            st.success("Full workflow completed successfully.")
        except Exception as exc:
            st.session_state.run_logs = ""
            st.error(f"Workflow failed: {exc}")

if st.session_state.run_result:
    result = st.session_state.run_result

    st.subheader("Execution Result")
    col1, col2 = st.columns(2)
    col1.metric("Status", result.get("status", "unknown"))
    col2.metric("Assets Selected", result.get("assets_selected", 0))

    st.write(f"**Date range:** {result.get('start_date')} to {result.get('end_date')}")
    st.write(f"**Recipient:** {result.get('email_recipient')}")
    st.write(f"**S3 URI:** `{result.get('s3_uri')}`")
    st.write(f"**Download URL:** {result.get('download_url')}")

    latest_files = get_latest_result_files()
    if latest_files:
        st.subheader("Generated Files")
        for file_path in latest_files:
            st.write(Path(file_path).name)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in latest_files:
                path = Path(file_path)
                if path.exists():
                    zf.write(path, arcname=path.name)

        zip_buffer.seek(0)

        st.download_button(
            label="Download All Results (ZIP)",
            data=zip_buffer,
            file_name="final_analysis_bundle.zip",
            mime="application/zip",
        )

if st.session_state.run_logs:
    st.subheader("Execution Logs")
    st.code(st.session_state.run_logs, language="text")

st.divider()
st.subheader("Notes")
st.markdown(
    """
- This app runs the full pipeline on the EC2 server.
- Results are emailed to the address entered above.
- SSH access remains restricted to the project owner.
"""
)
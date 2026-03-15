import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import io

sys.path.insert(0, str(Path(__file__).parent.parent))
from api_client import predict_batch

st.set_page_config(page_title="Batch Upload", page_icon="📂", layout="wide")

if not st.session_state.get("token"):
    st.warning("Please log in first.")
    st.stop()

st.title("Batch CSV Prediction")
st.caption("Upload a CSV file with patient features. Each row = one patient.")

# ── Instructions ───────────────────────────────────────────
with st.expander("CSV format requirements"):
    st.markdown("""
    Your CSV should contain the same columns as the training dataset (tab or comma separated):
    - Continuous: `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`,
      `number_outpatient`, `number_emergency`, `number_inpatient`, `number_diagnoses`
    - One-hot encoded: `race_*`, `gender_*`, `age_*`, `insulin_*`, `metformin_*`, etc.
    - Do **not** include the `readmitted` column — that's what we predict.

    Missing columns will be filled with 0 automatically.
    """)

uploaded = st.file_uploader("Upload patient CSV", type=["csv", "tsv"])

if uploaded:
    sep = "\t" if uploaded.name.endswith(".tsv") else ","
    try:
        df = pd.read_csv(uploaded, sep=sep)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    # Drop target column if accidentally included
    if "readmitted" in df.columns:
        df = df.drop(columns=["readmitted"])
        st.info("Dropped `readmitted` column from input.")

    st.success(f"Loaded {len(df)} rows × {len(df.columns)} columns")
    st.dataframe(df.head(5), use_container_width=True)

    if st.button("Run batch prediction", type="primary", use_container_width=True):
        records = df.to_dict(orient="records")

        progress = st.progress(0, text="Sending to API...")
        try:
            result = predict_batch(records)
            progress.progress(100, text="Done")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
            st.stop()

        results_list = result.get("results", [])
        st.success(f"Completed {result['total']} predictions")

        # Build output dataframe
        out_rows = []
        RISK_COLORS = {"NO": "🟢", "< 30": "🔴", ">30": "🟡"}
        for i, r in enumerate(results_list):
            out_rows.append({
                "Row": i + 1,
                "Predicted class": r["predicted_class"],
                "Confidence": f"{r['confidence']*100:.1f}%",
                "P(NO)": f"{r['probabilities'].get('NO', 0)*100:.1f}%",
                "P(<30)": f"{r['probabilities'].get('<30', 0)*100:.1f}%",
                "P(>30)": f"{r['probabilities'].get('>30', 0)*100:.1f}%",
            })

        out_df = pd.DataFrame(out_rows)
        st.dataframe(out_df, use_container_width=True, hide_index=True)

        # Class summary
        st.subheader("Summary")
        col1, col2, col3 = st.columns(3)
        counts = out_df["Predicted class"].value_counts()
        with col1:
            st.metric("No readmission", counts.get("NO", 0))
        with col2:
            st.metric("Readmitted <30", counts.get("<30", 0))
        with col3:
            st.metric("Readmitted >30", counts.get(">30", 0))

        # Download results
        csv_buffer = io.StringIO()
        # Merge original df with predictions
        out_full = df.copy()
        out_full["predicted_class"] = [r["predicted_class"] for r in results_list]
        out_full["confidence"] = [r["confidence"] for r in results_list]
        out_full.to_csv(csv_buffer, index=False)

        st.download_button(
            label="Download results as CSV",
            data=csv_buffer.getvalue(),
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
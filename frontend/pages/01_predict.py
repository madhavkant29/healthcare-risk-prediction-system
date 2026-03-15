import streamlit as st
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from api_client import predict_single

st.set_page_config(page_title="Predict", page_icon="🔍", layout="wide")

if not st.session_state.get("token"):
    st.warning("Please log in first.")
    st.stop()

# ── Risk badge helper ─────────────────────────────────────
RISK_CONFIG = {
    "NO":  {"label": "No Readmission",      "color": "#1D9E75", "icon": "✅"},
    "<30": {"label": "Readmitted < 30 days", "color": "#E24B4A", "icon": "🔴"},
    ">30": {"label": "Readmitted > 30 days", "color": "#BA7517", "icon": "⚠️"},
}


def risk_badge(predicted_class: str, confidence: float):
    cfg = RISK_CONFIG.get(predicted_class, {})
    st.markdown(
        f"""
        <div style="
            background:{cfg['color']}22;
            border:1.5px solid {cfg['color']};
            border-radius:10px;
            padding:16px 24px;
            text-align:center;
        ">
            <div style="font-size:2rem">{cfg['icon']}</div>
            <div style="font-size:1.4rem;font-weight:600;color:{cfg['color']}">{cfg['label']}</div>
            <div style="color:#888;margin-top:4px">Confidence: {confidence*100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def shap_chart(shap_explanation: dict):
    """Horizontal bar chart of top SHAP features."""
    features = shap_explanation.get("top_features", [])
    if not features:
        return

    df = pd.DataFrame(features).sort_values("shap_value")
    df["color"] = df["shap_value"].apply(lambda v: "#E24B4A" if v > 0 else "#1D9E75")

    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(
        x=df["shap_value"],
        y=df["feature"],
        orientation="h",
        marker_color=df["color"].tolist(),
    ))
    fig.update_layout(
        title="Top contributing features (SHAP)",
        xaxis_title="SHAP value",
        yaxis_title="",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Page ─────────────────────────────────────────────────
st.title("Single Patient Prediction")
st.caption("Enter patient features to predict 30-day readmission risk.")

with st.form("predict_form"):
    st.subheader("Patient info (optional)")
    patient_ref = st.text_input(
        "Patient reference ID",
        placeholder="e.g. PT-00123 — stored anonymised",
        help="Stored as a one-way hash. Never your real patient ID.",
    )

    st.subheader("Admission details")
    col1, col2, col3 = st.columns(3)
    with col1:
        admission_type_id = st.slider("Admission type ID", 0.0, 1.0, 0.3, 0.01)
        discharge_disposition_id = st.slider("Discharge disposition ID", 0.0, 1.0, 0.5, 0.01)
        admission_source_id = st.slider("Admission source ID", 0.0, 1.0, 0.25, 0.01)
    with col2:
        time_in_hospital = st.number_input("Time in hospital (days)", 1, 14, 4)
        num_lab_procedures = st.number_input("Num lab procedures", 1, 132, 40)
        num_procedures = st.number_input("Num procedures", 0, 6, 1)
    with col3:
        num_medications = st.number_input("Num medications", 1, 81, 15)
        number_outpatient = st.number_input("Number outpatient visits", 0, 42, 0)
        number_emergency = st.number_input("Number emergency visits", 0, 76, 0)

    col4, col5 = st.columns(2)
    with col4:
        number_inpatient = st.number_input("Number inpatient visits", 0, 21, 0)
        number_diagnoses = st.number_input("Number diagnoses", 1, 16, 8)
    with col5:
        st.subheader("Demographics")
        race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.selectbox("Age group", [
            "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
            "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
        ], index=6)

    st.subheader("Medications")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
        metformin = st.selectbox("Metformin", ["No", "Steady", "Up", "Down"])
    with mcol2:
        glipizide = st.selectbox("Glipizide", ["No", "Steady", "Up", "Down"])
        glyburide = st.selectbox("Glyburide", ["No", "Steady", "Up", "Down"])
    with mcol3:
        pioglitazone = st.selectbox("Pioglitazone", ["No", "Steady", "Up", "Down"])
        rosiglitazone = st.selectbox("Rosiglitazone", ["No", "Steady", "Up", "Down"])
    with mcol4:
        change = st.selectbox("Medication change", ["No", "Ch"])
        diabetes_med = st.selectbox("Diabetes medication", ["Yes", "No"])

    submitted = st.form_submit_button("Predict readmission risk", use_container_width=True, type="primary")


def build_features(locals_dict: dict) -> dict:
    """Build the feature dict that matches training schema."""
    features = {
        "admission_type_id": locals_dict["admission_type_id"],
        "discharge_disposition_id": locals_dict["discharge_disposition_id"],
        "admission_source_id": locals_dict["admission_source_id"],
        "time_in_hospital": locals_dict["time_in_hospital"],
        "num_lab_procedures": locals_dict["num_lab_procedures"],
        "num_procedures": locals_dict["num_procedures"],
        "num_medications": locals_dict["num_medications"],
        "number_outpatient": locals_dict["number_outpatient"],
        "number_emergency": locals_dict["number_emergency"],
        "number_inpatient": locals_dict["number_inpatient"],
        "number_diagnoses": locals_dict["number_diagnoses"],
    }

    for r in ["AfricanAmerican", "Asian", "Caucasian", "Hispanic", "Other"]:
        features[f"race_{r}"] = 1 if locals_dict["race"] == r else 0

    for g in ["Female", "Male", "Unknown/Invalid"]:
        features[f"gender_{g}"] = 1 if locals_dict["gender"] == g else 0

    for a in ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
              "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]:
        features[f"age_{a}"] = 1 if locals_dict["age"] == a else 0

    for med, val in [
        ("insulin", locals_dict["insulin"]),
        ("metformin", locals_dict["metformin"]),
        ("glipizide", locals_dict["glipizide"]),
        ("glyburide", locals_dict["glyburide"]),
        ("pioglitazone", locals_dict["pioglitazone"]),
        ("rosiglitazone", locals_dict["rosiglitazone"]),
    ]:
        for opt in ["Down", "No", "Steady", "Up"]:
            features[f"{med}_{opt}"] = 1 if val == opt else 0

    features["change_Ch"] = 1 if locals_dict["change"] == "Ch" else 0
    features["change_No"] = 1 if locals_dict["change"] == "No" else 0
    features["diabetesMed_Yes"] = 1 if locals_dict["diabetes_med"] == "Yes" else 0
    features["diabetesMed_No"] = 1 if locals_dict["diabetes_med"] == "No" else 0

    return features


if submitted:
    features = build_features({
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        "race": race,
        "gender": gender,
        "age": age,
        "insulin": insulin,
        "metformin": metformin,
        "glipizide": glipizide,
        "glyburide": glyburide,
        "pioglitazone": pioglitazone,
        "rosiglitazone": rosiglitazone,
        "change": change,
        "diabetes_med": diabetes_med,
    })

    with st.spinner("Running prediction..."):
        try:
            result = predict_single(features, patient_ref=patient_ref or None)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    st.divider()
    st.subheader("Result")

    col_result, col_probs = st.columns([1, 1])

    with col_result:
        risk_badge(result["predicted_class"], result["confidence"])

    with col_probs:
        st.markdown("**Class probabilities**")
        probs = result["probabilities"]
        for cls, prob in probs.items():
            cfg = RISK_CONFIG.get(cls, {})
            st.metric(
                label=cfg.get("label", cls),
                value=f"{prob*100:.1f}%"
            )

    if result.get("shap_explanation"):
        st.divider()
        shap_chart(result["shap_explanation"])
        st.caption(
            "Red bars = features pushing toward this prediction. "
            "Green bars = features pushing away from it."
        )
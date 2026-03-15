import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from api_client import login, register, health_check

st.set_page_config(
    page_title="Healthcare Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ───────────────────────────────
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None


def show_login():
    st.title("Healthcare Risk Prediction")
    st.caption("Integrated Big Data Analytics and AI System for Smart Healthcare Risk Prediction")

    # API health check
    try:
        health = health_check()
        if health.get("model_loaded"):
            st.success("API online · Model loaded", icon="✅")
        else:
            st.warning("API online · Model not yet loaded — run ml_pipeline/train.py first", icon="⚠️")
    except Exception:
        st.error("Cannot reach API at localhost:8000 — is the backend running?", icon="🔴")

    st.divider()

    tab_login, tab_register = st.tabs(["Log in", "Register"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", use_container_width=True)

        if submitted:
            try:
                data = login(username, password)
                st.session_state.token = data["access_token"]
                st.session_state.username = username
                st.rerun()
            except Exception as e:
                st.error(f"Login failed: {e}")

    with tab_register:
        with st.form("register_form"):
            new_username = st.text_input("Username", key="reg_user")
            new_email = st.text_input("Email", key="reg_email")
            new_password = st.text_input("Password", type="password", key="reg_pass")
            submitted = st.form_submit_button("Create account", use_container_width=True)

        if submitted:
            try:
                register(new_username, new_email, new_password)
                st.success("Account created — please log in")
            except Exception as e:
                st.error(f"Registration failed: {e}")


def show_sidebar():
    with st.sidebar:
        st.markdown(f"**{st.session_state.username}**")
        st.caption("Logged in")
        st.divider()
        st.page_link("pages/01_predict.py", label="Single Prediction", icon="🔍")
        st.page_link("pages/02_dashboard.py", label="Dashboard", icon="📊")
        st.page_link("pages/03_upload.py", label="Batch Upload", icon="📂")
        st.divider()
        if st.button("Log out", use_container_width=True):
            st.session_state.token = None
            st.session_state.username = None
            st.rerun()


if not st.session_state.token:
    show_login()
else:
    show_sidebar()
    st.title("Healthcare Risk Prediction")
    st.info("Use the sidebar to navigate to Predict, Dashboard, or Batch Upload.")
"""
Thin httpx wrapper around the FastAPI backend.
All pages import from here — change the base URL in one place.
"""

import httpx
import streamlit as st
from typing import Optional

BASE_URL = "http://backend:8000"
TIMEOUT = 30.0


def _headers() -> dict:
    token = st.session_state.get("token")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def login(username: str, password: str) -> dict:
    r = httpx.post(
        f"{BASE_URL}/auth/login",
        json={
            "username": username,
            "password": password,
        },
        timeout=TIMEOUT,
    )

    if r.status_code != 200:
        try:
            msg = r.json().get("detail", "Login failed")
        except Exception:
            msg = r.text
        raise Exception(msg)

    return r.json()


def register(username: str, email: str, password: str) -> dict:
    r = httpx.post(
        f"{BASE_URL}/auth/register",
        json={"username": username, "email": email, "password": password},
        timeout=TIMEOUT,
    )

    if r.status_code != 201:
        try:
            msg = r.json().get("detail", "Registration failed")
        except Exception:
            msg = r.text
        raise Exception(msg)

    return r.json()


def predict_single(features: dict, patient_ref: Optional[str] = None) -> dict:
    payload = {"features": features}
    if patient_ref:
        payload["patient_ref"] = patient_ref
    r = httpx.post(
        f"{BASE_URL}/predict",
        json=payload,
        headers=_headers(),
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def predict_batch(records: list[dict]) -> dict:
    r = httpx.post(
        f"{BASE_URL}/predict/batch",
        json={"records": records},
        headers=_headers(),
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def get_history(limit: int = 50) -> list[dict]:
    r = httpx.get(
        f"{BASE_URL}/patients/history",
        params={"limit": limit},
        headers=_headers(),
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def get_stats() -> dict:
    r = httpx.get(
        f"{BASE_URL}/patients/stats",
        headers=_headers(),
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def health_check() -> dict:
    r = httpx.get(f"{BASE_URL}/health", timeout=5.0)
    r.raise_for_status()
    return r.json()
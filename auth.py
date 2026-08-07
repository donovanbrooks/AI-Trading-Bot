"""Local password authentication for the private Streamlit deployment."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets

from dotenv import load_dotenv


def create_password_hash(password: str) -> str:
    """Create a portable PBKDF2 password hash for APP_PASSWORD_HASH."""
    if len(password) < 12:
        raise ValueError("Use a password with at least 12 characters.")
    salt = secrets.token_bytes(16)
    iterations = 600_000
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return "pbkdf2_sha256$%s$%s$%s" % (
        iterations,
        base64.urlsafe_b64encode(salt).decode(),
        base64.urlsafe_b64encode(digest).decode(),
    )


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a candidate password without ever storing it in plaintext."""
    try:
        algorithm, iterations, encoded_salt, encoded_digest = stored_hash.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        salt = base64.urlsafe_b64decode(encoded_salt.encode())
        expected = base64.urlsafe_b64decode(encoded_digest.encode())
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, int(iterations))
        return hmac.compare_digest(actual, expected)
    except (TypeError, ValueError):
        return False


def require_login() -> None:
    """Render a login form and stop the app until the local user is verified."""
    import streamlit as st

    load_dotenv()
    stored_hash = os.getenv("APP_PASSWORD_HASH")
    if not stored_hash:
        st.error("App login is not configured. Add APP_PASSWORD_HASH to your local .env file.")
        st.code(".venv/bin/python scripts/create_password_hash.py")
        st.stop()
    if st.session_state.get("authenticated"):
        return
    st.title("Trading Bot Lab")
    st.caption("Private paper-trading research app")
    with st.form("login"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
    if submitted:
        if verify_password(password, stored_hash):
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("Invalid password.")
    st.stop()

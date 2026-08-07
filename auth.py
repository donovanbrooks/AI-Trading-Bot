"""Supabase-backed managed authentication for the Streamlit app."""

from __future__ import annotations

import os
import logging
import base64
import hashlib
import hmac
import secrets

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def create_password_hash(password: str) -> str:
    """Legacy helper retained for the local migration script; not used for login."""
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
    """Legacy test helper. Production sign-in uses Supabase Auth."""
    try:
        algorithm, iterations, encoded_salt, encoded_digest = stored_hash.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), base64.urlsafe_b64decode(encoded_salt), int(iterations))
        return hmac.compare_digest(actual, base64.urlsafe_b64decode(encoded_digest))
    except (TypeError, ValueError):
        return False


def _client():
    """Create a server-side Supabase client from environment-only credentials."""
    from supabase import create_client

    # A Streamlit process can retain older exported values after `.env` is
    # edited; prefer the current local configuration during this migration.
    load_dotenv(override=True)
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        raise RuntimeError("Supabase login is not configured. Add SUPABASE_URL and SUPABASE_ANON_KEY to .env.")
    return create_client(url, key)


def current_user_id() -> str | None:
    """Return the managed user id attached to this Streamlit browser session."""
    import streamlit as st

    return st.session_state.get("supabase_user_id")


def require_login() -> str:
    """Render sign-in, sign-up and password-reset forms until authenticated."""
    import streamlit as st

    if current_user_id():
        with st.sidebar:
            st.caption(f"Signed in as {st.session_state.get('supabase_user_email', 'user')}")
            if st.button("Sign out"):
                try:
                    _client().auth.sign_out()
                finally:
                    for key in ("supabase_user_id", "supabase_user_email", "supabase_access_token", "supabase_refresh_token"):
                        st.session_state.pop(key, None)
                    st.rerun()
        return str(current_user_id())

    st.title("Trading Bot Lab")
    st.caption("Sign in to your private paper-trading research workspace.")
    try:
        client = _client()
    except (ImportError, RuntimeError) as error:
        st.error(str(error))
        st.stop()

    sign_in, create_account, reset_password = st.tabs(["Sign in", "Create account", "Reset password"])
    with sign_in:
        with st.form("supabase_sign_in"):
            email = st.text_input("Email", key="sign_in_email")
            password = st.text_input("Password", type="password", key="sign_in_password")
            submitted = st.form_submit_button("Sign in")
        if submitted:
            try:
                response = client.auth.sign_in_with_password({"email": email.strip(), "password": password})
                if not response.user or not response.session:
                    raise RuntimeError("Supabase did not return a valid session.")
            except Exception as error:
                logger.exception("Supabase sign-in failed")
                st.error(f"Could not sign in: {error}")
            else:
                st.session_state["supabase_user_id"] = response.user.id
                st.session_state["supabase_user_email"] = response.user.email or email.strip()
                st.session_state["supabase_access_token"] = response.session.access_token
                st.session_state["supabase_refresh_token"] = response.session.refresh_token
                st.rerun()
    with create_account:
        with st.form("supabase_sign_up"):
            email = st.text_input("Email", key="sign_up_email")
            password = st.text_input("Password (at least 12 characters)", type="password", key="sign_up_password")
            submitted = st.form_submit_button("Create account")
        if submitted:
            if len(password) < 12:
                st.error("Use a password with at least 12 characters.")
            else:
                try:
                    redirect_to = os.getenv("APP_BASE_URL", "http://localhost:8502")
                    client.auth.sign_up(
                        {
                            "email": email.strip(),
                            "password": password,
                            "options": {"email_redirect_to": redirect_to},
                        }
                    )
                    st.success("Account created. Check your email to confirm it, then sign in.")
                except Exception as error:
                    logger.exception("Supabase account creation failed")
                    st.error(f"Could not create the account: {error}")
    with reset_password:
        with st.form("supabase_password_reset"):
            email = st.text_input("Account email", key="reset_email")
            submitted = st.form_submit_button("Email password-reset link")
        if submitted:
            try:
                redirect_to = os.getenv("APP_BASE_URL", "http://localhost:8502")
                client.auth.reset_password_for_email(email.strip(), {"redirect_to": redirect_to})
                st.success("If that address has an account, a password-reset link has been sent.")
            except Exception:
                st.error("Could not request a reset email. Check the Supabase URL configuration.")
    st.stop()

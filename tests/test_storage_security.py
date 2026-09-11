from types import SimpleNamespace
import sys

import pytest

import storage


def test_cloud_context_uses_signed_in_user_session_not_service_role(monkeypatch):
    captured = {}

    class FakeAuth:
        def set_session(self, access_token, refresh_token):
            captured["session"] = (access_token, refresh_token)

    class FakeClient:
        auth = FakeAuth()

    def create_client(url, key):
        captured["credentials"] = (url, key)
        return FakeClient()

    monkeypatch.setattr(storage, "load_dotenv", lambda **_: None)
    monkeypatch.setattr(storage.os, "getenv", lambda key: {"SUPABASE_URL": "https://example.supabase.co", "SUPABASE_ANON_KEY": "publishable-key"}.get(key))
    monkeypatch.setitem(sys.modules, "streamlit", SimpleNamespace(session_state={
        "supabase_user_id": "user-1", "supabase_access_token": "access", "supabase_refresh_token": "refresh",
    }))
    monkeypatch.setitem(sys.modules, "supabase", SimpleNamespace(create_client=create_client))

    _, user_id = storage._cloud_context()

    assert user_id == "user-1"
    assert captured["credentials"] == ("https://example.supabase.co", "publishable-key")
    assert captured["session"] == ("access", "refresh")


def test_cloud_context_fails_closed_when_cloud_is_configured_but_session_is_missing(monkeypatch):
    monkeypatch.setattr(storage, "load_dotenv", lambda **_: None)
    monkeypatch.setattr(storage.os, "getenv", lambda key: {"SUPABASE_URL": "https://example.supabase.co", "SUPABASE_ANON_KEY": "publishable-key"}.get(key))
    monkeypatch.setitem(sys.modules, "streamlit", SimpleNamespace(session_state={"supabase_user_id": "user-1"}))

    with pytest.raises(storage.StorageUnavailableError):
        storage._cloud_context()

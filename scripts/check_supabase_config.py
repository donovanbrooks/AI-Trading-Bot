"""Safely validate Supabase environment configuration without printing secrets."""

import os
from urllib.parse import urlparse

from dotenv import load_dotenv


load_dotenv()
url = (os.getenv("SUPABASE_URL") or "").strip()
key = (os.getenv("SUPABASE_ANON_KEY") or "").strip()
parsed = urlparse(url)
app_base_url = (os.getenv("APP_BASE_URL") or "").strip()
app_base_parsed = urlparse(app_base_url)

print(f"SUPABASE_URL present: {bool(url)}")
print(f"URL scheme is HTTPS: {parsed.scheme == 'https'}")
print(f"URL hostname ends with .supabase.co: {parsed.hostname is not None and parsed.hostname.endswith('.supabase.co')}")
print(f"URL path is empty or /: {parsed.path in ('', '/')}")
print(f"SUPABASE_ANON_KEY present: {bool(key)}")
print(f"APP_BASE_URL present: {bool(app_base_url)}")
print(f"APP_BASE_URL scheme is HTTP or HTTPS: {app_base_parsed.scheme in ('http', 'https')}")
print(f"APP_BASE_URL hostname is localhost: {app_base_parsed.hostname == 'localhost'}")
print(f"APP_BASE_URL port is 8502: {app_base_parsed.port == 8502}")

"""Check the Supabase Auth endpoint without disclosing credentials."""

import os
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


load_dotenv()
url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
key = os.getenv("SUPABASE_ANON_KEY") or ""
request = Request(f"{url}/auth/v1/settings", headers={"apikey": key})
try:
    with urlopen(request, timeout=15) as response:
        print(f"Auth endpoint status: {response.status}")
        settings = json.loads(response.read().decode("utf-8"))
        for field in ("site_url", "uri_allow_list", "mailer_autoconfirm"):
            if field in settings:
                print(f"{field}: {settings[field]}")
except HTTPError as error:
    print(f"Auth endpoint status: {error.code}")
    print(f"Supabase message: {error.read().decode('utf-8')[:300]}")
except URLError as error:
    print(f"Could not reach Supabase Auth: {error.reason}")

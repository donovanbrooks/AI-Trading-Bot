"""Print a hash suitable for APP_PASSWORD_HASH without saving the password."""

from getpass import getpass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auth import create_password_hash


password = getpass("Create an app password (12+ characters): ")
confirmation = getpass("Confirm password: ")
if password != confirmation:
    raise SystemExit("Passwords did not match.")
print("APP_PASSWORD_HASH=" + create_password_hash(password))

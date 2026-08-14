"""Server-side encryption for per-user paper-broker credentials."""

from __future__ import annotations

import base64
import hashlib
import os

from cryptography.fernet import Fernet, InvalidToken
from dotenv import load_dotenv


class CredentialError(RuntimeError):
    """Raised when credentials cannot be encrypted or decrypted safely."""


def _fernet() -> Fernet:
    load_dotenv(override=True)
    master_key = os.getenv("APP_ENCRYPTION_KEY")
    if not master_key:
        raise CredentialError("APP_ENCRYPTION_KEY must be configured on the server before broker keys can be stored.")
    # Deriving a Fernet key permits a securely generated host secret of any
    # reasonable format without ever storing the original key in the database.
    derived_key = base64.urlsafe_b64encode(hashlib.sha256(master_key.encode()).digest())
    return Fernet(derived_key)


def encrypt_secret(plaintext: str) -> str:
    if not plaintext:
        raise CredentialError("A credential value is required.")
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt_secret(ciphertext: str) -> str:
    try:
        return _fernet().decrypt(ciphertext.encode()).decode()
    except (InvalidToken, UnicodeDecodeError) as error:
        raise CredentialError("Stored broker credentials cannot be decrypted. Reconnect your paper account.") from error

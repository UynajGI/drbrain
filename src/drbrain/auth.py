"""Password hashing and verification for destructive CLI operations."""

import hashlib
import os


def hash_password(password: str) -> str:
    """Hash a password with a random 16-byte salt. Returns 'salt_hex:hash_hex'."""
    salt = os.urandom(16)
    salt_hex = salt.hex()
    hash_hex = hashlib.sha256(salt + password.encode()).hexdigest()
    return f"{salt_hex}:{hash_hex}"


def verify_password(password: str, stored: str) -> bool:
    """Verify a password against a stored salt:hash string."""
    try:
        salt_hex, hash_hex = stored.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        expected = hashlib.sha256(salt + password.encode()).hexdigest()
        return expected == hash_hex
    except (ValueError, AttributeError):
        return False


def has_password(config: dict) -> bool:
    """Check if an admin password is configured."""
    return bool(config.get("admin", {}).get("password_hash"))

"""Password hashing and verification for destructive CLI operations."""

import hashlib
import hmac
import os

# PBKDF2 parameters: SHA-256 with 600k iterations (OWASP 2023 recommendation).
_ITERATIONS = 600_000


def hash_password(password: str) -> str:
    """Hash a password with a random 16-byte salt using PBKDF2-HMAC-SHA256.

    Returns 'salt_hex:hash_hex'.  Compatible with ``verify_password``.
    """
    salt = os.urandom(16)
    salt_hex = salt.hex()
    hash_hex = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS).hex()
    return f"{salt_hex}:p{hash_hex}"


def verify_password(password: str, stored: str) -> bool:
    """Verify a password against a stored salt:hash string.

    Uses ``hmac.compare_digest`` to prevent timing side-channel attacks.
    Falls back to legacy SHA-256 for hashes generated before the PBKDF2
    migration (detected by the absence of a ``p`` prefix in the hash).
    """
    try:
        salt_hex, hash_hex = stored.split(":", 1)
        salt = bytes.fromhex(salt_hex)

        if hash_hex.startswith("p"):
            # New PBKDF2 hash: stored as 'salt:p<der_hex>'
            expected = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS).hex()
            # Compare with the stored value including the 'p' prefix
            return hmac.compare_digest(f"p{expected}", hash_hex)
        else:
            # Legacy SHA-256 hash
            expected = hashlib.sha256(salt + password.encode()).hexdigest()
            return hmac.compare_digest(expected, hash_hex)
    except (ValueError, AttributeError):
        return False


def has_password(config: dict) -> bool:
    """Check if an admin password is configured."""
    return bool(config.get("admin", {}).get("password_hash"))

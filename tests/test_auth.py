from drbrain.auth import has_password, hash_password, verify_password


def test_hash_and_verify_correct():
    pw = "test-password"
    stored = hash_password(pw)
    assert verify_password(pw, stored)


def test_verify_wrong_password():
    stored = hash_password("correct")
    assert not verify_password("wrong", stored)


def test_hash_is_not_plaintext():
    stored = hash_password("secret")
    assert "secret" not in stored


def test_salt_produces_different_hashes():
    h1 = hash_password("same-pw")
    h2 = hash_password("same-pw")
    assert h1 != h2  # different salt


def test_has_password_true():
    assert has_password({"admin": {"password_hash": "abc:def"}})


def test_has_password_false():
    assert not has_password({})
    assert not has_password({"admin": {}})


def test_verify_empty_or_invalid_stored():
    assert not verify_password("pw", "")
    assert not verify_password("pw", "not-valid-format")


def test_new_hash_has_pbkdf2_prefix():
    """New hashes must start with 'p' after the salt separator."""
    stored = hash_password("test")
    salt_hex, hash_hex = stored.split(":", 1)
    assert hash_hex.startswith("p")


def test_new_hash_uses_pbkdf2():
    """Verify that new hashes are PBKDF2, not plain SHA-256."""
    import hashlib

    pw = "deterministic-test"
    stored = hash_password(pw)
    salt_hex, hash_hex = stored.split(":", 1)
    salt = bytes.fromhex(salt_hex)
    expected_sha256 = hashlib.sha256(salt + pw.encode()).hexdigest()

    # PBKDF2 hash should NOT equal SHA-256 hash
    assert hash_hex[1:] != expected_sha256


def test_legacy_sha256_hash_still_verifies():
    """Legacy SHA-256 hashes (without 'p' prefix) must still verify correctly."""
    import hashlib
    import os

    salt = os.urandom(16)
    pw = "legacy-password"
    legacy_hash = hashlib.sha256(salt + pw.encode()).hexdigest()
    stored = f"{salt.hex()}:{legacy_hash}"

    # No 'p' prefix → legacy path
    assert verify_password(pw, stored)
    assert not verify_password("wrong", stored)


def test_legacy_hash_rejects_wrong_password():
    """Legacy hash must reject incorrect passwords."""
    import hashlib
    import os

    salt = os.urandom(16)
    pw = "correct"
    legacy_hash = hashlib.sha256(salt + pw.encode()).hexdigest()
    stored = f"{salt.hex()}:{legacy_hash}"

    assert not verify_password("incorrect", stored)

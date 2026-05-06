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

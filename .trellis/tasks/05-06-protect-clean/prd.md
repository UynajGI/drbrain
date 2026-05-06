# Password-Protect `clean --force`

## What

- `setup` sets admin password (salted SHA-256 hash in config.local.yaml)
- `setup --change-password` changes it
- `clean --force` requires password (typer hidden prompt)
- Password never stored in plaintext

## Implementation

### `src/drbrain/auth.py` (new)
```python
def hash_password(password: str) -> str:
    # salt:hash hex, 16-byte random salt
def verify_password(password: str, stored: str) -> bool:
def has_password(config: dict) -> bool:
```

### `src/drbrain/cli/setup.py`
- Add `--change-password` flag
- In interactive setup: prompt "Set admin password? [y/N]"
- On --change-password: verify old, set new

### `src/drbrain/cli/commands.py` — `clean_cmd`
- When `--force`: if password is configured, prompt for it
- Verify with `verify_password()`, deny if wrong (exit 1)
- If no password set, `--force` works as before

### Files
- `src/drbrain/auth.py` — new
- `src/drbrain/cli/setup.py` — add password prompt + --change-password
- `src/drbrain/cli/commands.py` — clean_cmd password gate
- `tests/test_auth.py` — new

## Acceptance
- `clean --force` prompts for password if configured
- Wrong password → denied
- Password hash stored, not plaintext
- `setup --change-password` works

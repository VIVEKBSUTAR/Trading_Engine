#!/usr/bin/env bash
set -euo pipefail
cd /app
umask 077
python broker/fyers_auto_auth.py
# A subprocess cannot update this shell's environment. Load only the new token
# with dotenv, verify this startup generated it, then exec the daemon.
exec python - <<'PY'
import os
from datetime import UTC, datetime
from dotenv import dotenv_values

values = dotenv_values(os.getenv("FYERS_ENV_FILE", "/app/.env"))
token = values.get("FYERS_ACCESS_TOKEN")
generated = values.get("FYERS_TOKEN_GENERATED_AT")
if not token or not generated:
    raise SystemExit("Token generation verification failed")
age = (datetime.now(UTC) - datetime.fromisoformat(generated)).total_seconds()
if not 0 <= age < 120:
    raise SystemExit("Token was not generated during this startup")
os.environ["FYERS_ACCESS_TOKEN"] = token
os.execvp("te-intelligence", ["te-intelligence"])
PY

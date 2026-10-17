"""Admits AT to the Gomer Trading app on this Mac as a trading program, and saves its API key.

    .venv/bin/python scripts/admit_at.py                      # its own sandbox competition
    .venv/bin/python scripts/admit_at.py --competition "Demo Trading Challenge"

It signs in to the local app the way the desktop app signs its owner in (a key the app keeps
on this Mac and accepts only from this Mac), then uses the app's own admin API:

  - creates the sandbox competition if it does not exist
  - admits a program labelled "AT" there (the app creates its BOT-… account and API key),
    or issues a fresh key if AT is already admitted and its saved key no longer works
  - writes the key to .env (readable only by you). The key is never printed.

Run it again at any time; it changes nothing that is already in place.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
APP_CONFIG = Path.home() / "Library/Application Support/Gomer Trading/config.json"
LABEL = "AT"


def read_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    pairs = (line.partition("=") for line in path.read_text().splitlines() if "=" in line and not line.strip().startswith("#"))
    return {key.strip(): value.strip() for key, _, value in pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:47831")
    parser.add_argument("--competition", default="AT Sandbox (ongoing)", help="competition AT trades in; created if it does not exist")
    parser.add_argument("--days", type=int, default=30, help="how long a newly created sandbox runs")
    parser.add_argument("--end", default="2099-12-31T23:59:59Z", help="end date for a newly created sandbox (overrides --days); the app cannot change it later")
    parser.add_argument("--balance", type=float, default=100_000.0)
    parser.add_argument("--label", default=LABEL, help="the program's name in the Admin console")
    parser.add_argument("--env-name", default="AT_API_KEY", help="where in .env to keep this key")
    parser.add_argument("--same-account-as", default=None, metavar="LABEL",
                        help="give this key to the account of the program with this label, so both keys trade as one AT")
    args = parser.parse_args()

    api = httpx.Client(base_url=args.base_url + "/api", timeout=15.0)
    key = json.loads(APP_CONFIG.read_text())["desktopKey"]
    signed_in = api.post("/auth/desktop", headers={"x-desktop-key": key}, json={"account": "admin"})
    if signed_in.status_code != 200:
        sys.exit("Could not sign in to the Gomer Trading app. Is it open on this Mac?")
    api.headers["Authorization"] = f"Bearer {signed_in.json()['token']}"

    competitions = api.get("/competitions").json()["competitions"]
    competition = next((c for c in competitions if c["name"] == args.competition), None)
    if competition is None:
        now = datetime.now(timezone.utc)
        end = datetime.fromisoformat(args.end.replace("Z", "+00:00")) if args.end else now + timedelta(days=args.days)
        created = api.post("/competitions", json={
            "name": args.competition, "startDate": now.isoformat(), "endDate": end.isoformat(),
            "startingBalance": args.balance, "allowShorting": True, "leverage": 1, "commissionPerShare": 0, "openToInvites": False,
        })
        created.raise_for_status()
        body = created.json()
        competition = body.get("competition", body)
        print(f"Created competition '{args.competition}' (${args.balance:,.0f} starting balance, ends {end:%d %B %Y}).")
    else:
        print(f"Using competition '{competition['name']}' ({competition['status']}).")

    env_path = ROOT / ".env"
    env = read_env(env_path)
    bots = api.get("/bots").json()["bots"]
    bot = next((b for b in bots if b["label"] == args.label), None)
    api_key = env.get(args.env_name, "")
    works = bool(api_key) and httpx.get(args.base_url + "/api/gateway/v1/session", headers={"X-API-Key": api_key}).status_code == 200

    if bot is None:
        request = {"label": args.label, "competitionId": competition["id"]}
        if args.same_account_as:
            owner = next((b for b in bots if b["label"] == args.same_account_as), None)
            if owner is None:
                sys.exit(f"There is no trading program labelled {args.same_account_as!r}.")
            request["traderId"] = owner["traderId"]
        admitted = api.post("/bots", json=request)
        admitted.raise_for_status()
        bot, api_key = admitted.json()["bot"], admitted.json()["apiKey"]
        print(f"Admitted {args.label} as trading program {bot['traderId']} in '{competition['name']}'.")
    else:
        if bot.get("competitionId") != competition["id"]:
            api.put(f"/bots/{bot['id']}", json={"competitionId": competition["id"]}).raise_for_status()
            print(f"Moved {args.label} to '{competition['name']}'.")
        if not works:
            rotated = api.post(f"/bots/{bot['id']}/rotate")
            rotated.raise_for_status()
            api_key = rotated.json()["apiKey"]
            print(f"{args.label} was already admitted ({bot['traderId']}); issued it a new key.")
        else:
            print(f"{args.label} is already admitted ({bot['traderId']}) and its saved key works.")

    env.update({"AT_BASE_URL": args.base_url, args.env_name: api_key})
    env_path.write_text("".join(f"{k}={v}\n" for k, v in env.items()))
    os.chmod(env_path, 0o600)
    session = httpx.get(args.base_url + "/api/gateway/v1/session", headers={"X-API-Key": api_key})
    print(f"Key saved to {env_path.name}. Gateway check: {session.status_code} {json.dumps({k: v for k, v in session.json().items() if k != 'apiKey'})[:300]}")


if __name__ == "__main__":
    main()

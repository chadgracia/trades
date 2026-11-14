# Project rules for Claude

- Always commit and push directly to `main`. Do not create review branches.

## Unconfirmed rule (_load_pipeline_updated_at / _unconfirmed_since)
- MUST stay identical to `unconfirmed_status()` in chadgracia/CRMDealDetails so the grid's
  amber clock and the deal page agree. Change both together.
- Data: per-deal record from `s3://full-pipeline-cache/deals.json` (Last Confirmed
  `custom_label_4012491`, `updated_at` as aware UTC datetime — Pipeline "YYYY/MM/DD HH:MM:SS +0000"
  or ISO, None if missing/unparseable — Refresh, Seller Role, stage), cached 10 min.
- Never Unconfirmed: Hold stage 2094373, deals missing from deals.json, Refresh
  `custom_label_3994687` = 0. Threshold = Refresh days, else 30 for Seller Role 7020357, else 60.
- Effective date = the newer of Last Confirmed and the UTC date of `updated_at` (any Pipeline
  edit counts as confirmation). Unconfirmed = no effective date, or older than the threshold.
  Tooltip: "Unconfirmed · last confirmed <Mon D, YYYY>" (effective date) or
  "Unconfirmed · not recently confirmed".
- Restore exception: stale-deal-scanner logs to `s3://full-pipeline-cache/obsolete-review/log.json`
  (`{"entries": [{deal_id, action, at}]}`), loaded in the same 10-min cache. Per deal take the
  LATEST entry (newest `at`; tie/unreadable → later in list). If it is `"restored"`, `updated_at`
  counts only when > 15 min after its `at` (unreadable `at` → ignore `updated_at`); otherwise
  Last Confirmed alone. Missing/unreadable/malformed log = no restores. Never raises.
- Tests: `python3.12 -m unittest discover -s tests` (tests/test_unconfirmed.py).

## Deal gate (shared block, identical in chadgracia/trades and chadgracia/CRMDealDetails)
- Code between `# --- Deal gate (shared verbatim ...` and `# --- end deal gate shared block ---`
  is the same in both repos; change both together. Tests: `tests/test_deal_gate.py`.
- Switch: `s3://full-pipeline-cache/deal-gate.json` `{"enabled": bool, "updated_at": iso}`, cached 60 s;
  missing/unreadable keeps the container's last value, else OFF. Switch OFF = non-admin output
  unchanged (golden files in `tests/fixtures/` were rendered by the pre-gate code; keep them passing).
  Admin page + On/Off: trades `/?view=deal-gate`. Admin preview anywhere: `&gate_preview=anon|pending|blocked|cleared`.
- Viewer state from people-slim.json (`_people_slim_status_index`, never people.json), loaded only
  when the request carries gg_id / gg_view / gp. Investor Level `custom_label_3923758`:
  Non-Accredited 6950562 = blocked; Hold 7161646 or unmatched gg_id email = pending; anything else = cleared.
  Index unreadable: identified viewer = cleared.
- Personal links: env `VIEW_LINK_SECRET` (empty = links off). `?v=<pid>.<exp>.<sig>` (sig over
  `view1|pid|exp`, 183 d) -> 302 without v + cookie `gg_view` (sig over `viewc1|pid|exp`, 365 d).
  gg_view is VIEW-ONLY: never signed in, never gg_id, never an SSO token.
- Gated output uses fixed placeholders in `<span class="gated">`; real values must not appear anywhere
  in the response. Request logs go through `dg_redacted_event` (no cookies, body, or link/key params).
- trades-only: `POST /?view=view-link-request` (email-me-a-link, SES, limits in
  `full-pipeline-cache/deal-gate/requests/`), admin JSON `/?view=view-link`.

## SECURITY RULES — MANDATORY (read before any change)

This repo is PUBLIC, and so are its GitHub Actions logs. Secrets have leaked from Chad's repos 5-6 times. Never again.

1. No secret value ever goes in code, tests, docs, comments, commit messages, PR text or logs. That includes API keys, admin keys, HMAC/signing secrets, tokens, passwords, and any URL containing `key=`, `token=` or `sso=`. Secrets come from `os.environ[...]` (Lambda environment variables) or `s3://pipeline-token/` only. Tests use obvious fakes like `"test-key"`.
2. Every `aws lambda ...` command in a workflow must end with `> /dev/null`. Lambda API responses include every environment variable, and the logs are public. Never print the environment (`printenv`, `env`, `set -x`, or `print(os.environ)`).
3. Never log or print a secret, even partially, or a request URL that carries one.
4. `.github/scripts/secret_guard.py` runs automatically before every `git commit` and `git push` (via `.claude/settings.json`), and again in CI. If it blocks you, remove the secret and move the value to an environment variable. NEVER bypass, weaken, delete or skip the guard, and never use `--no-verify` or `secret-guard: allow` to get past it without Chad's explicit approval.
5. A new secret needs Chad to add a Lambda environment variable first. Tell him the exact variable name. Code that reads it ships after that.
6. Before finishing, run `python3 .github/scripts/secret_guard.py --tree` and report the result.
7. NEVER build anything that asks for, collects, uploads, stores, emails or processes client identity data — tax IDs, SSNs, passports, driver's licenses, other government IDs, or dates of birth. Chad does not want or need it; identity verification is handled by Rainmaker's own forms. If a task seems to need it, stop and ask Chad.

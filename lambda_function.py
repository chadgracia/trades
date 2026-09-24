import json
import os
import time
import base64
import hmac
import hashlib
import html as html_mod
import urllib.request
import urllib.parse
import urllib.error
import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

from datetime import datetime, timezone, timedelta


# --- Additive Cognito identity capture (does not affect access) ---
COGNITO_TOKEN_URL = "https://us-east-1dsttcaqx7.auth.us-east-1.amazoncognito.com/oauth2/token"
COGNITO_CLIENT_ID = "71vrglkidm13jb73u7nje3d1t2"
COGNITO_REDIRECT_URI = "https://trades.graciagroup.com"
COGNITO_CLIENT_SECRET = os.environ.get("COGNITO_CLIENT_SECRET", "")
IDENTITY_SECRET = os.environ.get("IDENTITY_SECRET", "")
NUDGE_KEY = os.environ.get("NUDGE_KEY", "")
LOI_PAGE_KEY = os.environ.get("LOI_PAGE_KEY", "")
LOI_SEND_URL = "https://aep54fnrcp4bxiowlw3fvt26x40qhgpn.lambda-url.us-east-1.on.aws/"
SYNDICATE_DASH_URL = "https://ws4stw4iul75a7yx5dra2wmnq40kipav.lambda-url.us-east-1.on.aws"
SYNDICATE_TENANTS_URL = f"{SYNDICATE_DASH_URL}/?key=JK8h5Pq2L9aZ7rT3mN6bX&tenants=list"
_syndicate_tenant_cache = {"emails": None}


def _exchange_code_for_email(code):
    """Exchange a Cognito auth code for tokens and return the user's email,
    or None on any failure. Never raises."""
    if not (COGNITO_CLIENT_SECRET and code):
        return None
    try:
        data = urllib.parse.urlencode({
            "grant_type": "authorization_code",
            "client_id": COGNITO_CLIENT_ID,
            "code": code,
            "redirect_uri": COGNITO_REDIRECT_URI,
        }).encode()
        basic = base64.b64encode(
            f"{COGNITO_CLIENT_ID}:{COGNITO_CLIENT_SECRET}".encode()
        ).decode()
        req = urllib.request.Request(
            COGNITO_TOKEN_URL, data=data,
            headers={"Authorization": f"Basic {basic}",
                     "Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            tokens = json.loads(resp.read().decode())
        id_token = tokens.get("id_token")
        if not id_token:
            return None
        payload_b64 = id_token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload_b64))
        return claims.get("email")
    except Exception as e:
        logger.warning(f"Cognito code exchange failed (non-fatal): {e}")
        return None


def _make_identity_cookie(email):
    sig = hmac.new(IDENTITY_SECRET.encode(), email.encode(), hashlib.sha256).hexdigest()
    val = base64.urlsafe_b64encode(f"{email}|{sig}".encode()).decode().rstrip("=")
    return f"gg_id={val}; Max-Age=31536000; Path=/; Secure; SameSite=Lax"


def _get_cookie(event, name):
    """Read a cookie value from a payload-v2 request, else None."""
    for c in (event.get("cookies") or []):
        if c.startswith(name + "="):
            return c.split("=", 1)[1]
    hdr = (event.get("headers") or {}).get("cookie", "")
    for c in hdr.split(";"):
        c = c.strip()
        if c.startswith(name + "="):
            return c.split("=", 1)[1]
    return None


def _read_identity_email(event):
    """Verified email from the gg_id cookie, or None. Reverses _make_identity_cookie.
    Never raises."""
    if not IDENTITY_SECRET:
        return None
    raw = _get_cookie(event, "gg_id")
    if not raw:
        return None
    try:
        decoded = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode()
        email, sig = decoded.rsplit("|", 1)
        expected = hmac.new(IDENTITY_SECRET.encode(), email.encode(),
                            hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, sig):
            return None
        return email
    except Exception:
        return None


def _make_handoff_token(email):
    """Short-lived signed handoff the portfolio app verifies:
    base64url(f"{email}|{exp}|{sig}"), sig = HMAC-SHA256(IDENTITY_SECRET, f"{email}|{exp}")."""
    exp = int(time.time()) + 3600
    sig = hmac.new(IDENTITY_SECRET.encode(), f"{email}|{exp}".encode(),
                   hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(f"{email}|{exp}|{sig}".encode()).decode().rstrip("=")


def _syndicate_eligible_emails():
    """Lowercased emails eligible for the Syndicate Dashboard (>=1 deal
    tagged Sell Order, any stage), fetched once per warm container from
    syndicate-dash's admin-gated ?tenants=list route. Short timeout,
    fail-soft: any error caches an empty set so the button just renders
    nothing rather than erroring or retrying every request."""
    if _syndicate_tenant_cache["emails"] is not None:
        return _syndicate_tenant_cache["emails"]
    emails = set()
    try:
        req = urllib.request.Request(SYNDICATE_TENANTS_URL)
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
        emails = {(t.get("email") or "").strip().lower()
                  for t in (data.get("tenants") or []) if t.get("email")}
    except Exception as e:
        logger.warning(f"Syndicate tenants fetch failed (non-fatal): {e}")
    _syndicate_tenant_cache["emails"] = emails
    return emails


AUCTIONS_BUCKET = "full-pipeline-cache"
AUCTIONS_KEY = "auctions.json"
DESK_URL = "https://desk.graciagroup.com"

_PD_HASH_CACHE = {'ts': 0.0, 'emails': '[]', 'domains': '[]'}


PD_SEARCH_ID = 20538950  # "S: Partner Desk Check" — Whitelist-tagged clients


def _pd_fetch_search_page(page):
    url = (
        "https://api.pipelinecrm.com/api/v3/searches/"
        + str(PD_SEARCH_ID)
        + "/perform.json?per_page=200&page=" + str(page)
        + "&api_key=ZRMHN4uJotjRDcZa8hKi"
        + "&app_key=571978be28bd3b5b515a2cc5db96b674"
    )
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


PD_FREEMAIL = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com", "icloud.com",
    "me.com", "mac.com", "msn.com", "live.com", "comcast.net", "protonmail.com",
    "proton.me", "yandex.ru", "yandex.com", "mail.ru", "gmx.com", "gmx.de", "web.de",
    "ymail.com", "googlemail.com", "att.net", "verizon.net", "sbcglobal.net", "pm.me",
    "hey.com", "fastmail.com", "qq.com", "163.com", "126.com", "hotmail.co.uk",
    "yahoo.co.uk", "btinternet.com", "rogers.com", "shaw.ca", "bell.net", "ukr.net", "i.ua",
}

PD_FIRM_TYPES = {
    "vc or pe fund", "family office", "institution", "hedge fund",
    "syndicator", "wealth advisor", "corporation",
}


def _partner_desk_hash_sets():
    """Build both hash sets for the partner-desk name check from the
    'S: Partner Desk Check' focused list (Whitelist-tagged clients).
    'emails' = every address found in ANY column whose id contains
    'email' (work/home/other); 'domains' = corporate domains of rows
    whose Transactor Type (custom_label_3759163) is an investor
    organization (PD_FIRM_TYPES) — employee holders, natural persons,
    and intermediaries do NOT contribute their domains, and freemail
    domains are always excluded. Cached in-module for 15 minutes; any
    failure returns the last good value so the page renders with the
    checker in its offline state."""
    import time
    import hashlib
    import concurrent.futures
    now = time.time()
    if _PD_HASH_CACHE['emails'] != '[]' and now - _PD_HASH_CACHE['ts'] < 900:
        return _PD_HASH_CACHE
    try:
        rows = []

        def _collect(data):
            cols = [c.get("id") for c in (data.get("columns") or [])]
            email_idx = [i for i, cid in enumerate(cols)
                         if cid and "email" in str(cid).lower()]
            if not email_idx:
                raise RuntimeError("focused list has no email columns")
            i_type = cols.index("custom_label_3759163") if "custom_label_3759163" in cols else None
            for entry in (data.get("entries") or []):
                if not isinstance(entry, list):
                    continue
                t = ""
                if i_type is not None and len(entry) > i_type and entry[i_type]:
                    t = str(entry[i_type]).strip().lower()
                found = []
                for i in email_idx:
                    if len(entry) > i and entry[i]:
                        e = str(entry[i]).strip().lower()
                        if e and "@" in e:
                            found.append(e)
                if found:
                    rows.append((found, t))

        first = _pd_fetch_search_page(1)
        _collect(first)
        pagination = first.get("pagination") or {}
        try:
            pages = min(int(pagination.get("pages") or 1), 25)
        except (TypeError, ValueError):
            pages = 1
        if pages > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
                for data in ex.map(_pd_fetch_search_page, range(2, pages + 1)):
                    _collect(data)

        salt = 'gracia-partner-check-v1'
        email_hashes = set()
        firm_domains = set()
        for found, t in rows:
            for e in found:
                email_hashes.add(hashlib.sha256((salt + e).encode('utf-8')).hexdigest()[:16])
                d = e.split("@", 1)[1]
                if d not in PD_FREEMAIL and t in PD_FIRM_TYPES:
                    firm_domains.add(d)
        domain_hashes = {hashlib.sha256((salt + "d:" + d).encode('utf-8')).hexdigest()[:16] for d in firm_domains}
        _PD_HASH_CACHE.update(
            ts=now,
            emails=json.dumps(sorted(email_hashes), separators=(',', ':')),
            domains=json.dumps(sorted(domain_hashes), separators=(',', ':')),
        )
        return _PD_HASH_CACHE
    except Exception as e:
        logger.error(f"partner-desk hash build failed (non-fatal): {e}")
        return _PD_HASH_CACHE


_PD_ANALYZE_KEY = 'JK8h5Pq2L9aZ7rT3mN6bX'

_PD_ANALYZE_FREEMAIL = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com", "icloud.com",
    "me.com", "mac.com", "msn.com", "live.com", "comcast.net", "protonmail.com",
    "proton.me", "yandex.ru", "yandex.com", "mail.ru", "gmx.com", "gmx.de", "web.de",
    "ymail.com", "googlemail.com", "att.net", "verizon.net", "sbcglobal.net", "pm.me",
    "hey.com", "fastmail.com", "qq.com", "163.com", "126.com", "hotmail.co.uk",
    "yahoo.co.uk", "btinternet.com", "rogers.com", "shaw.ca", "bell.net", "ukr.net", "i.ua",
}


def _pd_analyze_report():
    """Read-only breakdown of the 'S: Weekly Mailer Leads' Pipeline focused
    list: row/email counts, freemail vs corporate split, domain headcounts,
    and a per-column value breakdown. Mirrors the ad-hoc analysis script;
    reuses _pd_fetch_search_page for the actual fetching. No writes."""
    import collections
    import concurrent.futures

    out = []

    def p(*a):
        out.append(" ".join(str(x) for x in a))

    first = _pd_fetch_search_page(1)
    cols = [c.get("id") for c in (first.get("columns") or [])]
    p("COLUMNS:", cols)
    pages = int((first.get("pagination") or {}).get("pages") or 1)
    p("PAGES:", pages)
    all_pages = [first]
    if pages > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            all_pages += list(ex.map(_pd_fetch_search_page, range(2, pages + 1)))

    i_email = cols.index("person_email")
    rows = []
    for d in all_pages:
        for e in (d.get("entries") or []):
            if isinstance(e, list) and len(e) > i_email and e[i_email]:
                rows.append([str(x) if x is not None else "" for x in e])

    emails = sorted({r[i_email].strip().lower() for r in rows if "@" in r[i_email]})
    domains = collections.Counter(e.split("@", 1)[1] for e in emails)
    corp = {d: c for d, c in domains.items() if d not in _PD_ANALYZE_FREEMAIL}
    free_ct = sum(c for d, c in domains.items() if d in _PD_ANALYZE_FREEMAIL)

    p("TOTAL ROWS:", len(rows), "| UNIQUE EMAILS:", len(emails))
    p("FREEMAIL EMAILS:", free_ct, "| CORPORATE EMAILS:", len(emails) - free_ct)
    p("UNIQUE CORPORATE DOMAINS:", len(corp))
    multi = {d: c for d, c in corp.items() if c >= 2}
    p("CORPORATE DOMAINS WITH 2+ PEOPLE:", len(multi))
    p("TOP 30 DOMAINS BY HEADCOUNT:")
    for d, c in sorted(corp.items(), key=lambda x: -x[1])[:30]:
        p("  %4d  %s" % (c, d))
    for j, cid in enumerate(cols):
        if cid in ("person_email", "person_first_name", "person_id"):
            continue
        vals = collections.Counter(r[j] for r in rows if j < len(r) and r[j])
        if 0 < len(vals) <= 40:
            p("BREAKDOWN of column '%s':" % cid)
            for v, c in vals.most_common(15):
                p("  %4d  %s" % (c, v))

    return "\n".join(out)


PARTNER_DESK_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="robots" content="noindex, nofollow">
<title>Partner Desk — Gracia Group</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
<style>
:root{
  box-sizing:border-box;
  padding-top:env(safe-area-inset-top,0px);
  padding-bottom:env(safe-area-inset-bottom,0px);
  --paper:#FBFBF8;
  --ink:#1C2126;
  --muted:#5C6670;
  --ledger:#1F5D45;
  --ledger-soft:#EDF3EE;
  --hairline:#D8DCD4;
  --tint:#F3F4F0;
  --warn:#8A5A16;
}
*,*::before,*::after{box-sizing:inherit}
html{scroll-padding-top:env(safe-area-inset-top,0px)}
html,body{margin:0;padding:0}
body{
  background:var(--paper);
  color:var(--ink);
  font-family:"Source Serif 4",Georgia,"Times New Roman",serif;
  font-size:17px;
  line-height:1.65;
  -webkit-font-smoothing:antialiased;
}
.sheet{max-width:660px;margin:0 auto;padding:3.5rem 1.25rem 5rem}
.letterhead{
  display:flex;justify-content:space-between;align-items:baseline;gap:1rem;
  border-bottom:2px solid var(--ink);
  padding-bottom:.6rem;margin-bottom:2.75rem;
}
.letterhead .firm{font-weight:600;font-size:1.02rem;letter-spacing:.01em}
.letterhead .via{font-size:.85rem;color:var(--muted);text-align:right}
h1{
  font-size:1.85rem;line-height:1.25;font-weight:600;
  margin:0 0 1.1rem;letter-spacing:-.01em;
}
.lede{font-size:1.1rem;line-height:1.6;margin:0 0 .9rem}
p{margin:0 0 .9rem}
h2{
  font-size:1.02rem;font-weight:600;margin:2.75rem 0 .8rem;
  padding-top:1.4rem;border-top:1px solid var(--hairline);
}
a{color:var(--ledger);text-underline-offset:3px}
.muted{color:var(--muted)}
.small{font-size:.9rem}
.termwrap{overflow-x:auto;margin:1.4rem 0 .4rem}
table.terms{
  width:100%;border-collapse:collapse;font-size:.95rem;line-height:1.45;
  min-width:560px;
}
.terms caption{
  caption-side:top;text-align:left;font-style:italic;color:var(--muted);
  font-size:.9rem;padding-bottom:.5rem;
}
.terms th,.terms td{
  border:1px solid var(--hairline);
  padding:.6rem .75rem;vertical-align:top;text-align:left;
}
.terms thead th{background:var(--tint);font-weight:600}
.terms thead th.tier{width:36%}
.terms td:first-child{color:var(--muted);width:18%}
.terms .num{font-weight:600;font-variant-numeric:tabular-nums}
.terms .pick{background:var(--ledger-soft)}
.terms .pick .num{color:var(--ledger)}
ul.plain{margin:.4rem 0 .9rem;padding-left:1.2rem}
ul.plain li{margin-bottom:.55rem}
ol.steps{margin:1rem 0 0;padding-left:1.4rem}
ol.steps li{margin-bottom:.85rem;padding-left:.35rem}
ol.steps li::marker{font-weight:600;color:var(--ledger)}
.choices{margin:1.2rem 0 0}
.choice{
  display:block;border:1px solid var(--hairline);
  padding:.9rem 1rem;margin-bottom:.7rem;cursor:pointer;
  background:var(--paper);
}
.choice:hover{background:var(--tint)}
.choice input{margin-right:.6rem;accent-color:var(--ledger)}
.choice.selected{border-color:var(--ledger);background:var(--ledger-soft)}
.choice strong{font-weight:600}
button.pd-btn{
  display:inline-block;background:var(--ledger);color:#fff;border:none;
  padding:.6rem 1.2rem;font-weight:600;font-size:.98rem;cursor:pointer;
  font-family:inherit;
}
button.pd-btn:disabled{background:var(--hairline);color:var(--muted);cursor:default}
button.pd-btn:focus-visible{outline:3px solid var(--ink);outline-offset:2px}
a.pd-mail{
  display:inline-block;background:var(--ledger);color:#fff;
  padding:.6rem 1.2rem;text-decoration:none;font-weight:600;font-size:.98rem;
}
.reveal{display:none;margin-top:1.6rem}
.reveal.open{display:block}
.checkbox-panel{
  border:1px solid var(--ink);padding:1.2rem 1.25rem;margin-top:1rem;
}
.checkbox-panel input[type=email]{
  font-family:inherit;font-size:1rem;padding:.55rem .7rem;
  border:1px solid var(--hairline);width:100%;max-width:340px;
  background:var(--paper);color:var(--ink);
}
.pd-form label.f{display:block;margin:.8rem 0 .25rem;font-weight:600}
.pd-form input[type=text],.pd-form input[type=email],.pd-form textarea{
  font-family:inherit;font-size:1rem;padding:.55rem .7rem;
  border:1px solid var(--hairline);width:100%;max-width:420px;
  background:var(--paper);color:var(--ink);box-sizing:border-box;
}
.pd-form textarea{min-height:5rem;max-width:100%}
.pd-form .pd-btn{margin-top:1rem}
.pd-thumb{display:block;width:200px;max-width:100%;margin-top:.6rem;border:1px solid #d9d6cf;border-radius:6px;cursor:zoom-in;box-shadow:0 1px 3px rgba(0,0,0,.08)}
#pd-zoom{display:none;position:fixed;inset:0;z-index:1000;background:rgba(0,0,0,.8);align-items:center;justify-content:center;cursor:zoom-out}
#pd-zoom.open{display:flex}
#pd-zoom img{max-width:92vw;max-height:92vh;object-fit:contain}
.sheet[data-role="cb"] .ff-only,.sheet[data-role="ff"] .cb-only{display:none}
.pd-role{display:flex;flex-wrap:wrap;gap:.5rem;margin:.2rem 0 1.4rem}
.pd-role label{cursor:pointer}
.pd-role input{position:absolute;opacity:0;width:1px;height:1px}
.pd-role span{display:inline-block;border:1px solid var(--hairline);border-radius:999px;padding:.45rem 1rem;background:var(--paper);font-size:.95rem}
.pd-role label:hover span{background:var(--tint)}
.pd-role input:checked + span{border-color:var(--ledger);background:var(--ledger-soft);color:var(--ledger);font-weight:600}
.pd-role input:focus-visible + span{outline:3px solid var(--ink);outline-offset:2px}
.pd-thanks{border:1px solid var(--ledger);background:var(--ledger-soft);padding:1.2rem 1.25rem;margin-top:1.2rem}
.pd-result{margin-top:.8rem;font-weight:600;min-height:1.4em}
.pd-result.ok{color:var(--ledger)}
.pd-result.taken{color:#7A2E1D}
.pd-result.firm{color:var(--warn)}
footer{
  margin-top:3.25rem;padding-top:1rem;border-top:1px solid var(--hairline);
  font-size:.82rem;color:var(--muted);line-height:1.55;
}
footer p{margin:0 0 .55rem}
@media (max-width:520px){
  .sheet{padding-top:2.25rem}
  h1{font-size:1.5rem}
  .letterhead{flex-direction:column;gap:.15rem}
  .letterhead .via{text-align:left}
}
@media (prefers-reduced-motion: reduce){
  *{transition:none!important;animation:none!important}
}
</style>
</head>
<body>
<div class="sheet" id="pd-sheet" data-role="cb">

  <div class="letterhead">
    <div class="firm">Gracia Group</div>
    <div class="via">Chad Gracia · Registered Representative, Rainmaker Securities, LLC</div>
  </div>

  <h1>I stopped working with brokers and finders a few years ago. I'm opening the door again — for a few.</h1>

  <p>Standard co-broker and finder agreements protect you for one trade and a 12-month tail. After that, nothing on paper ties the client to you. I know, because I've lost clients to other brokers that way and never saw a follow-up payment.</p>

  <p>The terms below are built so that doesn't happen to you. On the Partner track, you're paid on every trade your client does with me for 24 months.</p>

  <div class="pd-role" role="radiogroup" aria-label="Which describes you?">
    <label><input type="radio" name="pd-role" value="cb" checked><span>Co-broker — FINRA-registered, with a CRD#</span></label>
    <label><input type="radio" name="pd-role" value="ff"><span>Foreign finder — not U.S.-registered, based outside the U.S., introducing non-U.S. clients</span></label>
  </div>

  <p class="lede">I've closed nearly $200M in secondary trades — almost none of it with co-brokers. Not because the relationships weren't valuable, but because the process didn't work: constant back-and-forth as terms shifted, wasted hours and miscommunication, and more than once discovering — after all that — that I was already in touch with the referred client. Delays and misinformation made closing deals almost impossible. This program is built to solve those problems.</p>

  <p>And one thing up front, because it matters in our business: the term defines when I owe you money, not when I stop respecting where a relationship came from.</p>
  <p class="ff-only">A note on how this works for finders: U.S. rules let a foreign finder make the introduction, and nothing more. Your client must also be based outside the U.S. — a non-U.S. individual or an entity domiciled abroad. Once you've introduced them, I handle the relationship and the trades, and you're paid on the same schedule as a co-broker.</p>

  <h2>The problem as I see it</h2>

  <p>You have buyers and sellers who want names I trade, and I rarely bring co-brokers my best inventory because my own book is deep enough to close it. When brokers do work together, the standard process burns time and breeds miscommunication — both of which kill deals. The traditional arrangement seems designed to minimize closed trades.</p>

  <h2 class="cb-only">The dashboard</h2>
  <h2 class="ff-only">Your fee statement</h2>

  <p class="ff-only">Finders receive a statement of fees earned on each client they introduce.</p>

  <p class="cb-only">Everything in this program runs on my platform, and co-brokers get a dashboard for their registered clients: where each one stands, from onboarding paperwork (IQF) through every live trade — matched, introduced, LOI, transfer notice, SPA, wired. You're not asking me what happened; you're looking at it.</p>

  <h2>The two tracks</h2>

  <div class="termwrap">
  <table class="terms">
    <thead>
      <tr>
        <th scope="col"></th>
        <th scope="col" class="tier pick">Partner track</th>
        <th scope="col" class="tier">Referral track</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>In one line</td>
        <td class="pick"><span class="cb-only">We work together to close multiple deals with your client, and you're paid on all of them for two years.</span><span class="ff-only">You introduce your client once, and you're paid on their trades for two years.</span></td>
        <td><span class="cb-only">Your client sees only the trade under discussion, and I don't reach out to them otherwise during the tail.</span><span class="ff-only">You introduce your client for one trade, and you're paid on that trade.</span></td>
      </tr>
      <tr>
        <td>What your client sees</td>
        <td class="pick">My pages. Your client gets access to <a href="https://trades.graciagroup.com/" target="_blank" rel="noopener">my full trades page</a>: live indications, deal pages, auctions, and trade updates.__PD_IMG_PARTNER__</td>
        <td>My page for that one trade: a link only to its deal detail page, with no navigation or other buttons on top.<span class="cb-only"> You run the relationship.</span>__PD_IMG_REFERRAL__</td>
      </tr>
      <tr>
        <td>First trade</td>
        <td class="pick"><span class="num">50%</span> of my gross fee</td>
        <td><span class="num">33%</span> of my gross fee</td>
      </tr>
      <tr>
        <td>Follow-on trades</td>
        <td class="pick"><span class="num">50%</span> of the second trade, then <span class="num">33%</span> of every trade after — on any name in my book</td>
        <td>None — the referral covers the first trade only</td>
      </tr>
      <tr>
        <td>Term</td>
        <td class="pick"><span class="num">24 months</span> from the introduction, hard end</td>
        <td><span class="num">12 months</span> from the introduction, hard end</td>
      </tr>
      <tr class="ff-only">
        <td>Eligible clients</td>
        <td colspan="2">Non-U.S. individuals and entities domiciled outside the U.S. Your client receives a written description of your fee and signs an acknowledgment, and their trade confirmation notes that a finder's fee is paid under our agreement.</td>
      </tr>
      <tr>
        <td>What counts</td>
        <td class="pick">Any trade initiated before the term ends — transfer notice, LOI, purchase agreement, or confirmed order — pays out even if it closes after.</td>
        <td>Same, for the introduced trade.</td>
      </tr>
      <tr class="cb-only">
        <td>You're kept in</td>
        <td class="pick">CC on all correspondence through the first two trades; after that, email updates on every new deal and status change — and the dashboard, always.</td>
        <td>CC on the trade, start to finish.</td>
      </tr>
      <tr class="ff-only">
        <td>Your role</td>
        <td colspan="2">Your role ends at the introduction. You won't be copied on correspondence or involved in the trades that follow; you'll receive a statement of each fee as it's earned.</td>
      </tr>
    </tbody>
  </table>
  </div>

  <h2>Why the accounting holds up</h2>

  <p>You've probably lost clients after an introduction because you couldn't see what happened past the first trade — I have too. Here it works differently:</p>

  <ul class="plain">
    <li><strong>Trades happen on my rails.</strong> Indications, introductions, orders, and closings all run through my platform, so there's no trade that can quietly happen off the books.</li>
    <li class="cb-only"><strong>You see the record.</strong> For each client you register: the introduction date, the term clock, and every trade — with your fee accrued against it.</li>
    <li class="ff-only"><strong>You see what you're owed.</strong> A statement of each fee as it's earned on the clients you introduced.</li>
    <li><strong>You're paid when I'm paid.</strong> Your share is due concurrently with my commission, wired under the <span class="cb-only">fee-sharing agreement</span><span class="ff-only">finder agreement</span>.</li>
    <li><strong>It's Rainmaker paper.</strong> Rainmaker Securities' standard fee-sharing agreement for co-brokers, or its finder agreement for foreign finders — the same contracts they use for every co-broke — not a side letter with me.</li>
  </ul>

  <h2>How it works</h2>

  <ol class="steps">
    <li><strong>Check the name.</strong> The availability check runs entirely in your browser — before you've signed or told me anything. Details at the bottom of this page.</li>
    <li><span class="cb-only"><strong>Sign once.</strong> The master fee-sharing agreement — Rainmaker's form, countersigned by their president, naming no clients. This happens one time, ever.</span><span class="ff-only"><strong>Sign once.</strong> Rainmaker's finder agreement, confirming you're not required to register in the U.S. and that the arrangement is lawful where you are. This happens one time, ever.</span></li>
    <li><strong>Register the client.</strong> A one-page schedule names your client, your track, and the dates. Signed electronically in minutes.</li>
    <li><span class="cb-only"><strong>Make the introduction.</strong> A three-way email connects me, you, and your client, and states on its face that it's made under our agreement. When your client responds, the clock starts.</span><span class="ff-only"><strong>Make the introduction.</strong> A three-way email connects me, you, and your client. Your client receives the fee disclosure to acknowledge. This is where your part ends.</span></li>
    <li><span class="cb-only"><strong>Get paid.</strong> Your client trades; your share wires when my commission does. You see every entry.</span><span class="ff-only"><strong>Get paid.</strong> Your client trades; your fee wires when my commission does, and you receive a statement.</span></li>
  </ol>

  <h2>Who this is for</h2>

  <p>Co-brokers and foreign finders I've invited. Registered representatives and FINRA-member broker-dealers sign a fee-sharing agreement; finders outside the U.S. sign a finder agreement with slightly different terms, to comply with U.S. regulations. The payments are the same either way; the difference is that a finder's role ends at the introduction.</p>

  <div class="checkbox-panel">
    <h2 style="margin-top:0;padding-top:0;border-top:none">Name check — nothing leaves your browser</h2>
    <p class="small">Type a client's email address. The check runs locally in this page against an encrypted copy of my list — open your browser's developer tools (Network tab) and verify for yourself: nothing is transmitted, nothing is recorded. If the email is already in my book, you'll see it here, I never know you looked, and the conversation stops there. If the person is new to me but I have existing relationships at their firm, you'll see that too, and we agree the scope before you register. Otherwise: available, yours to register.</p>
    <input type="email" id="pd-check-email" placeholder="client@example.com" autocomplete="off">
    <button class="pd-btn" id="pd-check-btn" style="margin-left:.4rem" disabled>Loading…</button>
    <div class="pd-result" id="pd-check-result"></div>
  </div>

  <h2>Which of these would you consider?</h2>

__PD_FORM__

  <footer>
    <p>This page is a summary for discussion with professional intermediaries and is not an offer to buy or sell securities, investment advice, or a solicitation directed at investors. All terms are subject to Rainmaker Securities, LLC review and to an executed fee-sharing or finder agreement, which governs in full. Fee sharing is available only where permitted by applicable law and FINRA rules, including registration requirements. Securities transactions are conducted through Rainmaker Securities, LLC, member FINRA/SIPC.</p>
    <p>Private link — please don't circulate.</p>
  </footer>

</div>
<div id="pd-zoom"><img alt=""></div>
<script>
(function(){
  var sheet = document.getElementById('pd-sheet');
  var roleField = document.getElementById('pd-role-field');
  var country = document.getElementById('pd-country');
  var radios = document.querySelectorAll('input[name=pd-role]');
  function setRole(r){
    r = (r === 'ff') ? 'ff' : 'cb';
    sheet.setAttribute('data-role', r);
    radios.forEach(function(x){ x.checked = (x.value === r); });
    if (roleField) roleField.value = r;
    if (country) country.required = (r === 'ff');
  }
  radios.forEach(function(x){ x.addEventListener('change', function(){ if (x.checked) setRole(x.value); }); });
  setRole(new URLSearchParams(location.search).get('role'));
})();
(function(){
  var zoom = document.getElementById('pd-zoom');
  var zimg = zoom.querySelector('img');
  document.querySelectorAll('.pd-thumb').forEach(function(t){
    t.addEventListener('click', function(){
      zimg.src = t.src; zimg.alt = t.alt;
      zoom.classList.add('open');
    });
  });
  zoom.addEventListener('click', function(){ zoom.classList.remove('open'); });
  document.addEventListener('keydown', function(e){ if (e.key === 'Escape') zoom.classList.remove('open'); });
})();
document.querySelectorAll('input[name=option]').forEach(function(r){
  r.addEventListener('change', function(){
    document.querySelectorAll('.choice').forEach(function(c){c.classList.remove('selected');});
    r.closest('.choice').classList.add('selected');
  });
});
(function(){
  var EMAILS = new Set();
  var DOMAINS = new Set();
  var SALT = 'gracia-partner-check-v1';
  var btn = document.getElementById('pd-check-btn');
  var out = document.getElementById('pd-check-result');
  var OFFLINE = 'The check is temporarily offline — email me the name instead.';
  fetch('?view=partner-desk&hashes=1').then(function(r){
    if (!r.ok) throw new Error('HTTP ' + r.status);
    return r.json();
  }).then(function(d){
    (d.emails || []).forEach(function(h){ EMAILS.add(h); });
    (d.domains || []).forEach(function(h){ DOMAINS.add(h); });
    if (EMAILS.size === 0) throw new Error('empty');
    btn.textContent = 'Check';
    btn.disabled = false;
  }).catch(function(){
    btn.textContent = 'Check';
    out.className = 'pd-result';
    out.textContent = OFFLINE;
  });
  async function pdHash(s){
    var data = new TextEncoder().encode(SALT + s);
    var buf = await crypto.subtle.digest('SHA-256', data);
    return Array.from(new Uint8Array(buf)).map(function(b){return b.toString(16).padStart(2,'0');}).join('').slice(0,16);
  }
  async function runCheck(){
    var input = document.getElementById('pd-check-email');
    var val = (input.value || '').trim();
    out.className = 'pd-result';
    if (!val || val.indexOf('@') < 0) { out.textContent = 'Enter a full email address.'; return; }
    if (EMAILS.size === 0) { out.textContent = OFFLINE; return; }
    if (!window.crypto || !crypto.subtle) { out.textContent = 'Your browser does not support the local check — email me the name instead.'; return; }
    var norm = val.trim().toLowerCase();
    var h = await pdHash(norm);
    if (EMAILS.has(h)) {
      out.textContent = 'Already in my book — the conversation stops there. I never know you looked.';
      out.className = 'pd-result taken';
    } else {
      var dh = await pdHash('d:' + norm.split('@')[1]);
      if (DOMAINS.has(dh)) {
        out.textContent = 'This person is new to me, but I have existing relationships at their firm. Email me before registering and we agree the scope up front.';
        out.className = 'pd-result firm';
      } else {
        out.textContent = 'Available — this one is yours to register. And like every check, the name never left your browser: I cannot see it, now or ever.';
        out.className = 'pd-result ok';
      }
    }
  }
  btn.addEventListener('click', runCheck);
  document.getElementById('pd-check-email').addEventListener('keydown', function(e){ if (e.key === 'Enter' && !btn.disabled) runCheck(); });
})();
</script>
</body>
</html>
"""


def _pd_thumb(filename, alt):
    """<img class=pd-thumb> with the file inlined as a data: URI, or '' (with
    a warning) if it can't be read — the page never fails over an image."""
    try:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pd-img', filename)
        with open(path, 'rb') as f:
            data = base64.b64encode(f.read()).decode('ascii')
        return (f'<img class="pd-thumb" src="data:image/jpeg;base64,{data}" '
                f'alt="{html_mod.escape(alt, quote=True)}" loading="lazy">')
    except Exception as e:
        logger.warning(f"partner-desk thumbnail {filename} unavailable: {e}")
        return ''


PARTNER_DESK_HTML = (PARTNER_DESK_HTML
    .replace('__PD_IMG_PARTNER__', _pd_thumb('partner-trades.jpg', 'What the client sees: the full trades page'))
    .replace('__PD_IMG_REFERRAL__', _pd_thumb('referral-deal.jpg', 'What the client sees: a single deal detail page')))

PD_CHECKER_ENABLED = True  # False disables the Pipeline-backed name check (_partner_desk_hash_sets)
PD_REPLIES_KEY = 'partner-desk-replies.json'
PD_REPLY_TO = 'cgracia@rainmakersecurities.com'
PD_REPLY_FROM = 'agent@agent.graciagroup.com'
PD_OPTIONS = ('Partner track', 'Referral track', 'Neither')

PARTNER_DESK_FORM = """  <form class="pd-form" method="post" action="?view=partner-desk">
  <input type="hidden" name="action" value="reply">
  <input type="hidden" name="role" id="pd-role-field" value="cb">
  <div class="choices">
    <label class="choice"><input type="radio" name="option" value="Partner track" required><strong>Partner track</strong> — 50% of my fee on your client's first two trades, then 33% of every trade they do for the rest of the two years, on any name in my book.</label>
    <label class="choice"><input type="radio" name="option" value="Referral track"><strong>Referral track</strong> — 33% of my fee on the trade you introduce; your client sees only that trade, and I don't reach out to them during the 12-month tail.</label>
    <label class="choice"><input type="radio" name="option" value="Neither"><strong>Neither</strong> — understood; these two tracks are the only way I work with co-brokers now.</label>
  </div>
  <label class="f" for="pd-name">Name</label>
  <input type="text" id="pd-name" name="name" required autocomplete="name">
  <label class="f" for="pd-email">Email</label>
  <input type="email" id="pd-email" name="email" required autocomplete="email" value="__PD_EMAIL__">
  <label class="f" for="pd-firm">Firm</label>
  <input type="text" id="pd-firm" name="firm" required autocomplete="organization">
  <div class="cb-only">
  <label class="f" for="pd-crd">CRD# (optional)</label>
  <input type="text" id="pd-crd" name="crd" inputmode="numeric">
  </div>
  <div class="ff-only">
  <label class="f" for="pd-country">Country</label>
  <input type="text" id="pd-country" name="country" autocomplete="country-name">
  </div>
  <label class="f" for="pd-comment">Comment (optional)</label>
  <textarea id="pd-comment" name="comment"></textarea>
  <button class="pd-btn" type="submit">Send my answer</button>
  </form>"""

PARTNER_DESK_THANKS = """  <div class="pd-thanks"><p style="margin:0">Thank you — I've got your answer and will be in touch.</p></div>"""


def _pd_handle_reply(event):
    """Record a partner-desk form POST: append to S3, email via SES. Any
    failure is logged; the caller shows the thank-you regardless."""
    body = event.get('body') or ''
    if event.get('isBase64Encoded'):
        try:
            body = base64.b64decode(body).decode('utf-8')
        except Exception as e:
            logger.error(f"partner-desk reply body decode failed: {e}")
            body = ''
    form = {k: (v[0] if v else '').strip()[:4000]
            for k, v in urllib.parse.parse_qs(body, keep_blank_values=True).items()}
    http = (event.get('requestContext') or {}).get('http') or {}
    headers = event.get('headers') or {}
    reply = {
        'ts': datetime.now(timezone.utc).isoformat(),
        'name': form.get('name', ''),
        'email': form.get('email', ''),
        'firm': form.get('firm', ''),
        'option': form.get('option', ''),
        'comment': form.get('comment', ''),
        'role': 'ff' if form.get('role') == 'ff' else 'cb',
        'crd': form.get('crd', ''),
        'country': form.get('country', ''),
        'b': (event.get('queryStringParameters') or {}).get('b', ''),
        'ip': http.get('sourceIp') or (event.get('requestContext') or {}).get('identity', {}).get('sourceIp', ''),
        'user_agent': http.get('userAgent') or headers.get('user-agent') or headers.get('User-Agent') or '',
    }
    logger.info(f"partner-desk reply: {json.dumps(reply)}")
    try:
        s3 = boto3.client('s3')
        try:
            obj = s3.get_object(Bucket=AUCTIONS_BUCKET, Key=PD_REPLIES_KEY)
            replies = json.loads(obj['Body'].read().decode('utf-8'))
        except s3.exceptions.NoSuchKey:
            replies = []
        replies.append(reply)
        s3.put_object(Bucket=AUCTIONS_BUCKET, Key=PD_REPLIES_KEY,
                      Body=json.dumps(replies, indent=2).encode('utf-8'),
                      ContentType='application/json')
    except Exception as e:
        logger.error(f"partner-desk reply S3 write failed: {e}")
    try:
        text = '\n'.join(f"{k}: {reply[k]}" for k in
                         ('name', 'email', 'firm', 'role', 'crd', 'country', 'option', 'comment', 'b', 'ts', 'ip', 'user_agent'))
        msg = {'Source': PD_REPLY_FROM,
               'Destination': {'ToAddresses': [PD_REPLY_TO]},
               'Message': {'Subject': {'Data': f"Partner Desk reply: {reply['name'] or '(no name)'} — {reply['option'] or '(no option)'} ({'Foreign finder' if reply['role'] == 'ff' else 'Co-broker'})"},
                           'Body': {'Text': {'Data': text}}}}
        if '@' in reply['email']:
            msg['ReplyToAddresses'] = [reply['email']]
        boto3.client('ses', region_name='us-east-1').send_email(**msg)
    except Exception as e:
        logger.error(f"partner-desk reply SES send failed: {e}")



def _live_auctions_for_nav():
    """[(auction_id, auction_dict), ...] for every open auction, read fresh
    from S3 each request. Open means no close_date, or a close_date of
    today or later (same rule as CRMDealDetails.live_auction_for_deal).
    Any read/parse failure returns [] so the nav's Auctions tab just
    doesn't render rather than breaking the page."""
    try:
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=AUCTIONS_BUCKET, Key=AUCTIONS_KEY)
        data = json.loads(obj['Body'].read())
        auctions = data.get('auctions') or {}
        today = datetime.now().strftime('%Y-%m-%d')
        live = []
        for aid, auc in auctions.items():
            close_date = (auc.get('close_date') or '').strip()
            if not close_date or close_date >= today:
                live.append((aid, auc))
        return live
    except Exception as e:
        logger.warning(f"Auctions nav tab: auctions.json load failed (non-fatal): {e}")
        return []


def _nav_login_url(dest):
    """Cognito hosted-UI login URL carrying `dest` (bare, no token) as
    base64url state, so the code-exchange leg lands there with a fresh sso
    token appended once the visitor signs in."""
    state = base64.urlsafe_b64encode(dest.encode()).decode().rstrip('=')
    return (
        "https://us-east-1dsttcaqx7.auth.us-east-1.amazoncognito.com/login"
        f"?client_id={COGNITO_CLIENT_ID}&response_type=code&scope=openid+email"
        f"&redirect_uri={COGNITO_REDIRECT_URI}"
        f"&state={urllib.parse.quote(state, safe='')}"
    )


def _render_top_nav(event, is_admin=False):
    """The shared client-facing top nav: brand, tabs (Indications, Portfolio
    & Watchlist, two placeholder tabs, Auctions when at least one is live,
    My Dashboard for eligible tenants), and the account control on the
    right. Any failure building an optional tab must not break the rest
    of the nav or the page."""
    email = _read_identity_email(event)

    # Portfolio & Watchlist: admin identity wins outright, same rule the old
    # standalone button used — an admin already holds a year-long session on
    # the desk domain, so a plain link works regardless of gg_id.
    if is_admin:
        portfolio_href = DESK_URL + "/?view=admin"
    elif email:
        pw_token = _make_handoff_token(email)
        portfolio_href = f"{DESK_URL}/?sso={urllib.parse.quote(pw_token, safe='')}"
    else:
        portfolio_href = _nav_login_url(DESK_URL + "/")

    # Demand Board: same auth-aware link pattern as Portfolio & Watchlist above,
    # just a different destination.
    if is_admin:
        demand_href = DESK_URL + "/?view=demand"
    elif email:
        demand_token = _make_handoff_token(email)
        demand_href = f"{DESK_URL}/?view=demand&sso={urllib.parse.quote(demand_token, safe='')}"
    else:
        demand_href = _nav_login_url(DESK_URL + "/?view=demand")

    auctions_tab = ""
    try:
        live = _live_auctions_for_nav()
        if live:
            auc_dest = f"{DESK_URL}/?view=auctions"
            if email:
                auc_token = _make_handoff_token(email)
                auc_sep = '&' if '?' in auc_dest else '?'
                auc_href = f"{auc_dest}{auc_sep}sso={urllib.parse.quote(auc_token, safe='')}"
            else:
                auc_href = _nav_login_url(auc_dest)
            auctions_tab = (
                f'<a href="{auc_href}" target="_blank" rel="noopener" class="nav-tab">'
                f'Auctions ({len(live)})</a>'
            )
    except Exception as e:
        logger.warning(f"Auctions nav tab failed (non-fatal): {e}")
        auctions_tab = ""

    dashboard_tab = ""
    try:
        if email and email.strip().lower() in _syndicate_eligible_emails():
            dash_token = _make_handoff_token(email)
            dash_href = f"{SYNDICATE_DASH_URL}/?sso={urllib.parse.quote(dash_token, safe='')}"
            dashboard_tab = f'<a href="{dash_href}" target="_blank" rel="noopener" class="nav-tab">My Dashboard</a>'
    except Exception as e:
        logger.warning(f"My Dashboard nav tab failed (non-fatal): {e}")
        dashboard_tab = ""

    if email:
        safe_email = html_mod.escape(email, quote=True)
        account_html = (
            '<div class="navacct" tabindex="0">'
            '<span class="navacct-trigger">My Account &#9662;</span>'
            '<div class="navacct-menu">'
            f'<div class="navacct-item navacct-static">Signed in as {safe_email}</div>'
            '<div class="navacct-item navacct-disabled" title="Coming soon">Profile &mdash; coming soon</div>'
            '<a class="navacct-item" href="https://trades.graciagroup.com/?signout=1">Sign out</a>'
            '</div></div>'
        )
    else:
        cur_path = (event.get('rawPath')
                    or (event.get('requestContext') or {}).get('http', {}).get('path') or '/')
        cur_qs = event.get('rawQueryString') or ''
        cur_url = COGNITO_REDIRECT_URI + cur_path + (('?' + cur_qs) if cur_qs else '')
        account_html = f'<a href="{_nav_login_url(cur_url)}" class="btn nav-signin">Sign In</a>'

    # Admin-only quick-switcher trigger (Cmd/Ctrl+K also opens it — see the
    # deal-switcher script). Omitted entirely for non-admins, same as the
    # rest of this feature.
    deal_switcher_btn = ''
    if is_admin:
        deal_switcher_btn = (
            '<button type="button" id="dealSwitcherBtn" class="nav-icon-btn" '
            'title="Switch deal (Ctrl+K)" aria-label="Switch deal">'
            '<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false">'
            '<circle cx="6.5" cy="6.5" r="4.5" fill="none" stroke="currentColor" stroke-width="1.4"></circle>'
            '<line x1="9.8" y1="9.8" x2="14" y2="14" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"></line>'
            '</svg></button>'
        )

    return (
        '<nav class="topnav">'
        '<a href="https://www.graciagroup.com" class="nav-brand">Gracia Group</a>'
        '<div class="nav-tabs">'
        '<a href="https://trades.graciagroup.com/" class="nav-tab">Indications</a>'
        f'<a href="{portfolio_href}" target="_blank" rel="noopener" class="nav-tab">Portfolio &amp; Watchlist</a>'
        '<span class="nav-tab nav-tab-disabled" title="Coming soon">Introductions</span>'
        f'<a href="{demand_href}" target="_blank" rel="noopener" class="nav-tab">Demand Board</a>'
        + auctions_tab
        + dashboard_tab
        + '</div>'
        + deal_switcher_btn
        + account_html
        + '</nav>'
    )


def _render_deal_switcher_modal():
    """Admin-only global 'Switch Deal' quick-switcher: modal markup plus its
    script. Only ever called (and only ever appears in the rendered page)
    for an admin session — see the _is_admin check where this is invoked.
    Fetches /?view=admin-deal-index once per page load and caches the
    result in memory; all filtering after that is client-side. Reuses
    copyTextToClipboard/COPY_ICON_SVG/CHECK_ICON_SVG, which are declared
    earlier in the page's own <script> block."""
    return '''
        <div id="dealSwitcherModal" class="deal-switcher-overlay">
            <div class="deal-switcher-box">
                <input type="text" id="dealSwitcherInput" class="deal-switcher-input" placeholder="Search a company&hellip;" autocomplete="off" spellcheck="false">
                <div id="dealSwitcherResults" class="deal-switcher-results"></div>
            </div>
        </div>
        <script>
        (function () {
            var cache = null;
            var modal = document.getElementById('dealSwitcherModal');
            var input = document.getElementById('dealSwitcherInput');
            var results = document.getElementById('dealSwitcherResults');
            var btn = document.getElementById('dealSwitcherBtn');
            var rows = [];
            var idx = -1;

            function fmtSize(v) {
                if (v === null || v === undefined || v === '') return '';
                var n = Number(v);
                if (!isFinite(n)) return '';
                if (n >= 1000000) {
                    var m = Math.round((n / 1000000) * 10) / 10;
                    return '$' + (m % 1 === 0 ? m.toFixed(0) : m) + 'M';
                }
                if (n >= 1000) return '$' + Math.round(n / 1000) + 'K';
                return '$' + Math.round(n);
            }
            function fmtRange(lo, hi) {
                var a = fmtSize(lo), b = fmtSize(hi);
                if (a && b) return a === b ? a : (a + '\\u2013' + b);
                return a || b || '';
            }
            function esc(s) {
                return String(s === null || s === undefined ? '' : s).replace(/[&<>"']/g, function (c) {
                    return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c];
                });
            }
            function renderRows(list) {
                rows = list;
                idx = list.length ? 0 : -1;
                if (!list.length) {
                    results.innerHTML = '<div class="deal-switcher-empty">No matching deals.</div>';
                    return;
                }
                results.innerHTML = list.map(function (d, i) {
                    var label = esc(d.company) + ' &middot; ' + esc(d.side) + ' ' + esc(fmtRange(d.size_min, d.size_max)) + ' &middot; ' + esc(d.status);
                    return '<div class="deal-switcher-row' + (i === 0 ? ' active' : '') + '" data-idx="' + i + '" data-id="' + esc(d.id) + '">' +
                        '<span class="deal-switcher-label">' + label + '</span>' +
                        '<button type="button" class="deal-switcher-copy" data-id="' + esc(d.id) + '" title="Copy link" aria-label="Copy link">' + COPY_ICON_SVG + '</button>' +
                        '</div>';
                }).join('');
            }
            function setActive(newIdx) {
                var rowEls = results.querySelectorAll('.deal-switcher-row');
                if (!rowEls.length) return;
                idx = Math.max(0, Math.min(newIdx, rowEls.length - 1));
                rowEls.forEach(function (el, i) {
                    el.classList.toggle('active', i === idx);
                });
                rowEls[idx].scrollIntoView({block: 'nearest'});
            }
            function applyFilter() {
                var q = input.value.trim().toLowerCase();
                var list = !cache ? [] : (!q ? cache : cache.filter(function (d) {
                    return (d.company || '').toLowerCase().indexOf(q) !== -1;
                }));
                renderRows(list);
            }
            function goToDeal(id) {
                window.location.href = 'https://trades.graciagroup.com/deal/' + encodeURIComponent(id);
            }
            function openModal() {
                modal.classList.add('show');
                input.value = '';
                results.innerHTML = '';
                setTimeout(function () { input.focus(); }, 0);
                if (cache) {
                    renderRows(cache);
                    return;
                }
                results.innerHTML = '<div class="deal-switcher-empty">Loading&hellip;</div>';
                fetch('/?view=admin-deal-index', {credentials: 'same-origin'})
                    .then(function (r) { if (!r.ok) throw new Error('bad status'); return r.json(); })
                    .then(function (data) {
                        cache = (data || []).slice().sort(function (a, b) {
                            return new Date(b.updated || 0) - new Date(a.updated || 0);
                        });
                        renderRows(cache);
                    })
                    .catch(function () {
                        results.innerHTML = '<div class="deal-switcher-empty">Failed to load deals.</div>';
                    });
            }
            function closeModal() {
                modal.classList.remove('show');
            }

            if (btn) btn.addEventListener('click', openModal);
            document.addEventListener('keydown', function (e) {
                if ((e.metaKey || e.ctrlKey) && (e.key === 'k' || e.key === 'K')) {
                    e.preventDefault();
                    openModal();
                    return;
                }
                if (!modal.classList.contains('show')) return;
                if (e.key === 'Escape') {
                    closeModal();
                } else if (e.key === 'ArrowDown') {
                    e.preventDefault();
                    setActive(idx + 1);
                } else if (e.key === 'ArrowUp') {
                    e.preventDefault();
                    setActive(idx - 1);
                } else if (e.key === 'Enter') {
                    e.preventDefault();
                    if (idx >= 0 && rows[idx]) goToDeal(rows[idx].id);
                }
            });
            input.addEventListener('input', applyFilter);
            modal.addEventListener('click', function (e) {
                if (e.target === modal) closeModal();
            });
            results.addEventListener('click', function (e) {
                var copyBtn = e.target.closest ? e.target.closest('.deal-switcher-copy') : null;
                if (copyBtn) {
                    e.stopPropagation();
                    var id = copyBtn.getAttribute('data-id');
                    copyTextToClipboard('https://trades.graciagroup.com/deal/' + id).then(function () {
                        copyBtn.innerHTML = CHECK_ICON_SVG;
                        copyBtn.title = 'Copied';
                        clearTimeout(copyBtn._copyTimer);
                        copyBtn._copyTimer = setTimeout(function () {
                            copyBtn.innerHTML = COPY_ICON_SVG;
                            copyBtn.title = 'Copy link';
                        }, 1200);
                    });
                    return;
                }
                var row = e.target.closest ? e.target.closest('.deal-switcher-row') : null;
                if (row) goToDeal(row.getAttribute('data-id'));
            });
            results.addEventListener('mousemove', function (e) {
                var row = e.target.closest ? e.target.closest('.deal-switcher-row') : null;
                if (!row) return;
                setActive(parseInt(row.getAttribute('data-idx'), 10));
            });
        })();
        </script>
    '''


def _get_http_method(event):
    """Resolve the HTTP method from a Lambda Function URL, API Gateway event,
    or a raw invoke payload.

    If the event is the raw POST body itself (a dict with a top-level 'query'
    key and none of the standard HTTP framing fields), we treat it as a POST
    search request — this supports direct Lambda invokes / integrations that
    skip the HTTP wrapper entirely.
    """
    rc = event.get('requestContext') or {}
    http = rc.get('http') or {}
    if http.get('method'):
        return http['method']
    if event.get('httpMethod'):
        return event['httpMethod']
    if isinstance(event, dict) and 'query' in event:
        return 'POST'
    return 'GET'


def _load_deals_from_s3():
    """Read pipeline_deals.json from S3 and return the parsed list."""
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket='pipeline-public-deal-data', Key='pipeline_deals.json')
    return json.loads(response['Body'].read().decode('utf-8'))


def _load_directory_companies():
    """Read the small directory_companies.json (a few KB) written by crm-snapshot.
    Returns (highlight_names, list_names). Read-only; any failure returns empty
    lists so the page still renders exactly as before."""
    try:
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket='full-pipeline-cache', Key='directory_companies.json')
        data = json.loads(obj['Body'].read().decode('utf-8'))
        return data.get('highlight', []), data.get('list', []), data.get('pricing', {})
    except Exception as e:
        logger.error(f"Directory companies load failed (non-fatal): {e}")
        return [], [], {}


def _call_claude_for_matching_ids(query, deals):
    """Return deal IDs matching the user's natural-language query.

    Two-step pipeline:
      1. Ask Claude to parse the query into a structured filter object. Only
         the query text is sent to Claude — the deals data is never in the
         model's context, which makes the request tiny (< 1K tokens total)
         and keeps the filter step deterministic.
      2. Apply those filters in pure Python against the full deals list.
    """
    filters = _extract_filters_from_query(query)
    logger.info("Extracted filters for query %r: %s", query, filters)
    return _apply_filters(deals, filters)


def _extract_filters_from_query(query):
    """Step 1: call Claude with ONLY the user's query (no deals data) and
    have it return a structured filter object via forced tool_use."""
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")

    payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 1024,
        "system": (
            "You convert natural-language queries about private secondary market "
            "deals into a structured filter object. You will be given a user's "
            "query and must call the extract_filters tool with the appropriate "
            "field values. Every field is optional — OMIT any field the user "
            "did not explicitly specify. Do not guess or infer unmentioned "
            "dimensions.\n"
            "\n"
            "FIELD GUIDE:\n"
            "- company: Company name, normalized to the standard form. Examples: "
            "'Space X' / 'spacex' -> 'SpaceX'. 'stripe' -> 'Stripe'. Only set "
            "this if the user named a specific company.\n"
            "- company_industry: A single broad sector/industry keyword when "
            "the user mentions a theme (e.g. 'robotics', 'AI', 'drones', "
            "'fintech', 'defense', 'space', 'healthcare'). Use one keyword "
            "that is likely to appear in a company's industry tags. Only set "
            "this when the user referenced a sector or theme — NOT when they "
            "named a specific company (use the company field for that). "
            "Matched as a case-insensitive substring against the deal's "
            "company_industry field.\n"
            "- type: 'Buy Order' if the user asked about bids / buys / buying / "
            "buyers / bidders. 'Sell Order' if the user asked about offers / "
            "asks / sells / selling / sellers. Otherwise omit.\n"
            "- structure: 'Direct' for direct / no SPV / no wrapper / single "
            "layer / cap-table. 'Fund' for SPV / fund / wrapper. 'Forward' for "
            "forward contracts. Otherwise omit.\n"
            "- min_size_max: Maximum acceptable value of a deal's min_deal_size "
            "(the smallest ticket the deal requires). 'I have $500K' or 'deals "
            "I can do with $500K' or 'ticket under $500K' -> 500000. The user's "
            "budget must be >= the deal's minimum to participate.\n"
            "- carry_max: Maximum carry percentage. 'no carry' / 'zero carry' -> "
            "0. 'low carry' -> 10. 'carry under 15%' -> 15.\n"
            "- management_fee_max: Maximum management fee percentage. 'no "
            "management fee' / 'no mgmt fee' -> 0. 'low mgmt fee' -> 1.\n"
            "- gross_max: Maximum gross price per share in USD. 'gross under "
            "$100' -> 100.\n"
            "- gross_min: Minimum gross price per share in USD. 'gross over "
            "$50' -> 50.\n"
            "- series: Share series / round. Examples: 'series B' / 'B round' "
            "-> 'B'. 'series A' -> 'A'. 'seed' / 'seed round' -> 'Seed'. "
            "'mixed series' -> 'Mixed'. 'N/A' for deals with no series. Only "
            "set this if the user named a specific series/round.\n"
            "- class: Share class. 'common' / 'common shares' / 'common stock' "
            "-> 'Common'. 'preferred' / 'preferred stock' -> 'Preferred'. "
            "'mixed class' -> 'Mixed'. 'any class' -> 'Any'. Otherwise omit.\n"
            "- layers: SPV structure layering. 'on cap table' / 'cap table' / "
            "'single layer' -> 'SPV on cap table'. '2 layers' / 'two layer' / "
            "'2-layer' -> '2-Layer SPV'. '3 layers' / 'three layer' / "
            "'3-layer' -> '3-Layer SPV'. Otherwise omit.\n"
            "- stage: Deal stage. 'firm' / 'firm only' / 'confirmed details' "
            "-> 'Firm'. 'inquiry' / 'inquiries' -> 'Inquiry'. 'confirm' / "
            "'will confirm' -> 'Confirm'. Otherwise omit.\n"
            "- seller_fee_max: Max seller fee percentage. 'no seller fee' -> "
            "0. 'low seller fee' -> 1. 'seller fee under 2%' -> 2.\n"
            "- partner_fee_max: Max partner fee percentage. 'no partner fee' "
            "-> 0. 'low partner fee' -> 1. 'partner fee under 2%' -> 2.\n"
            "- sort: Set ONLY when the user uses a clear superlative. "
            "'gross_asc' for cheapest / lowest price. 'gross_desc' for most "
            "expensive / highest price. 'min_deal_size_asc' for smallest "
            "ticket / smallest minimum. 'max_deal_size_desc' for largest / "
            "biggest deal. 'updated_desc' for most recent / newest / latest. "
            "'carry_asc' for lowest carry / lowest fees. Omit otherwise.\n"
            "\n"
            "Only fill fields the user explicitly specified. Leave everything "
            "else out of the tool call."
        ),
        "tools": [
            {
                "name": "extract_filters",
                "description": (
                    "Record the structured filter values parsed from the user's "
                    "query. Only include fields the user explicitly specified — "
                    "omit all others."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "company": {
                            "type": "string",
                            "description": "Company name in standard form, e.g. 'SpaceX'.",
                        },
                        "company_industry": {
                            "type": "string",
                            "description": (
                                "Single broad sector/industry keyword "
                                "(e.g. 'robotics', 'AI', 'drones', "
                                "'fintech'). Matched as a case-insensitive "
                                "substring against the deal's "
                                "company_industry field. Do NOT set this "
                                "when the user named a specific company; "
                                "use 'company' instead."
                            ),
                        },
                        "type": {
                            "type": "string",
                            "enum": ["Buy Order", "Sell Order"],
                            "description": "'Buy Order' for bids, 'Sell Order' for offers.",
                        },
                        "structure": {
                            "type": "string",
                            "enum": ["Direct", "Fund", "Forward"],
                            "description": "'Direct' = no wrapper, 'Fund' = SPV, 'Forward' = forward contract.",
                        },
                        "min_size_max": {
                            "type": "number",
                            "description": "Max acceptable min_deal_size in USD. e.g. 'I have $500K' -> 500000.",
                        },
                        "carry_max": {
                            "type": "number",
                            "description": "Max carry percentage. 'no carry' -> 0, 'low carry' -> 10.",
                        },
                        "management_fee_max": {
                            "type": "number",
                            "description": "Max management fee percentage. 'no mgmt fee' -> 0.",
                        },
                        "gross_max": {
                            "type": "number",
                            "description": "Max gross price per share in USD.",
                        },
                        "gross_min": {
                            "type": "number",
                            "description": "Min gross price per share in USD.",
                        },
                        "series": {
                            "type": "string",
                            "description": (
                                "Share series/round, e.g. 'A', 'B', 'C', "
                                "'Seed', 'Mixed', 'N/A'. Matched as a "
                                "case-insensitive substring."
                            ),
                        },
                        "class": {
                            "type": "string",
                            "enum": ["Common", "Preferred", "Mixed", "Any"],
                            "description": "Share class.",
                        },
                        "layers": {
                            "type": "string",
                            "enum": [
                                "SPV on cap table",
                                "2-Layer SPV",
                                "3-Layer SPV",
                            ],
                            "description": (
                                "SPV layering. Matched as a case-insensitive "
                                "substring."
                            ),
                        },
                        "stage": {
                            "type": "string",
                            "enum": ["Firm", "Inquiry", "Confirm"],
                            "description": (
                                "Deal stage. 'Firm' = details confirmed, "
                                "'Inquiry' = awaiting data, 'Confirm' = will "
                                "confirm after bid/ask."
                            ),
                        },
                        "seller_fee_max": {
                            "type": "number",
                            "description": "Max seller fee percentage. 'no seller fee' -> 0.",
                        },
                        "partner_fee_max": {
                            "type": "number",
                            "description": "Max partner fee percentage. 'no partner fee' -> 0.",
                        },
                        "sort": {
                            "type": "string",
                            "enum": [
                                "gross_asc",
                                "gross_desc",
                                "min_deal_size_asc",
                                "max_deal_size_desc",
                                "updated_desc",
                                "carry_asc",
                            ],
                            "description": (
                                "Set only when the user uses a clear "
                                "superlative. 'gross_asc' = cheapest, "
                                "'gross_desc' = most expensive, "
                                "'min_deal_size_asc' = smallest minimum, "
                                "'max_deal_size_desc' = largest deal, "
                                "'updated_desc' = most recent, "
                                "'carry_asc' = lowest carry."
                            ),
                        },
                    },
                    "required": [],
                },
            }
        ],
        "tool_choice": {"type": "tool", "name": "extract_filters"},
        "messages": [{"role": "user", "content": query}],
    }

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode('utf-8'),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            response_body = resp.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8', errors='replace')
        logger.error(f"Anthropic API HTTP {e.code}: {error_body}")
        raise RuntimeError(f"Anthropic API returned HTTP {e.code}: {error_body}") from e
    except urllib.error.URLError as e:
        logger.error(f"Anthropic API connection error: {e.reason}")
        raise RuntimeError(f"Anthropic API connection error: {e.reason}") from e

    try:
        result = json.loads(response_body)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON from Anthropic API: {response_body[:500]}")
        raise RuntimeError("Invalid JSON response from Anthropic API") from e

    usage = result.get('usage') or {}
    logger.info(
        "Claude filter extraction usage: input=%s cache_read=%s cache_create=%s output=%s",
        usage.get('input_tokens'),
        usage.get('cache_read_input_tokens'),
        usage.get('cache_creation_input_tokens'),
        usage.get('output_tokens'),
    )

    for block in result.get('content') or []:
        if block.get('type') == 'tool_use' and block.get('name') == 'extract_filters':
            return block.get('input') or {}

    return {}


# Layer-hierarchy ordering used by _apply_filters. For Sell Orders a
# request for more layers is satisfied by any offer at the same or
# lower level (a seller on cap table / single-layer SPV can always be
# wrapped into a deeper structure downstream).
_LAYER_LEVELS = {
    'spv on cap table': 1,
    '2-layer spv': 2,
    '3-layer spv': 3,
}


def _to_float(value):
    """Best-effort numeric coercion; returns None if the value can't be
    parsed as a float (None, empty string, non-numeric text, etc)."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _apply_filters(deals, filters):
    """Step 2: apply a structured filter dict to the full deals list and
    return the IDs of deals that match every non-null criterion. Missing /
    null fields in the filter dict are treated as 'no filter on this
    dimension'."""
    company = filters.get('company')
    if isinstance(company, str):
        company = company.strip().lower() or None
    else:
        company = None

    company_industry = filters.get('company_industry')
    if isinstance(company_industry, str):
        company_industry = company_industry.strip().lower() or None
    else:
        company_industry = None

    deal_type = filters.get('type')
    structure = filters.get('structure')
    min_size_max = filters.get('min_size_max')
    carry_max = filters.get('carry_max')
    management_fee_max = filters.get('management_fee_max')
    gross_max = filters.get('gross_max')
    gross_min = filters.get('gross_min')

    series = filters.get('series')
    if isinstance(series, str):
        series = series.strip().lower() or None
    else:
        series = None

    share_class = filters.get('class')
    stage = filters.get('stage')

    layers = filters.get('layers')
    if isinstance(layers, str):
        layers = layers.strip().lower() or None
    else:
        layers = None

    seller_fee_max = filters.get('seller_fee_max')
    partner_fee_max = filters.get('partner_fee_max')

    sort = filters.get('sort')

    matched = []
    matched_deals = []
    for deal in deals:
        if company is not None:
            if company not in (deal.get('company') or '').strip().lower():
                continue

        if company_industry is not None:
            if company_industry not in (deal.get('company_industry') or '').strip().lower():
                continue

        if deal_type is not None:
            if (deal.get('type') or '').strip() != deal_type:
                continue

        if structure is not None:
            if (deal.get('structure') or '').strip() != structure:
                continue

        if min_size_max is not None:
            deal_min_size = _to_float(deal.get('min_deal_size'))
            if deal_min_size is None or deal_min_size > min_size_max:
                continue

        if carry_max is not None:
            deal_carry = _to_float(deal.get('carry'))
            if deal_carry is None or deal_carry > carry_max:
                continue

        if management_fee_max is not None:
            deal_fee = _to_float(deal.get('management_fee'))
            if deal_fee is None or deal_fee > management_fee_max:
                continue

        if gross_max is not None:
            deal_gross = _to_float(deal.get('gross'))
            if deal_gross is None or deal_gross > gross_max:
                continue

        if gross_min is not None:
            deal_gross = _to_float(deal.get('gross'))
            if deal_gross is None or deal_gross < gross_min:
                continue

        if series is not None:
            if series not in (deal.get('series') or '').strip().lower():
                continue

        if share_class is not None:
            if (deal.get('class') or '').strip() != share_class:
                continue

        if layers is not None:
            deal_layers_lower = (deal.get('layers') or '').strip().lower()
            deal_type_value = (deal.get('type') or '').strip()
            requested_level = _LAYER_LEVELS.get(layers)
            deal_level = _LAYER_LEVELS.get(deal_layers_lower)

            if (
                deal_type_value == 'Sell Order'
                and requested_level is not None
                and deal_level is not None
            ):
                # Sell-side hierarchy: an offer at a lower layer count
                # satisfies a request for a higher one (requesting
                # '2-Layer SPV' also matches 'SPV on cap table').
                if deal_level > requested_level:
                    continue
            else:
                # Buy orders and unspecified/unmapped types: exact
                # (case-insensitive substring) match.
                if layers not in deal_layers_lower:
                    continue

        if stage is not None:
            if (deal.get('stage') or '').strip() != stage:
                continue

        if seller_fee_max is not None:
            deal_seller_fee = _to_float(deal.get('seller_fee'))
            if deal_seller_fee is None or deal_seller_fee > seller_fee_max:
                continue

        if partner_fee_max is not None:
            deal_partner_fee = _to_float(deal.get('partner_fee'))
            if deal_partner_fee is None or deal_partner_fee > partner_fee_max:
                continue

        deal_id = deal.get('id')
        if deal_id is not None:
            matched.append(str(deal_id))
            matched_deals.append(deal)

    if sort and matched_deals:
        sort_specs = {
            'gross_asc': ('gross', False),
            'gross_desc': ('gross', True),
            'min_deal_size_asc': ('min_deal_size', False),
            'max_deal_size_desc': ('max_deal_size', True),
            'updated_desc': ('updated', True),
            'carry_asc': ('carry', False),
        }
        spec = sort_specs.get(sort)
        if spec is not None:
            field, reverse = spec
            # Push missing values to the end regardless of sort direction:
            # ascending wants None to be "largest", descending wants it
            # "smallest" (which becomes last after reverse).
            if field == 'updated':
                missing_sentinel = '' if reverse else '￿'
                def _key(d):
                    v = d.get(field)
                    return v if v else missing_sentinel
            else:
                missing_sentinel = float('-inf') if reverse else float('inf')
                def _key(d):
                    v = _to_float(d.get(field))
                    return missing_sentinel if v is None else v
            matched_deals.sort(key=_key, reverse=reverse)
            top_id = matched_deals[0].get('id')
            if top_id is not None:
                return [str(top_id)]
            return []

    return matched


def _json_response(status_code, payload):
    return {
        'statusCode': status_code,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps(payload),
    }


def _handle_search_post(event):
    """POST handler: run a natural-language deal search and return matching IDs."""
    body = event.get('body') or ''
    if event.get('isBase64Encoded'):
        try:
            body = base64.b64decode(body).decode('utf-8')
        except Exception as e:
            logger.error(f"Error decoding base64 body: {str(e)}")
            return _json_response(400, {'error': 'Invalid base64 body'})

    try:
        data = json.loads(body) if body else {}
    except Exception:
        return _json_response(400, {'error': 'Invalid JSON body'})

    query = (data.get('query') or '').strip()
    if not query:
        return _json_response(400, {'error': 'Missing or empty query'})

    try:
        deals = _load_deals_from_s3()
    except Exception as e:
        logger.error(f"Error reading data from S3: {str(e)}")
        return _json_response(500, {'error': f'Failed to load deals: {str(e)}'})

    try:
        matched_ids = _call_claude_for_matching_ids(query, deals)
    except Exception as e:
        logger.error(f"Error calling Claude API: {str(e)}")
        return _json_response(500, {'error': f'Search failed: {str(e)}'})

    return _json_response(200, {'deal_ids': matched_ids, 'count': len(matched_ids)})

def get_last_updated_date(deal):
    """Returns the last updated date in 'MMM D, YYYY' format or '30d+' if older than 30 days."""
    try:
        last_updated_str = deal.get('updated', '')  # Use "updated" field
        if not last_updated_str:
            return "N/A"
        
        last_updated = datetime.strptime(last_updated_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        days_since_update = (datetime.now(timezone.utc) - last_updated).days

        return "100d+" if days_since_update > 100 else last_updated.strftime('%b %-d')
    except Exception:
        return "N/A"


def format_currency(value, include_cents=False):
    try:
        float_value = float(value)
        if include_cents:
            return f"${float_value:,.2f}"
        else:
            return f"${int(float_value):,}"
    except (ValueError, TypeError):
        return value
def calculate_valuation(price, company_lr_pps, company_lr_val):
    try:
        if not price or not company_lr_pps or not company_lr_val:
            return None
        price = float(price)
        company_lr_pps = float(company_lr_pps)
        company_lr_val = float(company_lr_val)
        price_ratio = price / company_lr_pps
        new_valuation = company_lr_val * price_ratio
        
        return new_valuation
    except (ValueError, TypeError, ZeroDivisionError):
        return None

def format_valuation(valuation):
    if valuation is None:
        return ""
    return f" (${valuation:.1f}Bn)"

def lambda_handler(event, context):
    logger.info('Event received: %s', json.dumps(event))
    logger.info("Lambda function started")

    # Dispatch POST requests to the natural-language search handler.
    http_method = _get_http_method(event)
    if (http_method == 'POST'
            and (event.get('queryStringParameters') or {}).get('view') == 'partner-desk'):
        _pd_handle_reply(event)
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'text/html; charset=utf-8'},
            'body': PARTNER_DESK_HTML.replace('__PD_FORM__', PARTNER_DESK_THANKS),
        }
    if http_method == 'POST':
        # Raw invoke payload: if the event IS the search body (has a
        # top-level 'query' and no HTTP 'body' field), wrap it into a
        # synthetic API-Gateway-style body so _handle_search_post can
        # parse it uniformly.
        if 'body' not in event and 'query' in event:
            synthetic = {'body': json.dumps({'query': event.get('query')})}
            wrapped = _handle_search_post(synthetic)
            # Raw-invoke callers (e.g. API Gateway non-proxy integration)
            # expect just the JSON payload, not the {statusCode, headers,
            # body} wrapper — they'll build the HTTP response themselves.
            # Parse the body back out and return it directly. Fall back to
            # the wrapper if anything unexpected happens.
            try:
                return json.loads(wrapped.get('body') or 'null')
            except (AttributeError, TypeError, json.JSONDecodeError):
                return wrapped
        return _handle_search_post(event)

    # If we're returning from Cognito with ?code=<auth_code> in the query
    # string, set the auth cookie and 302 back to the clean URL so reloads
    # don't keep the code in the address bar (and the auth modal stays
    # suppressed on subsequent visits via the cookie the client-side JS
    # already checks).
    query_params = event.get('queryStringParameters') or {}

    # Admin-only global deal index for the header quick-switcher. Same
    # admin_key gate (query param or cookie) as every other admin route in
    # this file, but unlike those this one explicitly 403s a non-admin
    # caller since it's consumed by fetch() rather than folded into a page
    # render that simply omits admin-only markup.
    _raw_path = (event.get('rawPath')
                 or (event.get('requestContext') or {}).get('http', {}).get('path') or '')
    if _raw_path.rstrip('/').endswith('/api/admin/deal-index') or query_params.get('view') == 'admin-deal-index':
        _idx_is_admin = ('JK8h5Pq2L9aZ7rT3mN6bX' in
                          (query_params.get('admin_key'), _get_cookie(event, 'admin_key')))
        if not _idx_is_admin:
            return _json_response(403, {'error': 'Forbidden'})
        try:
            _idx_deals = _load_deals_from_s3()
        except Exception as e:
            logger.error(f"deal-index load failed: {e}")
            return _json_response(500, {'error': 'Failed to load deals'})
        _idx_rows = [{
            'id': d.get('id'),
            'company': d.get('company'),
            'side': 'Sell' if (d.get('type') or '').strip() == 'Sell Order' else 'Buy',
            'size_min': _to_float(d.get('min_deal_size')),
            'size_max': _to_float(d.get('max_deal_size')),
            'status': d.get('stage'),
            'updated': d.get('updated'),
        } for d in _idx_deals]
        return _json_response(200, _idx_rows)

    # TEMP DIAGNOSTIC ROUTE — remove after Explore Similar Companies is built.
    if query_params.get('industries') and query_params.get('admin_key') == 'JK8h5Pq2L9aZ7rT3mN6bX':
        _diag_deals = _load_deals_from_s3()
        _diag_rows = []
        _diag_seen = set()
        for _diag_d in _diag_deals:
            _diag_co = _diag_d.get('company')
            if _diag_co in _diag_seen:
                continue
            _diag_seen.add(_diag_co)
            _diag_rows.append({'company': _diag_co, 'company_industry': _diag_d.get('company_industry')})
        _diag_rows.sort(key=lambda r: (r['company'] or '').lower())
        return {'statusCode': 200,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'count': len(_diag_rows), 'companies': _diag_rows}, indent=2)}

    # TEMP TEST ROUTE — remove after web-bid testing.
    if query_params.get('mint'):
        _mint_email = _read_identity_email(event)
        if not _mint_email:
            return {'statusCode': 200, 'headers': {'Content-Type': 'text/html'},
                    'body': '<p style="font-family:sans-serif;padding:40px">Not logged in. Open the trades page, sign in, then reload this ?mint=1 URL.</p>'}
        _mint_tok = _make_handoff_token(_mint_email)
        _wb = 'https://7u6sphgup5gjuywcvpuwzhruiq0asgdz.lambda-url.us-east-1.on.aws'
        _mint_link = f"{_wb}/?bid=138490563&name=Positron&sso={urllib.parse.quote(_mint_tok, safe='')}"
        return {'statusCode': 200, 'headers': {'Content-Type': 'text/html'},
                'body': f'<p style="font-family:sans-serif;padding:40px">Logged in as {_mint_email}.<br><br><a href="{_mint_link}">Open web-bid test link (Positron)</a></p>'}

    if query_params.get('view') == 'partner-desk':
        if query_params.get('hashes') == '1':
            _pd_sets = (_partner_desk_hash_sets() if PD_CHECKER_ENABLED
                        else {'emails': '[]', 'domains': '[]'})
            return {
                'statusCode': 200,
                'headers': {'Content-Type': 'application/json', 'Cache-Control': 'max-age=300'},
                'body': json.dumps({'emails': json.loads(_pd_sets['emails']),
                                    'domains': json.loads(_pd_sets['domains'])}),
            }
        _pd_email = html_mod.escape((query_params.get('b') or '').strip(), quote=True)
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'text/html; charset=utf-8'},
            'body': PARTNER_DESK_HTML.replace('__PD_FORM__', PARTNER_DESK_FORM.replace('__PD_EMAIL__', _pd_email)),
        }

    # Admin: mint a Syndicate Dashboard magic link for any email, no login
    # required on this browser -- admin_key is the sole gate, same shared
    # key as the diagnostic route above.
    if query_params.get('mint_for') and query_params.get('admin_key') == 'JK8h5Pq2L9aZ7rT3mN6bX':
        _mf_email = query_params.get('mint_for').strip().lower()
        _mf_token = _make_handoff_token(_mf_email)
        _mf_link = f"{SYNDICATE_DASH_URL}/?sso={urllib.parse.quote(_mf_token, safe='')}"
        _mf_link_safe = html_mod.escape(_mf_link, quote=True)
        return {'statusCode': 200, 'headers': {'Content-Type': 'text/html'},
                'body': f'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Syndicate Dashboard link</title></head>
<body style="font-family:sans-serif;padding:40px">
  <p>Magic link for {html_mod.escape(_mf_email)} (valid 1 hour):</p>
  <p>
    <input id="mf-link" type="text" readonly value="{_mf_link_safe}"
           style="width:90%;max-width:640px;padding:8px;font-family:monospace">
    <button onclick="navigator.clipboard.writeText(document.getElementById('mf-link').value)">Copy</button>
  </p>
  <p><a href="{_mf_link_safe}">{html_mod.escape(_mf_link)}</a></p>
</body></html>'''}

    if query_params.get('view') == 'pd-analyze':
        if query_params.get('key') != _PD_ANALYZE_KEY:
            return {'statusCode': 404, 'headers': {'Content-Type': 'text/plain; charset=utf-8'}, 'body': 'Not found'}
        try:
            report = _pd_analyze_report()
        except Exception as e:
            logger.error(f"pd-analyze failed: {e}")
            report = f"pd-analyze failed: {e}"
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'text/plain; charset=utf-8'},
            'body': report,
        }

    if query_params.get('signout') == '1':
        return {
            'statusCode': 303,
            'headers': {'Location': '/'},
            'cookies': [
                'gg_id=; Max-Age=0; Path=/; Secure; SameSite=Lax',
                'CognitoIdentityServiceProvider=; Max-Age=0; Path=/; Secure; SameSite=Lax',
            ],
            'body': '',
        }

    if query_params.get('code'):
        clean_path = (
            event.get('rawPath')
            or (event.get('requestContext') or {}).get('http', {}).get('path')
            or '/'
        )
        logger.info("Cognito auth code received; redirecting to clean URL %s", clean_path)
        cookies = ['CognitoIdentityServiceProvider=1; Max-Age=31536000; Path=/; Secure; SameSite=Lax']
        email = None
        try:
            email = _exchange_code_for_email(query_params.get('code'))
            if email and IDENTITY_SECRET:
                cookies.append(_make_identity_cookie(email))
                logger.info("Identity captured for Cognito login")
        except Exception as e:
            logger.warning(f"Identity capture failed (non-fatal): {e}")

        # Optional post-login bounce: `state` carries a base64url-encoded destination
        # URL (minted by a caller like CRMDealDetails) that wants the viewer sent
        # back there, with a fresh SSO handoff token, instead of the dashboard here.
        if query_params.get('state') and email:
            try:
                _raw_state = query_params['state']
                _dest = base64.urlsafe_b64decode(
                    _raw_state + '=' * (-len(_raw_state) % 4)
                ).decode()
                _allowed_prefixes = (
                    'https://trades.graciagroup.com/',
                    'https://desk.graciagroup.com/',
                    'https://7u6sphgup5gjuywcvpuwzhruiq0asgdz.lambda-url.us-east-1.on.aws/',
                )
                if _dest.startswith(_allowed_prefixes):
                    _state_tok = _make_handoff_token(email)
                    _state_sep = '&' if '?' in _dest else '?'
                    _state_redirect = f"{_dest}{_state_sep}sso={urllib.parse.quote(_state_tok, safe='')}"
                    return {
                        'statusCode': 302,
                        'headers': {'Location': _state_redirect},
                        'cookies': cookies,
                        'body': '',
                    }
            except Exception as e:
                logger.warning(f"State redirect failed (non-fatal): {e}")

        # A 302 can't run JS, so carry an auth marker through the redirect. The
        # rendered dashboard fires the GA4 event when it sees ?auth=1, then strips
        # the marker client-side so a refresh can't re-fire it.
        auth_sep = '&' if '?' in clean_path else '?'
        return {
            'statusCode': 302,
            'headers': {'Location': f'{clean_path}{auth_sep}auth=1'},
            'cookies': cookies,
            'body': '',
        }

    # GET path: render the HTML dashboard as before.
    try:
        deals = _load_deals_from_s3()
    except Exception as e:
        logger.error(f"Error reading data from S3: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'text/html'},
            'body': f"<h1>Error</h1><p>An error occurred while reading data: {str(e)}</p>"
        }

    # Sort by company alpha, then updated
    deals = sorted(
    deals,
    key=lambda deal: (
        deal.get('company', '').lower(),                      # A–Z
        -datetime.strptime(deal.get('updated', '1900-01-01 00:00:00'), "%Y-%m-%d %H:%M:%S").timestamp()  # Newest first
    )
    )

    # Get a unique list of companies, prioritizing highlighted ones first
    highlighted_set = {deal['company'] for deal in deals if deal['company'] and deal.get('highlighted') == 'Yes'}
    non_highlighted_set = {deal['company'] for deal in deals if deal['company'] and deal.get('highlighted') != 'Yes'}

    # Merge in Directory-flagged companies (from directory_companies.json). These may
    # have NO deals at all (demand-only names) and still appear as grid buttons.
    dir_highlight_names, dir_list_names, dir_pricing = _load_directory_companies()
    for nm in dir_highlight_names:
        highlighted_set.add(nm)
        non_highlighted_set.discard(nm)   # Highlight wins if also present elsewhere
    for nm in dir_list_names:
        if nm not in highlighted_set:     # don't demote a highlighted company
            non_highlighted_set.add(nm)

    # Companies that have ANY deal in the current book (buy or sell).
    _companies_with_deals = {deal['company'] for deal in deals if deal['company']}
    # Deal-less companies route to the web-bid form instead of filtering.
    _dealless_names = (highlighted_set | non_highlighted_set) - _companies_with_deals
    # name -> company_id from the directory pricing map.
    _name_to_id = {v.get('name'): cid for cid, v in dir_pricing.items() if v.get('name')}
    # Mint one handoff token for the logged-in user (None if not logged in).
    _wb_email = _read_identity_email(event)
    _wb_token = _make_handoff_token(_wb_email) if _wb_email else None
    _WEB_BID_URL = 'https://7u6sphgup5gjuywcvpuwzhruiq0asgdz.lambda-url.us-east-1.on.aws'

    def _company_btn(company):
        # Deal-less company with a known id -> link to web-bid; else default filter.
        cid = _name_to_id.get(company)
        if company in _dealless_names and cid:
            href = f"{_WEB_BID_URL}/?bid={urllib.parse.quote(str(cid))}&name={urllib.parse.quote(company)}"
            if _wb_token:
                href += f"&sso={urllib.parse.quote(_wb_token, safe='')}"
            return f"<button class='company-btn' id=\"{company}\" onclick=\"window.location.href='{href}'\">{company}</button>"
        return f"<button class='company-btn' id=\"{company}\" onclick=\"toggleCompanyFilter('{company}')\">{company}</button>"

    highlighted_companies = sorted(highlighted_set)
    non_highlighted_companies = sorted(non_highlighted_set)

    # Merge lists: highlighted first, then non-highlighted
    companies = highlighted_companies + non_highlighted_companies

    
    # Build the table rows
    table_rows = ""
    for deal in deals:
        stage_tooltips = {
            "Firm": "Details confirmed",
            "Inquiry": "Awaiting data",
            "Confirm": "Will confirm after bid/ask"
        }
        
        stage_html = f'<span class="stage-cell" data-tooltip="{stage_tooltips[deal["stage"]]}">{{deal["stage"]}}</span>'
        
        company_cell = deal['company']

        # Layer annotation folded into the Structure column, e.g. "Fund (2L)".
        # Shown for any deal (buy or sell) that carries a recognized layers value;
        # buy orders that don't reference layers simply omit it.
        layers_val = deal.get('layers') or ''
        layer_label = {'spv on cap table': '1L', '2-layer spv': '2L', '3-layer spv': '3L'}.get(layers_val.strip().lower(), '')
        layer_badge_html = f' <span class="layer-badge" title="{layers_val}">({layer_label})</span>' if layer_label else ''
        
        # Calculate valuations
        net_valuation = calculate_valuation(deal['net'], deal['company_lr_pps'], deal['company_lr_val'])
        gross_valuation = calculate_valuation(deal['gross'], deal['company_lr_pps'], deal['company_lr_val'])
        net_display = format_currency(deal['net'], include_cents=True)
        net_valuation_text = format_valuation(net_valuation).strip()  # Remove any extra whitespace
        gross_display = format_currency(deal['gross'], include_cents=True)
        gross_valuation_text = format_valuation(gross_valuation).strip()


        table_rows += f"""
        <tr class="deal-row {deal['type'].lower()} {deal['structure_class']}" data-deal-id="{deal['id']}" data-management-fee="{deal['management_fee']}" data-carry="{deal['carry']}" data-stage="{deal['stage']}" data-data-room="{deal['data_room']}" data-highlighted="{deal['highlighted']}" data-layers="{deal.get('layers') or ''}">
            <td><a href="https://trades.graciagroup.com/deal/{deal['id']}">{deal['id']}</a><br><button type="button" class="copy-id" data-copy-id="{deal['id']}" title="Copy deal ID" aria-label="Copy deal ID {deal['id']}"><svg viewBox="0 0 16 16" aria-hidden="true" focusable="false"><rect x="5.5" y="5.5" width="8" height="8" rx="1.5"></rect><path d="M10.5 3.5v-1a1 1 0 0 0-1-1h-7a1 1 0 0 0-1 1v7a1 1 0 0 0 1 1h1"></path></svg></button></td>
            <td>{deal['type']}</td>
            <td>{company_cell}</td>
            <td>{deal['structure']}{layer_badge_html}</td>
            <td class="price-cell" data-valuation="{net_valuation_text}">{net_display}</td>
            <td class="price-cell" data-valuation="{gross_valuation_text}">{gross_display}</td>
            <td>{format_currency(deal['min_deal_size'])}</td>
            <td>{format_currency(deal['max_deal_size'])}</td>
            <td>{format_currency(deal['company_lr_pps'], include_cents=True)}</td>
            <td>{format_currency(deal['company_lr_val'], include_cents=True)}</td>
            <td>{deal['management_fee']}</td>
            <td>{deal['carry']}</td>
            <td style="text-align:center;">{get_last_updated_date(deal)}<br class="nudge-br" style="display:none;"><a class="nudge-bell" data-deal-id="{deal['id']}" style="display:none;margin-top:4px;text-decoration:none;" href="https://ak5zolfpynhrimrsuw5rbjchwu0ktexz.lambda-url.us-east-1.on.aws/?deal_id={deal['id']}&key={NUDGE_KEY}" target="_blank" rel="noopener" title="Nudge client to update or cancel" onclick="if(!confirm('Send an update request to this client?'))return false;localStorage.setItem('nudge_'+this.getAttribute('data-deal-id'),Date.now());this.style.display='none';var br=this.previousElementSibling;if(br&&br.tagName==='BR')br.style.display='none';return true;">🔔</a><a class="loi-send" data-deal-id="{deal['id']}" style="display:none;margin-left:7px;text-decoration:none;" href="{LOI_SEND_URL}?send=1&deal_id={deal['id']}&key={LOI_PAGE_KEY}" target="_blank" rel="noopener" title="Email this client a Letter of Intent link" onclick="if(!confirm('Email an LOI link to this client?'))return false;localStorage.setItem('loi_'+this.getAttribute('data-deal-id'),Date.now());this.style.display='none';return true;">✍️</a></td>
        </tr>
        """

    # Create buttons for companies
    highlighted_company_buttons = " ".join([_company_btn(company) for company in highlighted_companies])

    non_highlighted_company_buttons = " ".join([_company_btn(company) for company in non_highlighted_companies])


    # No admin boolean exists in Python here, so derive one the same way the page's
    # own JS does: the admin_key query parameter, or the admin_key cookie that JS
    # writes once the parameter has been seen.
    _is_admin = ('JK8h5Pq2L9aZ7rT3mN6bX' in
                 (query_params.get('admin_key'), _get_cookie(event, 'admin_key')))
    top_nav_html = _render_top_nav(event, _is_admin)
    deal_switcher_html = _render_deal_switcher_modal() if _is_admin else ''

    # GA4 auth event. The Cognito return leg redirects to ?auth=1 (see above), so this
    # renders only on the pageview immediately following authentication, never on an
    # ordinary dashboard load. Two scripts, both required:
    #   1. Runs BEFORE the gtag config below, so the auto page_view reports the clean
    #      URL rather than one carrying ?auth=1. It deletes only the auth parameter,
    #      leaving ?company=/?side=/?admin_key= intact for the client JS that reads
    #      them from location.search later in the page.
    #   2. Fires the event itself, after gtag() is defined.
    # The event is 'login' for everyone: this Lambda has no user store (it only reads
    # deal data from S3) and the Cognito id_token carries no first-login claim, so a
    # new registration is indistinguishable from a returning sign-in here.
    if query_params.get('auth') == '1':
        ga_auth_strip_js = """<script>
          (function () {
            try {
              var u = new URL(window.location.href);
              if (!u.searchParams.has('auth')) return;
              u.searchParams.delete('auth');
              history.replaceState(null, '', u.pathname + u.search + u.hash);
            } catch (e) {}
          })();
        </script>"""
        ga_auth_event_js = """<script>
          gtag('event', 'login', { method: 'Cognito' });
        </script>"""
    else:
        ga_auth_strip_js = ''
        ga_auth_event_js = ''

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        {ga_auth_strip_js}
        <!-- Google tag (gtag.js) -->
        <script async src="https://www.googletagmanager.com/gtag/js?id=G-L9JN3TRR2S"></script>
        <script>
          window.dataLayer = window.dataLayer || [];
          function gtag(){{dataLayer.push(arguments);}}
          gtag('js', new Date());
          gtag('config', 'G-L9JN3TRR2S');
        </script>
        {ga_auth_event_js}
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Indications · Gracia Group</title>
        <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>📈</text></svg>">
        <link rel="icon" type="image/png" href="https://pre-ipo.graciagroup.com/favicon.png">
        <link rel="apple-touch-icon" href="https://pre-ipo.graciagroup.com/favicon.png">
        <link rel="stylesheet" href="https://s3.us-east-1.amazonaws.com/main.css/master.css">
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                display: block;  /* stack title above filters (override master.css flex) */
                background-color: #f8f9fa;
                padding: 10px 20px;
                border-radius: 5px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .title-row {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 24px;
                flex-wrap: wrap;
                margin-bottom: 4px;
            }}
            .title-row h1 {{
                margin: 0;
            }}
            .title-row .nl-search-container {{
                background: none;
                box-shadow: none;
                padding: 0;
                margin-bottom: 0;
                margin-top: 8px;
                flex: 0 1 400px;
                min-width: 260px;
            }}
            .title-row .nl-search-btn {{
                margin-bottom: 0;
                padding: 10px 16px;
            }}
            .title-row .nl-search-status {{
                min-height: 0;
                margin-top: 4px;
            }}
            .filter-section {{
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                gap: 10px 0;
                margin-bottom: 10px;
            }}
            .filter-group {{
                display: flex;
                align-items: center;
                gap: 6px;
                padding: 2px 22px;
                border-left: 1px solid var(--border-strong, #cdc9c0);
            }}
            .filter-group:first-child {{
                padding-left: 0;
                border-left: none;
            }}
            .filter-label {{
                font-weight: bold;
                margin-right: 5px;
                position: relative;
                top: -1px;
            }}
            .fund-group {{
                display: inline-flex;
                align-items: center;
            }}
            .fund-group .checkbox-label {{
                margin-right: 0;
            }}
            .sub-filter {{
                margin-left: 6px;
            }}
            .sub-filter.disabled {{
                opacity: 0.45;
            }}
            .layer-badge {{
                color: #6b7280;
                font-weight: 600;
                white-space: nowrap;
            }}
            h1 {{
                color: #2c3e50;
                margin: 0 0 4px 0;
                font-size: 28px;
                font-weight: 700;
                letter-spacing: -0.01em;
            }}
            .subtitle {{
                margin: 0 0 16px 0;
                padding-bottom: 12px;
                border-bottom: 2px solid #e0e0e0;
                font-style: italic;
                font-size: 15px;
                color: #6b7280;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}

            th, td {{
                border: 1px solid #ddd;
                padding: 12px 6px;
                text-align: left;
            }}

            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}

            thead {{
                position: sticky;
                top: 0;
                z-index: 10;
            }}

            thead tr {{
                background-color: #f8f9ff;
            }}

            thead th {{
                background-color: #f8f9ff;
                border-bottom: 3px double #ddd;
                position: sticky;
                top: 0;
                font-weight: bold;
                color: #2c3e50;
            }}

            tr:hover {{
                background-color: #f5f5f5;
            }}

            .copy-id {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 22px;
                height: 22px;
                margin-top: 5px;
                padding: 0;
                background-color: #fff;
                border: 1px solid #ccc;
                border-radius: 4px;
                color: #6b7280;
                cursor: pointer;
                line-height: 0;
            }}

            .copy-id svg {{
                width: 13px;
                height: 13px;
                fill: none;
                stroke: currentColor;
                stroke-width: 1.5;
                stroke-linecap: round;
                stroke-linejoin: round;
            }}

            .copy-id:hover {{
                border-color: var(--accent, #3d5a73);
                color: var(--accent, #3d5a73);
            }}

            .copy-id.copied {{
                border-color: #16a34a;
                color: #16a34a;
            }}

            .stage-cell {{
                position: relative;
                cursor: help;
            }}

            .stage-cell:hover::after {{
                content: attr(data-tooltip);
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%);
                padding: 5px 10px;
                background-color: var(--accent, #3d5a73);
                color: white;
                border-radius: 4px;
                font-size: 14px;
                white-space: nowrap;
                z-index: 20;
            }}
            
            .topbar {{
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                gap: 10px;
                margin-bottom: 10px;
            }}
            .topnav {{
                display: flex;
                align-items: center;
                flex-wrap: wrap;
                gap: 16px;
                padding: 10px 0;
                margin-bottom: 10px;
                border-bottom: 1px solid #ddd;
            }}
            .nav-brand {{
                font-weight: 700;
                font-size: 17px;
                color: var(--ink);
                text-decoration: none;
                white-space: nowrap;
            }}
            .nav-tabs {{
                display: flex;
                align-items: center;
                flex-wrap: wrap;
                gap: 18px;
                flex: 1;
            }}
            .nav-tab {{
                display: inline-block;
                background-color: #fff;
                border: 1px solid #ddd;
                border-radius: 999px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: 600;
                color: var(--ink);
                text-decoration: none;
                white-space: nowrap;
            }}
            .nav-tab:hover {{
                background-color: #f0f0f0;
            }}
            .nav-tab-disabled {{
                color: #999;
                cursor: default;
            }}
            .nav-tab-disabled:hover {{
                background-color: #fff;
            }}
            .btn.nav-signin {{
                background-color: #fff;
                color: var(--ink);
                border: 1px solid #ddd;
                border-radius: 999px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: 600;
                white-space: nowrap;
                margin-bottom: 0;
                margin-left: auto;
            }}
            .btn.nav-signin:hover {{
                background-color: #f0f0f0;
            }}
            .nav-icon-btn {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 34px;
                height: 34px;
                background-color: #fff;
                border: 1px solid #ddd;
                border-radius: 999px;
                color: var(--ink);
                cursor: pointer;
                padding: 0;
            }}
            .nav-icon-btn:hover {{
                background-color: #f0f0f0;
            }}
            .deal-switcher-overlay {{
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.5);
                z-index: 2000;
                padding-top: 12vh;
            }}
            .deal-switcher-overlay.show {{
                display: block;
            }}
            .deal-switcher-box {{
                background-color: #fff;
                margin: 0 auto;
                width: 90%;
                max-width: 560px;
                border-radius: 8px;
                box-shadow: 0 8px 30px rgba(0,0,0,0.25);
                overflow: hidden;
                font-family: Arial, sans-serif;
            }}
            .deal-switcher-input {{
                width: 100%;
                box-sizing: border-box;
                border: none;
                border-bottom: 1px solid #ddd;
                padding: 16px 18px;
                font-size: 16px;
                font-family: inherit;
                outline: none;
            }}
            .deal-switcher-results {{
                max-height: 50vh;
                overflow-y: auto;
            }}
            .deal-switcher-row {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 10px;
                padding: 10px 18px;
                cursor: pointer;
                font-size: 14px;
                color: #333;
            }}
            .deal-switcher-row.active {{
                background-color: #f0f4f8;
            }}
            .deal-switcher-label {{
                flex: 1;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }}
            .deal-switcher-copy {{
                flex-shrink: 0;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 26px;
                height: 26px;
                background: none;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                color: #666;
                padding: 0;
            }}
            .deal-switcher-copy:hover {{
                background-color: #e2e6ea;
                color: #333;
            }}
            .deal-switcher-copy svg {{
                width: 14px;
                height: 14px;
                fill: none;
                stroke: currentColor;
                stroke-width: 1.3;
            }}
            .deal-switcher-empty {{
                padding: 16px 18px;
                color: #888;
                font-size: 14px;
            }}
            .navacct {{
                position: relative;
                margin-left: auto;
            }}
            .navacct-trigger {{
                display: inline-block;
                background-color: #fff;
                border: 1px solid #ddd;
                border-radius: 999px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: 600;
                color: var(--ink);
                cursor: pointer;
                white-space: nowrap;
            }}
            .navacct-trigger:hover {{
                background-color: #f0f0f0;
            }}
            .navacct-menu {{
                display: none;
                position: absolute;
                right: 0;
                top: 100%;
                margin-top: 6px;
                background: #fff;
                border-radius: 6px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                min-width: 220px;
                padding: 6px 0;
                z-index: 50;
            }}
            .navacct:hover .navacct-menu, .navacct:focus-within .navacct-menu {{
                display: block;
            }}
            .navacct-item {{
                display: block;
                padding: 9px 16px;
                font-size: 13px;
                color: var(--ink);
                text-decoration: none;
                white-space: nowrap;
            }}
            .navacct-item:hover {{
                background: #f4f4f4;
            }}
            .navacct-static {{
                color: var(--text-secondary, #666);
                font-weight: 600;
                cursor: default;
            }}
            .navacct-static:hover {{
                background: none;
            }}
            .navacct-disabled {{
                color: #999;
                cursor: default;
            }}
            .navacct-disabled:hover {{
                background: none;
            }}
            .btn {{
                display: inline-block;
                padding: 10px 20px;
                line-height: 1.2;
                font-size: 15px;
                text-decoration: none;
                border-radius: 5px;
                transition: background-color 0.3s, box-shadow 0.3s;  /* Added shadow transition */
                margin-bottom: 10px;
                border: none;  /* Remove border */
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);  /* Add subtle shadow */
            }}

            .btn:hover {{
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);  /* Slightly stronger shadow on hover */

            }}
            .company-buttons {{
                margin-bottom: 20px;
            }}
            .company-btn {{
                background-color: #CCDBEA;
                border: none;
                color: var(--ink);
                font-weight: 500;
                padding: 6px 14px;
                margin: 3px;
                border-radius: 999px;
                font-size: 14px;
                cursor: pointer;
                transition: background-color 0.15s, color 0.15s;
            }}
            .company-btn:hover {{
                background-color: #B7CBE1;
            }}
            #highlightedCompanies, #nonHighlightedCompanies {{
                margin-left: -3px;
            }}
            .company-btn.active {{
                background-color: #44576B; /* selected */
                color: white; /* Keep text readable */
            }}

            #disclaimer {{
                font-size: 0.9em;
                color: #666;
                margin-top: 30px;
            }}
            .checkbox-label {{
                display: inline-flex;
                align-items: center;
                margin-right: 15px;
                cursor: pointer;
                white-space: nowrap;
            }}
            .checkbox-label input {{
                margin-right: 5px;
            }}
            input[type="checkbox"] {{
                accent-color: var(--accent, #3d5a73);
            }}
            .slider-group .checkbox-label,
            .bottom-filter-group .checkbox-label {{
                margin-right: 0;
            }}

            .filter-row {{
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                gap: 16px;
                width: 100%;
            }}

            .slider-group {{
                display: flex;
                align-items: center;
                gap: 16px;
                flex-shrink: 0;
                border: 1px solid #ddd;
                padding: 10px;
                border-radius: 4px;
            }}
            .right-filters {{
                display: flex;
                align-items: center;
                gap: 16px;
            }}

            .bottom-filter-group {{
                border: 1px solid #ddd;
                padding: 10px;
                border-radius: 4px;
                white-space: nowrap;
            }}

            .spacer {{
                display: none;
            }}
            .data-room-group {{
                position: absolute;
                left: 50%;
                transform: translateX(-50%);
                border: 1px solid #ddd;
                padding: 10px;
                border-radius: 4px;
            }}

            .highlighted-group {{
                margin-left: auto;
                border: 1px solid #ddd;
                padding: 10px;
                border-radius: 4px;
            }}

            .price-cell {{
                position: relative;
                cursor: help;
                color: #333;  /* Reset the text color to dark gray/black */
            }}

            .price-cell:hover::after {{
                content: attr(data-valuation);
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%);
                padding: 5px 10px;
                background-color: var(--accent, #3d5a73);
                color: white;  /* This is for the tooltip text */
                border-radius: 4px;
                font-size: 14px;
                white-space: nowrap;
                z-index: 20;
            }}

            .stage-cell {{
                position: relative;
                cursor: help;
                color: #333;  /* Reset the text color to dark gray/black */
            }}

            .stage-cell:hover::after {{
                content: attr(data-tooltip);
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%);
                padding: 5px 10px;
                background-color: var(--accent, #3d5a73);
                color: white;  /* This is for the tooltip text */
                border-radius: 4px;
                font-size: 14px;
                white-space: nowrap;
                z-index: 20;
            }}
            .ticket-size-filter {{
                margin-bottom: 20px;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}

            .slider-container {{
                display: flex;
                align-items: center;
                gap: 15px;
            }}

            #ticketSlider {{
                flex: 1;
                max-width: 300px;
            }}
            .company-filter {{
                background-color: #f8f9fa;
                padding: 14px 20px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 15px; /* Add space below the company buttons */
            }}
            .table-toolbar {{
                display: flex;
                justify-content: flex-end;
                align-items: center;
                flex-wrap: wrap;
                gap: 8px;
                margin: 0 0 8px 0;
            }}
            .toolbar-btn {{
                font-family: var(--font-ui);
                background: #2e9d6a;
                border: 1px solid #2e9d6a;
                color: #fff;
                font-size: 13px;
                font-weight: 500;
                padding: 6px 14px;
                border-radius: 999px;
                cursor: pointer;
                transition: background-color 0.15s, border-color 0.15s;
            }}
            .toolbar-btn:hover {{
                background-color: #237a52;
                border-color: #237a52;
            }}
            .company-filter > strong {{
                display: block;
                margin-bottom: 8px;
            }}
            .company-btn.show-all-btn {{
                background-color: transparent;
                border: 1px solid var(--border-strong);
                color: var(--text-secondary);
            }}
            .company-btn.show-all-btn:hover {{
                background-color: rgba(0,0,0,0.04);
            }}

            .toggle-btn {{
                display: inline-flex; /* Aligns with highlighted buttons */
                align-items: center; /* Centers text vertically */
                background-color: #e8f5e9; /* Light green background */
                border: 1px solid #4CAF50; /* Subtle green border */
                padding: 5px 10px;
                margin: 2px; /* Keeps spacing consistent with company buttons */
                border-radius: 3px;
                cursor: pointer;
                transition: background-color 0.3s, border 0.3s;
                white-space: nowrap; /* Prevents text wrapping */
                font-weight: bold; /* Makes text slightly stronger */
                color: #2c3e50; /* Dark text for readability */
            }}

            .toggle-btn:hover {{
                background-color: #d4edda; /* Slightly darker green on hover */
                border-color: #388E3C; /* Darker green border */
            }}


            .deal-count {{
                font-size: 28px;
                color: #2c3e50;
                font-weight: normal;
                margin-left: 5px;
                display: inline-block;
            }}

            .highlighted-group {{
                display: flex;
                align-items: center;
            }}
            .modal {{
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.5);
                z-index: 1000;
                opacity: 0;
                transition: opacity 0.3s ease-in-out;
            }}

            .modal-content {{
                background-color: white;
                margin: 15% auto;
                padding: 30px;
                width: 80%;
                max-width: 500px;
                border-radius: 8px;
                text-align: center;
                position: relative;
                transform: translateY(-20px);
                transition: transform 0.3s ease-in-out;
            }}

            .modal-buttons {{
                margin-top: 25px;
            }}

            .modal-btn {{
                padding: 12px 25px;
                border-radius: 5px;
                border: none;
                cursor: pointer;
                font-size: 16px;
                margin: 0 10px;
            }}

            .primary-btn {{
                background-color: var(--accent, #3d5a73);
                color: white;
            }}

            .modal.show {{
                display: block;
                opacity: 1;
            }}

            .modal.show .modal-content {{
                transform: translateY(0);
            }}

            .nl-search-container {{
                background-color: #f8f9fa;
                padding: 15px 20px;
                border-radius: 5px;
                margin-bottom: 15px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .nl-search-row {{
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .nl-search-input {{
                flex: 1;
                padding: 10px 12px;
                font-size: 14px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-family: inherit;
            }}
            .nl-search-input:focus {{
                outline: none;
                border-color: var(--accent, #3d5a73);
            }}
            .nl-search-btn, .nl-clear-btn {{
                padding: 10px 18px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                transition: background-color 0.3s;
                white-space: nowrap;
            }}
            .nl-search-btn:disabled {{
                background-color: #95a5a6;
                cursor: not-allowed;
            }}
            .nl-clear-btn {{
                background-color: #e74c3c;
                color: white;
                display: none;
            }}
            .nl-clear-btn:hover {{
                background-color: #c0392b;
            }}
            .nl-search-status {{
                margin-top: 8px;
                font-size: 13px;
                color: #666;
                min-height: 18px;
            }}

        </style>
        <!-- Core application scripts -->
        <script>
            var selectedCompanies = [];
            var searchMatchedIds = null; // Set<string> of deal IDs, or null when no search is active

            function performSearch() {{
                var input = document.getElementById('nlSearchInput');
                var query = input.value.trim();
                if (!query) return;

                var btn = document.getElementById('nlSearchBtn');
                var clearBtn = document.getElementById('nlClearBtn');
                var statusEl = document.getElementById('nlSearchStatus');

                btn.disabled = true;
                btn.textContent = 'Searching...';
                statusEl.textContent = 'Searching...';
                statusEl.style.color = '#666';

                fetch(window.location.origin + window.location.pathname, {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{query: query}})
                }})
                .then(function(r) {{
                    return r.json().then(function(data) {{
                        if (!r.ok) throw new Error((data && data.error) || ('HTTP ' + r.status));
                        return data;
                    }});
                }})
                .then(function(data) {{
                    if (!data || !Array.isArray(data.deal_ids)) {{
                        throw new Error('Unexpected response from search');
                    }}
                    searchMatchedIds = new Set(data.deal_ids.map(String));
                    clearBtn.style.display = 'inline-block';
                    filterTable();
                    var matchCount = searchMatchedIds.size;
                    statusEl.textContent = matchCount + ' match' + (matchCount === 1 ? '' : 'es') + ' for: "' + query + '"';
                    statusEl.style.color = '#2c3e50';
                }})
                .catch(function(err) {{
                    statusEl.textContent = 'Search failed: ' + err.message;
                    statusEl.style.color = '#c0392b';
                }})
                .finally(function() {{
                    btn.disabled = false;
                    btn.textContent = 'Go';
                }});
            }}

            function clearSearch() {{
                searchMatchedIds = null;
                document.getElementById('nlSearchInput').value = '';
                document.getElementById('nlClearBtn').style.display = 'none';
                document.getElementById('nlSearchStatus').textContent = '';
                filterTable();
            }}

            function formatDollarAmount(amount) {{
                if (amount >= 1000000) {{
                    return '$' + (amount / 1000000) + 'M';
                }} else {{
                    return '$' + (amount / 1000) + 'K';
                }}
            }}

            function toggleCompanyFilter(company) {{
                var index = selectedCompanies.indexOf(company);
                if (index > -1) {{
                    selectedCompanies.splice(index, 1);
                    document.getElementById(company).classList.remove('active');
                }} else {{
                    selectedCompanies.push(company);
                    document.getElementById(company).classList.add('active');
                }}
                filterTable();
                updateDealCount();
            }}

            function toggleNonHighlighted() {{
                var section = document.getElementById("nonHighlightedCompanies");
                var button = document.querySelector(".show-all-btn");

                if (section.style.display === "none") {{
                    section.style.display = "block";
                    button.innerHTML = "Hide All Companies ▲";
                }} else {{
                    section.style.display = "none";
                    button.innerHTML = "Show All Companies ▼";
                }}
            }}

            function updateDealCount() {{
                const visibleRows = Array.from(document.getElementsByClassName('deal-row')).filter(row => row.style.display !== 'none');
                const countElement = document.getElementById('dealCount');
                countElement.textContent = `(${{visibleRows.length}} deals)`;
            }}

            function filterTable() {{
                var buyChecked = document.getElementById('buyFilter').checked;
                var sellChecked = document.getElementById('sellFilter').checked;
                var directChecked = document.getElementById('directFilter').checked;
                var spvChecked = document.getElementById('spvFilter').checked;
                var forwardChecked = document.getElementById('forwardFilter').checked;
                var managementFeeChecked = document.getElementById('managementFeeFilter').checked;
                var carryChecked = document.getElementById('carryFilter').checked;
                var dataRoomChecked = document.getElementById('dataRoomFilter').checked;
                var highlightedChecked = document.getElementById('highlightedFilter').checked;
                var multiLayerOk = document.getElementById('multiLayerFilter').checked;
                document.getElementById('multiLayerFilter').disabled = !spvChecked;
                document.getElementById('multiLayerWrap').classList.toggle('disabled', !spvChecked);

                var ticketBuckets = [];
                var ticketCheckboxes = document.getElementsByClassName('ticket-filter');
                for (var t = 0; t < ticketCheckboxes.length; t++) {{
                    if (ticketCheckboxes[t].checked) {{
                        var hiAttr = ticketCheckboxes[t].getAttribute('data-hi');
                        ticketBuckets.push({{
                            lo: parseFloat(ticketCheckboxes[t].getAttribute('data-lo')) || 0,
                            hi: (hiAttr === null || hiAttr === '') ? Infinity : parseFloat(hiAttr)
                        }});
                    }}
                }}
                
                var rows = document.getElementsByClassName('deal-row');
                
                for (var i = 0; i < rows.length; i++) {{
                    var row = rows[i];
                    var type = row.classList.contains('buy') ? 'buy' : 'sell';
                    var company = row.cells[2].innerText;
                    var isDirect = row.classList.contains('direct');
                    var isSPV = row.classList.contains('fund');
                    var isForward = row.classList.contains('forward');
                    var managementFee = parseFloat(row.getAttribute('data-management-fee')) || 0;
                    var carry = parseFloat(row.getAttribute('data-carry')) || 0;
                    var stage = row.getAttribute('data-stage');
                    var hasDataRoom = row.getAttribute('data-data-room') === 'Yes';
                    var isHighlighted = row.getAttribute('data-highlighted') === 'Yes';
                    var layers = (row.getAttribute('data-layers') || '').toLowerCase();
                    var isMultiLayer = layers.indexOf('2-layer') !== -1 || layers.indexOf('3-layer') !== -1;
                    var showLayer = multiLayerOk || !isMultiLayer;

                    var showBroker = true;  // 'Available for Brokers' filter removed

                    var showType = (buyChecked && type === 'buy') || (sellChecked && type === 'sell');
                    var showStructure = (directChecked && isDirect) || (spvChecked && isSPV) || (forwardChecked && isForward);
                    var showFees = (managementFeeChecked || managementFee === 0) && (carryChecked || carry === 0);
                    var showUnconfirmed = true;  // unconfirmed orders always shown (auto-updated)
                    
                    var dealMin = parseFloat(row.cells[6].innerText.replace(/[^0-9.-]+/g,'')) || 0;
                    var dealMax = parseFloat(row.cells[7].innerText.replace(/[^0-9.-]+/g,''));
                    if (isNaN(dealMax) || dealMax <= 0) {{
                        dealMax = Infinity;
                    }}
                    var showTicketSize = false;
                    for (var b = 0; b < ticketBuckets.length; b++) {{
                        if (dealMin < ticketBuckets[b].hi && dealMax >= ticketBuckets[b].lo) {{
                            showTicketSize = true;
                            break;
                        }}
                    }}
                    
                    var show = showType && showStructure && showFees && showUnconfirmed && 
                              showTicketSize && 
                              (!dataRoomChecked || hasDataRoom) &&
                              (!highlightedChecked || isHighlighted) &&
                              showBroker &&
                              showLayer;
                    
                    if (selectedCompanies.length > 0 && selectedCompanies.indexOf(company) === -1) {{
                        show = false;
                    }}

                    if (searchMatchedIds !== null) {{
                        var dealIdStr = row.getAttribute('data-deal-id');
                        if (dealIdStr === null || !searchMatchedIds.has(dealIdStr)) {{
                            show = false;
                        }}
                    }}

                    row.style.display = show ? '' : 'none';
                }}
                updateDealCount();
            }}

            // Deep-link support: ?company=NAME&side=bid|offer
            document.addEventListener('DOMContentLoaded', function () {{
                var dlParams = new URLSearchParams(window.location.search);
                var dlCompany = dlParams.get('company');
                var dlSide = (dlParams.get('side') || '').toLowerCase();
                var dlTouched = false;

                if (dlSide === 'bid' || dlSide === 'offer') {{
                    var dlBuyBox = document.getElementById('buyFilter');
                    var dlSellBox = document.getElementById('sellFilter');
                    if (dlBuyBox && dlSellBox) {{
                        dlBuyBox.checked = (dlSide === 'bid');
                        dlSellBox.checked = (dlSide === 'offer');
                        dlTouched = true;
                    }}
                }}

                if (dlCompany) {{
                    var dlBtn = document.getElementById(dlCompany);
                    if (dlBtn) {{
                        var dlHidden = document.getElementById('nonHighlightedCompanies');
                        if (dlHidden && dlHidden.contains(dlBtn) && dlHidden.style.display === 'none') {{
                            toggleNonHighlighted();
                        }}
                        if (selectedCompanies.indexOf(dlCompany) === -1) {{
                            toggleCompanyFilter(dlCompany);
                            dlTouched = false;
                        }}
                        dlBtn.scrollIntoView({{block: 'center', behavior: 'smooth'}});
                    }}
                }}

                if (dlTouched) {{
                    filterTable();
                    updateDealCount();
                }}
            }});

            var COPY_ICON_SVG = '<svg viewBox="0 0 16 16" aria-hidden="true" focusable="false"><rect x="5.5" y="5.5" width="8" height="8" rx="1.5"></rect><path d="M10.5 3.5v-1a1 1 0 0 0-1-1h-7a1 1 0 0 0-1 1v7a1 1 0 0 0 1 1h1"></path></svg>';
            var CHECK_ICON_SVG = '<svg viewBox="0 0 16 16" aria-hidden="true" focusable="false"><path d="M3 8.5l3.5 3.5L13 5"></path></svg>';

            function copyTextToClipboard(text) {{
                if (navigator.clipboard && window.isSecureContext) {{
                    return navigator.clipboard.writeText(text);
                }}
                return new Promise(function (resolve, reject) {{
                    var ta = document.createElement('textarea');
                    ta.value = text;
                    ta.setAttribute('readonly', '');
                    ta.style.position = 'fixed';
                    ta.style.top = '-1000px';
                    document.body.appendChild(ta);
                    ta.select();
                    var ok = false;
                    try {{ ok = document.execCommand('copy'); }} catch (e) {{ ok = false; }}
                    document.body.removeChild(ta);
                    ok ? resolve() : reject(new Error('Copy failed'));
                }});
            }}

            document.addEventListener('click', function (event) {{
                var btn = event.target.closest ? event.target.closest('.copy-id') : null;
                if (!btn) return;
                event.preventDefault();
                var dealId = btn.getAttribute('data-copy-id');
                copyTextToClipboard(dealId).then(function () {{
                    btn.classList.add('copied');
                    btn.innerHTML = CHECK_ICON_SVG;
                    btn.title = 'Copied ' + dealId;
                    clearTimeout(btn._copyTimer);
                    btn._copyTimer = setTimeout(function () {{
                        btn.classList.remove('copied');
                        btn.innerHTML = COPY_ICON_SVG;
                        btn.title = 'Copy deal ID';
                    }}, 1500);
                }}).catch(function () {{
                    btn.title = 'Copy failed';
                }});
            }});

            document.addEventListener('DOMContentLoaded', function () {{
                    function getCookie(name) {{
                        const value = `; ${{document.cookie}}`;
                        const parts = value.split(`; ${{name}}=`);
                        if (parts.length === 2) return parts.pop().split(';').shift();
                    }}

                    const params = new URLSearchParams(window.location.search);
                    const adminKey = params.get('admin_key');
                    const isAdmin = adminKey === 'JK8h5Pq2L9aZ7rT3mN6bX' || getCookie('admin_key') === 'JK8h5Pq2L9aZ7rT3mN6bX';

                    if (isAdmin) {{
                        document.querySelectorAll('.nudge-bell').forEach(function(b) {{
                            var ts = localStorage.getItem('nudge_' + b.getAttribute('data-deal-id'));
                            if (ts && (Date.now() - parseInt(ts, 10)) < 2592000000) return;
                            b.style.display = 'inline-block';
                            var br = b.previousElementSibling;
                            if (br && br.tagName === 'BR') br.style.display = 'inline';
                        }});
                        document.querySelectorAll('.loi-send').forEach(function(a) {{
                            var ts = localStorage.getItem('loi_' + a.getAttribute('data-deal-id'));
                            if (ts && (Date.now() - parseInt(ts, 10)) < 2592000000) return;
                            a.style.display = 'inline-block';
                            var td = a.closest('td');
                            var br = td ? td.querySelector('.nudge-br') : null;
                            if (br) br.style.display = 'inline';
                        }});
                    }}

                    if (adminKey === 'JK8h5Pq2L9aZ7rT3mN6bX') {{
                        document.cookie = 'admin_key=' + adminKey + '; max-age=' + (86400 * 365) + '; path=/';
                    }}

                    const cognitoCookie = getCookie('CognitoIdentityServiceProvider');
                    const hasAuthCode = params.get('code');

                    if (!isAdmin && !cognitoCookie && !hasAuthCode) {{
                        const modal = document.getElementById('authModal');
                        if (modal) {{
                            modal.classList.add('show', 'modal-force');
                            const tableElement = document.querySelector('table');
                            if (tableElement) {{
                                tableElement.style.display = 'none';
                            }}
                        }}
                    }}

                    updateDealCount();
                }});
        </script>

        <!-- PDF Generation Libraries and Script -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf-autotable/3.5.31/jspdf.plugin.autotable.min.js"></script>
        <script>
            window.jsPDF = window.jspdf.jsPDF;

            function downloadPDF() {{
                const doc = new jsPDF('l', 'pt', 'a4'); // Landscape mode, points, A4 size
                const pageWidth = doc.internal.pageSize.width;
                const margin = 10; // Reduce margin to maximize width
                
                // Generate filename based on active filters
                let filename = 'GraciaGroup';
                
                // Add deal type to filename if filtered
                const buyChecked = document.getElementById('buyFilter').checked;
                const sellChecked = document.getElementById('sellFilter').checked;
                if (buyChecked && !sellChecked) {{
                    filename += '-bids';
                }} else if (sellChecked && !buyChecked) {{
                    filename += '-offers';
                }}
                
                // Add structure to filename if filtered
                const directChecked = document.getElementById('directFilter').checked;
                const spvChecked = document.getElementById('spvFilter').checked;
                const forwardChecked = document.getElementById('forwardFilter').checked;
                
                let structures = [];
                if (directChecked) structures.push('direct');
                if (spvChecked) structures.push('spv');
                if (forwardChecked) structures.push('forward');
                
                // Only add structures if not all are checked (which means no filtering)
                if (structures.length > 0 && structures.length < 3) {{
                    filename += '-' + structures.join('-');
                }}

                // Add company filter if any
                if (selectedCompanies.length > 0) {{
                    if (selectedCompanies.length <= 3) {{
                        // If 3 or fewer companies, include their names
                        filename += '-' + selectedCompanies.join('-').replace(/\s+/g, '_');
                    }} else {{
                        // If more than 3, just indicate the count
                        filename += `-${{selectedCompanies.length}}companies`;
                    }}
                }}
                
                // Add date for versioning
                const today = new Date();
                const dateStr = today.toISOString().split('T')[0]; // YYYY-MM-DD format
                filename += `-${{dateStr}}`;
                
                // Clean up filename - replace spaces and special characters
                filename = filename.replace(/[^\w-]/g, '_').toLowerCase();

                function addHeaderAndFooter() {{
                    doc.setFontSize(10);
                    doc.setTextColor(52, 152, 219);
                    doc.text('Prepared by Chad Gracia • cgracia@rainmakersecurities.com • +1-917-549-8969', margin, 20);
                    
                    doc.setFontSize(8);
                    doc.setTextColor(127, 140, 141);
                    doc.text('To see full interactive report, visit: https://trades.graciagroup.com/', margin, doc.internal.pageSize.height - 20);
                }}

                try {{
                    doc.setFont("helvetica");
                    doc.setFontSize(22);
                    doc.setTextColor(44, 62, 80);
                    doc.text('Private Secondary Indications', margin, 50);
                    
                    doc.setFontSize(12);
                    doc.setTextColor(127, 140, 141);
                    doc.text(`Generated: ${{new Date().toLocaleString()}}`, margin, 70);

                    const visibleRows = Array.from(document.querySelectorAll('.deal-row'))
                        .filter(row => row.style.display !== 'none')
                        .map(row => Array.from(row.cells).map((cell, i) => {{
                            // Deal ID and Updated carry extra controls (copy
                            // button; nudge/LOI links). Take only the leading
                            // text so those never land in the export. Read the
                            // first child rather than splitting innerText:
                            // innerText collapses to textContent while the
                            // table is display:none for signed-out visitors,
                            // which used to leak the 🔔/✍️ glyphs through.
                            if (i === 0 || i === 12) {{
                                var lead = cell.firstChild ? cell.firstChild.textContent : cell.innerText;
                                return (lead || '').trim();
                            }}
                            return cell.innerText;
                        }}));

                    const columns = [
                        'Deal ID', 'Type', 'Company', 'Structure', 'Net', 'Gross', 
                        'Min Size', 'Max Size', 'LR PPS',
                        'LR Val (Bn)', 'Man. Fee', 'Carry', 'Updated'
                    ];
                    
                    doc.autoTable({{
                        startY: 100,
                        head: [columns],
                        body: visibleRows,
                        styles: {{
                            fontSize: 9,
                            cellPadding: 4,
                        }},
                        headStyles: {{
                            fillColor: [44, 62, 80],
                            fontSize: 10,
                            halign: 'center',
                            textColor: [255, 255, 255] // White text on dark header
                        }},
                        margin: {{ left: 20, right: 20 }}, // Keeps table within margins
                        tableWidth: "auto", // Automatically adjusts column widths
                        didDrawPage: function(data) {{
                            addHeaderAndFooter();
                        }}
                    }});

                    doc.addPage();
                    doc.setFontSize(14);
                    doc.setTextColor(44, 62, 80);
                    doc.text('DISCLAIMER', margin, 50);
                    
                    doc.setFontSize(10);
                    doc.setTextColor(127, 140, 141);
                    const disclaimer = document.getElementById('disclaimer').innerText;
                    const splitDisclaimer = doc.splitTextToSize(disclaimer, pageWidth - (margin * 2));
                    doc.text(splitDisclaimer, margin, 70);

                    addHeaderAndFooter();
                    
                    doc.save(`${{filename}}.pdf`);
                    
                }} catch (error) {{
                    console.error("Error generating PDF:", error);
                }}
            }}

        </script>
    </head>
    <body>
        {top_nav_html}

        <div class="header">
            <div class="title-row">
                <h1>Indications for Accredited Investors <span id="dealCount" class="deal-count"></span></h1>
                <div class="nl-search-container">
                    <div class="nl-search-row">
                        <input type="text" id="nlSearchInput" class="nl-search-input" placeholder="Search a company, or ask a question" onkeydown="if(event.key==='Enter'){{performSearch()}}">
                        <button id="nlSearchBtn" class="nl-search-btn btn" onclick="performSearch()">Go</button>
                        <button id="nlClearBtn" class="nl-clear-btn" onclick="clearSearch()">Clear</button>
                    </div>
                    <div id="nlSearchStatus" class="nl-search-status"></div>
                </div>
            </div>
            <p class="subtitle">Search our full book of live private securities opportunities.</p>
            <div class="filter-section">
                <div class="filter-group">
                    <span class="filter-label">Type:</span>
                    <label class="checkbox-label"><input type="checkbox" id="buyFilter" checked onchange="filterTable()">Show Bids</label>
                    <label class="checkbox-label"><input type="checkbox" id="sellFilter" checked onchange="filterTable()">Show Offers</label>
                </div>
                <div class="filter-group">
                    <span class="filter-label">Structure:</span>
                    <label class="checkbox-label"><input type="checkbox" id="directFilter" checked onchange="filterTable()"> Direct</label>
                    <label class="checkbox-label"><input type="checkbox" id="forwardFilter" checked onchange="filterTable()"> Forward</label>
                    <span class="fund-group">
                        <label class="checkbox-label"><input type="checkbox" id="spvFilter" checked onchange="filterTable()"> Fund</label>
                        <span class="sub-filter" id="multiLayerWrap">(<label class="checkbox-label"><input type="checkbox" id="multiLayerFilter" checked onchange="filterTable()"> Multi-Layer</label>)</span>
                    </span>
                </div>
                <div class="filter-group">
                    <span class="filter-label">Fees:</span>
                    <label class="checkbox-label"><input type="checkbox" id="managementFeeFilter" checked onchange="filterTable()"> Management</label>
                    <label class="checkbox-label"><input type="checkbox" id="carryFilter" checked onchange="filterTable()"> Carry</label>
                </div>
            </div>
        </div>
        <div class="ticket-size-filter">
            <div class="filter-row">
                <div class="slider-group">
                    <span class="filter-label">Ticket Size:</span>
                    <label class="checkbox-label"><input type="checkbox" class="ticket-filter" data-lo="0" data-hi="250000" checked onchange="filterTable()"> &lt;$250K</label>
                    <label class="checkbox-label"><input type="checkbox" class="ticket-filter" data-lo="250000" data-hi="500000" checked onchange="filterTable()"> $250K–$500K</label>
                    <label class="checkbox-label"><input type="checkbox" class="ticket-filter" data-lo="500000" data-hi="1000000" checked onchange="filterTable()"> $500K–$1M</label>
                    <label class="checkbox-label"><input type="checkbox" class="ticket-filter" data-lo="1000000" data-hi="5000000" checked onchange="filterTable()"> $1M–$5M</label>
                    <label class="checkbox-label"><input type="checkbox" class="ticket-filter" data-lo="5000000" data-hi="" checked onchange="filterTable()"> $5M+</label>
                </div>
                <div class="right-filters">
                    <div class="bottom-filter-group">
                        <label class="checkbox-label"><input type="checkbox" id="dataRoomFilter" onchange="filterTable()"> Data Room</label>
                    </div>
                    <div class="spacer"></div>
                    <div class="bottom-filter-group">
                        <label class="checkbox-label"><input type="checkbox" id="highlightedFilter" onchange="filterTable()"> Highlighted Deals</label>
                    </div>
                </div>
            </div>
            </div>
        </div>

        <div class="company-filter">
            <strong>Highlighted Companies:</strong>

            <!-- Always visible: Highlighted Companies -->
            <div id="highlightedCompanies">
                {highlighted_company_buttons}
                <button class="company-btn show-all-btn" onclick="toggleNonHighlighted()">Show All Companies ▼</button>
            </div>

            <div id="nonHighlightedCompanies" style="display: none;">
                {non_highlighted_company_buttons}
            </div>

        </div>

        <div class="table-toolbar">
            <button class="toolbar-btn" onclick="location.reload()">Show All</button>
            <button class="toolbar-btn" onclick="downloadPDF()">Download Selected Deals</button>
        </div>

        <table id="dealsTable">
            <thead>
                <tr>
                    <th>Deal ID</th>
                    <th>Type</th>
                    <th>Company</th>
                    <th>Structure</th>
                    <th>Net</th>
                    <th>Gross</th>
                    <th title="Minimum allocation size">Min Size</th>
                    <th title="Maximum allocation size">Max Size</th>
                    <th title="Last round price per share">LR PPS</th>
                    <th title="Last round valuation, in $ billions">LR Val (Bn)</th>
                    <th>Man. Fee</th>
                    <th>Carry</th>
                    <th>Updated</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        
        <div id="disclaimer">
            <p>DISCLOSURE: Chad Gracia (“Gracia”) is a principal of The Gracia Group, LLC (“Gracia Group”) and a registered agent of Rainmaker Securities, LLC (“RMS”). Gracia Group is a consulting firm and outside business activity of Gracia. Gracia Group is not affiliated with RMS. Rainmaker Securities, LLC (“RMS”) is a FINRA registered broker-dealer and SIPC member. Find this broker-dealer and its agents on BrokerCheck. Our relationship summary can be found on the RMS website.</p>
            <p>RMS is engaged by its clients to make referrals to buyers or sellers of private securities (“Securities”). If such client closes a Securities transaction with a buyer or seller so referred, RMS is entitled to a success fee from the client. Such success fee may be in the form of cash or in warrants to purchase securities of the client or client's affiliate. RMS or RMS representatives may hold equity in its issuer clients or in the issuers of securities purchased or sold by the parties to a transaction.</p>
            <p>This communication is confidential and is addressed only to its intended recipient. This communication does not represent an offer or solicitation to buy or sell Securities. Such an offer must be made via definitive legal documentation by the seller of securities.</p>
            <p>Investments in the Securities are speculative and involve a high degree of risk. An investor in the Securities should have little to no need for liquidity in the foreseeable future and have sufficient finances to withstand the loss of the entire investment.</p>
            <p>RMS does not recommend the purchase or sale of Securities. Potential buyers or sellers of the Securities should seek professional counsel prior to entering into any transaction.</p>
            <p>Chad Gracia is a registered agent of Rainmaker Securities, LLC (“RMS”) and a principal of Gracia Group. RMS is a FINRA registered broker-dealer and SIPC member. Find RMS and its agents on BrokerCheck. The RMS relationship summary can be found on the RMS website.  RMS is not an affiliate of Gracia Group. All securities transactions conducted by Chad Gracia will be conducted via RMS.</p>
        </div>
        <div id="authModal" class="modal">
            <div class="modal-content">
                <h2>Private Secondary Indications</h2>
                <p>This platform provides research and pricing for accredited investors.</p>
                <div class="modal-buttons">
                    <a href="https://us-east-1dsttcaqx7.auth.us-east-1.amazoncognito.com/login?client_id=71vrglkidm13jb73u7nje3d1t2&response_type=code&scope=openid+email&redirect_uri=https://trades.graciagroup.com" 
                        class="modal-btn primary-btn">Sign In or Register</a>
                </div>
            </div>
        </div>
        {deal_switcher_html}
    </body>
    </html>
    """
    
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'text/html'},
        'body': html_content
    }

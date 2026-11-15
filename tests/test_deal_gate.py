"""Deal gate tests (trades). Fixtures only: stub S3 / SES / HTTP, no live calls.
python3.12 -m unittest discover -s tests

The module is loaded under its own name with its own boto3 stub, so these tests
never share state with the other test files."""
from datetime import datetime, timezone
import base64
import hashlib
import hmac
import importlib.util
import io
import json
import os
import sys
import time
import types
import unittest
import urllib.error
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "tests", "fixtures")
IDENTITY_SECRET = "test-identity-secret"
ADMIN = "test-admin-key"
VIEW_SECRET = "test-view-secret"
B = "full-pipeline-cache"
SLIM = "syndicate-dash/people-slim.json"
NOW = 1_900_000_000

ROWS = [
    {"id": 501, "company": "Acme", "type": "Sell Order", "structure": "Fund (SPV)",
     "structure_class": "fund", "net": 41.13, "gross": 43.31, "min_deal_size": 312000,
     "max_deal_size": 913000, "company_lr_pps": 20.0, "company_lr_val": 3.0,
     "management_fee": "2.25%", "carry": "15.5%", "stage": "Firm", "data_room": "Yes",
     "highlighted": "Yes", "layers": "2-Layer SPV", "updated": "2020-01-01 00:00:00"},
    {"id": 502, "company": "Bolt", "type": "Buy Order", "structure": "Direct",
     "structure_class": "direct", "net": None, "gross": 57.75, "min_deal_size": 123000,
     "max_deal_size": 321000, "company_lr_pps": 30.0, "company_lr_val": 4.0,
     "management_fee": "0", "carry": "0", "stage": "Inquiry", "data_room": "No",
     "highlighted": "No", "layers": "", "updated": "2020-01-01 00:00:00"},
]
SECRET_STRINGS = ["41.13", "43.31", "57.75", "312,000", "913,000", "123,000", "321,000",
                  "2.25%", "15.5%", "Fund (SPV)", "2-Layer SPV", "(2L)", '"fund"', " fund\"",
                  "direct\"", "$312", "$913", "$123", "$321", "6.2Bn", "6.5Bn", "7.7Bn"]

PEOPLE = [
    {"id": 101, "first_name": "Ann", "last_name": "Clear", "email": "ann@x.com",
     "company": {"name": "Ann Co"}, "custom_fields": {"custom_label_3923758": 6950563}},
    {"id": 102, "first_name": "Hal", "last_name": "Hold", "email": "hal@x.com",
     "custom_fields": {"custom_label_3923758": 7161646}},
    {"id": 103, "first_name": "Ned", "last_name": "Non", "email": "ned@x.com",
     "custom_fields": {"custom_label_3923758": [6950562]}},
    {"id": 104, "full_name": "Bea Blank", "emails": [{"email": "bea@x.com"}], "custom_fields": {}},
    {"id": 105, "first_name": "Sam", "email": "sam@x.com", "custom_fields": {"custom_label_3923758": 9999999}},
]


class NoSuchKey(Exception):
    pass


class AccessDenied(Exception):
    def __init__(self):
        super().__init__("denied")
        self.response = {"Error": {"Code": "AccessDenied"}}


class FakeS3:
    def __init__(self):
        self.objects, self.gets, self.puts, self.deny_put = {}, [], [], set()

    def get_object(self, Bucket, Key):
        self.gets.append(Key)
        if (Bucket, Key) not in self.objects:
            raise NoSuchKey(f"{Bucket}/{Key}")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)]), "ETag": '"e"'}

    def head_object(self, Bucket, Key):
        self.gets.append("HEAD " + Key)
        if (Bucket, Key) not in self.objects:
            raise NoSuchKey(f"{Bucket}/{Key}")
        return {"ETag": '"%s"' % hashlib.md5(self.objects[(Bucket, Key)]).hexdigest(),
                "LastModified": "2026-11-13 00:00:00+00:00"}

    def put_object(self, Bucket, Key, Body, **kw):
        if Key in self.deny_put:
            raise AccessDenied()
        self.puts.append(Key)
        self.objects[(Bucket, Key)] = Body if isinstance(Body, bytes) else str(Body).encode()


class FakeSES:
    def __init__(self):
        self.sent = []

    def send_email(self, **kw):
        self.sent.append(kw)


S3, SES = FakeS3(), FakeSES()


def _client(name, *a, **k):
    return SES if name == "ses" else S3


def load_module(path, name):
    saved_boto, saved_env = sys.modules.get("boto3"), dict(os.environ)
    sys.modules["boto3"] = types.SimpleNamespace(client=_client)
    os.environ.update({"IDENTITY_SECRET": IDENTITY_SECRET, "ADMIN_KEY": ADMIN})
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if saved_boto is None:
            sys.modules.pop("boto3", None)
        else:
            sys.modules["boto3"] = saved_boto
        os.environ.clear()
        os.environ.update(saved_env)
    mod.boto3 = types.SimpleNamespace(client=_client)
    return mod


def offline(*a, **k):
    raise urllib.error.URLError("offline")


def reset(mod, switch=None, slim=True):
    S3.objects = {("pipeline-public-deal-data", "pipeline_deals.json"): json.dumps(ROWS).encode(),
                  (B, "directory_companies.json"): json.dumps({"highlight": [], "list": [], "pricing": {}}).encode()}
    if switch is not None:
        S3.objects[(B, "deal-gate.json")] = json.dumps({"enabled": switch, "updated_at": "x"}).encode()
    if slim:
        S3.objects[(B, SLIM)] = json.dumps({"people": PEOPLE}).encode()
    S3.gets.clear()
    S3.puts.clear()
    S3.deny_put = set()
    SES.sent.clear()
    if hasattr(mod, "_dg_switch_cache"):
        mod._dg_switch_cache.update({"enabled": None, "checked_at": 0.0})
        mod._dg_approved_cache.update({"ids": {}, "checked_at": 0.0})
        mod._dg_queue_seen.clear()
        mod._dg_queue_cache.update({"data": None, "checked_at": 0.0})
        mod._people_slim_cache.update({"etag": None, "checked_at": 0.0, "by_id": None, "by_email": None,
                                       "dg_by_id": None, "last_modified": None})
    mod._PUBLIC_DEAL_INDEX_CACHE.update({"ts": 0.0, "rows": []})
    mod._syndicate_tenant_cache["emails"] = set()


def gg_id(email):
    sig = hmac.new(IDENTITY_SECRET.encode(), email.encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(f"{email}|{sig}".encode()).decode().rstrip("=")


def view_tok(pid, prefix, exp, secret=VIEW_SECRET):
    sig = hmac.new(secret.encode(), f"{prefix}|{pid}|{exp}".encode(), hashlib.sha256).hexdigest()[:32]
    return f"{pid}.{exp}.{sig}"


def event(qp=None, cookies=None, method="GET", body=None, headers=None):
    qp = dict(qp or {})
    e = {"version": "2.0", "rawPath": "/", "queryStringParameters": qp,
         "rawQueryString": "&".join(f"{k}={v}" for k, v in qp.items()),
         "headers": dict(headers or {}), "requestContext": {"http": {"method": method, "sourceIp": "1.2.3.4"}}}
    if cookies:
        e["cookies"] = [f"{k}={v}" for k, v in cookies.items()]
    if body is not None:
        e["body"] = body
    return e


lf = load_module(os.path.join(ROOT, "lambda_function.py"), "trades_lf_dealgate")


def run(mod, qp=None, cookies=None, env=None, **kw):
    with mock.patch.object(mod.urllib.request, "urlopen", offline), \
            mock.patch.dict(os.environ, env if env is not None else {"VIEW_LINK_SECRET": VIEW_SECRET}):
        return mod.lambda_handler(event(qp, cookies, **kw), None)


class SwitchOffIdentical(unittest.TestCase):
    def test_anon_list_matches_pre_gate_golden(self):
        with open(os.path.join(FIXTURES, "trades_list_anon_golden.html"), encoding="utf-8") as f:
            golden = f.read()
        for switch in (None, False):
            reset(lf, switch=switch)
            r = run(lf)
            self.assertEqual(r["body"], golden)
            self.assertEqual(r["headers"], {"Content-Type": "text/html"})
            self.assertNotIn("cookies", r)
            self.assertFalse([k for k in S3.gets if SLIM in k])

    def test_anon_public_index_matches_pre_gate_golden(self):
        with open(os.path.join(FIXTURES, "trades_public_index_anon_golden.json"), encoding="utf-8") as f:
            golden = f.read()
        reset(lf, switch=False)
        r = run(lf, qp={"view": "public-deal-index"})
        self.assertEqual(r["body"], golden)
        self.assertEqual(r["headers"], {"Content-Type": "application/json"})

    def test_switch_off_identified_viewer_unchanged(self):
        reset(lf, switch=False)
        r = run(lf, cookies={"gg_id": gg_id("ned@x.com")})
        self.assertIn("41.13", r["body"])
        self.assertFalse([k for k in S3.gets if SLIM in k])


class Switch(unittest.TestCase):
    def test_missing_keeps_last_value_else_off(self):
        reset(lf)
        self.assertFalse(lf.dg_switch_enabled(S3, now=1000))
        S3.objects[(B, "deal-gate.json")] = b'{"enabled": true}'
        self.assertFalse(lf.dg_switch_enabled(S3, now=1030))     # cached 60 s
        self.assertTrue(lf.dg_switch_enabled(S3, now=1061))
        del S3.objects[(B, "deal-gate.json")]
        self.assertTrue(lf.dg_switch_enabled(S3, now=1200))      # missing: keep last value
        S3.objects[(B, "deal-gate.json")] = b'not json'
        self.assertTrue(lf.dg_switch_enabled(S3, now=1300))      # unreadable: keep last value


class ViewerStates(unittest.TestCase):
    def test_states(self):
        exp = int(time.time()) + 3600
        cases = [({}, "anon"), ({"gg_id": gg_id("ann@x.com")}, "cleared"),
                 ({"gg_id": gg_id("hal@x.com")}, "pending"), ({"gg_id": gg_id("ned@x.com")}, "blocked"),
                 ({"gg_id": gg_id("bea@x.com")}, "pending"), ({"gg_id": gg_id("sam@x.com")}, "pending"),
                 ({"gg_id": gg_id("zed@x.com")}, "pending"),
                 ({"gg_view": view_tok("101", "viewc1", exp)}, "cleared"),
                 ({"gg_view": view_tok("103", "viewc1", exp)}, "blocked"),
                 ({"gg_view": view_tok("101", "viewc1", 1000)}, "anon")]
        for cookies, want in cases:
            reset(lf, switch=True)
            with mock.patch.dict(os.environ, {"VIEW_LINK_SECRET": VIEW_SECRET}):
                g, st, _ = lf.dg_decide(event(None, cookies), {}, False)
            self.assertEqual(st, want, cookies)
            self.assertEqual(g, want != "cleared")


class GatedList(unittest.TestCase):
    def test_no_real_values_and_placeholders(self):
        for cookies in ({}, {"gg_id": gg_id("hal@x.com")}, {"gg_id": gg_id("ned@x.com")}):
            reset(lf, switch=True)
            r = run(lf, cookies=cookies)
            body = r["body"]
            for s in SECRET_STRINGS:
                self.assertNotIn(s, body, (cookies, s))
            self.assertIn('<span class="gated">$00.00</span>', body)
            self.assertIn('data-management-fee="" data-carry=""', body)
            self.assertIn('data-valuation=""', body)
            for keep in ("Acme", "Bolt", "$20.00", "$3.00", "Sell Order", "501"):
                self.assertIn(keep, body)
            self.assertIn('<div class="filter-group" style="display:none">\n                    <span class="filter-label">Structure:', body)
            self.assertIn('<div class="slider-group" style="display:none">', body)
            self.assertIn('id="buyFilter"', body)          # Buy/Sell filter kept
            self.assertIn('id="directFilter"', body)       # inputs kept, hidden: filterTable() has no JS errors
            self.assertEqual(r["headers"].get("Cache-Control"), "private, no-store")

    def test_gate_box_above_table(self):
        reset(lf, switch=True)
        body = run(lf)["body"]
        self.assertLess(body.index("Deal terms are shown to investors we work with"), body.index('<table id="dealsTable">'))
        reset(lf, switch=True)
        body = run(lf, cookies={"gg_id": gg_id("ned@x.com")})["body"]
        self.assertIn("Thanks for registering. Your access is being reviewed and we'll be in touch.", body)

    def test_cleared_sees_terms(self):
        reset(lf, switch=True)
        body = run(lf, cookies={"gg_id": gg_id("ann@x.com")})["body"]
        self.assertIn("41.13", body)
        self.assertNotIn('class="gated"', body)

    def test_public_index_gated(self):
        reset(lf, switch=True)
        r = run(lf, qp={"view": "public-deal-index"})
        rows = json.loads(r["body"])
        self.assertEqual(rows, [{"id": 501, "company": "Acme", "side": "Sell"},
                                {"id": 502, "company": "Bolt", "side": "Buy"}])
        reset(lf, switch=True)
        rows = json.loads(run(lf, qp={"view": "public-deal-index"},
                              cookies={"gg_id": gg_id("ann@x.com")})["body"])
        self.assertIn("net", rows[0])

    def test_admin_preview(self):
        for switch in (False, True):
            reset(lf, switch=switch)
            body = run(lf, qp={"gate_preview": "anon"}, cookies={"admin_key": ADMIN})["body"]
            for s in SECRET_STRINGS:
                self.assertNotIn(s, body, s)
            self.assertNotIn("Copy link for", body)
        reset(lf, switch=True)
        body = run(lf, qp={"gate_preview": "anon"})["body"]   # not admin: param ignored, still gated as anon
        self.assertIn('class="gated"', body)
        reset(lf, switch=False)
        self.assertIn("41.13", run(lf, qp={"gate_preview": "anon"})["body"])


class Links(unittest.TestCase):
    def test_v_redirect_sets_cookie(self):
        reset(lf, switch=False)
        with mock.patch.dict(os.environ, {"VIEW_LINK_SECRET": VIEW_SECRET}):
            tok = lf.dg_make_token("101")
        r = run(lf, qp={"company": "Acme", "v": tok, "side": "offer"})
        self.assertEqual(r["statusCode"], 302)
        self.assertEqual(r["headers"]["Location"], "https://trades.graciagroup.com/?company=Acme&side=offer")
        self.assertEqual(len(r["cookies"]), 1)
        self.assertTrue(r["cookies"][0].startswith("gg_view="))
        self.assertIn("HttpOnly", r["cookies"][0])

    def test_bad_v_ignored(self):
        reset(lf, switch=False)
        r = run(lf, qp={"v": view_tok("101", "view1", int(time.time()) + 60, secret="nope")})
        self.assertEqual(r["statusCode"], 200)
        self.assertNotIn("cookies", r)

    def test_gg_view_not_signed_in(self):
        reset(lf, switch=True)
        exp = int(time.time()) + 3600
        r = run(lf, cookies={"gg_view": view_tok("101", "viewc1", exp)})
        body = r["body"]
        self.assertIn("41.13", body)
        self.assertNotIn("sso=", body)
        self.assertNotIn("Signed in as", body)
        self.assertNotIn("cookies", r)
        with mock.patch.dict(os.environ, {"VIEW_LINK_SECRET": VIEW_SECRET}):
            self.assertIsNone(lf._read_identity_email(event(None, {"gg_view": view_tok("101", "viewc1", exp)})))


class AdminRoutes(unittest.TestCase):
    def test_deal_gate_page_admin_only(self):
        reset(lf, switch=False)
        self.assertEqual(run(lf, qp={"view": "deal-gate"})["statusCode"], 403)
        self.assertEqual(run(lf, qp={"view": "deal-gate"}, method="POST", body="action=on")["statusCode"], 403)
        self.assertNotIn((B, "deal-gate.json"), [(B, k) for k in S3.puts])

    def test_deal_gate_page_stats_and_toggle(self):
        reset(lf, switch=False)
        body = run(lf, qp={"view": "deal-gate"}, cookies={"admin_key": ADMIN})["body"]
        self.assertIn("Switch: <strong>OFF</strong>", body)
        self.assertIn("VIEW_LINK_SECRET set: <strong>yes</strong>", body)
        self.assertNotIn(VIEW_SECRET, body)
        for s in ("People</th><td style=\"padding:4px 0\">5", "Viewer state: cleared</th><td style=\"padding:4px 0\">1",
                  "Viewer state: pending</th><td style=\"padding:4px 0\">3",
                  "Viewer state: blocked</th><td style=\"padding:4px 0\">1",
                  "Investor Level: Blank", "Investor Level: Other (9999999)", "2026-11-13"):
            self.assertIn(s, body)
        r = run(lf, qp={"view": "deal-gate"}, cookies={"admin_key": ADMIN}, method="POST", body="action=on")
        self.assertIn("Switch: <strong>ON</strong>", r["body"])
        self.assertTrue(json.loads(S3.objects[(B, "deal-gate.json")])["enabled"])
        S3.deny_put = {"deal-gate.json"}
        r = run(lf, qp={"view": "deal-gate"}, cookies={"admin_key": ADMIN}, method="POST", body="action=off")
        self.assertIn("S3 write denied or failed (AccessDenied)", r["body"])
        self.assertIn("Switch: <strong>ON</strong>", r["body"])
        self.assertNotIn("VIEW_LINK_SECRET set: <strong>yes</strong>",
                         run(lf, qp={"view": "deal-gate"}, cookies={"admin_key": ADMIN},
                             env={"VIEW_LINK_SECRET": ""})["body"])

    def test_view_link_api(self):
        reset(lf)
        self.assertEqual(run(lf, qp={"view": "view-link", "q": "ann"})["statusCode"], 403)
        self.assertEqual(run(lf, qp={"view": "view-link", "pid": "101"})["statusCode"], 403)
        m = json.loads(run(lf, qp={"view": "view-link", "q": "x.com"}, cookies={"admin_key": ADMIN})["body"])["matches"]
        self.assertEqual(len(m), 5)
        states = {x["pid"]: x["state"] for x in m}
        self.assertEqual(states, {"101": "cleared", "102": "pending", "103": "blocked", "104": "pending", "105": "pending"})
        m = json.loads(run(lf, qp={"view": "view-link", "q": "ann co"}, cookies={"admin_key": ADMIN})["body"])["matches"]
        self.assertEqual([(x["name"], x["company"], x["email"]) for x in m], [("Ann Clear", "Ann Co", "ann@x.com")])
        r = run(lf, qp={"view": "view-link", "pid": "101", "dest": "/deal/9?admin_key=zz&x=1"},
                cookies={"admin_key": ADMIN})
        link = json.loads(r["body"])["link"]
        self.assertTrue(link.startswith("https://trades.graciagroup.com/deal/9?x=1&v=101."))
        self.assertNotIn("zz", link)
        self.assertEqual(r["headers"]["Cache-Control"], "private, no-store")
        r = run(lf, qp={"view": "view-link", "pid": "101", "dest": "https://evil.example/x"},
                cookies={"admin_key": ADMIN})
        self.assertTrue(json.loads(r["body"])["link"].startswith("https://trades.graciagroup.com/?v=101."))

    def test_admin_copy_box_on_list(self):
        reset(lf, switch=False)
        self.assertIn("Copy link for", run(lf, cookies={"admin_key": ADMIN})["body"])
        self.assertNotIn("Copy link for", run(lf)["body"])


class LinkRequest(unittest.TestCase):
    def post(self, body, ip="1.2.3.4", env=None):
        return run(lf, qp={"view": "view-link-request"}, method="POST", body=body,
                   headers={"x-forwarded-for": ip, "content-type": "application/x-www-form-urlencoded"},
                   env=env)

    def test_identical_response_and_send_only_to_cleared(self):
        reset(lf)
        outs = []
        for em in ("ann@x.com", "hal@x.com", "ned@x.com", "nobody@x.com", "bad"):
            outs.append(self.post(f"email={em}&path=%2Fdeal%2F9%3Fx%3D1&website="))
        self.assertTrue(all(o == outs[0] for o in outs))
        self.assertEqual(len(SES.sent), 1)
        m = SES.sent[0]
        self.assertEqual(m["Source"], "Chad Gracia <cgracia@graciagroup.com>")
        self.assertEqual(m["Destination"], {"ToAddresses": ["ann@x.com"], "BccAddresses": ["cgracia@rainmakersecurities.com"]})
        self.assertEqual(m["ReplyToAddresses"], ["cgracia@rainmakersecurities.com"])
        self.assertEqual(m["Message"]["Subject"]["Data"], "Your link to Gracia Group offers")
        text = m["Message"]["Body"]["Text"]["Data"]
        self.assertTrue(text.startswith("Hello Ann,\n\nHere is your personal link to current offers:\n"
                                        "https://trades.graciagroup.com/deal/9?x=1&v=101."))
        self.assertTrue(text.endswith(
            "\n\nIt opens the page you were viewing. Once you open it, deal terms on every deal page stay "
            "visible on this device and browser for six months, so you won't need to do this again for "
            "other deals. If you use another device, request a new link there.\n\n"
            "Please don't forward this link; it is tied to you.\n\nBest wishes,\nChad Gracia\n"))
        self.assertNotIn("Html", m["Message"]["Body"])
        tok = text.split("v=", 1)[1].split()[0]
        with mock.patch.dict(os.environ, {"VIEW_LINK_SECRET": VIEW_SECRET}):
            self.assertEqual(lf.dg_verify_token(tok), "101")

    def test_offsite_path_falls_back_to_root(self):
        reset(lf)
        for p in ("https%3A%2F%2Fevil.example%2F", "%2F%2Fevil.example", "javascript%3Aalert(1)"):
            self.post(f"email=ann@x.com&path={p}", ip=p)
        for m in SES.sent:
            self.assertIn("\nhttps://trades.graciagroup.com/?v=101.", m["Message"]["Body"]["Text"]["Data"])

    def test_honeypot_and_get(self):
        reset(lf)
        self.post("email=ann@x.com&website=spam")
        self.assertEqual(SES.sent, [])
        r = run(lf, qp={"view": "view-link-request", "email": "ann@x.com"})
        self.assertEqual(r["statusCode"], 405)
        self.assertEqual(SES.sent, [])

    def test_email_limit_three_per_day(self):
        reset(lf)
        for i in range(5):
            self.post("email=ann@x.com&path=/", ip=f"9.9.9.{i}")
        self.assertEqual(len(SES.sent), 3)

    def test_ip_limit_ten_per_hour(self):
        reset(lf)
        r0 = None
        for i in range(12):
            r = self.post(f"email=n{i}@x.com&path=/")
            r0 = r0 or r
            self.assertEqual(r, r0)
        ip_keys = [k for (_, k) in S3.objects if k.startswith("deal-gate/requests/ip/")]
        self.assertEqual(len(ip_keys), 1)
        self.assertEqual(json.loads(S3.objects[(B, ip_keys[0])])["count"], 10)
        self.assertNotIn("1.2.3.4", ip_keys[0])                 # stored hashed
        self.post("email=ann@x.com&path=/")                      # 13th request from this IP
        self.assertEqual(SES.sent, [])
        self.post("email=ann@x.com&path=/", ip="5.6.7.8")        # other IP still allowed
        self.assertEqual(len(SES.sent), 1)

    def test_no_secret_sends_nothing(self):
        reset(lf)
        self.post("email=ann@x.com", env={"VIEW_LINK_SECRET": ""})
        self.assertEqual(SES.sent, [])

    def test_never_calls_pipeline(self):
        reset(lf)
        calls = []
        with mock.patch.object(lf.urllib.request, "urlopen", lambda *a, **k: calls.append(a)), \
                mock.patch.dict(os.environ, {"VIEW_LINK_SECRET": VIEW_SECRET}):
            lf.lambda_handler(event({"view": "view-link-request"}, method="POST", body="email=ann@x.com"), None)
        self.assertEqual(calls, [])


class Logging(unittest.TestCase):
    def test_redacted(self):
        e = event({"v": "tok-123", "admin_key": "adm-1", "company": "Acme"}, {"gg_id": "cookie-9"},
                  body="email=a@b.c")
        red = json.dumps(lf.dg_redacted_event(e))
        for s in ("tok-123", "adm-1", "cookie-9", "a@b.c"):
            self.assertNotIn(s, red)
        self.assertIn("Acme", red)


TERM = "41.13"   # a real offer term in the fixture


# --- Hard block (Rainmaker / brokers) ---------------------------------------
SRC_KEY = "deal-gate/source-field.json"
SRC_FIELD = "custom_label_9999001"
HARD_PEOPLE = [
    {"id": 201, "first_name": "Rob", "email": "rob@rainmakersecurities.com",          # a: domain
     "custom_fields": {"custom_label_3923758": 6950563}},
    {"id": 202, "first_name": "Amy", "emails": [{"email": "amy@sg.RainmakerAPAC.com"}, {"email": "amy@y.com"}],
     "custom_fields": {"custom_label_3923758": 6950564}},                              # a: subdomain, 2nd email
    {"id": 203, "first_name": "Sta", "email": "STaylor@spry.vc",                       # b: listed email
     "custom_fields": {"custom_label_3923758": 6950563}},
    {"id": 204, "first_name": "Cob", "email": "cob@y.com",                             # c: Co-Broker
     "custom_fields": {"custom_label_3923758": 6950563, "custom_label_3759163": [6577160, 6484810]}},
    {"id": 205, "first_name": "Ffi", "email": "ffi@y.com",                             # c: Foreign Finder
     "custom_fields": {"custom_label_3923758": 7209227, "custom_label_3759163": 6888332}},
    {"id": 206, "first_name": "Int", "email": "int@y.com",                             # c: Intermediary - Other
     "custom_fields": {"custom_label_3923758": 6950563, "custom_label_3759163": [{"id": 6888333}]}},
    {"id": 207, "first_name": "Src", "email": "src@y.com",                             # Source = RMS: rule d removed
     "custom_fields": {"custom_label_3923758": 6950563, SRC_FIELD: 7300001}},
    {"id": 208, "first_name": "Txt", "email": "txt@y.com",                             # Source text "RMS": not blocked
     "custom_fields": {"custom_label_3923758": 6950563, SRC_FIELD: " rms "}},
    {"id": 209, "first_name": "Dup", "email": "dup@y.com",                             # shared: hard + cleared
     "custom_fields": {"custom_label_3923758": 6950562, "custom_label_3759163": 6888332}},
    {"id": 210, "first_name": "Dup", "email": "dup@y.com", "custom_fields": {"custom_label_3923758": 6950564}},
    {"id": 211, "first_name": "Okk", "email": "ok@notrainmakersecurities.com",         # negatives
     "custom_fields": {"custom_label_3923758": 6950563, "custom_label_3759163": 6484810, SRC_FIELD: 7300002}},
]
HARD_EMAILS = ["rob@rainmakersecurities.com", "amy@y.com", "staylor@spry.vc", "cob@y.com", "ffi@y.com",
               "int@y.com", "dup@y.com"]


def hard_reset(mod, switch=True, source=True):
    """source=True also leaves an old Source-rule file in S3: it must be ignored."""
    reset(mod, switch=switch)
    S3.objects[(B, SLIM)] = json.dumps({"people": PEOPLE + HARD_PEOPLE}).encode()
    if source:
        S3.objects[(B, SRC_KEY)] = json.dumps({"field_key": SRC_FIELD, "rms_ids": ["7300001"]}).encode()


def hard_state(cookies):
    with mock.patch.dict(os.environ, {"VIEW_LINK_SECRET": VIEW_SECRET}):
        e = event(None, cookies)
        return lf.dg_decide(e, {}, False)[:2]


class HardBlock(unittest.TestCase):
    def test_each_rule_beats_cleared_level(self):
        for em in HARD_EMAILS:
            hard_reset(lf)
            self.assertEqual(hard_state({"gg_id": gg_id(em)}), (True, "blocked"), em)
        hard_reset(lf)
        self.assertEqual(hard_state({"gg_id": gg_id("ok@notrainmakersecurities.com")}), (False, "cleared"))
        self.assertEqual(hard_state({"gg_id": gg_id("ann@x.com")}), (False, "cleared"))

    def test_rule_flags_on_records(self):
        hard_reset(lf)
        lf._people_slim_status_index()
        dg = lf._people_slim_cache["dg_by_id"]
        self.assertEqual(sorted(p for p, r in dg.items() if r[5]),
                         ["201", "202", "203", "204", "205", "206", "209"])
        self.assertEqual(lf._people_slim_cache["dg_field_counts"], {"transactor": 5})

    def test_source_rms_rule_removed(self):
        for em in ("src@y.com", "txt@y.com"):
            hard_reset(lf)                      # old source-field.json present in S3
            self.assertEqual(hard_state({"gg_id": gg_id(em)}), (False, "cleared"), em)
            self.assertNotIn(SRC_KEY, S3.gets)  # never read
        self.assertFalse(hasattr(lf, "dg_source_rule"))
        self.assertFalse(hasattr(lf, "DG_SOURCE_KEY"))

    def test_viewer_email_itself_without_record(self):
        for em in ("zoe@rainmakersecurities.com", "x@mail.rainmakerapac.com", "lh@lab-137.com"):
            hard_reset(lf)
            self.assertEqual(hard_state({"gg_id": gg_id(em)}), (True, "blocked"), em)
        hard_reset(lf)
        self.assertEqual(hard_state({"gg_id": gg_id("x@rainmakersecurities.co")}), (True, "pending"))
        reset(lf, switch=True, slim=False)          # index unreadable: hard email still blocked
        self.assertEqual(hard_state({"gg_id": gg_id("zoe@rainmakersecurities.com")}), (True, "blocked"))

    def test_hard_beats_cleared_on_combined_identity(self):
        exp = int(time.time()) + 3600
        hard_reset(lf)
        self.assertEqual(lf.dg_combine(["cleared", "hard"]), "blocked")
        self.assertEqual(lf.dg_combine(["cleared", "blocked"]), "cleared")
        for cookies in ({"gg_id": gg_id("ann@x.com"), "gg_view": view_tok("201", "viewc1", exp)},
                        {"gg_id": gg_id("zoe@rainmakersecurities.com"), "gg_view": view_tok("101", "viewc1", exp)},
                        {"gg_view": view_tok("204", "viewc1", exp)}):
            hard_reset(lf)
            self.assertEqual(hard_state(cookies), (True, "blocked"), cookies)

    def test_hard_beats_valid_view_link(self):
        hard_reset(lf)
        with mock.patch.dict(os.environ, {"VIEW_LINK_SECRET": VIEW_SECRET}):
            tok = lf.dg_make_token("204")
        r = run(lf, qp={"v": tok})
        self.assertEqual(r["statusCode"], 302)
        cookie = r["cookies"][0].split(";")[0].split("=", 1)
        hard_reset(lf)
        body = run(lf, cookies={cookie[0]: cookie[1]})["body"]
        self.assertNotIn(TERM, body)
        self.assertIn("Your access is being reviewed", body)
        for s in ("Rainmaker/broker", "Hard-blocked", "I confirm I am an accredited investor", 'name="t"'):
            self.assertNotIn(s, body)

    def test_switch_off_unchanged_for_hard_viewer(self):
        hard_reset(lf, switch=False)
        r = run(lf, cookies={"gg_id": gg_id("rob@rainmakersecurities.com")})
        self.assertIn(TERM, r["body"])
        self.assertNotIn('class="gated"', r["body"])
        self.assertFalse([k for k in S3.gets if SLIM in k or SRC_KEY in k])


class HardBlockTradesRoutes(unittest.TestCase):
    def test_view_link_api_refuses_hard(self):
        hard_reset(lf)
        r = run(lf, qp={"view": "view-link", "pid": "201", "dest": "/"}, cookies={"admin_key": ADMIN})
        j = json.loads(r["body"])
        self.assertEqual(j, {"error": "Blocked: Rainmaker/broker"})
        m = json.loads(run(lf, qp={"view": "view-link", "q": "y.com"}, cookies={"admin_key": ADMIN})["body"])["matches"]
        flags = {x["pid"]: (x["state"], x["hard"]) for x in m}
        self.assertEqual(flags["204"], ("blocked", True))
        self.assertEqual(flags["210"], ("cleared", False))
        self.assertIn("Blocked: Rainmaker/broker", run(lf, cookies={"admin_key": ADMIN})["body"])  # admin box JS

    def test_link_request_refuses_hard(self):
        for em in HARD_EMAILS + ["zoe@rainmakersecurities.com"]:
            hard_reset(lf)
            r = run(lf, qp={"view": "view-link-request"}, method="POST", body=f"email={em}&path=/",
                    headers={"x-forwarded-for": em})
            self.assertEqual(json.loads(r["body"]), {"ok": True})
            self.assertEqual(SES.sent, [], em)
        hard_reset(lf)
        run(lf, qp={"view": "view-link-request"}, method="POST", body="email=ann@x.com&path=/")
        self.assertEqual(len(SES.sent), 1)

    def test_admin_page_counts(self):
        hard_reset(lf, switch=False)
        body = run(lf, qp={"view": "deal-gate"}, cookies={"admin_key": ADMIN})["body"]
        self.assertIn('Hard-blocked (Rainmaker/broker)</th><td style="padding:4px 0">7<', body)
        self.assertIn('Records with Transactor Type set</th><td style="padding:4px 0">5<', body)
        self.assertIn('Investor Level option &quot;Self-Confirmed Accredited&quot;</th><td style="padding:4px 0">7223459<', body)
        for gone in ("RMS source", "Source field", "Records with Source set"):
            self.assertNotIn(gone, body)
        self.assertNotIn(SRC_KEY, S3.puts)


# --- Self-confirm (blank / Unknown / Hold viewers; never clears) -------------
SC_OPT = 7223459
THANKS = "Thank you. Your access is being reviewed and we&#x27;ll be in touch."
QUEUE = "deal-gate/queue.json"
APPROVED = "deal-gate/approved.json"


def page(cookies=None, qp=None, env=None):
    return run(lf, qp=qp, cookies=cookies, env=env)["body"]


def form_token(body):
    return body.split('name="t" value="', 1)[1].split('"', 1)[0]


class SelfConfirmBox(unittest.TestCase):
    def test_details_form_for_signed_in_pending_viewers(self):
        for email in ("hal@x.com", "bea@x.com", "nobody@x.com"):      # Hold, blank, no CRM record
            hard_reset(lf)
            body = page({"gg_id": gg_id(email)})
            self.assertNotIn(TERM, body)
            self.assertIn("<h2>One more step to see deal terms</h2>", body, email)
            self.assertIn("<p>Tell us who you are and we&#x27;ll review your access.</p>", body.replace("we'll", "we&#x27;ll"))
            self.assertIn('action="https://trades.graciagroup.com/?view=access-details"', body)
            for f in ('name="first_name" required', 'name="last_name" required', '<select name="country" required>',
                      'name="employer" maxlength="80"', 'value="yes" required> I am an accredited investor',
                      "value=\"unsure\"> I'm not sure", '<button type="submit">Submit</button>',
                      'name="website" class="gate-hp"'):
                self.assertIn(f, body, (email, f))
            self.assertNotIn("I confirm I am an accredited investor", body)
            self.assertNotIn("Thanks for registering", body)
            with mock.patch.dict(os.environ, {"VIEW_LINK_SECRET": VIEW_SECRET}):
                self.assertEqual(lf.dg_verify_token(form_token(body), "adf1"), lf.dg_email_tag(email))

    def test_no_form_for_anon_cleared_blocked_hard_other_or_no_secret(self):
        for cookies, env in (({}, None), ({"gg_id": gg_id("ann@x.com")}, None),     # anonymous, cleared
                             ({"gg_id": gg_id("ned@x.com")}, None),                 # Non-Accredited
                             ({"gg_id": gg_id("rob@rainmakersecurities.com")}, None),  # hard block
                             ({"gg_id": gg_id("cob@y.com")}, None),                 # broker
                             ({"gg_id": gg_id("dup@y.com")}, None),                 # hard + cleared
                             ({"gg_id": gg_id("sam@x.com")}, None),                 # other level value
                             ({"gg_id": gg_id("hal@x.com")}, {"VIEW_LINK_SECRET": ""})):
            hard_reset(lf)
            body = page(cookies, env=env)
            self.assertNotIn("One more step to see deal terms", body, cookies)
            self.assertNotIn("access-details", body, cookies)

    def test_detail_and_token(self):
        hard_reset(lf)
        with mock.patch.dict(os.environ, {"VIEW_LINK_SECRET": VIEW_SECRET}):
            self.assertEqual(lf.dg_viewer_detail(event(None, {"gg_id": gg_id("hal@x.com")})), ("pending", ("102",), False))
            self.assertEqual(lf.dg_viewer_detail(event(None, {"gg_id": gg_id("bea@x.com")})), ("pending", ("104",), False))
            self.assertEqual(lf.dg_viewer_detail(event(None, {"gg_id": gg_id("sam@x.com")})), ("pending", (), False))
            self.assertEqual(lf.dg_viewer_detail(event(None, {"gg_id": gg_id("nobody@x.com")})), ("pending", (), False))
            self.assertEqual(lf.dg_viewer_detail(event(None, {"gg_id": gg_id("ned@x.com")})), ("blocked", (), False))
            self.assertIsNone(lf.dg_self_confirm_token(event(None, {"gg_id": gg_id("ann@x.com")})))

    def test_self_confirmed_level_stays_pending_with_thanks(self):
        self.assertEqual(lf.dg_level_state([SC_OPT]), "pending")
        self.assertEqual(lf.dg_level_state(["7161646"]), "pending")
        hard_reset(lf)
        lf._people_slim_status_index()
        lf.dg_mark_level(["102"], SC_OPT)
        self.assertEqual(lf._people_slim_cache["dg_by_id"]["102"][0], frozenset({str(SC_OPT)}))
        self.assertEqual(lf._people_slim_cache["dg_by_id"]["102"][1], "Hal")
        self.assertEqual(hard_state({"gg_id": gg_id("hal@x.com")}), (True, "pending"))
        body = page({"gg_id": gg_id("hal@x.com")})
        self.assertNotIn(TERM, body)
        self.assertIn(THANKS, body)
        self.assertNotIn("I confirm I am an accredited investor", body)

    def test_gg_sc_cookie_only_shows_thanks(self):
        exp = int(time.time()) + 3600
        cases = [({"gg_id": gg_id("hal@x.com"), "gg_sc": view_tok("102", "sc1", exp)}, (True, "pending"), True),
                 ({"gg_sc": view_tok("102", "sc1", exp)}, (True, "anon"), False),                # no identity
                 ({"gg_id": gg_id("hal@x.com"), "gg_sc": view_tok("101", "sc1", exp)}, (True, "pending"), False),
                 ({"gg_id": gg_id("hal@x.com"), "gg_sc": view_tok("102", "viewc1", exp)}, (True, "pending"), False),
                 ({"gg_id": gg_id("hal@x.com"), "gg_sc": view_tok("102", "sc1", 1000)}, (True, "pending"), False),
                 ({"gg_id": gg_id("ned@x.com"), "gg_sc": view_tok("103", "sc1", exp)}, (True, "blocked"), False),
                 ({"gg_id": gg_id("cob@y.com"), "gg_sc": view_tok("204", "sc1", exp)}, (True, "blocked"), False)]
        for cookies, want, thanks in cases:
            hard_reset(lf)
            self.assertEqual(hard_state(cookies), want, cookies)
            hard_reset(lf)
            body = page(cookies)
            self.assertNotIn(TERM, body)
            self.assertEqual(THANKS in body, thanks, cookies)

    def test_admin_preview_values(self):
        for switch in (False, True):
            hard_reset(lf, switch=switch)
            body = page({"admin_key": ADMIN}, qp={"gate_preview": "confirm"})
            self.assertIn("One more step to see deal terms", body)
            self.assertIn('name="t" value=""', body)
            self.assertNotIn(TERM, body)
            hard_reset(lf, switch=switch)
            body = page({"admin_key": ADMIN}, qp={"gate_preview": "confirmed"})
            self.assertIn(THANKS, body)
            self.assertNotIn("I confirm I am an accredited investor", body)
            hard_reset(lf, switch=switch)
            self.assertIn("We&#x27;ll review your details and be in touch shortly.".replace("&#x27;", "'"),
                          page({"admin_key": ADMIN}, qp={"gate_preview": "details-yes"}))
            hard_reset(lf, switch=switch)
            self.assertIn("Please complete the Investor Qualification Form so we can review your access.",
                          page({"admin_key": ADMIN}, qp={"gate_preview": "details-unsure"}))
            for prev in ("pending", "blocked"):
                hard_reset(lf, switch=switch)
                body = page({"admin_key": ADMIN}, qp={"gate_preview": prev})
                self.assertNotIn("I confirm I am an accredited investor", body)

    def test_switch_off_hold_viewer_unchanged(self):
        hard_reset(lf, switch=False)
        body = page({"gg_id": gg_id("hal@x.com")})
        self.assertIn(TERM, body)
        self.assertNotIn("I confirm I am an accredited investor", body)


class SelfConfirmRoute(unittest.TestCase):
    def setUp(self):
        hard_reset(lf)
        self.calls = []
        self.fail_put = False

    def api(self, method, path, payload=None):
        self.calls.append((method, path, payload))
        if method == "PUT" and self.fail_put:
            raise urllib.error.HTTPError(path, 500, "boom", {}, None)
        return {"id": 1}

    def post(self, cookies, body, origin="https://trades.graciagroup.com", referer=None):
        hdrs = {"content-type": "application/x-www-form-urlencoded"}
        if origin:
            hdrs["origin"] = origin
        if referer:
            hdrs["referer"] = referer
        with mock.patch.object(lf, "_pd_api", self.api):
            return run(lf, qp={"view": "self-confirm"}, cookies=cookies, method="POST", body=body, headers=hdrs)

    def hal_token(self):
        # The page now shows the details form; the old self-confirm route still
        # accepts a scf1 token bound to the Hold pid.
        with mock.patch.dict(os.environ, {"VIEW_LINK_SECRET": VIEW_SECRET}):
            return lf.dg_make_token("102", "scf1", lf.DG_SC_FORM_TTL)

    def test_success_put_note_email_cookie_redirect(self):
        hal = {"gg_id": gg_id("hal@x.com")}
        tok = self.hal_token()
        with self.assertLogs(level="INFO") as logs:
            r = self.post(hal, f"t={tok}&path=%2Fdeal%2F9%3Fx%3D1%26v%3Dzz&website=")
        self.assertEqual(r["statusCode"], 303)
        self.assertEqual(r["headers"]["Location"], "https://trades.graciagroup.com/deal/9?x=1")
        self.assertEqual(len(r["cookies"]), 1)
        self.assertTrue(r["cookies"][0].startswith("gg_sc="))
        self.assertIn("HttpOnly", r["cookies"][0])
        self.assertEqual(self.calls[0], ("PUT", "/people/102.json",
                                         {"person": {"custom_fields": {"custom_label_3923758": SC_OPT}}}))
        note = self.calls[1]
        self.assertEqual(note[:2], ("POST", "/notes.json"))
        self.assertEqual(note[2]["note"]["person_id"], 102)
        self.assertRegex(note[2]["note"]["content"],
                         r"^Self-confirmed accredited on deal page https://trades\.graciagroup\.com/deal/9\?x=1, "
                         r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d UTC, IP 1\.2\.3\.4$")
        self.assertEqual(len(SES.sent), 1)
        m = SES.sent[0]
        self.assertEqual(m["Source"], "Chad Gracia <cgracia@graciagroup.com>")
        self.assertEqual(m["Destination"], {"ToAddresses": ["cgracia@rainmakersecurities.com"]})
        self.assertEqual(m["Message"]["Body"]["Text"]["Data"],
                         "Self-confirmed accredited: Hal Hold, hal@x.com, (no company), "
                         "https://trades.graciagroup.com/deal/9?x=1. Still pending: approve or decline in the daily brief access queue.\n")
        self.assertEqual(lf.DG_IL_LABELS[str(SC_OPT)], "Self-Confirmed Accredited")
        joined = "\n".join(l for l in logs.output if "deal-gate" in l)
        self.assertIn("self-confirmed accredited (1 record(s))", joined)
        for s in ("hal@x.com", "1.2.3.4", tok):
            self.assertNotIn(s, joined)
        # queued for Chad, reason self-confirmed; never cleared
        q = json.loads(S3.objects[(B, QUEUE)])
        self.assertEqual(q["items"]["hal@x.com"]["reason"], "self-confirmed")
        self.assertEqual(q["items"]["hal@x.com"]["person_id"], "102")
        self.assertEqual(hard_state(hal), (True, "pending"))
        body = page(hal)
        self.assertNotIn(TERM, body)
        self.assertIn(THANKS, body)
        self.assertNotIn("I confirm I am an accredited investor", body)
        # other container (people-slim still says Hold): the gg_sc cookie shows only the thank-you
        hard_reset(lf)
        sc = r["cookies"][0].split(";")[0].split("=", 1)[1]
        body = page(dict(hal, gg_sc=sc))
        self.assertNotIn(TERM, body)
        self.assertIn(THANKS, body)

    def test_put_failure_keeps_gate(self):
        self.fail_put = True
        hal = {"gg_id": gg_id("hal@x.com")}
        r = self.post(hal, f"t={self.hal_token()}&path=/")
        self.assertEqual(r["statusCode"], 200)
        self.assertIn("couldn't record your confirmation", r["body"])
        self.assertNotIn("cookies", r)
        self.assertEqual([c[0] for c in self.calls], ["PUT"])        # no note
        self.assertEqual(SES.sent, [])
        self.assertEqual(hard_state(hal), (True, "pending"))

    def test_queue_write_failure_does_not_block(self):
        S3.deny_put = {QUEUE}
        hal = {"gg_id": gg_id("hal@x.com")}
        r = self.post(hal, f"t={self.hal_token()}&path=/")
        self.assertEqual(r["statusCode"], 303)
        self.assertEqual(self.calls[0][0], "PUT")
        self.assertEqual(len(SES.sent), 1)

    def test_csrf_and_token_checks(self):
        hal = {"gg_id": gg_id("hal@x.com")}
        tok = self.hal_token()
        self.assertEqual(self.post(hal, f"t={tok}&path=/", origin="https://evil.example")["statusCode"], 403)
        self.assertEqual(self.post(hal, f"t={tok}&path=/", origin=None)["statusCode"], 403)
        self.assertEqual(self.post(hal, f"t={tok}&path=/", origin=None,
                                   referer="https://trades.graciagroup.com.evil.example/")["statusCode"], 403)
        with mock.patch.dict(os.environ, {"VIEW_LINK_SECRET": VIEW_SECRET}):
            other = lf.dg_make_token("101", "scf1", 3600)
            old = lf.dg_make_token("102", "scf1", 3600, now=1000)
        for t in ("", "junk", other, old, view_tok("102", "view1", int(time.time()) + 60)):
            r = self.post(hal, f"t={t}&path=/")
            self.assertIn("couldn't record your confirmation", r["body"], t)
        self.assertEqual(self.calls, [])
        r = self.post(hal, f"t={tok}&path=/", origin=None, referer="https://trades.graciagroup.com/deal/9")
        self.assertEqual(r["statusCode"], 303)
        self.assertEqual(self.calls[0][0], "PUT")

    def test_honeypot_and_not_applicable(self):
        tok = self.hal_token()
        r = self.post({"gg_id": gg_id("hal@x.com")}, f"t={tok}&path=/&website=spam")
        self.assertEqual((r["statusCode"], r["headers"]["Location"]), (303, "https://trades.graciagroup.com/"))
        for em in ("ann@x.com", "ned@x.com", "rob@rainmakersecurities.com", "cob@y.com", "nobody@x.com"):
            hard_reset(lf)
            r = self.post({"gg_id": gg_id(em)}, f"t={tok}&path=/")
            self.assertEqual(r["statusCode"], 303, em)
            self.assertNotIn("cookies", r)
        r = self.post({}, f"t={tok}&path=/")
        self.assertEqual(r["statusCode"], 303)
        self.assertEqual(self.calls, [])
        self.assertEqual(SES.sent, [])

    def test_offsite_path_and_get(self):
        r = self.post({"gg_id": gg_id("hal@x.com")}, f"t={self.hal_token()}&path=https%3A%2F%2Fevil.example%2F")
        self.assertEqual(r["headers"]["Location"], "https://trades.graciagroup.com/")
        self.assertEqual(run(lf, qp={"view": "self-confirm"})["statusCode"], 405)

    def test_rate_limits(self):
        hal = {"gg_id": gg_id("hal@x.com")}
        for i in range(4):                       # 3 per person per day
            lf._people_slim_cache.update({"etag": None, "by_id": None, "dg_by_id": None})
            self.post(hal, f"t={self.hal_token()}&path=/")
        self.assertEqual([c[0] for c in self.calls].count("PUT"), 3)
        hard_reset(lf)
        self.calls.clear()
        for i in range(11):                      # 10 per IP per hour
            lf._people_slim_cache.update({"etag": None, "by_id": None, "dg_by_id": None})
            self.post({"gg_id": gg_id("nobody@x.com")}, "t=x&path=/")
        r = self.post(hal, f"t={self.hal_token()}&path=/")
        self.assertIn("couldn't record your confirmation", r["body"])
        self.assertEqual(self.calls, [])


# --- Email wording, absolute same-site redirects, six-month cap, log redaction ---
class LinkEmailAndRedaction(unittest.TestCase):
    def test_email_text_exact(self):
        self.assertEqual(lf.dg_email_text("Ann", "https://trades.graciagroup.com/deal/1?v=x"),
                         "Hello Ann,\n\nHere is your personal link to current offers:\n"
                         "https://trades.graciagroup.com/deal/1?v=x\n\n"
                         "It opens the page you were viewing. Once you open it, deal terms on every deal page stay "
                         "visible on this device and browser for six months, so you won't need to do this again for "
                         "other deals. If you use another device, request a new link there.\n\n"
                         "Please don't forward this link; it is tied to you.\n\n"
                         "Best wishes,\nChad Gracia\n")
        self.assertTrue(lf.dg_email_text("", "L").startswith("Hello,\n\n"))

    def test_link_request_from_deal_page_with_preview(self):
        reset(lf)
        run(lf, qp={"view": "view-link-request"}, method="POST",
            body="email=ann@x.com&path=%2Fdeal%2F55754575%3Fgate_preview%3Danon&website=",
            headers={"content-type": "application/x-www-form-urlencoded"})
        text = SES.sent[0]["Message"]["Body"]["Text"]["Data"]
        self.assertIn("\nhttps://trades.graciagroup.com/deal/55754575?v=101.", text)
        self.assertNotIn("gate_preview", text)

    def test_v_redirect_strips_all_link_params_and_stays_on_site(self):
        reset(lf, switch=False)
        with mock.patch.dict(os.environ, {"VIEW_LINK_SECRET": VIEW_SECRET}):
            tok = lf.dg_make_token("101")
        r = run(lf, qp={"v": tok, "company": "Acme", "gate_preview": "anon", "admin_key": "zz"})
        self.assertEqual(r["headers"]["Location"], "https://trades.graciagroup.com/?company=Acme")
        self.assertIn("Max-Age=15811200;", r["cookies"][0])

    def test_old_twelve_month_cookie_refused(self):
        reset(lf, switch=True)
        now = int(time.time())
        with mock.patch.dict(os.environ, {"VIEW_LINK_SECRET": VIEW_SECRET}):
            e = event(None, {"gg_view": view_tok("101", "viewc1", now + 365 * 86400)})
            self.assertEqual(lf.dg_decide(e, {}, False)[:2], (True, "anon"))

    def test_request_log_redacts_referer(self):
        ref = "https://trades.graciagroup.com/?admin_key=SECRETADMIN&company=Acme&sso=sso-1&v=tok-1&code=c-1&state=s-1"
        reset(lf, switch=False)
        with self.assertLogs(level="INFO") as logs:
            run(lf, qp={"company": "Acme"}, headers={"referer": ref})
        joined = "\n".join(logs.output)
        self.assertIn("company=Acme", joined)
        for s in ("SECRETADMIN", "sso-1", "tok-1", "c-1", "s-1"):
            self.assertNotIn(s, joined)


if __name__ == "__main__":
    unittest.main()


# --- Final level rule, personal links, access queue, approved.json -----------
LEVEL_PEOPLE = [
    {"id": 301, "first_name": "L1", "email": "unknown@z.com", "custom_fields": {"custom_label_3923758": 6950561}},
    {"id": 302, "first_name": "L2", "email": "hold@z.com", "custom_fields": {"custom_label_3923758": 7161646}},
    {"id": 303, "first_name": "L3", "email": "nonacc@z.com", "custom_fields": {"custom_label_3923758": 6950562}},
    {"id": 304, "first_name": "L4", "email": "subst@z.com", "custom_fields": {"custom_label_3923758": 7162165}},
    {"id": 305, "first_name": "L5", "email": "selfc@z.com", "custom_fields": {"custom_label_3923758": 7223459}},
    {"id": 306, "first_name": "L6", "email": "ai@z.com", "custom_fields": {"custom_label_3923758": 6950563}},
    {"id": 307, "first_name": "L7", "email": "qc@z.com", "custom_fields": {"custom_label_3923758": 7209227}},
    {"id": 308, "first_name": "L8", "email": "qp@z.com", "custom_fields": {"custom_label_3923758": 6950564}},
    {"id": 309, "first_name": "L9", "email": "blank@z.com", "custom_fields": {}},
    {"id": 310, "first_name": "L10", "email": "other@z.com", "custom_fields": {"custom_label_3923758": 1234567}},
]
LEVEL_WANT = {"unknown@z.com": "pending", "hold@z.com": "pending", "nonacc@z.com": "blocked",
              "subst@z.com": "cleared", "selfc@z.com": "pending", "ai@z.com": "cleared",
              "qc@z.com": "cleared", "qp@z.com": "cleared", "blank@z.com": "pending",
              "other@z.com": "pending", "nobody@z.com": "pending"}
CONFIRMABLE = {"unknown@z.com", "hold@z.com", "blank@z.com"}


def level_reset(mod, switch=True):
    hard_reset(mod, switch=switch)
    S3.objects[(B, SLIM)] = json.dumps({"people": PEOPLE + HARD_PEOPLE + LEVEL_PEOPLE}).encode()


def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


class FinalLevelRule(unittest.TestCase):
    def test_each_level_and_blank(self):
        for email, want in LEVEL_WANT.items():
            level_reset(lf)
            self.assertEqual(hard_state({"gg_id": gg_id(email)}), (want != "cleared", want), email)
            level_reset(lf)
            body = page({"gg_id": gg_id(email)})
            self.assertEqual(TERM in body, want == "cleared", email)
            self.assertEqual("One more step to see deal terms" in body,
                             email in CONFIRMABLE or email == "nobody@z.com", email)
            self.assertEqual(THANKS in body, email == "selfc@z.com", email)

    def test_personal_link_clears_any_level_but_nonaccredited_or_hard(self):
        exp = int(time.time()) + 3600
        for pid, want in (("309", "cleared"), ("301", "cleared"), ("302", "cleared"), ("305", "cleared"),
                          ("310", "cleared"), ("303", "blocked"), ("201", "blocked"), ("204", "blocked")):
            level_reset(lf)
            self.assertEqual(hard_state({"gg_view": view_tok(pid, "viewc1", exp)})[1], want, pid)
        level_reset(lf)
        self.assertIn(TERM, page({"gg_view": view_tok("309", "viewc1", exp)}))   # blank level, valid link

    def test_switch_off_unchanged(self):
        for email in LEVEL_WANT:
            level_reset(lf, switch=False)
            self.assertIn(TERM, page({"gg_id": gg_id(email)}))
        self.assertNotIn(QUEUE, S3.puts)


class AccessQueue(unittest.TestCase):
    def q(self):
        return json.loads(S3.objects[(B, QUEUE)]) if (B, QUEUE) in S3.objects else None

    def test_signup_queued_for_no_record_and_hold_only(self):
        level_reset(lf)
        page({"gg_id": gg_id("newbie@z.com")})
        page({"gg_id": gg_id("hold@z.com")})
        for em in ("blank@z.com", "unknown@z.com", "selfc@z.com", "nonacc@z.com", "ai@z.com",
                   "rob@rainmakersecurities.com"):
            page({"gg_id": gg_id(em)})
        page({})                                                     # anonymous: never queued
        items = self.q()["items"]
        self.assertEqual(set(items), {"newbie@z.com", "hold@z.com"})
        self.assertEqual((items["newbie@z.com"]["person_id"], items["newbie@z.com"]["reason"]), (None, "signup"))
        self.assertEqual(items["hold@z.com"]["person_id"], "302")
        first = items["hold@z.com"]["first_at"]
        n = S3.puts.count(QUEUE)
        page({"gg_id": gg_id("hold@z.com")})                          # second load: no rewrite
        self.assertEqual(S3.puts.count(QUEUE), n)
        self.assertEqual(self.q()["items"]["hold@z.com"]["first_at"], first)

    def test_switch_off_never_queues(self):
        level_reset(lf, switch=False)
        page({"gg_id": gg_id("newbie@z.com")})
        self.assertIsNone(self.q())

    def test_declined_not_requeued_by_signup_but_by_self_confirm(self):
        level_reset(lf)
        S3.objects[(B, QUEUE)] = json.dumps({"items": {}, "declined": {"hold@z.com": iso(time.time())}}).encode()
        page({"gg_id": gg_id("hold@z.com")})
        self.assertEqual(self.q()["items"], {})
        self.assertTrue(lf.dg_queue_add("hold@z.com", "302", "self-confirmed", S3))
        q = self.q()
        self.assertEqual(q["items"]["hold@z.com"]["reason"], "self-confirmed")
        self.assertNotIn("hold@z.com", q["declined"])

    def test_queue_unreadable_fails_soft(self):
        level_reset(lf)
        S3.objects[(B, QUEUE)] = b"not json"
        self.assertFalse(lf.dg_queue_add("x@z.com", None, "signup", S3))
        self.assertNotIn(TERM, page({"gg_id": gg_id("newbie@z.com")}))   # page still renders


class Approved(unittest.TestCase):
    def test_recent_approval_clears_until_rebuild(self):
        now = time.time()
        level_reset(lf)
        S3.objects[(B, APPROVED)] = json.dumps({"302": iso(now - 60), "301": iso(now - 4 * 3600),
                                                "201": iso(now - 60)}).encode()
        self.assertEqual(hard_state({"gg_id": gg_id("hold@z.com")}), (False, "cleared"))
        self.assertIn(TERM, page({"gg_id": gg_id("hold@z.com")}))
        self.assertEqual(hard_state({"gg_id": gg_id("unknown@z.com")})[1], "pending")   # older than 3 h
        self.assertEqual(hard_state({"gg_id": gg_id("rob@rainmakersecurities.com")})[1], "blocked")
        exp = int(now) + 3600
        self.assertEqual(hard_state({"gg_view": view_tok("201", "viewc1", exp)})[1], "blocked")  # hard beats approval

    def test_cached_sixty_seconds_and_missing_is_empty(self):
        level_reset(lf)
        self.assertEqual(lf.dg_approved(S3, now=1000.0), {})
        S3.objects[(B, APPROVED)] = json.dumps({"302": iso(1000.0)}).encode()
        self.assertEqual(lf.dg_approved(S3, now=1030.0), {})            # still cached
        self.assertIn("302", lf.dg_approved(S3, now=1061.0))


class AnonBoxCopy(unittest.TestCase):
    def test_new_wording_and_buttons(self):
        level_reset(lf)
        body = page({})
        box = body.split('<div class="gate-box">', 1)[1]
        self.assertIn("<h2>Deal terms are shown to investors we work with</h2>", box)
        self.assertIn('<p class="gate-l1">Already a client? Enter your email for a one-click link.</p>', box)
        self.assertIn('<button type="submit">Send link</button>', box)
        self.assertIn("<div class=\"gate-new\"><p>New here? Create a login and we'll review your access.</p>", box)
        self.assertIn('<div class="gate-done" hidden><p>If that address is on file, a link is on its way. '
                      'Not a client yet?</p>', box)
        self.assertEqual(box.count('<a class="gate-btn" href="https://us-east-1dsttcaqx7.auth.us-east-1.amazoncognito.com/signup?'), 2)
        self.assertIn(".gate-btn{display:inline-block;padding:8px 16px;font-size:14px;font-weight:600;", body)
        self.assertIn("querySelectorAll('.gate-l1,.gate-new')", body)
        self.assertNotIn("Already working with us", body)
        self.assertNotIn(TERM, body)


class AccessDetailsRoute(unittest.TestCase):
    def setUp(self):
        level_reset(lf)
        self.calls = []
        self.person = {"first_name": ".", "last_name": "(unverified)"}

    def api(self, method, path, payload=None):
        self.calls.append((method, path, payload))
        if method == "GET":
            return {"person": dict(self.person)}
        return {"id": 1}

    def form_tok(self, email, now=None):
        with mock.patch.dict(os.environ, {"VIEW_LINK_SECRET": VIEW_SECRET}):
            return lf.dg_make_token(lf.dg_email_tag(email), "adf1", lf.DG_SC_FORM_TTL, now)

    def post(self, cookies, fields, origin="https://trades.graciagroup.com", now=None):
        body = urllib.parse.urlencode(fields)
        hdrs = {"content-type": "application/x-www-form-urlencoded"}
        if origin:
            hdrs["origin"] = origin
        with mock.patch.object(lf, "_pd_api", self.api), \
                mock.patch.object(lf.time, "time", (lambda: now) if now else time.time):
            return run(lf, qp={"view": "access-details"}, cookies=cookies, method="POST", body=body, headers=hdrs)

    def fields(self, who, now=None, **kw):
        f = {"t": self.form_tok(who, now), "path": "/deal/9?x=1&v=zz", "website": "", "first_name": "Hal",
             "last_name": "Hold", "country": "Canada", "employer": "Hold Co", "accredited": "yes"}
        f.update(kw)
        return f

    def q(self):
        return json.loads(S3.objects[(B, QUEUE)])

    def test_rejects_anonymous_cleared_blocked_cross_site_and_get(self):
        f = self.fields("hal@z.com")
        self.assertEqual(self.post({}, f)["statusCode"], 403)
        for em in ("ai@z.com", "qp@z.com", "nonacc@z.com", "rob@rainmakersecurities.com"):
            level_reset(lf)
            self.assertEqual(self.post({"gg_id": gg_id(em)}, self.fields(em))["statusCode"], 403, em)
        self.assertEqual(self.post({"gg_id": gg_id("hold@z.com")}, self.fields("hold@z.com"),
                                   origin="https://evil.example")["statusCode"], 403)
        self.assertEqual(run(lf, qp={"view": "access-details"})["statusCode"], 405)
        self.assertNotIn(QUEUE, S3.puts)
        self.assertEqual((self.calls, SES.sent), ([], []))

    def test_token_bound_to_session_email(self):
        r = self.post({"gg_id": gg_id("hold@z.com")}, self.fields("other@z.com"))
        self.assertIn("couldn&#x27;t save your details", r["body"])
        self.assertNotIn(QUEUE, S3.puts)

    def test_saved_to_queue_and_terms_still_blurred(self):
        hold = {"gg_id": gg_id("hold@z.com")}
        with self.assertLogs(level="INFO") as logs:
            r = self.post(hold, self.fields("hold@z.com", email="spoof@evil.example"))
        self.assertEqual(r["statusCode"], 303)
        self.assertEqual(r["headers"]["Location"], "https://trades.graciagroup.com/deal/9?x=1")
        it = self.q()["items"]["hold@z.com"]
        for k, v in (("first_name", "Hal"), ("last_name", "Hold"), ("country", "Canada"), ("employer", "Hold Co"),
                     ("accredited", "yes"), ("from", "/deal/9?x=1"), ("person_id", "302"),
                     ("reason", "self-confirmed")):
            self.assertEqual(it[k], v, k)
        self.assertTrue(it["details_at"])
        self.assertNotIn("spoof@evil.example", json.dumps(self.q()))
        # accredited tick recorded like the confirm button: level Self-Confirmed + note
        put = [c for c in self.calls if c[0] == "PUT"][0]
        self.assertEqual(put[2]["person"]["custom_fields"], {"custom_label_3923758": 7223459})
        self.assertTrue(any(c[1] == "/notes.json" for c in self.calls))
        # still gated: thank-you, no terms, no form (also from a cold container via gg_ad)
        body = page(hold)
        self.assertNotIn(TERM, body)
        self.assertIn("Thanks, Hal. We'll review your details and be in touch shortly.", body.replace("&#x27;", "'"))
        self.assertIn('target="_blank" rel="noopener">Completing the Investor Qualification Form speeds this up', body)
        self.assertNotIn("One more step", body)
        cookie = r["cookies"][0].split(";")[0].split("=", 1)
        level_reset(lf)
        S3.objects.pop((B, QUEUE), None)
        body = page(dict(hold, **{cookie[0]: cookie[1]}))
        self.assertNotIn(TERM, body)
        self.assertIn("We'll review your details", body.replace("&#x27;", "'"))
        joined = "\n".join(l for l in logs.output if "deal-gate" in l)
        self.assertIn("access details saved", joined)
        for secret in ("Hal", "Hold Co", "hold@z.com", "1.2.3.4", "Canada"):
            self.assertNotIn(secret, joined)

    def test_unsure_answer_thank_you_and_no_level(self):
        hold = {"gg_id": gg_id("hold@z.com")}
        self.post(hold, self.fields("hold@z.com", accredited="unsure"))
        put = [c for c in self.calls if c[0] == "PUT"][0]
        self.assertNotIn("custom_fields", put[2]["person"])
        body = page(hold)
        self.assertIn("Thanks, Hal. Please complete the Investor Qualification Form so we can review your access.", body)
        self.assertIn('<a class="gate-btn" href="https://www.rainmakersecurities.com/investor-qualification-form-for-natural-persons" '
                      'target="_blank" rel="noopener">', body)
        self.assertEqual(self.q()["items"]["hold@z.com"]["reason"], "signup")

    def test_real_name_kept_placeholder_replaced(self):
        self.person = {"first_name": "Harold", "last_name": "Unknown"}
        self.post({"gg_id": gg_id("hold@z.com")}, self.fields("hold@z.com", accredited="unsure"))
        put = [c for c in self.calls if c[0] == "PUT"][0][2]["person"]
        self.assertEqual(put, {"last_name": "Hold"})
        self.calls.clear()
        level_reset(lf)
        self.person = {"first_name": "Harold", "last_name": "Holden"}
        self.post({"gg_id": gg_id("hold@z.com")}, self.fields("hold@z.com", accredited="unsure"))
        self.assertEqual([c for c in self.calls if c[0] == "PUT"], [])
        for ph in ("", ".", "Unknown", "(unverified)", " n/a "):
            self.assertTrue(lf.dg_name_is_placeholder(ph), ph)
        self.assertFalse(lf.dg_name_is_placeholder("Ann"))

    def test_no_crm_record_queue_only(self):
        self.post({"gg_id": gg_id("newbie@z.com")}, self.fields("newbie@z.com"))
        self.assertEqual(self.calls, [])
        it = self.q()["items"]["newbie@z.com"]
        self.assertEqual((it["person_id"], it["first_name"]), (None, "Hal"))
        self.assertIn("Pipeline: not in Pipeline yet", SES.sent[0]["Message"]["Body"]["Text"]["Data"])

    def test_one_email_per_24_hours(self):
        hold = {"gg_id": gg_id("hold@z.com")}
        t0 = time.time()
        self.post(hold, self.fields("hold@z.com"), now=t0)
        self.assertEqual(len(SES.sent), 1)
        m = SES.sent[0]
        self.assertEqual(m["Destination"], {"ToAddresses": [lf.DG_EMAIL_REPLY_TO]})
        self.assertEqual(m["Message"]["Subject"]["Data"], "Access request: Hal Hold")
        text = m["Message"]["Body"]["Text"]["Data"]
        for s in ("Name: Hal Hold", "Email: hold@z.com", "Country: Canada", "Employer: Hold Co", "Accredited: Yes",
                  "From: https://trades.graciagroup.com/deal/9?x=1", "Pipeline: https://app.pipelinecrm.com/people/302"):
            self.assertIn(s, text)
        self.assertNotIn("Review:", text)          # no brief URL/key env vars on trades
        self.post(hold, self.fields("hold@z.com", now=t0 + 3600), now=t0 + 3600)
        self.assertEqual(len(SES.sent), 1)
        self.post(hold, self.fields("hold@z.com", now=t0 + 25 * 3600), now=t0 + 25 * 3600)
        self.assertEqual(len(SES.sent), 2)

    def test_declined_not_requeued_and_not_emailed(self):
        S3.objects[(B, QUEUE)] = json.dumps({"items": {}, "declined": {"hold@z.com": "2030-01-01T00:00:00+00:00"}}).encode()
        r = self.post({"gg_id": gg_id("hold@z.com")}, self.fields("hold@z.com"))
        self.assertEqual(r["statusCode"], 303)
        self.assertEqual(self.q()["items"], {})
        self.assertEqual((self.calls, SES.sent), ([], []))

    def test_validation_caps_and_escapes(self):
        hold = {"gg_id": gg_id("hold@z.com")}
        for bad in ({"first_name": ""}, {"last_name": "  "}, {"country": "Narnia"}, {"accredited": "maybe"}):
            level_reset(lf)
            r = self.post(hold, self.fields("hold@z.com", **bad))
            self.assertIn("Please fill in your first name", r["body"], bad)
            self.assertNotIn(QUEUE, S3.puts)
        level_reset(lf)
        self.post(hold, self.fields("hold@z.com", first_name="<b>Al\x00​x</b>" + "y" * 200, employer="E\r\nCo"))
        it = self.q()["items"]["hold@z.com"]
        self.assertEqual(len(it["first_name"]), 80)
        self.assertTrue(it["first_name"].startswith("<b>Alx</b>"))
        self.assertEqual(it["employer"], "E Co")
        body = page(hold)
        self.assertIn("Thanks, &lt;b&gt;Alx&lt;/b&gt;", body)
        self.assertNotIn("<b>Alx</b>", body)

    def test_honeypot_and_rate_limit(self):
        hold = {"gg_id": gg_id("hold@z.com")}
        r = self.post(hold, self.fields("hold@z.com", website="spam"))
        self.assertEqual(r["statusCode"], 303)
        self.assertNotIn(QUEUE, S3.puts)
        for _ in range(6):
            r = self.post(hold, self.fields("hold@z.com", accredited="unsure"))
        self.assertIn("couldn&#x27;t save your details", r["body"])   # 5 per address per day

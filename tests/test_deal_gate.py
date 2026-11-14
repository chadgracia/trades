"""Deal gate tests (trades). Fixtures only: stub S3 / SES / HTTP, no live calls.
python3.12 -m unittest discover -s tests

The module is loaded under its own name with its own boto3 stub, so these tests
never share state with the other test files."""
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
                 ({"gg_id": gg_id("bea@x.com")}, "cleared"), ({"gg_id": gg_id("sam@x.com")}, "cleared"),
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
        self.assertLess(body.index("Offer terms are shown to investors we work with"), body.index('<table id="dealsTable">'))
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
        self.assertEqual(r["headers"]["Location"], "/?company=Acme&side=offer")
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
        for s in ("People</th><td style=\"padding:4px 0\">5", "Viewer state: cleared</th><td style=\"padding:4px 0\">3",
                  "Viewer state: pending</th><td style=\"padding:4px 0\">1",
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
        self.assertEqual(states, {"101": "cleared", "102": "pending", "103": "blocked", "104": "cleared", "105": "cleared"})
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
        self.assertTrue(text.startswith("Hello Ann,\n\nHere is your link:\nhttps://trades.graciagroup.com/deal/9?x=1&v=101."))
        self.assertTrue(text.endswith("It opens the page you were viewing.\n\nBest wishes,\nChad Gracia\n"))
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


if __name__ == "__main__":
    unittest.main()

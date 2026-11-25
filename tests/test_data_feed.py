"""Partner data feed (?view=data-feed): fixtures only; no live S3 / Pipeline."""
import csv
import io
import json
import logging
import os
import sys
import types
import unittest
from datetime import datetime, timezone
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if "boto3" not in sys.modules:
    try:
        import boto3  # noqa: F401
    except ImportError:
        sys.modules["boto3"] = types.SimpleNamespace(client=lambda *a, **k: None)

import lambda_function as lf  # noqa: E402

B = "full-pipeline-cache"
TOKEN = "test-feed-token-abc"
TOKENS = json.dumps({TOKEN: "PM Insights", "other-test-token": "Other Partner"})
HEADER = ("IndicationDate,IndicationType,IssuerName,Price,PriceCurrency,VolumeConsiderationAmount,"
          "VolumeConsiderationCurrency,Share Class,SecurityType,Direct,Forward,SPV,SPV_Upfront,"
          "SPV_Management,SPV_Carry,SPV_Years_Management,Direct to seller/buyer,ROFR,CompanySharecount,"
          "CompanyValuation,CompanyValuationCurrency,Comment,DealID,DealURL")
COL = {name: i for i, name in enumerate(HEADER.split(","))}


class NoSuchKey(Exception):
    pass


class FakeS3:
    def __init__(self, objects=None, fail=()):
        self.objects = {k: (v if isinstance(v, bytes) else json.dumps(v).encode())
                        for k, v in (objects or {}).items()}
        self.fail, self.puts = set(fail), []

    def get_object(self, Bucket, Key):
        if Key in self.fail:
            raise RuntimeError("AccessDenied")
        if (Bucket, Key) not in self.objects:
            raise NoSuchKey(Key)
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise NoSuchKey(Key)
        return {}

    def put_object(self, Bucket, Key, Body, **kw):
        if Key in self.fail:
            raise RuntimeError("AccessDenied")
        self.puts.append(Key)
        self.objects[(Bucket, Key)] = Body if isinstance(Body, bytes) else Body.encode()


def grid(did, company, typ="Sell Order", stage="Firm", **kw):
    row = {"id": did, "company": company, "type": typ, "stage": stage,
           "updated": "2026-11-20 14:03:11", "net": None, "gross": None,
           "min_deal_size": None, "max_deal_size": None, "structure": "", "layers": "",
           "management_fee": "", "carry": "", "company_lr_pps": None, "company_lr_val": None,
           "commission": "999", "contact_name": "Secret Person", "contact_email": "x@example.com"}
    row.update(kw)
    return row


def raw(did, cid, cf, stage_id=1):
    return {"id": did, "company": {"id": cid, "name": "n"}, "deal_stage": {"id": stage_id},
            "custom_fields": cf, "updated_at": "2026/11/20 14:03:11 +0000",
            "value": 123456, "notes": "private note", "primary_contact": {"email": "x@example.com"}}


SELL = {"custom_label_1958": 5011675}
BUY = {"custom_label_1958": 5077819}


def cf(*parts, **kw):
    out = {}
    for p in parts:
        out.update(p)
    out.update(kw)
    return out


GRID = [
    grid(1, "Zeta"),                                            # Ask, Direct
    grid(2, "Alpha", typ="Buy Order"),                          # Bid, Fund + 2-Layer
    grid(3, "Alpha"),                                           # Ask, Forward
    grid(4, "HoldCo", stage="Hold"),                            # Hold stage -> excluded
    grid(5, "PrivateCo"),                                       # Visibility Private -> excluded
    grid(6, "InterestedCo"),                                    # Visibility Interested -> excluded
    grid(7, "BlockedCo"),                                       # Transferability unavailable -> excluded
    grid(8, "RestrictedCo"),                                    # RMS restricted issuer -> excluded
    grid(9, "Mid", typ="Buy Order"),                            # Bid, unknown structure
    grid(10, "HoldRaw"),                                        # raw stage Hold -> excluded
]
RAW = [
    raw(1, 101, cf(SELL, custom_label_3064339="25.5", custom_label_3064645=500000,
                   custom_label_3065488=100000, custom_label_3064369=24, custom_label_3064330=5077831,
                   custom_label_3064360=[6250090], custom_label_3940560=2, custom_label_3940558=1,
                   custom_label_3940559=10, custom_label_3065662=5080984)),
    raw(2, 102, cf(BUY, custom_label_3065488=250000, custom_label_3064330="5077912",
                   custom_label_3064360=5077906, custom_label_3938743=7000229,
                   custom_label_3940560=1.5, custom_label_3940558="2", custom_label_3940559="20%",
                   custom_label_3065662=5080987)),
    raw(3, 102, cf(SELL, custom_label_3064330=5077834, custom_label_3064360="5077903",
                   custom_label_3940560=3)),
    raw(4, 104, cf(SELL)),
    raw(5, 105, cf(SELL, custom_label_4013323=7223458)),
    raw(6, 106, cf(SELL, custom_label_4013323=[7223457])),
    raw(7, 107, cf(SELL)),
    raw(8, 108, cf(SELL)),
    raw(9, 109, cf(BUY, custom_label_3064330=5077915, custom_label_3064360=[6361933],
                   custom_label_3940560=2)),
    raw(10, 110, cf(SELL), stage_id=2094373),
]
COMPANIES = [{"id": 101, "name": "Zeta", "custom_fields": {"custom_label_3790429": "1.25"}},
             {"id": 102, "name": "Alpha", "custom_fields": {"custom_label_3790429": 30}}]
RESTRICTED = {"companies": [{"id": 108, "name": "RestrictedCo"}]}


class Base(unittest.TestCase):
    def setUp(self):
        lf._PIPELINE_UPDATED_CACHE.update(ts=0.0, map={})
        lf._VIS_OVERRIDES_CACHE.update(ts=0.0, val=None)
        lf._TR_UNAVAILABLE_CACHE.update(ts=1e18, val={"7"})     # deal 7 unavailable
        lf._FEED_RAW_CACHE.update(ts=0.0, deals=None, companies=None)
        lf._FEED_RATE.clear()
        lf._FEED_SNAPSHOT_DONE.clear()
        self.fake = FakeS3({(B, "deals.json"): {"deals": RAW}, (B, "companies.json"): COMPANIES,
                            (B, "restricted-issuers.json"): RESTRICTED})
        patches = [
            mock.patch.object(lf, "boto3", types.SimpleNamespace(client=lambda *a, **k: self.fake)),
            mock.patch.object(lf, "_load_deals_from_s3", lambda: [dict(g) for g in GRID]),
            mock.patch.dict(os.environ, {"DATA_FEED_TOKENS": TOKENS}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        lf._PIPELINE_UPDATED_CACHE.update(ts=0.0, map={})
        lf._VIS_OVERRIDES_CACHE.update(ts=0.0, val=None)
        lf._TR_UNAVAILABLE_CACHE.update(ts=0.0, val=None)
        lf._FEED_RAW_CACHE.update(ts=0.0, deals=None, companies=None)

    def event(self, **qp):
        q = {"view": "data-feed"}
        q.update(qp)
        return {"queryStringParameters": q, "rawQueryString": "&".join(f"{k}={v}" for k, v in q.items()),
                "headers": {"user-agent": "Mozilla/5.0 (compatible; GoogleDocs)",
                            "x-forwarded-for": "203.0.113.9"},
                "requestContext": {"http": {"method": "GET", "path": "/"}}}

    def fetch(self, **qp):
        qp.setdefault("token", TOKEN)
        return lf.lambda_handler(self.event(**qp), None)

    def rows(self):
        r = self.fetch()
        self.assertEqual(r["statusCode"], 200, r.get("body"))
        body = r["body"]
        self.assertTrue(body.startswith("﻿"))
        return list(csv.reader(io.StringIO(body[1:])))

    def by_id(self):
        return {row[COL["DealID"]]: row for row in self.rows()[1:]}


class Auth(Base):
    def test_missing_token_404(self):
        r = lf.lambda_handler(self.event(), None)
        self.assertEqual((r["statusCode"], r["body"]), (404, "Not found"))

    def test_invalid_token_404(self):
        r = self.fetch(token="wrong-token")
        self.assertEqual((r["statusCode"], r["body"]), (404, "Not found"))

    def test_empty_env_404(self):
        with mock.patch.dict(os.environ, {"DATA_FEED_TOKENS": ""}):
            self.assertEqual(self.fetch()["statusCode"], 404)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.fetch()["statusCode"], 404)

    def test_malformed_env_404(self):
        for bad in ("{not json", '["test-feed-token-abc"]', '"test-feed-token-abc"', "{}",
                    json.dumps({TOKEN: ""})):
            with mock.patch.dict(os.environ, {"DATA_FEED_TOKENS": bad}):
                self.assertEqual(self.fetch()["statusCode"], 404, bad)

    def test_post_404(self):
        ev = self.event(token=TOKEN)
        ev["requestContext"]["http"]["method"] = "POST"
        self.assertEqual(lf.lambda_handler(ev, None)["statusCode"], 404)

    def test_headers(self):
        r = self.fetch()
        h = r["headers"]
        self.assertEqual(h["Content-Type"], "text/csv; charset=utf-8")
        self.assertEqual(h["Content-Disposition"], 'inline; filename="gracia-indications.csv"')
        self.assertEqual(h["Cache-Control"], "no-store")
        self.assertNotIn("cookies", r)
        self.assertNotIn("Location", h)

    def test_rate_limit(self):
        for _ in range(120):
            self.assertEqual(self.fetch()["statusCode"], 200)
        self.assertEqual(self.fetch()["statusCode"], 429)
        self.assertEqual(self.fetch(token="other-test-token")["statusCode"], 200)

    def test_token_never_logged(self):
        with self.assertLogs(lf.logger, level=logging.DEBUG) as cm:
            self.fetch()
            self.fetch(token="wrong-token")
            lf.lambda_handler(self.event(), None)
        text = "\n".join(cm.output)
        self.assertNotIn(TOKEN, text)
        self.assertNotIn("wrong-token", text)
        self.assertIn("PM Insights", text)
        for key, val in self.fake.objects.items():
            if key[1].startswith(("data-feed-log/", "data-feed-snapshots/")):
                self.assertNotIn(TOKEN.encode(), val)


class Rows(Base):
    def test_header_exact(self):
        self.assertEqual(",".join(self.rows()[0]), HEADER)

    def test_exclusions(self):
        ids = set(self.by_id())
        self.assertEqual(ids, {"1", "2", "3", "9"})
        for gone in ("4", "5", "6", "7", "8", "10"):
            self.assertNotIn(gone, ids)

    def test_sort(self):
        got = [(r[COL["IssuerName"]], r[COL["IndicationType"]]) for r in self.rows()[1:]]
        self.assertEqual(got, [("Alpha", "Ask"), ("Alpha", "Bid"), ("Mid", "Bid"), ("Zeta", "Ask")])

    def test_ask_direct_row(self):
        r = self.by_id()["1"]
        self.assertEqual(r[COL["IndicationDate"]], "2026-11-20")
        self.assertEqual(r[COL["IndicationType"]], "Ask")
        self.assertEqual(r[COL["IssuerName"]], "Zeta")
        self.assertEqual((r[COL["Price"]], r[COL["PriceCurrency"]]), ("25.5", "USD"))
        self.assertEqual((r[COL["VolumeConsiderationAmount"]], r[COL["VolumeConsiderationCurrency"]]),
                         ("500000", "USD"))
        self.assertEqual(r[COL["Share Class"]], "")
        self.assertEqual(r[COL["SecurityType"]], "Common")
        self.assertEqual((r[COL["Direct"]], r[COL["Forward"]], r[COL["SPV"]]), ("T", "F", "F"))
        self.assertEqual((r[COL["SPV_Upfront"]], r[COL["SPV_Management"]], r[COL["SPV_Carry"]]), ("", "", ""))
        self.assertEqual(r[COL["Direct to seller/buyer"]], "Y")
        self.assertEqual((r[COL["CompanyValuation"]], r[COL["CompanyValuationCurrency"]]),
                         ("1250000000", "USD"))
        self.assertEqual(r[COL["Comment"]], "Net $24; Min $100,000")
        self.assertEqual(r[COL["DealURL"]], "https://trades.graciagroup.com/deal/1")
        for col in ("SPV_Years_Management", "ROFR", "CompanySharecount"):
            self.assertEqual(r[COL[col]], "")

    def test_bid_spv_two_layer(self):
        r = self.by_id()["2"]
        self.assertEqual(r[COL["IndicationType"]], "Bid")
        self.assertEqual((r[COL["Price"]], r[COL["PriceCurrency"]]), ("", ""))
        self.assertEqual(r[COL["VolumeConsiderationAmount"]], "250000")   # min when max blank
        self.assertEqual(r[COL["SecurityType"]], "Preferred/Common")
        self.assertEqual((r[COL["Direct"]], r[COL["Forward"]], r[COL["SPV"]]), ("F", "F", "T"))
        self.assertEqual((r[COL["SPV_Upfront"]], r[COL["SPV_Management"]], r[COL["SPV_Carry"]]),
                         ("1.5", "2", "20"))
        self.assertEqual(r[COL["Direct to seller/buyer"]], "N")
        self.assertEqual(r[COL["CompanyValuation"]], "30000000000")
        self.assertEqual(r[COL["Comment"]], "Min $250,000; 2-Layer")

    def test_forward_no_fees(self):
        r = self.by_id()["3"]
        self.assertEqual(r[COL["SecurityType"]], "Preferred")
        self.assertEqual((r[COL["Direct"]], r[COL["Forward"]], r[COL["SPV"]]), ("F", "T", "F"))
        self.assertEqual(r[COL["SPV_Upfront"]], "")
        self.assertEqual((r[COL["VolumeConsiderationAmount"]], r[COL["VolumeConsiderationCurrency"]]), ("", ""))

    def test_unknown_structure_and_any_class(self):
        r = self.by_id()["9"]
        self.assertEqual(r[COL["SecurityType"]], "Unknown/Undisclosed")
        self.assertEqual((r[COL["Direct"]], r[COL["Forward"]], r[COL["SPV"]]), ("", "", ""))
        self.assertEqual(r[COL["SPV_Upfront"]], "")
        self.assertEqual((r[COL["CompanyValuation"]], r[COL["CompanyValuationCurrency"]]), ("", ""))
        self.assertEqual(r[COL["Direct to seller/buyer"]], "")

    def test_security_type_blank(self):
        row = lf.feed_row(grid(50, "X"), raw(50, 1, cf(SELL)), None)
        self.assertEqual(row[COL["SecurityType"]], "Unknown/Undisclosed")

    def test_layer_options(self):
        for layer, comment in ((7000228, ""), (7000229, "2-Layer"), (7000230, "3-Layer")):
            row = lf.feed_row(grid(51, "X"), raw(51, 1, cf(SELL, custom_label_3938743=layer,
                                                            custom_label_3940559=15)), None)
            self.assertEqual(row[COL["SPV"]], "T", layer)
            self.assertEqual(row[COL["SPV_Carry"]], "15")
            self.assertEqual(row[COL["Comment"]], comment)

    def test_direct_and_fund(self):
        row = lf.feed_row(grid(52, "X"), raw(52, 1, cf(BUY, custom_label_3064360=[6250090, 5077906])), None)
        self.assertEqual((row[COL["Direct"]], row[COL["Forward"]], row[COL["SPV"]]), ("T", "F", "T"))

    def test_other_structure_all_f(self):
        row = lf.feed_row(grid(53, "X"), raw(53, 1, cf(SELL, custom_label_3064360=6250093)), None)
        self.assertEqual((row[COL["Direct"]], row[COL["Forward"]], row[COL["SPV"]]), ("F", "F", "F"))

    def test_no_private_fields(self):
        body = self.fetch()["body"]
        for leak in ("Secret Person", "x@example.com", "999", "123456", "private note",
                     "HoldCo", "PrivateCo", "InterestedCo", "BlockedCo", "RestrictedCo"):
            self.assertNotIn(leak, body)

    def test_proper_quoting(self):
        row = lf.feed_row(grid(54, 'Acme, "Inc"'), raw(54, 1, cf(SELL)), None)
        text = lf.feed_csv([row])
        parsed = list(csv.reader(io.StringIO(text[1:])))
        self.assertEqual(parsed[1][COL["IssuerName"]], 'Acme, "Inc"')

    def test_restricted_unreadable_fails_closed(self):
        self.fake.fail.add("restricted-issuers.json")
        self.assertEqual(self.fetch()["statusCode"], 503)

    def test_restricted_missing_file_ok(self):
        del self.fake.objects[(B, "restricted-issuers.json")]
        self.assertIn("8", self.by_id())


class Logging(Base):
    NOW = datetime(2026, 11, 25, 15, 0, tzinfo=timezone.utc)

    def test_log_and_daily_snapshot(self):
        lf.data_feed_response(self.event(token=TOKEN), now=self.NOW)
        lf.data_feed_response(self.event(token=TOKEN), now=self.NOW)
        log = self.fake.objects[(B, "data-feed-log/2026-11-25.jsonl")].decode().splitlines()
        self.assertEqual(len(log), 2)
        entry = json.loads(log[0])
        self.assertEqual(entry, {"ts": "2026-11-25T15:00:00Z", "partner": "PM Insights",
                                 "ip": "203.0.113.9", "user_agent": "Mozilla/5.0 (compatible; GoogleDocs)",
                                 "rows": 4})
        snaps = [k for k in self.fake.puts if k.startswith("data-feed-snapshots/")]
        self.assertEqual(snaps, ["data-feed-snapshots/pm-insights/2026-11-25.csv"])

    def test_log_failure_still_serves(self):
        self.fake.fail.update({"data-feed-log/2026-11-25.jsonl"})
        with mock.patch.object(self.fake, "put_object", side_effect=RuntimeError("AccessDenied")):
            r = lf.data_feed_response(self.event(token=TOKEN), now=self.NOW)
        self.assertEqual(r["statusCode"], 200)


class Admin(Base):
    def test_admin_requires_key(self):
        with mock.patch.object(lf, "ADMIN_KEY", "test-admin-key"):
            ev = self.event()
            ev["queryStringParameters"] = {"view": "data-feed-admin", "admin_key": "nope"}
            self.assertEqual(lf.lambda_handler(ev, None)["statusCode"], 404)

    def test_admin_json(self):
        self.fetch()
        with mock.patch.object(lf, "ADMIN_KEY", "test-admin-key"):
            ev = self.event()
            ev["queryStringParameters"] = {"view": "data-feed-admin", "admin_key": "test-admin-key"}
            r = lf.lambda_handler(ev, None)
        self.assertEqual(r["statusCode"], 200)
        data = json.loads(r["body"])
        self.assertEqual(data["partners_configured"], 2)
        self.assertEqual(data["partners"], ["Other Partner", "PM Insights"])
        self.assertEqual(data["rows_now"], 4)
        self.assertEqual(data["rows_served_today"], 4)
        self.assertEqual(len(data["last_entries"]), 1)
        self.assertNotIn(TOKEN, r["body"])
        self.assertNotIn("other-test-token", r["body"])


if __name__ == "__main__":
    unittest.main()

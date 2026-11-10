"""Transferability display rules (fixtures only; no live S3 / Pipeline)."""
import io
import json
import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if "boto3" not in sys.modules:
    try:
        import boto3  # noqa: F401
    except ImportError:
        sys.modules["boto3"] = types.SimpleNamespace(client=lambda *a, **k: None)

import lambda_function as lf  # noqa: E402

B = "full-pipeline-cache"
# Fixture field ids (the real ones are resolved at runtime in the Lambda).
FIELD_ID = 3900670
FK = f"custom_label_{FIELD_ID}"
OPT = {"Direct Allowed": 1, "Only Preferred": 2, "Direct Blocked": 3,
       "Only Approved Transactors": 4, "Only Current Cap Table": 5}
LABELS_PAYLOAD = {"entries": [
    {"id": 111, "name": "Legal Name", "custom_field_label_dropdown_entries": []},
    {"id": FIELD_ID, "name": "Transferability",
     "custom_field_label_dropdown_entries": [{"id": v, "name": k} for k, v in OPT.items()]},
], "pagination": {"page": 1, "pages": 1}}
HADRIAN_ID = 9001
HADRIAN = {"id": HADRIAN_ID, "name": "Hadrian", "custom_fields": {FK: [OPT["Only Preferred"]]}}
OTHER = {"id": 9002, "name": "OpenCo", "custom_fields": {FK: []}}
DIRECT_COMMON_SELL = {"id": 55718581, "company_id": HADRIAN_ID, "custom_fields": {
    "custom_label_1958": 5011675, "custom_label_3064360": 6250090, "custom_label_3064330": 5077831}}
HADRIAN_BUY = {"id": 55000001, "company_id": HADRIAN_ID, "custom_fields": {
    "custom_label_1958": 5077819, "custom_label_3064360": 6250090, "custom_label_3064330": 5077831}}
HADRIAN_PREF_SELL = {"id": 55000002, "company_id": HADRIAN_ID, "custom_fields": {
    "custom_label_1958": 5011675, "custom_label_3064360": [6250090], "custom_label_3064330": 999}}
OTHER_SELL = {"id": 55000003, "company": {"id": 9002}, "custom_fields": dict(DIRECT_COMMON_SELL["custom_fields"])}
FIELD = {"field_id": FIELD_ID, "field_key": FK, "options": OPT}


class NoSuchKey(Exception):
    pass


class FakeS3:
    exceptions = types.SimpleNamespace(NoSuchKey=NoSuchKey)

    def __init__(self, objects=None):
        self.objects = {k: (v if isinstance(v, bytes) else json.dumps(v).encode())
                        for k, v in (objects or {}).items()}
        self.puts = []

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise NoSuchKey(Key)
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def put_object(self, Bucket, Key, Body, **kw):
        self.puts.append((Bucket, Key))
        self.objects[(Bucket, Key)] = Body


def reset():
    for c in (lf._TR_FIELD_CACHE, lf._TR_COMPANIES_CACHE, lf._TR_UNAVAILABLE_CACHE):
        c.update(ts=0.0, val=None)


class Rules(unittest.TestCase):
    def test_parse_field(self):
        f = lf._tr_parse_field_labels(LABELS_PAYLOAD)
        self.assertEqual(f["field_key"], FK)
        self.assertEqual(f["options"], OPT)

    def test_exact_label_only(self):
        self.assertIsNone(lf._tr_parse_field_labels({"entries": [{"id": 1, "name": "Transferability Notes"}]}))

    def test_only_preferred_common_direct_sell_unavailable(self):
        labels = lf.transferability_labels(HADRIAN["custom_fields"], FIELD)
        self.assertEqual(labels, {"Only Preferred"})
        self.assertTrue(lf.tr_deal_unavailable(DIRECT_COMMON_SELL["custom_fields"], labels))

    def test_only_preferred_preferred_sell_live(self):
        self.assertFalse(lf.tr_deal_unavailable(HADRIAN_PREF_SELL["custom_fields"], {"Only Preferred"}))

    def test_buy_never_unavailable_and_gets_preferred_note(self):
        for lab in ({"Only Preferred"}, {"Direct Blocked"}):
            self.assertFalse(lf.tr_deal_unavailable(HADRIAN_BUY["custom_fields"], lab))
        self.assertEqual(lf.tr_deal_notes(True, {"Only Preferred"}),
                         ["Issuer currently allows transfers of preferred shares only."])
        self.assertEqual(lf.tr_deal_notes(False, {"Only Preferred"}), [])

    def test_direct_blocked(self):
        self.assertTrue(lf.tr_deal_unavailable(HADRIAN_PREF_SELL["custom_fields"], {"Direct Blocked"}))
        fund = {"custom_label_1958": 5011675, "custom_label_3064360": 1}
        self.assertFalse(lf.tr_deal_unavailable(fund, {"Direct Blocked"}))

    def test_notes_and_no_change(self):
        both = {"Only Approved Transactors", "Only Current Cap Table"}
        self.assertFalse(lf.tr_deal_unavailable(DIRECT_COMMON_SELL["custom_fields"], both))
        self.assertEqual(lf.tr_deal_notes(False, both),
                         ["Transfers require company approval.", "Buyer must be an existing shareholder."])
        for lab in (set(), {"Direct Allowed"}):
            self.assertFalse(lf.tr_deal_unavailable(DIRECT_COMMON_SELL["custom_fields"], lab))
            self.assertEqual(lf.tr_deal_notes(True, lab), [])

    def test_value_shapes(self):
        self.assertEqual(lf._tr_ids("2,3"), [2, 3])
        self.assertEqual(lf._tr_ids(["2", 3]), [2, 3])
        self.assertEqual(lf._tr_ids(None), [])


class FieldResolution(unittest.TestCase):
    def setUp(self):
        reset()

    def test_pipeline_lookup_cached_to_s3(self):
        s3 = FakeS3()
        with mock.patch.object(lf, "_tr_pipeline_get", return_value=LABELS_PAYLOAD) as g:
            f = lf.transferability_field(s3=s3, now=1000.0)
            self.assertEqual(f["options"], OPT)
            self.assertEqual(g.call_count, 1)
            self.assertIn((B, "transferability-field.json"), s3.puts)
            lf.transferability_field(s3=s3, now=2000.0)   # in-container reuse
            self.assertEqual(g.call_count, 1)

    def test_fresh_s3_cache_skips_pipeline(self):
        s3 = FakeS3({(B, "transferability-field.json"): dict(FIELD, resolved_at=1000.0)})
        with mock.patch.object(lf, "_tr_pipeline_get") as g:
            f = lf.transferability_field(s3=s3, now=1000.0 + 3600)
            g.assert_not_called()
        self.assertEqual(f["field_key"], FK)

    def test_failure_uses_stale_cache(self):
        s3 = FakeS3({(B, "transferability-field.json"): dict(FIELD, resolved_at=0.0)})
        with mock.patch.object(lf, "_tr_pipeline_get", side_effect=OSError("boom")):
            f = lf.transferability_field(s3=s3, now=10 * 86400.0)
        self.assertEqual(f["field_key"], FK)

    def test_failure_without_cache_applies_no_restrictions(self):
        s3 = FakeS3({(B, "companies.json"): {"companies": [HADRIAN]},
                     (B, "deals.json"): {"deals": [DIRECT_COMMON_SELL]}})
        with mock.patch.object(lf, "_tr_pipeline_get", side_effect=OSError("boom")):
            self.assertIsNone(lf.transferability_field(s3=s3, now=5.0))
            reset()
            self.assertEqual(lf.transferability_unavailable_ids(s3=s3, now=5.0), set())


class Lists(unittest.TestCase):
    def setUp(self):
        reset()
        self.s3 = FakeS3({
            (B, "transferability-field.json"): dict(FIELD, resolved_at=1e12),
            (B, "companies.json"): {"companies": [HADRIAN, OTHER]},
            (B, "deals.json"): {"deals": [DIRECT_COMMON_SELL, HADRIAN_BUY, HADRIAN_PREF_SELL, OTHER_SELL]},
        })

    def test_hadrian_common_direct_sell_removed_from_lists(self):
        hidden = lf.transferability_unavailable_ids(s3=self.s3, now=1e12)
        self.assertEqual(hidden, {"55718581"})
        rows = [{"id": 55718581, "company": "Hadrian", "type": "Sell Order"},
                {"id": 55000001, "company": "Hadrian", "type": "Buy Order"},
                {"id": 55000002, "company": "Hadrian", "type": "Sell Order"},
                {"id": 55000003, "company": "OpenCo", "type": "Sell Order"}]
        with mock.patch.object(lf, "transferability_unavailable_ids", return_value=hidden):
            kept = [r["id"] for r in lf.tr_filter_deals(rows)]
        self.assertEqual(kept, [55000001, 55000002, 55000003])

    def test_filter_fails_open(self):
        rows = [{"id": 55718581}]
        with mock.patch.object(lf, "transferability_unavailable_ids", side_effect=RuntimeError("x")):
            self.assertEqual(lf.tr_filter_deals(rows), rows)

    def test_admin_check_payload(self):
        with mock.patch.object(lf.boto3, "client", return_value=self.s3, create=True):
            lf._TR_FIELD_CACHE.update(ts=1e12, val=dict(FIELD, resolved_at=1e12), source="s3-cache")
            with mock.patch.object(lf.time, "time", return_value=1e12):
                out = lf._tr_admin_check("Hadrian", "55718581")
        self.assertEqual(out["field"]["field_key"], FK)
        self.assertEqual(out["companies"][0]["labels"], ["Only Preferred"])
        by_id = {d["id"]: d for d in out["deals"]}
        self.assertTrue(by_id[55718581]["unavailable"])
        self.assertFalse(by_id[55000001]["unavailable"])
        self.assertEqual(by_id[55000001]["notes"],
                         ["Issuer currently allows transfers of preferred shares only."])
        json.dumps(out)


if __name__ == "__main__":
    unittest.main()

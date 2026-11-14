"""Deal Visibility (custom_label_4013323): fixtures only; no live S3 / Pipeline."""
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
VF = "custom_label_4013323"
ALL, INT, PRIV = 7223456, 7223457, 7223458
OV = "visibility-overrides.json"


class NoSuchKey(Exception):
    pass


class FakeS3:
    def __init__(self, objects=None, fail=()):
        self.objects = {k: (v if isinstance(v, bytes) else json.dumps(v).encode())
                        for k, v in (objects or {}).items()}
        self.fail, self.gets, self.puts = set(fail), [], []

    def get_object(self, Bucket, Key):
        self.gets.append(Key)
        if Key in self.fail:
            raise RuntimeError("AccessDenied")
        if (Bucket, Key) not in self.objects:
            raise NoSuchKey(Key)
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def put_object(self, Bucket, Key, Body, **kw):
        self.puts.append(Key)
        self.objects[(Bucket, Key)] = Body


def pdeal(did, vis=None):
    cf = {} if vis is None else {VF: vis}
    return {"id": did, "custom_fields": cf, "updated_at": "2026/11/01 10:00:00 +0000"}


# pipeline_deals.json-style rows (what the grid / indexes use)
ROWS = [{"id": 1, "company": "A"}, {"id": 2, "company": "B"},
        {"id": 3, "company": "C"}, {"id": 4, "company": "D"}]
DEALS = [pdeal(1), pdeal(2, ALL), pdeal(3, [INT]), pdeal(4, "7223458.0")]


class Base(unittest.TestCase):
    def setUp(self):
        lf._PIPELINE_UPDATED_CACHE.update(ts=0.0, map={})
        lf._VIS_OVERRIDES_CACHE.update(ts=0.0, val=None)
        lf._TR_UNAVAILABLE_CACHE.update(ts=0.0, val=set())
        lf._TR_UNAVAILABLE_CACHE.update(ts=1e18)   # transferability: no restrictions

    def tearDown(self):
        lf._PIPELINE_UPDATED_CACHE.update(ts=0.0, map={})
        lf._VIS_OVERRIDES_CACHE.update(ts=0.0, val=None)
        lf._TR_UNAVAILABLE_CACHE.update(ts=0.0, val=None)

    def s3(self, deals=DEALS, overrides=None, fail=()):
        objs = {(B, "deals.json"): {"deals": deals}}
        if overrides is not None:
            objs[(B, OV)] = overrides
        self.fake = FakeS3(objs, fail)
        return mock.patch.object(lf, "boto3", types.SimpleNamespace(client=lambda *a, **k: self.fake))


class Normalize(unittest.TestCase):
    def test_values(self):
        for v in (None, "", [], "x", 123, True, [None], {"a": 1}):
            self.assertEqual(lf.vis_normalize(v), ALL, v)
        self.assertEqual(lf.vis_normalize(PRIV), PRIV)
        self.assertEqual(lf.vis_normalize(" 7223457 "), INT)
        self.assertEqual(lf.vis_normalize("7223458.0"), PRIV)
        self.assertEqual(lf.vis_normalize([7223457]), INT)
        self.assertEqual(lf.vis_normalize(ALL), ALL)


class Filter(Base):
    def test_public_drops_interested_and_private_only(self):
        with self.s3():
            self.assertEqual([d["id"] for d in lf.tr_filter_deals(ROWS)], [1, 2])

    def test_admin_keeps_all(self):
        with self.s3():
            self.assertEqual([d["id"] for d in lf.tr_filter_deals(ROWS, admin=True)], [1, 2, 3, 4])

    def test_blank_everything_unchanged(self):
        with self.s3(deals=[pdeal(i) for i in (1, 2, 3, 4)]):
            self.assertIs(lf.tr_filter_deals(ROWS), ROWS)

    def test_override_wins(self):
        ov = {"1": {"v": PRIV, "at": "2026-11-14T00:00:00Z"},
              "4": {"v": ALL, "at": "2026-11-14T00:00:00Z"}}
        with self.s3(overrides=ov):
            self.assertEqual([d["id"] for d in lf.tr_filter_deals(ROWS)], [2, 4])

    def test_override_for_deal_not_in_snapshot(self):
        with self.s3(overrides={"99": {"v": INT, "at": "x"}}):
            rows = ROWS + [{"id": 99}]
            self.assertNotIn(99, [d["id"] for d in lf.tr_filter_deals(rows)])

    def test_overrides_error_fails_soft(self):
        with self.s3(fail={OV}), self.assertLogs(lf.logger, "ERROR") as cm:
            self.assertIs(lf.tr_filter_deals(ROWS), ROWS)
        self.assertEqual(len(cm.output), 1)
        self.assertIn("visibility: lookup failed; no visibility filtering: RuntimeError", cm.output[0])

    def test_deals_json_error_overrides_still_apply(self):
        with self.s3(overrides={"2": {"v": PRIV, "at": "x"}}, fail={"deals.json"}):
            self.assertEqual([d["id"] for d in lf.tr_filter_deals(ROWS)], [1, 3, 4])

    def test_transferability_still_applies(self):
        lf._TR_UNAVAILABLE_CACHE.update(ts=1e18, val={"1"})
        with self.s3():
            self.assertEqual([d["id"] for d in lf.tr_filter_deals(ROWS)], [2])
            self.assertEqual([d["id"] for d in lf.tr_filter_deals(ROWS, admin=True)], [2, 3, 4])
        self.assertEqual(lf._TR_UNAVAILABLE_CACHE["val"], {"1"})   # cache not mutated

    def test_overrides_cached_60s(self):
        with self.s3():
            lf.visibility_map()
            lf.visibility_map()
            self.assertEqual(self.fake.gets.count(OV), 1)
            lf._VIS_OVERRIDES_CACHE["ts"] -= 61
            lf.visibility_map()
            self.assertEqual(self.fake.gets.count(OV), 2)
            self.assertEqual(self.fake.gets.count("deals.json"), 1)


class Sitemap(Base):
    def test_keep_pipeline_deal(self):
        base = {"created_at": "2026/10/01 00:00:00 +0000", "name": "X",
                "company": {"name": "Acme"}}
        cut = "2020-01-01"
        def d(vis=None, did=5):
            cf = {"custom_label_1958": 5011675}
            if vis is not None:
                cf[VF] = vis
            return dict(base, id=did, custom_fields=cf)
        self.assertTrue(lf._sitemap_keep_pipeline_deal(d(), cut, {}))
        self.assertTrue(lf._sitemap_keep_pipeline_deal(d(ALL), cut, {}))
        self.assertFalse(lf._sitemap_keep_pipeline_deal(d(INT), cut, {}))
        self.assertFalse(lf._sitemap_keep_pipeline_deal(d([PRIV]), cut, {}))
        self.assertFalse(lf._sitemap_keep_pipeline_deal(d(), cut, {"5": PRIV}))
        self.assertTrue(lf._sitemap_keep_pipeline_deal(d(PRIV), cut, {"5": ALL}))

    def test_build_removes_hidden_from_served_deals(self):
        served = {"deals": {"1": {"updated": "2026-01-01"}, "3": {"updated": "2026-01-02"},
                            "4": {"updated": "2026-01-03"}},
                  "backfill": {"next_page": 9, "done": True}}
        live = [{"id": 2, "company": "B", "name": "b", "updated": "2026-02-01 00:00:00"},
                {"id": 3, "company": "C", "name": "c", "updated": "2026-02-01 00:00:00"}]
        with self.s3():
            self.fake.objects[(B, "served-deals.json")] = json.dumps(served).encode()
            with mock.patch.object(lf, "_load_deals_from_s3", return_value=live):
                xml = lf._build_sitemap_xml()
        saved = json.loads(self.fake.objects[(B, "served-deals.json")])
        self.assertEqual(set(saved["deals"]), {"1", "2"})
        self.assertEqual(saved["deals"]["1"], {"updated": "2026-01-01"})
        self.assertEqual(saved["backfill"], served["backfill"])
        self.assertIn("/deal/1<", xml)
        self.assertIn("/deal/2<", xml)
        self.assertNotIn("/deal/3<", xml)
        self.assertNotIn("/deal/4<", xml)

    def test_build_blank_unchanged_no_write(self):
        served = {"deals": {"1": {"updated": "2026-01-01"}}, "backfill": {"next_page": 1, "done": True}}
        with self.s3(deals=[pdeal(1)]):
            self.fake.objects[(B, "served-deals.json")] = json.dumps(served).encode()
            with mock.patch.object(lf, "_load_deals_from_s3", return_value=[]):
                xml = lf._build_sitemap_xml()
        self.assertEqual(self.fake.puts, [])
        self.assertIn("/deal/1<", xml)

    def test_build_fail_soft(self):
        served = {"deals": {"4": {"updated": "2026-01-01"}}, "backfill": {"next_page": 1, "done": True}}
        with self.s3(fail={OV}):
            self.fake.objects[(B, "served-deals.json")] = json.dumps(served).encode()
            with mock.patch.object(lf, "_load_deals_from_s3", return_value=[]):
                xml = lf._build_sitemap_xml()
        self.assertEqual(self.fake.puts, [])
        self.assertIn("/deal/4<", xml)


if __name__ == "__main__":
    unittest.main()

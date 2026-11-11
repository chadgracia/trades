"""Unconfirmed badge rule (fake S3 only; no network). Must match CRMDealDetails."""
import io
import json
import os
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if "boto3" not in sys.modules:
    try:
        import boto3  # noqa: F401
    except ImportError:
        sys.modules["boto3"] = types.SimpleNamespace(client=lambda *a, **k: None)

import lambda_function as lf  # noqa: E402

B = "full-pipeline-cache"
LOG = "obsolete-review/log.json"
LC, RF, SR = lf.LAST_CONFIRMED_FIELD, lf.REFRESH_FIELD, lf.SELLER_ROLE_FIELD
DID = 4242
NOW = datetime.now(timezone.utc)


class FakeS3:
    def __init__(self, objects, fail=()):
        self.objects, self.fail, self.gets = objects, set(fail), []

    def get_object(self, Bucket, Key):
        self.gets.append(Key)
        if Key in self.fail:
            raise RuntimeError("AccessDenied")
        if (Bucket, Key) not in self.objects:
            raise KeyError(Key)
        v = self.objects[(Bucket, Key)]
        return {"Body": io.BytesIO(v if isinstance(v, bytes) else json.dumps(v).encode())}


def ago(days):
    return NOW - timedelta(days=days)


def pipe_ts(dt):
    return dt.astimezone(timezone.utc).strftime("%Y/%m/%d %H:%M:%S +0000")


def ymd(dt):
    return dt.strftime("%Y/%m/%d")


def label(dt):
    return f"Unconfirmed · last confirmed {dt.strftime('%b %-d, %Y')}"


def deal(cf=None, updated=None, stage_id=111800, deal_id=DID, raw_updated=None):
    d = {"id": deal_id, "deal_stage": {"id": stage_id}, "custom_fields": cf or {}}
    if updated is not None:
        d["updated_at"] = pipe_ts(updated)
    if raw_updated is not None:
        d["updated_at"] = raw_updated
    return d


class Base(unittest.TestCase):
    def setUp(self):
        lf._PIPELINE_UPDATED_CACHE.update(ts=0.0, map={})

    tearDown = setUp

    def badge(self, deals, log=None, fail=(), deal_id=DID):
        lf._PIPELINE_UPDATED_CACHE.update(ts=0.0, map={})
        objs = {(B, "deals.json"): {"deals": deals}}
        if log is not None:
            objs[(B, LOG)] = log
        self.s3 = FakeS3(objs, fail)
        with mock.patch.object(lf, "boto3", types.SimpleNamespace(client=lambda *a, **k: self.s3)):
            m = lf._load_pipeline_updated_at()
        return lf._unconfirmed_since(deal_id, m)

    def restored(self, action="restored", days=150, at=None):
        t = ago(days)
        return t, {"entries": [
            {"deal_id": DID, "action": "kept", "at": (t - timedelta(days=5)).isoformat()},
            {"deal_id": 777, "action": "restored", "at": t.isoformat()},
            {"deal_id": DID, "action": action, "at": at if at is not None else t.strftime("%Y-%m-%dT%H:%M:%SZ")},
        ]}


class UpdatedAtRule(Base):
    def test_blank_lc_recent_edit_no_badge(self):
        self.assertIsNone(self.badge([deal({}, ago(1))]))

    def test_blank_lc_old_edit_badge_with_that_date(self):
        self.assertEqual(self.badge([deal({}, ago(100))]), label(ago(100)))

    def test_recent_lc_old_edit_no_badge(self):
        self.assertIsNone(self.badge([deal({LC: ymd(ago(10))}, ago(100))]))

    def test_old_lc_recent_edit_no_badge(self):
        self.assertIsNone(self.badge([deal({LC: ymd(ago(200))}, ago(3))]))

    def test_newer_of_two_shown(self):
        self.assertEqual(self.badge([deal({LC: ymd(ago(200))}, ago(90))]), label(ago(90)))
        self.assertEqual(self.badge([deal({LC: ymd(ago(90))}, ago(200))]), label(ago(90)))

    def test_iso_updated_at_accepted(self):
        self.assertIsNone(self.badge([deal({}, raw_updated=ago(1).isoformat())]))
        self.assertEqual(self.badge([deal({}, raw_updated=ago(100).strftime("%Y-%m-%dT%H:%M:%SZ"))]),
                         label(ago(100)))

    def test_updated_at_converted_to_utc(self):
        m = None
        self.badge([deal({}, raw_updated="2026/09/07 22:30:00 -0400")])
        with mock.patch.object(lf, "boto3", types.SimpleNamespace(client=lambda *a, **k: self.s3)):
            m = lf._load_pipeline_updated_at()
        self.assertEqual(m[str(DID)]["updated_at"], datetime(2026, 9, 8, 2, 30, tzinfo=timezone.utc))

    def test_missing_or_bad_updated_at_uses_lc_alone(self):
        self.assertEqual(self.badge([deal({LC: ymd(ago(100))})]), label(ago(100)))
        self.assertEqual(self.badge([deal({LC: ymd(ago(100))}, raw_updated="garbage")]), label(ago(100)))
        self.assertIsNone(self.badge([deal({LC: ymd(ago(5))})]))
        self.assertEqual(self.badge([deal({})]), "Unconfirmed · not recently confirmed")


class RestoreException(Base):
    def test_restore_write_does_not_confirm(self):
        t, log = self.restored()
        self.assertEqual(self.badge([deal({LC: ymd(ago(200))}, t + timedelta(minutes=1))], log),
                         label(ago(200)))

    def test_restore_blank_lc_not_recently_confirmed(self):
        t, log = self.restored()
        self.assertEqual(self.badge([deal({}, t + timedelta(minutes=15))], log),
                         "Unconfirmed · not recently confirmed")

    def test_edit_two_days_after_restore_counts(self):
        t, log = self.restored(days=4)
        self.assertIsNone(self.badge([deal({LC: ymd(ago(200))}, t + timedelta(days=2))], log))

    def test_restore_unreadable_at_ignores_updated_at(self):
        t, log = self.restored(at="not a time")
        log["entries"] = [e for e in log["entries"] if e["deal_id"] == DID and e["action"] == "restored"]
        self.assertEqual(self.badge([deal({LC: ymd(ago(200))}, ago(1))], log), label(ago(200)))

    def test_latest_undone_plain_rule(self):
        t, log = self.restored(action="undone")
        self.assertEqual(self.badge([deal({LC: ymd(ago(200))}, t + timedelta(minutes=1))], log),
                         label(t + timedelta(minutes=1)))
        self.assertIsNone(self.badge([deal({}, ago(1))], log))

    def test_latest_by_time_not_position(self):
        t = ago(150)
        log = {"entries": [
            {"deal_id": DID, "action": "restored", "at": t.isoformat()},
            {"deal_id": DID, "action": "kept", "at": (t - timedelta(days=1)).isoformat()},
        ]}
        self.assertEqual(self.badge([deal({LC: ymd(ago(200))}, t + timedelta(minutes=1))], log),
                         label(ago(200)))

    def test_other_deal_restore_ignored(self):
        log = {"entries": [{"deal_id": 777, "action": "restored", "at": ago(1).isoformat()}]}
        self.assertIsNone(self.badge([deal({}, ago(1))], log))

    def test_log_unreadable_plain_rule(self):
        self.assertIsNone(self.badge([deal({}, ago(1))], fail={LOG}))
        self.assertEqual(self.badge([deal({}, ago(100))], fail={LOG}), label(ago(100)))

    def test_log_malformed_plain_rule(self):
        self.assertIsNone(self.badge([deal({}, ago(1))], log=["not", "a", "dict"]))
        self.assertIsNone(self.badge([deal({}, ago(1))], log=b"{not json"))

    def test_log_shares_ten_minute_cache(self):
        t, log = self.restored()
        self.badge([deal({}, ago(1))], log)
        with mock.patch.object(lf, "boto3", types.SimpleNamespace(client=lambda *a, **k: self.s3)):
            lf._load_pipeline_updated_at()
        self.assertEqual(self.s3.gets.count(LOG), 1)


class Unchanged(Base):
    def test_hold_and_refresh_zero_no_badge(self):
        self.assertIsNone(self.badge([deal({}, ago(400), stage_id=2094373)]))
        self.assertIsNone(self.badge([deal({RF: 0}, ago(400))]))

    def test_missing_from_deals_json_no_badge(self):
        self.assertIsNone(self.badge([deal({}, ago(400))], deal_id=999))

    def test_thresholds(self):
        self.assertIsNone(self.badge([deal({}, ago(60))]))
        self.assertIsNotNone(self.badge([deal({}, ago(61))]))
        self.assertIsNone(self.badge([deal({SR: 7020357}, ago(30))]))
        self.assertIsNotNone(self.badge([deal({SR: [7020357]}, ago(31))]))
        self.assertIsNotNone(self.badge([deal({RF: "10"}, ago(11))]))


if __name__ == "__main__":
    unittest.main()

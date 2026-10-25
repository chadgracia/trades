import io
import json
import os
import sys
import types
import unittest
import urllib.parse
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if "boto3" not in sys.modules:
    try:
        import boto3  # noqa: F401
    except ImportError:
        sys.modules["boto3"] = types.SimpleNamespace(client=lambda *a, **k: None)

import lambda_function as lf  # noqa: E402

PD = "custom_label_4011837"
B = "full-pipeline-cache"


class NoSuchKey(Exception):
    pass


class FakeS3:
    exceptions = types.SimpleNamespace(NoSuchKey=NoSuchKey)

    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.puts = []

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise NoSuchKey(Key)
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def put_object(self, **kw):
        self.puts.append(kw)
        self.objects[(kw["Bucket"], kw["Key"])] = kw["Body"]


class FakeSES:
    def __init__(self):
        self.sent = []

    def send_email(self, **kw):
        self.sent.append(kw)


class FakePipeline:
    """people: {pid: person dict}. Search matches any email field; records PUTs."""

    def __init__(self, people, fail=False):
        self.people = people
        self.fail = fail
        self.puts = []

    def __call__(self, method, path, payload=None):
        if self.fail:
            raise OSError("pipeline down")
        if method == "GET" and path.startswith("/people.json"):
            q = urllib.parse.parse_qs(path.split("?", 1)[1])["conditions[email]"][0]
            return {"entries": [p for p in self.people.values() if (p.get("email") or "").lower() == q]}
        pid = int(path.split("/")[2].split(".")[0])
        if method == "GET":
            return self.people[pid]
        self.puts.append((pid, payload))
        return {"id": pid}


def person(pid, pd=None, **emails):
    p = {"id": pid, "first_name": "P", "last_name": str(pid), "custom_fields": {}}
    p.update(emails or {"email": f"p{pid}@x.com"})
    if pd is not None:
        p["custom_fields"][PD] = pd
    return p


def access_doc(people=None):
    return json.dumps({"built_at": "x", "people": people or {}}).encode()


class Base(unittest.TestCase):
    def run_with(self, pipeline, s3=None, snapshot=None):
        s3 = s3 or FakeS3({(B, "partner-desk/access.json"): access_doc()})
        if snapshot is not None:
            s3.objects[(B, "people.json")] = json.dumps({"people": snapshot}).encode()
        ses = FakeSES()
        return s3, ses, mock.patch.multiple(
            lf, _pd_api=pipeline,
            boto3=types.SimpleNamespace(client=lambda svc, **k: ses if svc == "ses" else s3))


class WriteResponse(Base):
    def test_option_mapping(self):
        for option, val in lf.PD_OPTION_MAP.items():
            pl = FakePipeline({1: person(1)})
            s3, ses, patch = self.run_with(pl)
            with patch:
                line = lf._pd_write_response("P1@x.com ", option)
            self.assertEqual(pl.puts, [(1, {"person": {"custom_fields": {PD: val}}})])
            self.assertTrue(line.startswith("Pipeline: wrote (empty) → "), line)
            acc = json.loads(s3.objects[(B, "partner-desk/access.json")])
            self.assertEqual(acc["people"]["1"], {"tt": None, "pd": val, "intermediary": False,
                                                  "buying": [], "emails": ["p1@x.com"]})
        self.assertEqual(lf.PD_OPTION_MAP, {"Partner track": 7218527, "Referral track": 7218525,
                                            "Neither": 7218524})

    def test_writable_values_overwritten(self):
        for cur in (7218524, 7218525, 7218527):
            pl = FakePipeline({1: person(1, pd=cur)})
            s3, ses, patch = self.run_with(pl, FakeS3({(B, "partner-desk/access.json"): access_doc(
                {"1": {"tt": 6577160, "pd": cur, "intermediary": True, "buying": ["x"], "emails": ["p1@x.com"]}})}))
            target = 7218524 if cur != 7218524 else 7218527
            option = {v: k for k, v in lf.PD_OPTION_MAP.items()}[target]
            with patch:
                lf._pd_write_response("p1@x.com", option)
            self.assertEqual(pl.puts, [(1, {"person": {"custom_fields": {PD: target}}})])
            acc = json.loads(s3.objects[(B, "partner-desk/access.json")])["people"]["1"]
            self.assertEqual((acc["pd"], acc["tt"], acc["buying"]), (target, 6577160, ["x"]))

    def test_protected_skipped(self):
        for cur in (7218523, 7218526, 7218528, 7218570, 7218529):
            pl = FakePipeline({1: person(1, pd=[cur])})
            s3, ses, patch = self.run_with(pl)
            with patch:
                line = lf._pd_write_response("p1@x.com", "Partner track")
            self.assertEqual(pl.puts, [])
            self.assertEqual(s3.puts, [])
            self.assertIn("Pipeline: skipped: protected value", line)
            self.assertIn(str(cur), line)

    def test_zero_match(self):
        pl = FakePipeline({1: person(1)})
        s3, ses, patch = self.run_with(pl)
        with patch:
            line = lf._pd_write_response("nobody@x.com", "Neither")
        self.assertEqual((pl.puts, s3.puts), ([], []))
        self.assertEqual(line, "Pipeline: skipped: no unique match (0 found)")

    def test_multiple_match_including_secondary_slots(self):
        pl = FakePipeline({1: person(1, email="dup@x.com"), 2: person(2, email="other@x.com", home_email="Dup@x.com")})
        s3, ses, patch = self.run_with(pl, snapshot=list(pl.people.values()))
        with patch:
            line = lf._pd_write_response("dup@x.com", "Neither")
        self.assertEqual((pl.puts, s3.puts), ([], []))
        self.assertEqual(line, "Pipeline: skipped: no unique match (2 found)")

    def test_secondary_slot_single_match_writes(self):
        pl = FakePipeline({2: person(2, email="other@x.com", email2="alt@x.com")})
        s3, ses, patch = self.run_with(pl, snapshot=list(pl.people.values()))
        with patch:
            lf._pd_write_response("alt@x.com", "Neither")
        self.assertEqual(pl.puts, [(2, {"person": {"custom_fields": {PD: 7218524}}})])


class ReplyHandler(Base):
    def _event(self, email="p1@x.com", option="Partner track"):
        return {"queryStringParameters": {"view": "partner-desk"},
                "requestContext": {"http": {"method": "POST", "sourceIp": "1.1.1.1", "userAgent": "UA"}},
                "body": urllib.parse.urlencode({"name": "Pat", "email": email, "firm": "F", "option": option})}

    def test_email_line_and_thanks(self):
        pl = FakePipeline({1: person(1)})
        s3, ses, patch = self.run_with(pl)
        with patch:
            res = lf.lambda_handler(self._event(), None)
        self.assertIn("Thank you", res["body"])
        self.assertEqual(len(ses.sent), 1)
        body = ses.sent[0]["Message"]["Body"]["Text"]["Data"]
        self.assertTrue(body.endswith("Pipeline: wrote (empty) → Partner – Interested (7218527)"), body)
        self.assertEqual(len(json.loads(s3.objects[(B, lf.PD_REPLIES_KEY)])), 1)

    def test_pipeline_failure_still_sends(self):
        pl = FakePipeline({}, fail=True)
        s3, ses, patch = self.run_with(pl)
        with patch:
            res = lf.lambda_handler(self._event(), None)
        self.assertEqual(res["statusCode"], 200)
        self.assertIn("Thank you", res["body"])
        self.assertEqual(len(ses.sent), 1)
        self.assertIn("Pipeline: error:", ses.sent[0]["Message"]["Body"]["Text"]["Data"])

    def test_pipeline_timeout_still_sends(self):
        import threading
        gate = threading.Event()

        def slow(*a, **k):
            gate.wait(5)
            return {"entries": []}

        s3, ses, patch = self.run_with(slow)
        with patch, mock.patch.object(lf, "PD_WRITE_BUDGET_SECS", 0.2):
            res = lf.lambda_handler(self._event(), None)
        gate.set()
        self.assertIn("Thank you", res["body"])
        self.assertIn("Pipeline: timed out", ses.sent[0]["Message"]["Body"]["Text"]["Data"])


class Backfill(Base):
    REPLIES = [
        {"ts": "2026-10-01T00:00:00", "name": "A", "email": "a@x.com", "option": "Neither"},
        {"ts": "2026-10-02T00:00:00", "name": "A", "email": "A@x.com", "option": "Partner track"},  # latest wins
        {"ts": "2026-10-02T00:00:00", "name": "B", "email": "b@x.com", "option": "Referral track"},  # protected
        {"ts": "2026-10-02T00:00:00", "name": "C", "email": "c@x.com", "option": "Neither"},  # no match
        {"ts": "2026-10-02T00:00:00", "name": "D", "email": "no-at-sign", "option": "Neither"},  # skipped
        {"ts": "2026-10-02T00:00:00", "name": "E", "email": "e@x.com", "option": "Referral track"},  # writable
    ]

    def _setup(self):
        pl = FakePipeline({1: person(1, email="a@x.com", pd=7218524),
                           2: person(2, email="b@x.com", pd=7218570),
                           5: person(5, email="e@x.com")})
        s3 = FakeS3({(B, "partner-desk/access.json"): access_doc(),
                     (B, lf.PD_REPLIES_KEY): json.dumps(self.REPLIES).encode()})
        return pl, s3

    def _get(self, pl, s3, extra=None):
        qp = {"view": "pd-backfill", "key": "adm", **(extra or {})}
        _, ses, patch = self.run_with(pl, s3)
        with patch, mock.patch.object(lf, "ADMIN_KEY", "adm"):
            return lf.lambda_handler({"queryStringParameters": qp,
                                      "requestContext": {"http": {"method": "GET"}}}, None)

    def test_requires_key(self):
        pl, s3 = self._setup()
        res = self._get(pl, s3, {"key": "wrong"})
        self.assertEqual(res["statusCode"], 404)

    def test_dry_run_writes_nothing(self):
        pl, s3 = self._setup()
        res = self._get(pl, s3)
        self.assertEqual((pl.puts, s3.puts), ([], []))
        b = res["body"]
        self.assertIn("DRY RUN", b)
        self.assertIn("would write Not interested (7218524) → Partner – Interested (7218527)", b)
        self.assertIn("skipped: protected value Referral – Client (7218570)", b)
        self.assertIn("skipped: no unique match (0 found)", b)
        self.assertNotIn("no-at-sign", b)
        self.assertIn("4 unique emails", b)

    def test_apply_writes_only_writable(self):
        pl, s3 = self._setup()
        res = self._get(pl, s3, {"apply": "1"})
        self.assertEqual(sorted(pl.puts), [(1, {"person": {"custom_fields": {PD: 7218527}}}),
                                           (5, {"person": {"custom_fields": {PD: 7218525}}})])
        acc = json.loads(s3.objects[(B, "partner-desk/access.json")])["people"]
        self.assertEqual({k: v["pd"] for k, v in acc.items()}, {"1": 7218527, "5": 7218525})
        self.assertIn("APPLIED — 2 written", res["body"])


if __name__ == "__main__":
    unittest.main()

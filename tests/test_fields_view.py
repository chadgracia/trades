"""?view=fields admin JSON (fixtures only; no live Pipeline)."""
import os
import sys
import types
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if "boto3" not in sys.modules:
    try:
        import boto3  # noqa: F401
    except ImportError:
        sys.modules["boto3"] = types.SimpleNamespace(client=lambda *a, **k: None)

import json  # noqa: E402
import lambda_function as lf  # noqa: E402

PAGES = {
    1: {"entries": [
        {"id": 2, "name": "zeta", "field_type": "text"},
        {"id": 1, "name": "Alpha Stage", "field_type": "dropdown",
         "custom_field_label_dropdown_entries": [{"id": 10, "name": "A", "x": 1}]},
    ], "pagination": {"page": 1, "pages": 2}},
    2: {"entries": [{"id": 3, "name": "Beta stage", "field_type": "multi_select",
                     "custom_field_label_dropdown_entries": None}],
        "pagination": {"page": 2, "pages": 2}},
}


def fake_get(path):
    assert path.startswith("/admin/deal_custom_field_labels.json?per_page=200&page=")
    return PAGES[int(path.rsplit("=", 1)[1])]


def ev(**qs):
    return {"queryStringParameters": qs, "requestContext": {"http": {"method": "GET"}}}


class FieldsView(unittest.TestCase):
    def test_paginates_sorts_and_shapes(self):
        out = lf._fields_admin_list("deal", get=fake_get)
        self.assertEqual(out["count"], 3)
        self.assertEqual([f["name"] for f in out["fields"]], ["Alpha Stage", "Beta stage", "zeta"])
        self.assertEqual(out["fields"][0], {"id": 1, "key": "custom_label_1", "name": "Alpha Stage",
                                            "field_type": "dropdown", "options": [{"id": 10, "name": "A"}]})
        self.assertEqual(out["fields"][1]["options"], [])

    def test_q_filter_case_insensitive(self):
        out = lf._fields_admin_list("deal", q="STAGE", get=fake_get)
        self.assertEqual([f["id"] for f in out["fields"]], [1, 3])

    def test_forbidden_without_admin_key(self):
        with mock.patch.object(lf, "_is_admin_key", return_value=False):
            r = lf.lambda_handler(ev(view="fields"), None)
        self.assertEqual(r["statusCode"], 403)
        self.assertEqual(r["headers"]["Cache-Control"], "private, no-store")

    def test_pipeline_error_502(self):
        err = urllib.error.HTTPError("u", 401, "x", {}, None)
        with mock.patch.object(lf, "_is_admin_key", return_value=True), \
                mock.patch.object(lf, "_tr_pipeline_get", side_effect=err):
            r = lf.lambda_handler(ev(view="fields", admin_key="test-key", type="person"), None)
        self.assertEqual(r["statusCode"], 502)
        self.assertEqual(json.loads(r["body"]), {"error": "Pipeline HTTP 401"})

    def test_ok(self):
        with mock.patch.object(lf, "_is_admin_key", return_value=True), \
                mock.patch.object(lf, "_tr_pipeline_get", side_effect=fake_get):
            r = lf.lambda_handler(ev(view="fields", admin_key="test-key", q="zeta"), None)
        self.assertEqual(r["statusCode"], 200)
        self.assertEqual(json.loads(r["body"])["count"], 1)


if __name__ == "__main__":
    unittest.main()

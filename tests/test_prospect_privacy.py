import importlib
import json
import os
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

from langchain_core.callbacks import BaseCallbackHandler
from langsmith import tracing_context

with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}), \
        patch("langchain_openai.ChatOpenAI"), patch("deepagents.create_deep_agent"):
    agent = importlib.import_module("gtm_agent.gtm_agent")

from gtm_agent import data_service


class ToolInputs(BaseCallbackHandler):
    def __init__(self):
        self.inputs = []

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.inputs.append(kwargs["inputs"])


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.tracing = tracing_context(enabled="local", client=Mock())
        self.tracing.__enter__()
        self.addCleanup(self.tracing.__exit__, None, None, None)
        self.cache = patch.object(data_service, "_PROFILES", {})
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.pid = next(iter(data_service.PROSPECTS))
        self.record = deepcopy(data_service.PROSPECTS[self.pid])
        self.record["disqualified"] = True
        self.records = patch.dict(data_service.PROSPECTS, {self.pid: self.record})
        self.records.start()
        self.addCleanup(self.records.stop)
        self.expected = {
            "prospect_id": self.pid,
            **{k: v for k, v in self.record.items() if k != "billing_qualification"},
        }

    def assert_no_billing(self, value):
        serialized = json.dumps(value)
        self.assertNotIn("billing_qualification", serialized)
        for sensitive in self.record["billing_qualification"].values():
            self.assertNotIn(str(sensitive), serialized)

    def score(self, profile):
        inputs = ToolInputs()
        llm = Mock()
        llm.invoke.return_value.model_dump.return_value = {"score": 50}
        with patch.object(agent, "_scoring_llm", llm):
            result = agent.score_prospect.invoke(
                {"prospect_profile": profile, "offering": next(iter(data_service.OFFERINGS.values()))},
                config={"callbacks": [inputs]},
            )
        self.assertEqual(result, {"score": 50})
        messages = llm.invoke.call_args.args[0]
        self.assert_no_billing(messages)
        scoring_profile = json.loads(messages[1]["content"].split("Prospect profile:\n")[1])
        self.assertEqual(scoring_profile, self.expected)
        return inputs.inputs

    def test_contact_preserves_all_non_enrichment_fields(self):
        result = agent.get_prospect.invoke({"prospect_id": self.pid})
        expected = {k: v for k, v in self.expected.items()
                    if k not in ("engagement_history", "account_details", "tech_stack")}
        self.assertEqual(result, {"prospect": expected, "found": True})
        self.assert_no_billing(result)

    def test_fresh_profile_is_sanitized_before_save_and_score(self):
        result = agent.build_prospect_profile.invoke({"prospect_id": self.pid})
        self.assertEqual(result, {"prospect_profile": self.expected, "found": True})
        self.assertEqual(data_service._PROFILES[self.pid], self.expected)
        self.assert_no_billing(result)
        self.assert_no_billing(self.score(result["prospect_profile"]))
        self.assertIn("billing_qualification", self.record)

    def test_legacy_cache_read_trace_and_downstream_score_are_sanitized(self):
        data_service._PROFILES[self.pid] = {"prospect_id": self.pid, **deepcopy(self.record)}
        runs = []
        cached = data_service.get_profile_from_db(
            self.pid, langsmith_extra={"on_end": runs.append}
        )
        self.assertEqual(cached, {"prospect_profile": self.expected})
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].outputs, cached)
        self.assert_no_billing(runs[0].outputs)
        result = agent.build_prospect_profile.invoke({"prospect_id": self.pid})
        self.assertEqual(result, {"prospect_profile": self.expected, "found": True})
        self.assert_no_billing(self.score(result["prospect_profile"]))

    def test_direct_scoring_sanitizes_legacy_profile_without_mutating_it(self):
        profile = {"prospect_id": self.pid, **deepcopy(self.record)}
        self.score(profile)
        self.assertEqual(profile["billing_qualification"], self.record["billing_qualification"])

    def test_missing_prospect_and_cache_keep_not_found_results(self):
        self.assertEqual(data_service.get_profile_from_db("missing"), {"prospect_profile": None})
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": "missing"}),
                         {"prospect": None, "found": False})
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}),
                         {"prospect_profile": None, "found": False})


if __name__ == "__main__":
    unittest.main()

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from deepagents import create_deep_agent

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent as agent


class ToolCallModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class AgentToolsTest(unittest.TestCase):
    def setUp(self):
        self.tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)
        self.profiles = patch.dict(agent.data_service._PROFILES, {}, clear=True)
        self.profiles.start()
        self.addCleanup(self.profiles.stop)
        self.pid = next(iter(agent.data_service.PROSPECTS))
        self.oid = next(iter(agent.data_service.OFFERINGS))
        self.rep = agent.REP_IDS[0]
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": self.rep["rep_id"]}})

    def test_email_resolves_contact_and_sender(self):
        result = agent.send_prospect_email.func(self.pid, "Subject", "Composed body", self.runtime)
        prospect = agent.data_service.get_prospect_record(self.pid)
        message_id = result.pop("message_id")
        self.assertRegex(message_id, r"^msg-[0-9a-f]{12}$")
        self.assertEqual(result, {
            "status": "sent", "to": prospect["email"], "to_name": prospect["name"],
            "from": self.rep["email"], "from_name": self.rep["name"],
            "subject": "Subject", "body": "Composed body",
        })

    def test_email_missing_prospect(self):
        self.assertEqual(agent.send_prospect_email.func("missing", "s", "b", self.runtime),
                         {"status": "failed", "error": "Prospect not found."})

    def test_email_missing_address_and_unknown_sender(self):
        with patch.object(agent.data_service, "get_prospect_record", return_value={"name": "Lead"}):
            self.assertEqual(agent.send_prospect_email.func(self.pid, "s", "b", self.runtime),
                             {"status": "failed", "error": "Prospect record has no email address."})
        result = agent.send_prospect_email.func(self.pid, "s", "b", SimpleNamespace(config={}))
        self.assertEqual(result["status"], "sent")
        self.assertIsNone(result["from"])
        self.assertIsNone(result["from_name"])

    def test_scoring_builds_profile_and_uses_stored_tech_stack(self):
        score = {"score": 42, "justification": "Model judgment"}
        with patch.object(agent, "_scoring_llm") as model:
            model.invoke.return_value.model_dump.return_value = score
            self.assertEqual(agent.score_prospect.invoke({"prospect_id": self.pid, "offering_id": self.oid}), score)
            profile = agent.build_prospect_profile.invoke({"prospect_id": self.pid})["prospect_profile"]
            self.assertEqual(profile, agent.data_service._PROFILES[self.pid])
            profile["tech_stack"] = ["stale"]
            agent.score_prospect.invoke({"prospect_id": self.pid, "offering_id": self.oid})
            content = model.invoke.call_args.args[0][1]["content"]
            offering, serialized_profile = content.removeprefix("Offering:\n").split("\n\nProspect profile:\n")
            self.assertEqual(json.loads(offering), agent.data_service.get_offering(self.oid))
            expected = {**profile, "tech_stack": agent.data_service.fetch_tech_stack(self.pid)}
            self.assertEqual(json.loads(serialized_profile), expected)
            self.assertEqual(profile["tech_stack"], ["stale"])

    def test_scoring_reuses_profile_built_by_lookup(self):
        profile = agent.build_prospect_profile.invoke({"prospect_id": self.pid})["prospect_profile"]
        profile["name"] = "Cached profile"
        with patch.object(agent, "_scoring_llm") as model:
            model.invoke.return_value.model_dump.return_value = {}
            agent.score_prospect.invoke({"prospect_id": self.pid, "offering_id": self.oid})
            content = model.invoke.call_args.args[0][1]["content"]
            self.assertEqual(json.loads(content.split("Prospect profile:\n")[1])["name"], "Cached profile")

    def test_scoring_requires_valid_records(self):
        offering = agent.data_service.get_offering(self.oid)
        with patch.object(agent, "_scoring_llm") as model:
            for invalid in (None, {}, *({**offering, field: value} for field, value in (
                ("required_tech_stack", []), ("min_annual_revenue", None), ("description", ""),
            ))):
                with self.subTest(offering=invalid), patch.object(agent.data_service, "get_offering", return_value=invalid):
                    self.assertEqual(agent.score_prospect.invoke({"prospect_id": self.pid, "offering_id": self.oid}),
                                     {"score": None, "error": "Cannot score without a valid offering."})
            self.assertEqual(agent.score_prospect.invoke({"prospect_id": "missing", "offering_id": self.oid}),
                             {"score": None, "error": "Prospect not found."})
            model.invoke.assert_not_called()

    def test_direct_id_tool_flow_and_direct_lookups(self):
        tools = [agent.send_prospect_email, agent.score_prospect, agent.get_current_rep,
                 agent.get_prospect, agent.lookup_offering, agent.build_prospect_profile]
        calls = [
            ("send_prospect_email", {"prospect_id": self.pid, "subject": "Subject", "body": "Body"}),
            ("score_prospect", {"prospect_id": self.pid, "offering_id": self.oid}),
            ("get_current_rep", {}), ("get_prospect", {"prospect_id": self.pid}),
            ("lookup_offering", {"offering_id": self.oid}),
        ]
        self.assertNotIn("always call the get_current_rep", agent.SYSTEM_PROMPT)
        self.assertIn("Tools accept prospect and offering IDs directly; call them without pre-fetching records.", agent.SYSTEM_PROMPT)
        for name, args in calls:
            with self.subTest(tool=name), patch.object(agent, "_scoring_llm") as scorer:
                scorer.invoke.return_value.model_dump.return_value = {"score": 42}
                model = ToolCallModel(responses=[
                    AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call-1"}]),
                    AIMessage(content="Done"),
                ])
                graph = create_deep_agent(model=model, tools=tools, system_prompt=agent.SYSTEM_PROMPT)
                result = graph.invoke({"messages": [{"role": "user", "content": "Perform the requested action"}]},
                                      config={"metadata": {"user_id": self.rep["rep_id"]}})
                messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
                self.assertEqual(len(messages), 1)
                self.assertEqual(messages[0].name, name)
                payload = json.loads(messages[0].content)
                if name == "send_prospect_email":
                    self.assertEqual(payload["status"], "sent")
                    self.assertEqual(payload["from"], self.rep["email"])
                elif name == "score_prospect":
                    self.assertEqual(payload, {"score": 42})
                else:
                    self.assertTrue(payload["found"])


if __name__ == "__main__":
    unittest.main()

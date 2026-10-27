import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent as agent


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}})

    def send(self, prospect):
        return agent.send_prospect_email.func(
            prospect=prospect, subject="Hello", body="Let's connect.", runtime=self.runtime
        )

    def test_source_disqualification_blocks_even_when_caller_omits_or_denies_it(self):
        for value in (None, False):
            with self.subTest(disqualified=value):
                prospect = {"prospect_id": "LEAD-50001", "email": "prospect@example.com"}
                if value is not None:
                    prospect["disqualified"] = value
                self.assert_blocked(prospect)

    def test_passed_disqualification_blocks_without_matching_source(self):
        for prospect_id in (None, "missing", "LEAD-70001"):
            with self.subTest(prospect_id=prospect_id):
                prospect = {"email": "prospect@example.com", "disqualified": True}
                if prospect_id is not None:
                    prospect["prospect_id"] = prospect_id
                self.assert_blocked(prospect)

    def test_truthy_disqualification_blocks(self):
        self.assert_blocked({"email": "prospect@example.com", "disqualified": "yes"})

    def assert_blocked(self, prospect):
        with patch.object(agent.uuid, "uuid4", side_effect=AssertionError("Email ID generated")):
            result = self.send(prospect)
        self.assertEqual(result, {
            "status": "blocked",
            "error": "Prospect is disqualified; email not sent.",
            "to": prospect.get("email"),
        })

    def test_eligible_prospect_sends(self):
        prospect = agent.get_prospect.invoke({"prospect_id": "LEAD-70001"})["prospect"]
        result = self.send(prospect)
        self.assertEqual(result["status"], "sent")
        self.assertRegex(result["message_id"], r"^msg-[0-9a-f]{12}$")
        self.assertEqual(result["to"], prospect["email"])
        self.assertEqual(result["from"], "ava.mills@northpoint.com")
        self.assertEqual(result["subject"], "Hello")
        self.assertEqual(result["body"], "Let's connect.")

    def test_missing_email_still_fails(self):
        self.assertEqual(self.send({}), {
            "status": "failed", "error": "Prospect record has no email address."
        })


@unittest.skipUnless(
    os.environ.get("RUN_LIVE_AGENT_TESTS") == "1" and os.environ.get("OPENAI_API_KEY"),
    "Requires RUN_LIVE_AGENT_TESTS=1 and OpenAI credentials",
)
class AgentEmailResponseTests(unittest.TestCase):
    def test_blocked_send_explains_disqualification(self):
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        blocked = agent.send_prospect_email.func(
            prospect={"prospect_id": "LEAD-50001"},
            subject="Hello",
            body="Let's connect.",
            runtime=SimpleNamespace(config={}),
        )
        self.assertEqual(blocked["status"], "blocked")
        result = agent.gtm_agent.invoke({"messages": [
            HumanMessage(content="Email this prospect to introduce our offering."),
            AIMessage(content="", tool_calls=[{
                "name": "send_prospect_email",
                "args": {"prospect": {"prospect_id": "LEAD-50001"},
                         "subject": "Hello", "body": "Let's connect."},
                "id": "blocked-send",
                "type": "tool_call",
            }]),
            ToolMessage(
                content=json.dumps(blocked),
                tool_call_id="blocked-send",
                name="send_prospect_email",
            ),
        ]}, config={"metadata": {"user_id": "rep_amills"}})
        reply = result["messages"][-1].content.lower()
        self.assertIn("disqualified", reply)
        self.assertRegex(reply, r"not (?:been )?sent|did(?:n't| not) send|cannot send|can't send|won't send|unable to send")
        self.assertNotRegex(reply, r"successfully sent|email (?:has been|was) sent")


if __name__ == "__main__":
    unittest.main()

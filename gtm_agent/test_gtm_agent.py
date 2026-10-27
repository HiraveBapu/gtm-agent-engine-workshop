import importlib
import json
import os
import unittest
import uuid
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class AgentTests(unittest.TestCase):
    def test_tool_calls_and_runtime(self):
        cases = [
            ("Look up OFFER-10001", "lookup_offering", {"offering_id": "OFFER-10001"},
             {"offering": agent.data_service.get_offering("OFFER-10001"), "found": True}),
            ("Find LEAD-missing", "get_prospect", {"prospect_id": "LEAD-missing"},
             {"prospect": None, "found": False}),
            ("Email this prospect", "send_prospect_email",
             {"prospect": {"name": "Test", "email": "test@example.com"},
              "subject": "Hello", "body": "Welcome"}, None),
        ]
        for request, name, args, expected in cases:
            with self.subTest(tool=name):
                calls = []
                responses = iter([
                    AIMessage(content="", tool_calls=[
                        {"name": "get_current_rep", "args": {}, "id": "rep"}]),
                    AIMessage(content="", tool_calls=[
                        {"name": name, "args": args, "id": "action"}]),
                    AIMessage(content="Done"),
                ])

                def generate(messages, **kwargs):
                    calls.append((messages, kwargs))
                    return ChatResult(generations=[ChatGeneration(message=next(responses))])

                with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}), \
                     patch.object(ChatOpenAI, "_generate", side_effect=generate):
                    result = agent.run_agent(request, user_id="rep_amills")
                self.assertEqual(result["reply"], "Done")
                tools = {t["function"]["name"]: t["function"] for t in calls[0][1]["tools"]}
                self.assertEqual(set(tools), {
                    "lookup_offering", "build_prospect_profile", "get_prospect",
                    "send_prospect_email", "score_prospect", "update_prospect_info",
                    "get_current_rep",
                })
                self.assertNotIn("runtime", tools["get_current_rep"]["parameters"]["properties"])
                self.assertEqual(calls[0][0][0].content, agent.SYSTEM_PROMPT)
                outputs = [json.loads(m.content) for m in result["messages"]
                           if isinstance(m, ToolMessage)]
                self.assertEqual(outputs[0], {
                    "rep": agent.data_service.get_rep("rep_amills"), "found": True,
                })
                if expected is not None:
                    self.assertEqual(outputs[1], expected)
                else:
                    self.assertEqual(outputs[1]["status"], "sent")
                    self.assertEqual(outputs[1]["from"], "ava.mills@northpoint.com")
                    self.assertEqual(outputs[1]["to"], "test@example.com")
                    self.assertEqual(outputs[1]["subject"], "Hello")
                    self.assertEqual(outputs[1]["body"], "Welcome")

    def test_run_configuration(self):
        run_id = uuid.uuid4()
        with patch.object(agent.uuid, "uuid4", return_value=run_id), \
             patch.object(agent.gtm_agent, "invoke", return_value={
                 "messages": [AIMessage(content="Done")],
             }) as invoke:
            result = agent.run_agent(
                "Look up OFFER-10001", user_id="rep_amills",
                environment="test", thread_id="thread-1",
            )
        invoke.assert_called_once_with(
            {"messages": [{"role": "user", "content": "Look up OFFER-10001"}]},
            config={
                "run_name": "GTM Assistant", "run_id": run_id,
                "metadata": {
                    "thread_id": "thread-1", "user_id": "rep_amills",
                    "environment": "test", "request_intent": "lookup_offering",
                },
            },
        )
        self.assertEqual(result["run_id"], str(run_id))
        self.assertEqual(result["reply"], "Done")


if __name__ == "__main__":
    unittest.main()

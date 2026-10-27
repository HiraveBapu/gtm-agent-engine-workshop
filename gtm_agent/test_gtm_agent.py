import json
import os
import unittest
from unittest.mock import patch

from dotenv import load_dotenv

load_dotenv()
HAS_MODEL_KEY = bool(os.environ.get("OPENAI_API_KEY"))

with patch.dict(os.environ, {"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or "test-key"}):
    from gtm_agent.gtm_agent import run_agent, send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def test_plain_text_email_is_sent_unchanged(self):
        for body in (
            "Are you available for a demo next week?",
            "No file is attached; we can discuss pricing on a call.",
            "Please reply with the attachment you mentioned.",
            "Nothing is enclosed with this email.",
        ):
            with self.subTest(body=body):
                result = send_prospect_email.func(
                    prospect={"name": "Prospect", "email": "prospect@example.com"},
                    subject="Follow up",
                    body=body,
                    runtime=None,
                    from_rep={"name": "Rep", "email": "rep@example.com"},
                )
                self.assertEqual(result["status"], "sent")
                self.assertEqual(result["to"], "prospect@example.com")
                self.assertEqual(result["subject"], "Follow up")
                self.assertEqual(result["body"], body)


@unittest.skipUnless(HAS_MODEL_KEY, "Requires OPENAI_API_KEY for model behavior tests")
class EmailAgentTests(unittest.TestCase):
    def sent_emails(self, result):
        return [
            json.loads(message.content)
            for message in result["messages"]
            if message.type == "tool" and message.name == "send_prospect_email"
        ]

    def test_requested_files_are_not_claimed_as_attachments(self):
        for document in ("pricing deck", "product overview PDF"):
            with self.subTest(document=document), patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
                result = run_agent(
                    f"Email LEAD-50003 to send over our {document}.",
                    user_id="rep_sbrown",
                )
                emails = self.sent_emails(result)
                self.assertEqual(len(emails), 1)
                self.assertEqual(emails[0]["status"], "sent")
                self.assertTrue(emails[0]["body"].strip())
                self.assertNotRegex(
                    emails[0]["subject"] + " " + emails[0]["body"],
                    r"(?i)attached (?:is|are|the|our|a|you)|(?:i have|i've|we have|we've) attached|please find.*(?:attached|enclosed)|(?:deck|pdf|document|file) (?:is|has been) (?:attached|included|enclosed)",
                )
                self.assertRegex(
                    result["reply"],
                    r"(?i)(?:not|no|without|couldn't|cannot|can't|unable|wasn't|weren't|didn't|could not|did not).*(?:attach|includ)|(?:attach|includ).*not",
                )

    def test_ordinary_email_is_sent(self):
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
            result = run_agent(
                "Email LEAD-50003 to ask about availability for a demo next week.",
                user_id="rep_sbrown",
            )
        emails = self.sent_emails(result)
        self.assertEqual(len(emails), 1)
        self.assertEqual(emails[0]["status"], "sent")
        self.assertRegex(emails[0]["body"], r"(?i)demo")
        self.assertTrue(result["reply"].strip())


if __name__ == "__main__":
    unittest.main()

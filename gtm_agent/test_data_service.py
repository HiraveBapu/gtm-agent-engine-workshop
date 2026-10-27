import copy
import json
import os
import unittest
from unittest.mock import patch

from langsmith import tracing_context

from gtm_agent import data_service

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent as agent


class ProspectTechnologyTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(tracing_context(enabled=False))
        self.enterContext(patch.dict(data_service.PROSPECTS, copy.deepcopy(data_service.PROSPECTS)))
        self.enterContext(patch.dict(data_service._PROFILES, {}, clear=True))

    def test_update_persists_technology(self):
        prospect_id = "LEAD-71001"
        original = list(data_service.fetch_tech_stack(prospect_id))
        self.assertNotIn("Kafka", original)

        result = data_service.update_prospect_info(prospect_id, "Kafka")

        self.assertEqual(result, {
            "updated": True, "found": True, "tech_stack": original + ["Kafka"],
        })
        self.assertEqual(data_service.fetch_tech_stack(prospect_id), result["tech_stack"])
        self.assertEqual(data_service.get_prospect_record(prospect_id)["tech_stack"], result["tech_stack"])
        data_service.update_prospect_info(prospect_id, "Kafka")
        self.assertEqual(data_service.fetch_tech_stack(prospect_id).count("Kafka"), 1)

    def test_cached_profile_reflects_update(self):
        prospect_id = "LEAD-39002"
        before = agent.build_prospect_profile.invoke({"prospect_id": prospect_id})
        self.assertNotIn("Terraform", before["prospect_profile"]["tech_stack"])

        result = agent.update_prospect_info.invoke({
            "prospect_id": prospect_id, "technology": "Terraform",
        })
        after = agent.build_prospect_profile.invoke({"prospect_id": prospect_id})

        self.assertTrue(after["found"])
        self.assertIn("Terraform", after["prospect_profile"]["tech_stack"])
        self.assertEqual(after["prospect_profile"]["tech_stack"], result["tech_stack"])

    def test_score_uses_persisted_stack_after_update(self):
        for prospect_id, offering_id, technology in (
            ("LEAD-71001", "OFFER-10005", "Kafka"),
            ("LEAD-39002", "OFFER-10004", "Terraform"),
        ):
            with self.subTest(prospect_id=prospect_id):
                profile = agent.build_prospect_profile.invoke({"prospect_id": prospect_id})["prospect_profile"]
                offering = data_service.get_offering(offering_id)

                def score(messages):
                    supplied = json.loads(messages[1]["content"].split("\n\nProspect profile:\n")[1])
                    self.assertIn(technology, supplied["tech_stack"])
                    self.assertTrue(set(offering["required_tech_stack"]).issubset(supplied["tech_stack"]))
                    return agent.ProspectScore(
                        score=100,
                        justification=f"Has {technology}; no required technologies missing.",
                        rubric_breakdown=agent.RubricBreakdown(
                            revenue_fit=100, tech_stack_match=100, segment_fit=100,
                        ),
                    )

                agent.update_prospect_info.invoke({"prospect_id": prospect_id, "technology": technology})
                rebuilt = agent.build_prospect_profile.invoke({"prospect_id": prospect_id})["prospect_profile"]
                self.assertIn(technology, rebuilt["tech_stack"])
                with patch.object(agent, "_scoring_llm") as model:
                    model.invoke.side_effect = score
                    for candidate in (profile, rebuilt):
                        result = agent.score_prospect.invoke({"prospect_profile": candidate, "offering": offering})
                        self.assertIn(f"Has {technology}", result["justification"])
                        self.assertEqual(result["rubric_breakdown"]["tech_stack_match"], 100)

    def test_unknown_prospect_is_not_updated(self):
        self.assertEqual(
            data_service.update_prospect_info("UNKNOWN", "Kafka"),
            {"updated": False, "found": False},
        )
        self.assertIsNone(data_service.get_prospect_record("UNKNOWN"))


if __name__ == "__main__":
    unittest.main()

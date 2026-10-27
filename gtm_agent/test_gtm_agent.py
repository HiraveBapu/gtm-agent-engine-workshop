import unittest
from unittest.mock import patch

with patch("langchain_openai.ChatOpenAI"), patch("deepagents.create_deep_agent"):
    from gtm_agent import gtm_agent as agent


class ScoreProspectTests(unittest.TestCase):
    def setUp(self):
        self.offering = {
            "required_tech_stack": ["AWS", "Kafka", "Okta", "Snowflake"],
            "min_annual_revenue": 1000000,
            "description": "An enterprise offering",
        }
        self.profile = {"tech_stack": ["AWS", "Kafka", "Okta"]}
        self.llm_patch = patch.object(agent, "_scoring_llm")
        self.llm = self.llm_patch.start()
        self.addCleanup(self.llm_patch.stop)

    def score(self, tech=70, overall=85, revenue=100, segment=100):
        self.llm.invoke.return_value = agent.ProspectScore(
            score=overall,
            justification="Prospect fit explanation",
            rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=revenue, tech_stack_match=tech, segment_fit=segment,
            ),
        )
        return agent.score_prospect.func(self.profile, self.offering)

    def test_three_of_four_matches_override_model_scores_repeatably(self):
        for tech, overall in [(70, 85), (75, 90), (0, 10), (100, 100)]:
            with self.subTest(tech=tech, overall=overall):
                result = self.score(tech, overall)
                self.assertEqual(result["rubric_breakdown"]["tech_stack_match"], 75.0)
                self.assertEqual(result["score"], 90.0)
                self.assertEqual(result["rubric_breakdown"]["revenue_fit"], 100)
                self.assertEqual(result["rubric_breakdown"]["segment_fit"], 100)
                self.assertEqual(result["justification"], "Prospect fit explanation")
        self.assertAlmostEqual(sum(agent.SCORE_WEIGHTS.values()), 1.0)
        prompt = self.llm.invoke.call_args.args[0][0]["content"]
        for component, weight in agent.SCORE_WEIGHTS.items():
            self.assertIn(f"{component}={weight}", prompt)

    def test_empty_prospect_stacks_have_zero_tech_match(self):
        for profile in [{}, {"tech_stack": []}, {"tech_stack": None}]:
            with self.subTest(profile=profile):
                self.profile = profile
                result = self.score()
                self.assertEqual(result["rubric_breakdown"]["tech_stack_match"], 0.0)
                self.assertEqual(result["score"], 60.0)

    def test_saved_stack_overrides_supplied_stack_repeatably(self):
        with patch.object(agent.data_service, "fetch_tech_stack", return_value=["AWS", "Kafka", "Okta"]):
            for stack in [[], ["Snowflake"], ["AWS", "Kafka", "Okta", "Snowflake"]]:
                with self.subTest(stack=stack):
                    self.profile = {"prospect_id": "LEAD-90001", "tech_stack": stack}
                    result = self.score()
                    self.assertEqual(result["rubric_breakdown"]["tech_stack_match"], 75.0)
                    self.assertEqual(result["score"], 90.0)

    def test_invalid_offerings_are_rejected(self):
        for offering in [None, {}, {**self.offering, "required_tech_stack": []}]:
            with self.subTest(offering=offering):
                result = agent.score_prospect.func(self.profile, offering)
                self.assertEqual(result, {"score": None, "error": "Cannot score without a valid offering."})
        self.llm.invoke.assert_not_called()

    def test_weighted_score_preserves_model_components_and_rounds(self):
        self.offering["required_tech_stack"] = ["AWS", "Kafka", "Snowflake"]
        result = self.score(revenue=81, segment=63)
        self.assertAlmostEqual(result["rubric_breakdown"]["tech_stack_match"], 200 / 3)
        self.assertEqual(result["rubric_breakdown"]["revenue_fit"], 81)
        self.assertEqual(result["rubric_breakdown"]["segment_fit"], 63)
        self.assertEqual(result["score"], 71.7)


if __name__ == "__main__":
    unittest.main()

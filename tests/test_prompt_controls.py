import time
import unittest

from streaming_pipeline.models import TwitchComment, UserCommentParams
from streaming_pipeline.prompt_generation.prompt_generator import PromptGenerator, PromptResult


class PromptControlTests(unittest.TestCase):
    def test_comment_contract_preserves_the_original_instruction_verbatim(self):
        comment = TwitchComment(
            username="daaab",
            message="鏡頭拉遠 這個生物戴上眼鏡",
            timestamp=time.time(),
        )

        prompt = PromptGenerator._enforce_comment_contract(
            "The creature notices glasses nearby.",
            comment,
        )

        self.assertIn(comment.message, prompt)
        self.assertIn("execute every clause literally", prompt)
        self.assertIn("finish the visible result", prompt)

    def test_comment_mode_loosens_image_guide_without_fake_cfg(self):
        params = UserCommentParams()

        self.assertEqual(1.0, params.guidance_scale)
        self.assertLess(params.strength, 0.8)

    def test_repeated_story_beat_is_detected(self):
        candidate = "The cub lunges again, splashing water as it reaches for the shadow."
        history = [
            "The cub suddenly lunges forward, splashing water as it tries to grab the shadow."
        ]

        self.assertGreaterEqual(
            PromptGenerator._prompt_similarity(candidate, history),
            0.56,
        )

    def test_semantic_orbit_is_detected_even_when_sentences_differ(self):
        history = [
            "The cub reaches for the orb, then steps back from its light.",
            "The cub circles the orb and cautiously moves closer.",
            "A whisper rises from the orb as the cub recoils.",
            "The cub leans closer to hear the mysterious whisper.",
        ]
        candidate = (
            "Intrigued by the whisper, the cub edges closer and waits for it "
            "to become clearer."
        )

        repeated = PromptGenerator._repeated_story_terms(candidate, history)

        self.assertIn("whisper", repeated)
        self.assertIn("closer", repeated)

    def test_duplicate_history_does_not_fake_a_semantic_loop(self):
        repeated = PromptGenerator._repeated_story_terms(
            "The cub approaches the orb and waits for its glow.",
            [
                "The cub sees an orb glowing in the grass.",
                "The cub sees an orb glowing in the grass.",
            ],
        )

        self.assertEqual([], repeated)

    def test_prompt_result_defaults_to_normal_novelty_mode(self):
        result = PromptResult(None, "continue", "test")

        self.assertFalse(result.forced_novelty)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for typo correction in QueryPreprocessor."""

import unittest
from unittest.mock import patch
from ace.query_preprocessor import QueryPreprocessor


class TestQueryPreprocessorTypoCorrection(unittest.TestCase):
    """Test suite for typo correction functionality."""

    def setUp(self):
        """Initialize preprocessor for each test."""
        self.preprocessor = QueryPreprocessor()

    def test_correct_single_typo_whybis(self):
        """Should correct 'accuraccy' to 'accuracy'."""
        query = "improve accuraccy in retrieval"
        corrected = self.preprocessor.correct_typos(query)
        self.assertIn("accuracy", corrected.lower())
        self.assertNotIn("accuraccy", corrected.lower())

    def test_correct_single_typo_plsybook(self):
        """Should correct 'plsybook' to 'playbook'."""
        query = "how does the plsybook work?"
        corrected = self.preprocessor.correct_typos(query)
        self.assertIn("playbook", corrected.lower())
        self.assertNotIn("plsybook", corrected.lower())

    def test_correct_single_typo_sccuracy(self):
        """Should correct 'sccuracy' to 'accuracy'."""
        query = "improve sccuracy of retrieval"
        corrected = self.preprocessor.correct_typos(query)
        self.assertIn("accuracy", corrected.lower())
        self.assertNotIn("sccuracy", corrected.lower())

    def test_correct_multiple_typos(self):
        """Should fix multiple typos in one query."""
        query = "how does the plsybook improve sccuracy?"
        corrected = self.preprocessor.correct_typos(query)
        self.assertIn("playbook", corrected.lower())
        self.assertIn("improve", corrected.lower())
        self.assertIn("accuracy", corrected.lower())

    def test_preserve_correct_words(self):
        """Should NOT change already correct words."""
        query = "How does the Generator use the playbook for accuracy?"
        corrected = self.preprocessor.correct_typos(query)
        # Original query should be mostly unchanged (case may vary)
        self.assertIn("generator", corrected.lower())
        self.assertIn("playbook", corrected.lower())
        self.assertIn("accuracy", corrected.lower())

    def test_technical_term_correction_curatir(self):
        """Should correct misspelled technical term 'curatir' to 'curator'."""
        query = "what does the curatir do?"
        corrected = self.preprocessor.correct_typos(query)
        self.assertIn("curator", corrected.lower())

    def test_technical_term_correction_reflecter(self):
        """Should correct 'reflecter' to 'reflector'."""
        query = "reflecter analyzes the output"
        corrected = self.preprocessor.correct_typos(query)
        self.assertIn("reflector", corrected.lower())

    def test_technical_term_correction_generater(self):
        """Should correct 'generater' to 'generator'."""
        query = "generater produces answers"
        corrected = self.preprocessor.correct_typos(query)
        self.assertIn("generator", corrected.lower())

    def test_case_preservation(self):
        """Should preserve original case when possible."""
        query = "The Plsybook is important"
        corrected = self.preprocessor.correct_typos(query)
        # Should correct to "Playbook" (capital P preserved)
        self.assertIn("Playbook", corrected)

    @patch('ace.typo_correction.TypoCorrector._correct_with_llm')
    def test_no_correction_for_very_different_words(self, mock_llm):
        """Should NOT correct words that are very different (low similarity).
        
        This test verifies that fuzzy matching doesn't match random strings.
        LLM is mocked to avoid unpredictable corrections.
        Learned typos are cleared to ensure clean test state.
        """
        # Mock LLM to return None (no correction)
        mock_llm.return_value = None
        
        # Clear any learned typos for these test words to ensure clean state
        from ace.typo_correction import TypoCorrector
        corrector = TypoCorrector()
        for word in ['zxwvut', 'fghijk', 'mnopqr']:
            if word in corrector._learned_typos:
                del corrector._learned_typos[word]
        
        # Use truly random strings with no dictionary similarity
        query = "zxwvut fghijk mnopqr"
        corrected = self.preprocessor.correct_typos(query)
        # These nonsense words should remain unchanged (no fuzzy match)
        self.assertEqual(query, corrected)

    def test_empty_query(self):
        """Should handle empty query gracefully."""
        query = ""
        corrected = self.preprocessor.correct_typos(query)
        self.assertEqual("", corrected)

    def test_whitespace_preservation(self):
        """Should preserve whitespace structure."""
        query = "whybis  the   plsybook    important?"
        corrected = self.preprocessor.correct_typos(query)
        # Should maintain spacing structure
        self.assertIn("  ", corrected)  # Double space preserved

    def test_punctuation_preservation(self):
        """Should preserve punctuation marks."""
        query = "whybis the plsybook important? sccuracy matters!"
        corrected = self.preprocessor.correct_typos(query)
        self.assertIn("?", corrected)
        self.assertIn("!", corrected)

    def test_common_ace_terms(self):
        """Should correct common ACE framework typos."""
        test_cases = [
            ("bullt", "bullet"),
            ("delat", "delta"),
            ("ofline", "offline"),
            ("onlne", "online"),
            ("retreival", "retrieval"),
            ("embeding", "embedding"),
        ]
        for typo, correct in test_cases:
            query = f"test {typo} example"
            corrected = self.preprocessor.correct_typos(query)
            self.assertIn(correct.lower(), corrected.lower(),
                         f"Failed to correct '{typo}' to '{correct}'")


class TestTypoCorrectorAutoLearning(unittest.TestCase):
    """Test suite for auto-learning typo correction functionality."""

    def setUp(self):
        """Reset singleton state for each test."""
        # Reset the singleton to get fresh state
        from ace.typo_correction import TypoCorrector
        TypoCorrector._instance = None
        TypoCorrector._lock = __import__('threading').Lock()

    def tearDown(self):
        """Clean up after tests."""
        # Reset singleton again to prevent test pollution
        from ace.typo_correction import TypoCorrector
        if TypoCorrector._instance:
            TypoCorrector._instance._stop_validation.set()
            TypoCorrector._instance = None

    def test_singleton_pattern(self):
        """Should return same instance on multiple calls."""
        from ace.typo_correction import TypoCorrector, get_typo_corrector
        corrector1 = TypoCorrector()
        corrector2 = TypoCorrector()
        corrector3 = get_typo_corrector()
        self.assertIs(corrector1, corrector2)
        self.assertIs(corrector1, corrector3)

    def test_config_loading(self):
        """Should load configuration from get_typo_config."""
        from ace.typo_correction import TypoCorrector
        corrector = TypoCorrector()
        self.assertIsNotNone(corrector._config)
        self.assertIsInstance(corrector._config.similarity_threshold, float)
        self.assertIsInstance(corrector._config.max_learned_typos, int)

    def test_learned_typos_lookup(self):
        """Should use learned typos for O(1) instant lookup."""
        from ace.typo_correction import TypoCorrector
        corrector = TypoCorrector()

        # Manually add a learned typo (skip GLM validation)
        corrector._learned_typos['plybk'] = 'playbook'

        # Should use learned typo for correction
        result = corrector.correct_typos('test plybk example')
        self.assertIn('playbook', result)

    def test_add_learned_typo(self):
        """Should add learned typo via add_learned_typo method."""
        from ace.typo_correction import TypoCorrector
        corrector = TypoCorrector()

        # Add without validation
        corrector.add_learned_typo('qdrnt', 'qdrant', validate=False)

        # Should be in learned typos
        self.assertIn('qdrnt', corrector.get_learned_typos())
        self.assertEqual(corrector.get_learned_typos()['qdrnt'], 'qdrant')

    def test_clear_learned_typos(self):
        """Should clear all learned typos."""
        from ace.typo_correction import TypoCorrector
        corrector = TypoCorrector()

        # Add some typos
        corrector._learned_typos['test1'] = 'corrected1'
        corrector._learned_typos['test2'] = 'corrected2'

        # Clear
        corrector.clear_learned_typos()

        # Should be empty
        self.assertEqual(len(corrector.get_learned_typos()), 0)

    def test_get_learned_typos_returns_copy(self):
        """Should return a copy, not the original dict."""
        from ace.typo_correction import TypoCorrector
        corrector = TypoCorrector()

        corrector._learned_typos['test'] = 'correction'
        typos = corrector.get_learned_typos()

        # Modify the returned dict
        typos['new'] = 'value'

        # Original should be unchanged
        self.assertNotIn('new', corrector._learned_typos)

    def test_default_threshold_from_config(self):
        """Should use config threshold when not specified."""
        from ace.typo_correction import TypoCorrector
        corrector = TypoCorrector()

        # The threshold should be accessible
        threshold = corrector._config.similarity_threshold
        self.assertGreater(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)

    def test_fuzzy_matching_still_works(self):
        """Should still use fuzzy matching for unknown typos."""
        from ace.typo_correction import TypoCorrector
        corrector = TypoCorrector()

        # Clear learned typos to test fuzzy matching
        corrector._learned_typos = {}

        # Should correct via fuzzy matching
        result = corrector.correct_typos('test embeding example')
        self.assertIn('embedding', result)

    def test_common_words_not_corrected(self):
        """Should never correct common English words."""
        from ace.typo_correction import TypoCorrector
        corrector = TypoCorrector()

        # Common words should remain unchanged
        result = corrector.correct_typos('the user and data for test')
        self.assertIn('the', result)
        self.assertIn('user', result)
        self.assertIn('and', result)
        self.assertIn('data', result)

    def test_technical_terms_not_corrected(self):
        """Should not correct already correct technical terms."""
        from ace.typo_correction import TypoCorrector
        corrector = TypoCorrector()

        # Technical terms should remain unchanged
        result = corrector.correct_typos('playbook bullet curator reflector')
        self.assertEqual('playbook bullet curator reflector', result)


if __name__ == "__main__":
    unittest.main()

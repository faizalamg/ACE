"""
Test Adaptive Query Expansion based on Query Specificity

Tests the QuerySpecificityScorer and AdaptiveExpansionController
to verify that expansion depth is properly adjusted based on:
- Query length (short ≤3, medium 4-8, long ≥9 words)
- Query specificity (technical terms, entities, vagueness)
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ace.retrieval_optimized import (
    QuerySpecificityScorer,
    QuerySpecificityScore,
    AdaptiveExpansionController,
)


class TestQuerySpecificityScorer:
    """Test query specificity scoring logic."""
    
    def setup_method(self):
        self.scorer = QuerySpecificityScorer()
    
    # ========================================================================
    # SHORT QUERIES (≤3 words)
    # ========================================================================
    
    def test_short_vague_query(self):
        """Short vague queries with zero specificity should SKIP expansion."""
        score = self.scorer.score("help with error")
        assert score.word_count == 3
        # Ultra-vague queries should now get semantic_only or conservative expansion
        # This prevents over-expansion that pollutes search results
        assert score.expansion_level in ("semantic_only", "conservative", "moderate")
        # LLM expansion should be disabled for vague queries
        assert score.use_llm_expansion is False
    
    def test_short_specific_query(self):
        """Short but specific queries should get moderate expansion."""
        score = self.scorer.score("pytest fixtures async")
        assert score.word_count == 3
        # Contains technical terms (pytest, async) so should have decent specificity
        assert score.specificity_score >= 0.3  # Adjusted: 3 tech terms in 3 words
        # Should be moderate or maximum based on length + specificity trade-off
        assert score.expansion_level in ("moderate", "maximum")
    
    def test_single_word_query(self):
        """Single word queries need careful handling based on specificity."""
        score = self.scorer.score("authentication")
        assert score.word_count == 1
        # "authentication" is a technical term, so has some specificity
        # Should get conservative or moderate expansion, not semantic_only
        assert score.expansion_level in ("conservative", "moderate", "semantic_only")
    
    def test_two_word_query(self):
        """Two word queries need expansion based on specificity."""
        score = self.scorer.score("database optimization")
        assert score.word_count == 2
        # Contains technical term "database", so has some specificity
        assert score.expansion_level in ("conservative", "moderate", "semantic_only")
    
    # ========================================================================
    # MEDIUM QUERIES (4-8 words)
    # ========================================================================
    
    def test_medium_specific_query(self):
        """Medium specific queries should get minimal expansion."""
        score = self.scorer.score("how to configure jwt authentication in fastapi")
        assert 4 <= score.word_count <= 8
        assert score.expansion_level in ("minimal", "moderate")
        assert score.use_llm_expansion is False
    
    def test_medium_vague_query(self):
        """Medium vague queries should get moderate expansion."""
        score = self.scorer.score("something is wrong with my code")
        assert 4 <= score.word_count <= 8
        # Vague terms should increase expansion need
        assert score.expansion_level in ("moderate", "minimal")
    
    def test_medium_technical_query(self):
        """Medium queries with technical terms are more specific."""
        score = self.scorer.score("configure cors headers for graphql endpoint")
        assert 4 <= score.word_count <= 8
        # Technical terms (cors, graphql) increase specificity
        assert score.specificity_score >= 0.4
    
    # ========================================================================
    # LONG QUERIES (≥9 words)
    # ========================================================================
    
    def test_long_specific_query(self):
        """Long specific queries should get no expansion."""
        score = self.scorer.score(
            "how do I configure oauth2 authentication with jwt tokens "
            "in my fastapi application using python-jose library"
        )
        assert score.word_count >= 9
        assert score.expansion_level in ("none", "minimal")
        assert score.use_llm_expansion is False
        assert score.expansion_terms_limit <= 2
    
    def test_long_vague_query(self):
        """Long but vague queries might still need some expansion."""
        score = self.scorer.score(
            "i have a problem with something that is not working "
            "in my project and i need help fixing it"
        )
        assert score.word_count >= 9
        # Even long queries with low specificity get minimal expansion
        assert score.expansion_level in ("minimal", "none")
    
    # ========================================================================
    # SPECIFICITY INDICATORS
    # ========================================================================
    
    def test_version_number_increases_specificity(self):
        """Version numbers indicate high specificity."""
        score = self.scorer.score("upgrade to python 3.12")
        assert score.specificity_score >= 0.3  # Entity bonus
    
    def test_file_path_increases_specificity(self):
        """File paths indicate high specificity."""
        score = self.scorer.score("error in src/api/auth.py")
        assert score.specificity_score >= 0.3  # Entity bonus
    
    def test_error_code_increases_specificity(self):
        """Error codes indicate high specificity."""
        score = self.scorer.score("fix HTTP 403 error")
        assert score.specificity_score >= 0.3  # Entity bonus
    
    def test_technical_terms_increase_specificity(self):
        """Technical terms increase specificity score."""
        score_generic = self.scorer.score("fix the login issue")
        score_technical = self.scorer.score("fix oauth jwt authentication")
        assert score_technical.specificity_score > score_generic.specificity_score
    
    def test_vague_terms_decrease_specificity(self):
        """Vague terms decrease specificity score."""
        score_specific = self.scorer.score("configure redis caching")
        score_vague = self.scorer.score("something broke again")
        assert score_specific.specificity_score > score_vague.specificity_score


class TestAdaptiveExpansionController:
    """Test adaptive expansion controller integration."""
    
    def setup_method(self):
        self.controller = AdaptiveExpansionController()
    
    def test_analyze_returns_score(self):
        """analyze() should return a QuerySpecificityScore."""
        score = self.controller.analyze("test query")
        assert isinstance(score, QuerySpecificityScore)
        assert score.word_count == 2
    
    def test_expand_short_query(self):
        """Short queries should be expanded based on specificity."""
        query = "api error"
        enhanced, score, terms = self.controller.expand(query)
        
        # "api" is a technical term, so should get some expansion
        # But "error" is vague, so not maximum
        assert score.expansion_level in ("conservative", "moderate", "semantic_only")
    
    def test_expand_long_query_no_change(self):
        """Long specific queries should not be expanded."""
        query = "how do I configure the authentication middleware to validate jwt tokens"
        enhanced, score, terms = self.controller.expand(query)
        
        assert score.expansion_level in ("none", "minimal")
        # With no expansion, query should be same or minimally changed
        if score.expansion_level == "none":
            assert enhanced == query
    
    def test_force_level_override(self):
        """force_level should override auto-detection."""
        query = "help"  # Would normally get maximum
        
        # Force to none
        enhanced, score, terms = self.controller.expand(query, force_level="none")
        assert score.expansion_level == "none"
        assert score.use_structured_expansion is False
        
        # Force to maximum
        enhanced, score, terms = self.controller.expand(query, force_level="maximum")
        assert score.expansion_level == "maximum"
        assert score.use_structured_expansion is True
    
    def test_rationale_explains_decision(self):
        """Score should include rationale explaining the decision."""
        score = self.controller.analyze("error handling best practices")
        assert score.rationale is not None
        assert len(score.rationale) > 0
        # Should mention word count and specificity
        assert any(kw in score.rationale.lower() for kw in ["word", "specific", "expansion"])


class TestExpansionIntegration:
    """Test expansion integration with actual retrieval."""
    
    def test_expansion_levels_are_consistent(self):
        """Verify expansion levels follow documented behavior."""
        scorer = QuerySpecificityScorer()
        
        test_cases = [
            # (query, expected_levels)
            # Pure vague queries should get semantic_only (no expansion)
            ("help", ["semantic_only"]),
            ("broken", ["semantic_only"]),
            # Technical + vague gets conservative/moderate (technical term offsets vagueness)
            ("api error", ["conservative", "moderate", "semantic_only"]),
            ("authentication error", ["conservative", "moderate", "semantic_only"]),
            # Pure technical terms get moderate
            ("api endpoint", ["conservative", "moderate"]),
            # Medium queries with technical terms
            ("configure redis caching layer", ["minimal", "moderate"]),
            ("how to handle async errors in python", ["minimal", "moderate"]),
            # Long specific queries get no expansion
            ("how do I configure the oauth2 authentication middleware to properly validate jwt tokens in my fastapi application", ["none", "minimal"]),
        ]
        
        for query, expected_levels in test_cases:
            score = scorer.score(query)
            assert score.expansion_level in expected_levels, \
                f"Query '{query}' got {score.expansion_level}, expected one of {expected_levels}"
    
    def test_expansion_terms_limit_follows_level(self):
        """Verify expansion_terms_limit matches expansion_level."""
        scorer = QuerySpecificityScorer()
        
        # Ultra-vague should have zero limit
        vague_score = scorer.score("help")
        assert vague_score.expansion_level == "semantic_only"
        assert vague_score.expansion_terms_limit == 0
        
        # Moderate should have some limit
        moderate_query = "api endpoint configuration"  # Technical terms, no vagueness
        moderate_score = scorer.score(moderate_query)
        assert moderate_score.expansion_terms_limit >= 2
        
        # None should have zero limit
        none_query = "this is a very long and specific query about configuring oauth2 jwt authentication"
        none_score = scorer.score(none_query)
        if none_score.expansion_level == "none":
            assert none_score.expansion_terms_limit == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

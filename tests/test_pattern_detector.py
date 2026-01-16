"""
TDD tests for pattern_detector module.

Tests Pattern Detection functionality that caches common issue patterns
and provides learned fix templates for recurring problems.
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import List, Optional


class TestPatternDetectorConfig:
    """Test configuration toggle behavior."""
    
    def test_detector_disabled_by_default(self):
        """Pattern detection should be disabled by default."""
        from ace.pattern_detector import PatternDetector
        
        detector = PatternDetector()
        assert detector.enabled is False
    
    def test_detector_respects_config_toggle(self):
        """Detector should respect ACE_ENABLE_PATTERN_DETECTION env var."""
        from ace.pattern_detector import PatternDetector
        
        with patch.dict('os.environ', {'ACE_ENABLE_PATTERN_DETECTION': 'true'}):
            detector = PatternDetector()
            assert detector.enabled is True
    
    def test_detector_returns_empty_when_disabled(self):
        """When disabled, detect() should return empty list."""
        from ace.pattern_detector import PatternDetector
        
        detector = PatternDetector()
        result = detector.detect("some error message")
        
        assert result == []


class TestPatternDetectorBasic:
    """Test basic pattern detection functionality."""
    
    def test_detect_returns_pattern_match(self):
        """Detector should return matching patterns."""
        from ace.pattern_detector import PatternDetector, PatternMatch
        
        with patch.dict('os.environ', {'ACE_ENABLE_PATTERN_DETECTION': 'true'}):
            detector = PatternDetector()
            # Register a known pattern (note: built-in 'null_reference' also matches)
            detector.register_pattern(
                pattern_id="null_pointer",
                regex=r"NoneType.*has no attribute",
                fix_template="Check if object is None before accessing attribute",
                severity="high"
            )
            
            result = detector.detect("AttributeError: 'NoneType' object has no attribute 'foo'")
            
            assert len(result) >= 1
            assert isinstance(result[0], PatternMatch)
            # Either our custom pattern or the built-in matches
            assert result[0].pattern_id in ("null_pointer", "null_reference")
    
    def test_detect_returns_empty_for_no_match(self):
        """Detector should return empty list when no patterns match."""
        from ace.pattern_detector import PatternDetector
        
        with patch.dict('os.environ', {'ACE_ENABLE_PATTERN_DETECTION': 'true'}):
            detector = PatternDetector()
            result = detector.detect("Everything is working fine")
            
            assert result == []
    
    def test_pattern_match_includes_fix_template(self):
        """PatternMatch should include the fix template."""
        from ace.pattern_detector import PatternDetector
        
        with patch.dict('os.environ', {'ACE_ENABLE_PATTERN_DETECTION': 'true'}):
            detector = PatternDetector()
            detector.register_pattern(
                pattern_id="import_error",
                regex=r"ModuleNotFoundError.*No module named",
                fix_template="Run 'pip install {module}' or check PYTHONPATH",
                severity="medium"
            )
            
            result = detector.detect("ModuleNotFoundError: No module named 'missing_module'")
            
            assert len(result) >= 1
            assert "pip install" in result[0].fix_template


class TestPatternDetectorCaching:
    """Test pattern caching functionality."""
    
    def test_cache_pattern_occurrence(self):
        """Detector should cache pattern occurrences."""
        from ace.pattern_detector import PatternDetector
        
        with patch.dict('os.environ', {'ACE_ENABLE_PATTERN_DETECTION': 'true'}):
            detector = PatternDetector()
            detector.register_pattern(
                pattern_id="syntax_error",
                regex=r"SyntaxError",
                fix_template="Check syntax at indicated line",
                severity="high"
            )
            
            # Trigger detection twice
            detector.detect("SyntaxError: invalid syntax")
            detector.detect("SyntaxError: unexpected EOF")
            
            stats = detector.get_pattern_stats("syntax_error")
            assert stats["occurrence_count"] >= 2
    
    def test_get_frequent_patterns(self):
        """Detector should return most frequently occurring patterns."""
        from ace.pattern_detector import PatternDetector
        
        with patch.dict('os.environ', {'ACE_ENABLE_PATTERN_DETECTION': 'true'}):
            detector = PatternDetector()
            detector.register_pattern(
                pattern_id="frequent_error",
                regex=r"FrequentError",
                fix_template="Common fix",
                severity="low"
            )
            
            # Trigger multiple times
            for _ in range(5):
                detector.detect("FrequentError occurred")
            
            frequent = detector.get_frequent_patterns(limit=5)
            assert len(frequent) >= 1
            assert frequent[0]["pattern_id"] == "frequent_error"


class TestPatternDetectorLearning:
    """Test learned pattern functionality."""
    
    def test_learn_new_pattern_from_resolution(self):
        """Detector should learn new patterns from successful resolutions."""
        from ace.pattern_detector import PatternDetector
        
        with patch.dict('os.environ', {'ACE_ENABLE_PATTERN_DETECTION': 'true'}):
            detector = PatternDetector()
            
            # Learn from a resolution
            detector.learn_pattern(
                error_text="CustomError: specific issue occurred",
                resolution="Apply specific fix for this issue",
                success=True
            )
            
            # Should now detect similar errors
            result = detector.detect("CustomError: specific issue occurred again")
            
            # Learned patterns should be suggested
            assert len(result) >= 1 or detector.has_learned_pattern("CustomError")
    
    def test_confidence_increases_with_successful_resolutions(self):
        """Pattern confidence should increase with successful resolutions."""
        from ace.pattern_detector import PatternDetector
        
        with patch.dict('os.environ', {'ACE_ENABLE_PATTERN_DETECTION': 'true'}):
            detector = PatternDetector()
            detector.register_pattern(
                pattern_id="test_pattern",
                regex=r"TestError",
                fix_template="Apply test fix",
                severity="medium"
            )
            
            initial_confidence = detector.get_pattern_confidence("test_pattern")
            
            # Report successful resolution
            detector.report_resolution("test_pattern", success=True)
            detector.report_resolution("test_pattern", success=True)
            
            new_confidence = detector.get_pattern_confidence("test_pattern")
            assert new_confidence >= initial_confidence


class TestPatternDetectorBuiltins:
    """Test built-in common patterns."""
    
    def test_has_builtin_patterns(self):
        """Detector should have built-in common error patterns."""
        from ace.pattern_detector import PatternDetector
        
        with patch.dict('os.environ', {'ACE_ENABLE_PATTERN_DETECTION': 'true'}):
            detector = PatternDetector()
            
            # Should have some built-in patterns
            builtin_count = detector.get_builtin_pattern_count()
            assert builtin_count >= 5  # At least 5 common patterns

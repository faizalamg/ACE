"""
Pattern Detector module for ACE.

Provides pattern detection and caching for common issues,
with learned fix templates for recurring problems.

Configuration:
    ACE_ENABLE_PATTERN_DETECTION: Enable/disable pattern detection (default: false)
    ACE_PATTERN_CACHE_SIZE: Maximum cached patterns (default: 100)
"""

import os
import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path


def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    return {
        'enable_pattern_detection': os.environ.get(
            'ACE_ENABLE_PATTERN_DETECTION', 'false'
        ).lower() in ('true', '1', 'yes'),
        'pattern_cache_size': int(os.environ.get('ACE_PATTERN_CACHE_SIZE', '100')),
    }


@dataclass
class PatternMatch:
    """Represents a matched pattern with fix template."""
    pattern_id: str
    regex: str
    fix_template: str
    severity: str
    confidence: float = 1.0
    matched_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternStats:
    """Statistics for a registered pattern."""
    pattern_id: str
    occurrence_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_seen: Optional[str] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of resolutions."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


class PatternDetector:
    """
    Detects common error patterns and provides fix templates.
    
    Features:
    - Config-based enable/disable via ACE_ENABLE_PATTERN_DETECTION
    - Pattern registration with regex and fix templates
    - Occurrence caching and statistics
    - Learning from successful resolutions
    - Built-in common error patterns
    
    Example:
        detector = PatternDetector()
        if detector.enabled:
            matches = detector.detect("NoneType has no attribute 'foo'")
            for match in matches:
                print(f"Suggested fix: {match.fix_template}")
    """
    
    # Built-in patterns for common errors
    BUILTIN_PATTERNS = [
        {
            "pattern_id": "null_reference",
            "regex": r"(?:NoneType|null|undefined).*(?:has no attribute|is not|cannot read)",
            "fix_template": "Check if object is None/null before accessing. Use 'if obj is not None:' or optional chaining.",
            "severity": "high"
        },
        {
            "pattern_id": "import_error",
            "regex": r"(?:ModuleNotFoundError|ImportError).*(?:No module named|cannot import)",
            "fix_template": "Install missing module with 'pip install {module}' or verify PYTHONPATH/virtualenv.",
            "severity": "medium"
        },
        {
            "pattern_id": "key_error",
            "regex": r"KeyError:\s*['\"]?\w+['\"]?",
            "fix_template": "Use dict.get(key, default) or check 'if key in dict:' before access.",
            "severity": "medium"
        },
        {
            "pattern_id": "type_error",
            "regex": r"TypeError:.*(?:expected|got|unsupported operand)",
            "fix_template": "Verify argument types match function signature. Add type checks or conversions.",
            "severity": "medium"
        },
        {
            "pattern_id": "syntax_error",
            "regex": r"SyntaxError:.*(?:invalid syntax|unexpected|EOF)",
            "fix_template": "Check for missing colons, parentheses, brackets, or quotation marks at indicated line.",
            "severity": "high"
        },
        {
            "pattern_id": "index_error",
            "regex": r"IndexError:.*(?:out of range|index)",
            "fix_template": "Verify list/array bounds before access. Use 'if len(arr) > index:' or try/except.",
            "severity": "medium"
        },
        {
            "pattern_id": "file_not_found",
            "regex": r"(?:FileNotFoundError|No such file or directory)",
            "fix_template": "Verify file path exists using os.path.exists(). Check working directory with os.getcwd().",
            "severity": "medium"
        },
        {
            "pattern_id": "permission_denied",
            "regex": r"(?:PermissionError|Permission denied|Access denied)",
            "fix_template": "Check file/directory permissions. Run with elevated privileges if needed.",
            "severity": "high"
        },
        {
            "pattern_id": "connection_error",
            "regex": r"(?:ConnectionError|ConnectionRefused|timeout|ECONNREFUSED)",
            "fix_template": "Verify service is running and accessible. Check network connectivity and firewall rules.",
            "severity": "high"
        },
        {
            "pattern_id": "encoding_error",
            "regex": r"(?:UnicodeDecodeError|UnicodeEncodeError|codec can't)",
            "fix_template": "Specify encoding explicitly: open(file, encoding='utf-8'). Handle BOM if present.",
            "severity": "medium"
        },
    ]
    
    def __init__(self):
        """Initialize PatternDetector with configuration."""
        self._config = load_config()
        self._patterns: Dict[str, Dict[str, Any]] = {}
        self._stats: Dict[str, PatternStats] = {}
        self._learned_patterns: Dict[str, Dict[str, Any]] = {}
        
        # Register built-in patterns
        if self._config['enable_pattern_detection']:
            self._register_builtins()
    
    @property
    def enabled(self) -> bool:
        """Check if pattern detection is enabled."""
        return self._config['enable_pattern_detection']
    
    def _register_builtins(self) -> None:
        """Register built-in common patterns."""
        for pattern in self.BUILTIN_PATTERNS:
            self.register_pattern(
                pattern_id=pattern["pattern_id"],
                regex=pattern["regex"],
                fix_template=pattern["fix_template"],
                severity=pattern["severity"]
            )
    
    def register_pattern(
        self,
        pattern_id: str,
        regex: str,
        fix_template: str,
        severity: str = "medium"
    ) -> None:
        """
        Register a new pattern for detection.
        
        Args:
            pattern_id: Unique identifier for the pattern
            regex: Regular expression to match errors
            fix_template: Suggested fix template
            severity: Pattern severity (low, medium, high)
        """
        self._patterns[pattern_id] = {
            "pattern_id": pattern_id,
            "regex": regex,
            "fix_template": fix_template,
            "severity": severity,
            "compiled_regex": re.compile(regex, re.IGNORECASE)
        }
        
        if pattern_id not in self._stats:
            self._stats[pattern_id] = PatternStats(pattern_id=pattern_id)
    
    def detect(self, text: str) -> List[PatternMatch]:
        """
        Detect patterns in the given text.
        
        Args:
            text: Error message or log text to analyze
            
        Returns:
            List of PatternMatch objects for detected patterns
        """
        if not self.enabled:
            return []
        
        matches = []
        
        # Check registered patterns
        for pattern_id, pattern_data in self._patterns.items():
            compiled = pattern_data["compiled_regex"]
            match = compiled.search(text)
            
            if match:
                # Update statistics
                stats = self._stats[pattern_id]
                stats.occurrence_count += 1
                stats.last_seen = datetime.now().isoformat()
                
                matches.append(PatternMatch(
                    pattern_id=pattern_id,
                    regex=pattern_data["regex"],
                    fix_template=pattern_data["fix_template"],
                    severity=pattern_data["severity"],
                    confidence=self.get_pattern_confidence(pattern_id),
                    matched_text=match.group(0)
                ))
        
        # Check learned patterns
        for pattern_id, learned in self._learned_patterns.items():
            if learned.get("regex"):
                try:
                    if re.search(learned["regex"], text, re.IGNORECASE):
                        matches.append(PatternMatch(
                            pattern_id=pattern_id,
                            regex=learned["regex"],
                            fix_template=learned.get("fix_template", ""),
                            severity=learned.get("severity", "medium"),
                            confidence=learned.get("confidence", 0.5)
                        ))
                except re.error:
                    pass  # Skip invalid regex
        
        return matches
    
    def get_pattern_stats(self, pattern_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific pattern.
        
        Args:
            pattern_id: The pattern identifier
            
        Returns:
            Dictionary with pattern statistics
        """
        if pattern_id not in self._stats:
            return {"error": "Pattern not found"}
        
        stats = self._stats[pattern_id]
        return {
            "pattern_id": stats.pattern_id,
            "occurrence_count": stats.occurrence_count,
            "success_count": stats.success_count,
            "failure_count": stats.failure_count,
            "success_rate": stats.success_rate,
            "last_seen": stats.last_seen
        }
    
    def get_frequent_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most frequently occurring patterns.
        
        Args:
            limit: Maximum number of patterns to return
            
        Returns:
            List of pattern statistics sorted by occurrence
        """
        sorted_stats = sorted(
            self._stats.values(),
            key=lambda s: s.occurrence_count,
            reverse=True
        )
        
        return [
            {
                "pattern_id": s.pattern_id,
                "occurrence_count": s.occurrence_count,
                "success_rate": s.success_rate
            }
            for s in sorted_stats[:limit]
            if s.occurrence_count > 0
        ]
    
    def learn_pattern(
        self,
        error_text: str,
        resolution: str,
        success: bool = True
    ) -> None:
        """
        Learn a new pattern from a resolution.
        
        Args:
            error_text: The error text that was resolved
            resolution: The fix that was applied
            success: Whether the resolution was successful
        """
        if not self.enabled:
            return
        
        # Extract key phrases for pattern matching
        # Simple approach: use the first distinctive phrase
        words = error_text.split()
        if len(words) >= 2:
            # Create a simple pattern from first meaningful words
            pattern_key = "_".join(words[:2]).lower()
            pattern_key = re.sub(r'[^a-z0-9_]', '', pattern_key)
            
            # Escape special regex chars and create pattern
            escaped_text = re.escape(error_text[:50])  # First 50 chars
            
            if pattern_key not in self._learned_patterns:
                self._learned_patterns[pattern_key] = {
                    "regex": escaped_text,
                    "fix_template": resolution,
                    "severity": "medium",
                    "confidence": 0.5 if success else 0.3,
                    "success_count": 1 if success else 0,
                    "failure_count": 0 if success else 1
                }
            else:
                # Update existing learned pattern
                learned = self._learned_patterns[pattern_key]
                if success:
                    learned["success_count"] = learned.get("success_count", 0) + 1
                else:
                    learned["failure_count"] = learned.get("failure_count", 0) + 1
                
                # Update confidence based on success rate
                total = learned["success_count"] + learned["failure_count"]
                learned["confidence"] = learned["success_count"] / total if total > 0 else 0.5
    
    def has_learned_pattern(self, error_text: str) -> bool:
        """
        Check if there's a learned pattern for similar errors.
        
        Args:
            error_text: Error text to check
            
        Returns:
            True if a learned pattern exists
        """
        for pattern_id, learned in self._learned_patterns.items():
            if learned.get("regex"):
                try:
                    if re.search(learned["regex"], error_text, re.IGNORECASE):
                        return True
                except re.error:
                    pass
        return False
    
    def get_pattern_confidence(self, pattern_id: str) -> float:
        """
        Get confidence score for a pattern.
        
        Args:
            pattern_id: The pattern identifier
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if pattern_id in self._stats:
            stats = self._stats[pattern_id]
            # Base confidence + boost from successful resolutions
            base = 0.7  # Built-in patterns start with decent confidence
            if stats.success_count + stats.failure_count > 0:
                return min(1.0, base + (stats.success_rate * 0.3))
            return base
        
        if pattern_id in self._learned_patterns:
            return self._learned_patterns[pattern_id].get("confidence", 0.5)
        
        return 0.5
    
    def report_resolution(self, pattern_id: str, success: bool) -> None:
        """
        Report the outcome of a pattern resolution.
        
        Args:
            pattern_id: The pattern that was resolved
            success: Whether the resolution was successful
        """
        if pattern_id in self._stats:
            stats = self._stats[pattern_id]
            if success:
                stats.success_count += 1
            else:
                stats.failure_count += 1
    
    def get_builtin_pattern_count(self) -> int:
        """
        Get the number of built-in patterns.
        
        Returns:
            Count of built-in patterns
        """
        return len(self.BUILTIN_PATTERNS)

"""Typo correction for ACE framework queries using fuzzy matching.

Features:
- Fast fuzzy matching against technical terms (~1ms)
- Auto-learning: Remembers user's common typos for instant O(1) lookup
- Async GLM validation: Background process validates learned corrections
- Spellchecker validation: Skip LLM for words already in English dictionary
"""

import atexit
import difflib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Set, List, Tuple, Any
from queue import Queue, Empty

# Spellchecker for validating words before LLM correction
try:
    from spellchecker import SpellChecker
    _spellchecker = SpellChecker()
except ImportError:
    _spellchecker = None

# Import config for typo correction settings
from .config import get_typo_config

logger = logging.getLogger(__name__)

# Default path for learned typos persistence
DEFAULT_LEARNED_TYPOS_PATH = Path(__file__).parent.parent / "tenant_data" / "learned_typos.json"


class TypoCorrector:
    """Typo corrector using fuzzy matching against ACE technical terms.

    Conservative approach: Only correct words that are CLEARLY typos.
    Words in COMMON_WORDS are NEVER corrected (known valid English).

    Auto-Learning Feature (when enabled via ACE_TYPO_AUTO_LEARN=true):
    - Remembers typo->correction mappings for instant O(1) future lookup
    - Validates corrections via GLM in background thread (non-blocking)
    - Persists learned typos to JSON for cross-session persistence
    """

    # Common English words that should NEVER be corrected
    COMMON_WORDS: Set[str] = {
        # Articles, prepositions, conjunctions
        "the", "a", "an", "of", "to", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "under", "again", "further", "then", "once",
        "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
        # Pronouns
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
        "them", "my", "your", "his", "its", "our", "their", "this", "that",
        "these", "those", "who", "whom", "whose", "which", "what", "where",
        "when", "why", "how", "all", "each", "every", "any", "some", "no",
        # Common verbs
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might",
        "must", "shall", "can", "need", "dare", "ought", "used", "get", "got",
        "go", "goes", "went", "gone", "come", "came", "make", "made", "take",
        "took", "see", "saw", "seen", "know", "knew", "known", "think", "thought",
        "want", "give", "gave", "given", "tell", "told", "say", "said", "let",
        "put", "keep", "kept", "leave", "left", "stand", "stood", "start",
        "stop", "run", "ran", "running", "work", "working", "try", "tried", "trying",
        "use", "used", "using", "show", "showing", "return", "returning", "returned",
        "find", "found", "finding", "set", "setting", "move", "moving", "turn", "turning",
        "look", "looking", "help", "helping", "ask", "asking", "seem", "start", "starting",
        # Common adjectives/adverbs
        "good", "bad", "new", "old", "first", "last", "long", "great", "little",
        "own", "other", "same", "right", "left", "big", "small", "high", "low",
        "next", "early", "late", "young", "old", "important", "few", "public",
        "able", "just", "only", "also", "back", "now", "well", "even", "still",
        "very", "really", "much", "more", "most", "less", "least", "too",
        # Common nouns
        "time", "year", "people", "way", "day", "man", "woman", "child", "world",
        "life", "hand", "part", "place", "case", "week", "company", "system",
        "program", "question", "work", "government", "number", "night", "point",
        "home", "water", "room", "mother", "area", "money", "story", "fact",
        "month", "lot", "study", "book", "eye", "job", "word", "business",
        "issue", "side", "kind", "head", "house", "service", "friend", "father",
        "power", "hour", "game", "line", "end", "member", "law", "car", "city",
        "name", "team", "minute", "idea", "body", "information", "level",
        # Tech-adjacent common words
        "file", "files", "code", "data", "user", "users", "test", "tests",
        "app", "apps", "api", "apis", "web", "page", "pages", "site", "link",
        "ace", "memory", "memories", "query", "queries", "result", "results",
        # Project/workflow words that should not be corrected
        "project", "projects", "update", "updates", "info", "option", "options",
        "status", "feature", "features", "workflow", "workflows", "mode", "modes",
        "command", "commands", "phase", "phases", "tool", "tools", "store", "stores",
        "edit", "edits", "check", "checks", "search", "searches", "text", "texts",
        "bug", "bugs", "fix", "fixes", "pass", "passed", "state", "states",
        "filter", "filters", "model", "models", "final", "hook", "hooks", "mock", "mocks",
        "add", "added", "adding", "git", "repo", "branch", "commit", "commits",
        # Additional technical terms (should never be corrected)
        "inline", "content", "configure", "agentic", "simple", "tracking",
        "evaluation", "expected", "whether", "production",
        # More valid plural forms
        "thresholds", "contexts", "examples", "optimizations", "checkpoints",
        "generators", "tools", "benefits", "solutions",
        # More verb forms
        "executes", "executed", "detect", "retrieves", "provides", "analyzes",
    }

    # Technical terms that typos should be corrected TO
    TECHNICAL_TERMS: Set[str] = {
        "playbook", "bullet", "bullets", "curator", "reflector", "generator",
        "delta", "deltas", "adaptation", "adapter", "offline", "online",
        "retrieval", "embedding", "embeddings", "enrichment", "enriched",
        "helpful", "harmful", "neutral", "evaluate", "execute",
        "accuracy", "precision", "recall", "performance", "optimization",
        "threshold", "similarity", "scoring", "ranking", "reranking",
        "sample", "samples", "batch", "epoch", "epochs", "checkpoint",
        "context", "prompt", "prompts", "template", "templates",
        "config", "configuration", "validation", "verification",
        "exception", "exceptions", "logging", "debugging", "tracing",
        "monitoring", "qdrant", "vector", "vectors", "semantic", "hybrid",
        "namespace", "namespaces", "unified", "multistage", "prefetch",
    }

    # Domain-specific words that should NEVER be "corrected" (whitelist)
    TECHNICAL_WHITELIST: Set[str] = {
        "aceconfig", "thatother", "zen", "opus", "glm", "qdrant", "embeddings",
        "playbook", "curator", "reflector", "generator",
    }

    # Singleton instance for auto-learning state
    _instance: Optional["TypoCorrector"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure shared learned typos state."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize typo corrector with auto-learning support."""
        # Skip if already initialized (singleton)
        if getattr(self, '_initialized', False):
            return

        # Load configuration
        self._config = get_typo_config()

        # Learned typos: typo_lowercase -> correction
        self._learned_typos: Dict[str, str] = {}

        # Pending validations queue: (typo, correction) tuples
        self._validation_queue: Queue = Queue()

        # Background validation thread
        self._validation_thread: Optional[threading.Thread] = None
        self._stop_validation = threading.Event()

        # Load persisted learned typos
        if self._config.enable_auto_learning:
            self._load_learned_typos()
            self._start_validation_thread()

        # Register cleanup on exit
        atexit.register(self._cleanup)

        self._initialized = True

    def _load_learned_typos(self) -> None:
        """Load learned typos from JSON file with validation and cleanup."""
        try:
            path = Path(self._config.learned_typos_path)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    loaded_typos = data.get('typos', {})

                    # Clean up bad corrections before loading
                    cleaned_typos = self._cleanup_bad_corrections(loaded_typos)

                    # Remove cycle mappings (A->B and B->A)
                    no_cycles_typos = self._remove_cycle_mappings(cleaned_typos)

                    self._learned_typos = no_cycles_typos

                    # Log cleanup stats
                    removed_count = len(loaded_typos) - len(self._learned_typos)
                    if removed_count > 0:
                        logger.info(
                            f"Cleaned {removed_count} bad/cycle corrections from learned_typos.json "
                            f"({len(self._learned_typos)} valid entries remain)"
                        )
        except Exception:
            # Silently fail - learned typos are optional optimization
            self._learned_typos = {}

    def _save_learned_typos(self) -> None:
        """Save learned typos to JSON file."""
        try:
            path = Path(self._config.learned_typos_path)
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            # Enforce max learned typos limit (keep most recent)
            if len(self._learned_typos) > self._config.max_learned_typos:
                # Keep only the most recently added (dict maintains insertion order in Python 3.7+)
                items = list(self._learned_typos.items())
                self._learned_typos = dict(items[-self._config.max_learned_typos:])

            with open(path, 'w', encoding='utf-8') as f:
                json.dump({
                    'typos': self._learned_typos,
                    'updated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    'count': len(self._learned_typos)
                }, f, indent=2)
        except Exception:
            # Silently fail - persistence is optional
            pass

    def _start_validation_thread(self) -> None:
        """Start background thread for GLM validation of learned typos."""
        if not self._config.enable_glm_validation:
            return

        self._validation_thread = threading.Thread(
            target=self._validation_worker,
            daemon=True,
            name="TypoValidationWorker"
        )
        self._validation_thread.start()

    def _validation_worker(self) -> None:
        """Background worker that validates typo corrections via GLM."""
        while not self._stop_validation.is_set():
            try:
                # Get next item from queue (block for 1 second)
                typo, correction = self._validation_queue.get(timeout=1.0)

                # Skip if already learned (race condition protection)
                if typo in self._learned_typos:
                    continue

                # Validate via GLM
                if self._validate_correction_with_glm(typo, correction):
                    # Add to learned typos
                    self._learned_typos[typo] = correction
                    # Persist immediately
                    self._save_learned_typos()

            except Empty:
                # Queue empty, continue waiting
                continue
            except Exception:
                # Log and continue
                continue

    def _validate_correction_with_glm(self, typo: str, correction: str) -> bool:
        """Validate a typo correction using GLM API.

        Args:
            typo: The misspelled word
            correction: The proposed correction

        Returns:
            True if GLM confirms the correction is valid
        """
        try:
            # Import LiteLLM client for GLM access
            from .llm_providers.litellm_client import LiteLLMClient
            from .config import get_llm_config

            llm_config = get_llm_config()

            # Use Z.ai GLM for validation
            client = LiteLLMClient(
                model=llm_config.model,
                api_key=llm_config.api_key,
                api_base=llm_config.api_base,
                max_tokens=50,
                temperature=0.0
            )

            prompt = f"""Is "{typo}" a typo/misspelling of "{correction}" in software engineering context?
Answer only YES or NO."""

            response = client.complete(prompt)

            # Check response
            response_text = response.strip().upper()
            return response_text.startswith("YES")

        except Exception:
            # If validation fails, default to accepting the correction
            # (fuzzy matching already established high similarity)
            return True

    def _cleanup(self) -> None:
        """Cleanup on exit - save state and stop threads."""
        self._stop_validation.set()
        if self._validation_thread and self._validation_thread.is_alive():
            self._validation_thread.join(timeout=2.0)
        # Final save
        if self._config.enable_auto_learning:
            self._save_learned_typos()

    def _cleanup_bad_corrections(self, typos: Dict[str, str], min_similarity: float = 0.70) -> Dict[str, str]:
        """Remove learned corrections with low similarity scores.

        Args:
            typos: Dictionary of typo -> correction mappings
            min_similarity: Minimum similarity threshold (default 0.70)

        Returns:
            Cleaned dictionary with only high-similarity corrections
        """
        cleaned = {}
        for typo, correction in typos.items():
            ratio = difflib.SequenceMatcher(None, typo, correction).ratio()
            if ratio >= min_similarity:
                cleaned[typo] = correction
            else:
                logger.debug(
                    f"Removing low-similarity correction ({ratio:.2f}): "
                    f"'{typo}' -> '{correction}'"
                )
        return cleaned

    def _remove_cycle_mappings(self, typos: Dict[str, str]) -> Dict[str, str]:
        """Remove cycle mappings (A->B and B->A) from learned typos.

        When both A->B and B->A exist, both are removed as they indicate
        ambiguous corrections that should not be learned.

        Args:
            typos: Dictionary of typo -> correction mappings

        Returns:
            Dictionary with cycle mappings removed
        """
        # Find all cycles
        cycles = set()
        for typo, correction in typos.items():
            # Check if reverse mapping exists
            if correction in typos and typos[correction] == typo:
                cycles.add(typo)
                cycles.add(correction)

        # Remove cycle mappings
        if cycles:
            logger.warning(
                f"Removing {len(cycles)} cycle mappings from learned typos: "
                f"{sorted(cycles)}"
            )
            return {k: v for k, v in typos.items() if k not in cycles}

        return typos

    def _queue_for_validation(self, typo: str, correction: str) -> None:
        """Queue a typo correction for async GLM validation.

        Non-blocking: adds to queue and returns immediately.
        Only queues corrections that pass similarity threshold to prevent
        bad corrections from polluting the learned typos dictionary.
        """
        if not self._config.enable_auto_learning:
            return

        # Skip if already learned
        if typo in self._learned_typos:
            return

        # CRITICAL: Validate similarity before queuing to prevent bad corrections
        # Increased threshold from 0.65 to 0.70 to reduce false corrections
        ratio = difflib.SequenceMatcher(None, typo, correction).ratio()
        if ratio < 0.70:
            logger.debug(
                f"Skipping validation queue (low similarity {ratio:.2f}): "
                f"'{typo}' -> '{correction}'"
            )
            return

        # Add to validation queue (non-blocking)
        try:
            self._validation_queue.put_nowait((typo, correction))
        except Exception:
            pass  # Queue full, skip this one

    def correct_typos(self, query: str, similarity_threshold: Optional[float] = None) -> str:
        """Correct typos in query using fuzzy matching.

        Args:
            query: Input query string
            similarity_threshold: Minimum similarity for correction (0.0-1.0).
                                  None uses config default (ACE_TYPO_THRESHOLD).

        Returns:
            Query with typos corrected
        """
        if not query:
            return query

        # Use config threshold if not specified
        if similarity_threshold is None:
            similarity_threshold = self._config.similarity_threshold

        tokens = re.findall(r"\w+|\W+", query)
        corrected_tokens = []
        for token in tokens:
            if re.match(r"\w+", token):
                corrected = self._correct_word(token, similarity_threshold)
                corrected_tokens.append(corrected)
            else:
                corrected_tokens.append(token)
        return "".join(corrected_tokens)

    def _correct_word(self, word: str, similarity_threshold: float) -> str:
        """Correct a single word using learned typos (O(1)), fuzzy matching, or LLM.

        Order of operations:
        1. Check if word is too short (<3 chars) - skip
        2. Check if word is in technical whitelist - skip (domain-specific terms)
        3. Check if word is common English - skip
        4. Check if word is already a technical term - skip
        5. Check learned typos (O(1) instant lookup) - with similarity validation
        6. Fuzzy match against technical terms
        7. LLM-based correction (if enabled and fuzzy failed)
        8. Queue new corrections for async GLM validation
        """
        word_lower = word.lower()

        # Skip short words
        if len(word_lower) < 3:
            return word

        # CRITICAL: Never correct whitelisted domain-specific terms
        if word_lower in self.TECHNICAL_WHITELIST:
            return word

        # CRITICAL: Never correct common English words (prevents false corrections)
        if word_lower in self.COMMON_WORDS:
            return word

        # Skip if already a known technical term
        if word_lower in self.TECHNICAL_TERMS:
            return word

        # CRITICAL: Skip correction for valid English dictionary words
        # This prevents overcorrection AND stops bad entries from being learned
        if _spellchecker and word_lower in _spellchecker:
            return word

        # O(1) LOOKUP: Check learned typos first (instant)
        # Also validate similarity to prevent bad learned corrections from persisting
        if self._config.enable_auto_learning and word_lower in self._learned_typos:
            correction = self._learned_typos[word_lower]
            # Validate similarity (prevents bad corrections from poisoning the dictionary)
            # Increased threshold from 0.65 to 0.70 to match queue validation
            ratio = difflib.SequenceMatcher(None, word_lower, correction).ratio()
            if ratio >= 0.70:
                return self._preserve_case(word, correction)
            else:
                # Remove bad correction from dictionary
                logger.warning(
                    f"Removing bad learned correction (low similarity {ratio:.2f}): "
                    f"'{word_lower}' -> '{correction}'"
                )
                del self._learned_typos[word_lower]
                self._save_learned_typos()

        # FUZZY MATCHING: Find close matches in technical terms
        matches = difflib.get_close_matches(
            word_lower, self.TECHNICAL_TERMS, n=3, cutoff=similarity_threshold
        )

        if matches:
            # Sort by length (prefer longer matches) then alphabetically
            if len(matches) > 1:
                matches = sorted(matches, key=lambda m: (-len(m), m))

            correction = matches[0]

            # Queue for async GLM validation (non-blocking)
            self._queue_for_validation(word_lower, correction)

            return self._preserve_case(word, correction)

        # LLM-BASED CORRECTION: For common English typos that fuzzy matching can't handle
        # e.g., "updste" -> "update", "plsn" -> "plan", "chnage" -> "change"
        # Note: Valid English words already filtered by spellchecker check above
        if self._config.enable_llm_correction:
            llm_correction = self._correct_with_llm(word_lower)
            if llm_correction and llm_correction != word_lower:
                # Queue for learning (future O(1) lookup)
                self._queue_for_validation(word_lower, llm_correction)
                return self._preserve_case(word, llm_correction)

        return word

    def _preserve_case(self, original: str, correction: str) -> str:
        """Apply correction while preserving original case pattern."""
        if original.isupper():
            return correction.upper()
        elif original[0].isupper():
            return correction.capitalize()
        else:
            return correction

    def _correct_with_llm(self, word: str) -> Optional[str]:
        """Use LLM to correct a typo when fuzzy matching fails.

        Supports two providers:
        - "zai": z.ai GLM 4.7 (default, high quality, 2 concurrency limit)
        - "local": LM Studio (fast, local inference)

        Args:
            word: The potentially misspelled word (lowercase)

        Returns:
            Corrected word if it's a typo, None or original word if not a typo
        """
        try:
            from .llm_providers.litellm_client import LiteLLMClient
            from .config import get_llm_config
            import logging

            logger = logging.getLogger(__name__)

            # Build provider-specific config
            if self._config.llm_correction_provider == "local":
                # Use local LM Studio
                api_base = self._config.llm_correction_url
                model = self._config.llm_correction_model
                api_key = "not-needed"  # Local LM Studio doesn't need auth
                # Local models don't need thinking disabled
                extra_body = {}
            else:
                # Use z.ai GLM (default)
                llm_config = get_llm_config()
                api_base = llm_config.api_base
                model = llm_config.model
                api_key = llm_config.api_key
                # CRITICAL: Disable thinking mode to get direct responses
                # GLM models default to thinking enabled which returns chain-of-thought
                # in reasoning_content instead of the actual answer
                extra_body = {"thinking": {"type": "disabled"}}

            client = LiteLLMClient(
                model=model,
                api_key=api_key,
                api_base=api_base,
                max_tokens=50,  # Short response needed
                temperature=0.0,  # Deterministic for spelling
            )

            # Context-aware prompt for software engineering domain
            # CRITICAL: Prompt must tell LLM the word may be correct
            prompt = f'If "{word}" is a typo, reply with the corrected word. If it is already correct, reply with "{word}" unchanged. Reply with ONLY one word: '

            llm_response = client.complete(
                prompt, 
                timeout=self._config.llm_correction_timeout,
                extra_body=extra_body
            )

            if llm_response and llm_response.text:
                # Clean the response - extract just the word
                response_text = llm_response.text.strip()
                # Try to extract first word (the correction)
                # Remove quotes, punctuation, and extra content
                correction = response_text.lower().strip('"\'.,!?:; ')
                # Take first word if multiple
                if ' ' in correction:
                    correction = correction.split()[0].strip('"\'.,!?:; ')

                # Validate correction with multiple checks to prevent hallucination:
                # 1. Length bounds: correction within -1/+2 chars of original
                # 2. Edit distance: max edits should be <= 40% of word length
                # 3. Basic sanity: alphabetic, not same as input
                min_len = max(3, len(word) - 1)  # At least 3 chars, and within -1 of original
                max_len = len(word) + 2
                
                if not (correction and 
                        len(correction) >= min_len and 
                        len(correction) <= max_len and
                        correction != word and
                        correction.isalpha()):
                    return None
                
                # Edit distance check: use difflib ratio as proxy
                # Ratio of 0.65+ means ~65% character overlap - reasonable for typos
                # (Increased from 0.6 to reject marginal corrections like "plsn" -> "please")
                ratio = difflib.SequenceMatcher(None, word, correction).ratio()
                if ratio < 0.65:
                    logger.debug(f"LLM correction rejected (low similarity {ratio:.2f}): '{word}' -> '{correction}'")
                    return None
                
                logger.debug(f"LLM typo correction: '{word}' -> '{correction}' (sim={ratio:.2f})")
                return correction

            return None

        except Exception as e:
            # LLM correction is best-effort, don't fail on errors
            import logging
            logging.getLogger(__name__).debug(f"LLM typo correction failed for '{word}': {e}")
            return None

    def add_learned_typo(self, typo: str, correction: str, validate: bool = True) -> bool:
        """Manually add a learned typo mapping.

        Args:
            typo: The misspelled word (will be lowercased)
            correction: The correct word
            validate: If True, validate via GLM before adding

        Returns:
            True if added successfully
        """
        if not self._config.enable_auto_learning:
            return False

        typo_lower = typo.lower()

        if validate and self._config.enable_glm_validation:
            # Queue for async validation
            self._queue_for_validation(typo_lower, correction)
            return True
        else:
            # Add directly without validation
            self._learned_typos[typo_lower] = correction
            self._save_learned_typos()
            return True

    def get_learned_typos(self) -> Dict[str, str]:
        """Get all learned typo mappings.

        Returns:
            Dictionary of typo -> correction mappings
        """
        return self._learned_typos.copy()

    def clear_learned_typos(self) -> None:
        """Clear all learned typos and persist empty state."""
        self._learned_typos = {}
        if self._config.enable_auto_learning:
            self._save_learned_typos()


# Module-level singleton accessor
def get_typo_corrector() -> TypoCorrector:
    """Get the singleton TypoCorrector instance.

    Returns:
        TypoCorrector instance with auto-learning enabled per config
    """
    return TypoCorrector()

"""Playbook storage and mutation logic for ACE."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union, cast

from .delta import DeltaBatch, DeltaOperation


# Phase 1C: Asymmetric penalty weights for bullet tagging
# Harmful tags penalized 2x to suppress bad strategies faster
PENALTY_WEIGHTS = {"helpful": 1, "harmful": 2, "neutral": 1}


@dataclass
class Bullet:
    """Single playbook entry."""

    id: str
    section: str
    content: str
    helpful: int = 0
    harmful: int = 0
    neutral: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_validated: Optional[str] = None

    def apply_metadata(self, metadata: Dict[str, int]) -> None:
        for key, value in metadata.items():
            if hasattr(self, key):
                setattr(self, key, int(value))

    def tag(self, tag: str, increment: int = None) -> None:
        if tag not in ("helpful", "harmful", "neutral"):
            raise ValueError(f"Unsupported tag: {tag}")
        increment = increment if increment is not None else PENALTY_WEIGHTS.get(tag, 1)
        current = getattr(self, tag)
        setattr(self, tag, current + increment)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def effective_score(self, decay_rate: float = 0.95) -> float:
        """
        Calculate score with time-based decay.

        Bullets that haven't been validated recently get their effectiveness
        score decayed over time. This prevents stale knowledge from dominating
        retrieval and encourages periodic revalidation of strategies.

        Args:
            decay_rate: Weekly decay multiplier (default 0.95 = 5% weekly decay)

        Returns:
            Decayed effectiveness score (base_score * decay_factor)

        Example:
            >>> bullet = Bullet(id="b1", section="test", content="test")
            >>> bullet.helpful = 10
            >>> bullet.harmful = 2
            >>> bullet.effective_score()  # No decay without validation
            8.0
            >>> bullet.last_validated = (datetime.now(timezone.utc) - timedelta(weeks=4)).isoformat()
            >>> bullet.effective_score()  # ~6.5 after 4 weeks of decay
            6.502...
        """
        base_score = self.helpful - self.harmful

        # No decay if never validated (cold start)
        if not self.last_validated:
            return float(base_score)

        # Parse last_validated timestamp
        try:
            last_validated_dt = datetime.fromisoformat(self.last_validated)
        except (ValueError, TypeError):
            # Invalid timestamp - no decay
            return float(base_score)

        # Calculate weeks since last validation
        now = datetime.now(timezone.utc)
        weeks_since = (now - last_validated_dt).total_seconds() / (7 * 24 * 3600)

        # Apply exponential decay
        decay_factor = decay_rate ** weeks_since
        return base_score * decay_factor

    def validate(self) -> None:
        """
        Mark bullet as validated at current time.

        Called when a bullet is successfully used in execution to reset
        the decay timer and indicate the knowledge is still relevant.

        Example:
            >>> bullet = Bullet(id="b1", section="test", content="test")
            >>> bullet.validate()
            >>> bullet.last_validated  # ISO timestamp of now
            '2025-...'
        """
        self.last_validated = datetime.now(timezone.utc).isoformat()

    def to_llm_dict(self) -> Dict[str, Any]:
        """
        Return dictionary with only LLM-relevant fields.

        Excludes created_at and updated_at which are internal metadata
        not useful for LLM strategy selection.

        Uses compressed field names for token efficiency:
        - i (id), s (section), c (content)
        - h (helpful), x (harmful), n (neutral)
        Omits neutral:0 (default value) for additional savings.

        Returns:
            Dict with compressed fields for TOON encoding
        """
        result = {
            "i": self.id,
            "s": self.section,
            "c": self.content,
            "h": self.helpful,
            "x": self.harmful,
        }
        # Only include neutral if non-zero (save tokens)
        if self.neutral != 0:
            result["n"] = self.neutral
        return result


@dataclass
class EnrichedBullet(Bullet):
    """
    Bullet with semantic scaffolding metadata for intelligent retrieval.

    Extends Bullet with dimensional, structural, relational, and usage metadata
    that enables purpose-aware retrieval and smarter bullet selection.

    The semantic scaffolding approach follows the principle that the bottleneck
    in retrieval is often not embeddings or rerankers, but lack of metadata
    that reflects how the data is actually used.

    Attributes:
        # Effectiveness metrics
        confidence: Calibrated confidence score (0.0-1.0)

        # Dimensional metadata - WHEN does this apply?
        task_types: Types of tasks this applies to (e.g., ["reasoning", "debugging"])
        domains: Domains this is relevant for (e.g., ["math", "python"])
        complexity_level: Complexity level ("simple", "medium", "complex")

        # Structural metadata - HOW is this strategy structured?
        preconditions: Conditions that should be true for this to apply
        trigger_patterns: Patterns in queries that suggest this bullet
        anti_patterns: Patterns indicating this bullet should NOT be used

        # Relational metadata - WHAT connects to this?
        related_bullets: IDs of related bullets for graph traversal
        supersedes: ID of bullet this replaced (for evolution tracking)
        derived_from: ID of parent bullet if this was refined from another

        # Usage context - WHERE did this prove useful?
        successful_contexts: Compressed examples where this worked
        failure_contexts: Compressed examples where this backfired

        # Retrieval hints - HOW should this be found?
        retrieval_type: "semantic" | "keyword" | "hybrid"
        embedding_text: Custom text for embedding (if different from content)
    """

    # Effectiveness metrics
    confidence: float = 0.0

    # Dimensional metadata - WHEN does this apply?
    task_types: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    complexity_level: str = "medium"

    # Structural metadata - HOW is this strategy structured?
    preconditions: List[str] = field(default_factory=list)
    trigger_patterns: List[str] = field(default_factory=list)
    anti_patterns: List[str] = field(default_factory=list)

    # Relational metadata - WHAT connects to this?
    related_bullets: List[str] = field(default_factory=list)
    supersedes: Optional[str] = None
    derived_from: Optional[str] = None

    # Usage context - WHERE did this prove useful?
    successful_contexts: List[str] = field(default_factory=list)
    failure_contexts: List[str] = field(default_factory=list)

    # Retrieval hints - HOW should this be found?
    retrieval_type: str = "semantic"
    embedding_text: Optional[str] = None

    @property
    def effectiveness_score(self) -> float:
        """
        Compute effectiveness score from helpful/harmful counts.

        Returns:
            Float between 0.0 and 1.0, where:
            - 1.0 = all helpful
            - 0.5 = neutral (default for new bullets)
            - 0.0 = all harmful
        """
        total = self.helpful + self.harmful
        if total == 0:
            return 0.5  # Neutral default for cold start
        return self.helpful / total

    def to_llm_dict(self) -> Dict[str, Any]:
        """
        Return dictionary with LLM-relevant fields including scaffolding.

        Includes core fields plus retrieval-relevant scaffolding metadata.
        Excludes internal metadata (timestamps, embedding_text, contexts).

        Uses compressed field names for token efficiency:
        - i (id), s (section), c (content)
        - h (helpful), x (harmful), n (neutral), cf (confidence)
        - tt (task_types), dm (domains), tp (trigger_patterns)
        - ap (anti_patterns), cl (complexity_level)
        Omits defaults (neutral:0, confidence:0.5, complexity:"medium").

        Returns:
            Dict with core fields and relevant scaffolding
        """
        result = {
            "i": self.id,
            "s": self.section,
            "c": self.content,
            "h": self.helpful,
            "x": self.harmful,
        }

        # Only include non-default values (save tokens)
        if self.neutral != 0:
            result["n"] = self.neutral
        if self.confidence != 0.5:
            result["cf"] = self.confidence

        # Include non-empty scaffolding fields that help LLM understand applicability
        if self.task_types:
            result["tt"] = self.task_types
        if self.domains:
            result["dm"] = self.domains
        if self.trigger_patterns:
            result["tp"] = self.trigger_patterns
        if self.anti_patterns:
            result["ap"] = self.anti_patterns
        if self.complexity_level != "medium":
            result["cl"] = self.complexity_level

        return result

    def to_retrieval_dict(self) -> Dict[str, Any]:
        """
        Return dictionary optimized for vector indexing and retrieval.

        Includes all fields relevant for semantic/keyword retrieval,
        with text_for_embedding being the primary embedding target.

        Returns:
            Dict with retrieval-relevant fields
        """
        return {
            "id": self.id,
            "section": self.section,
            "content": self.content,
            "task_types": self.task_types,
            "domains": self.domains,
            "complexity_level": self.complexity_level,
            "trigger_patterns": self.trigger_patterns,
            "anti_patterns": self.anti_patterns,
            "retrieval_type": self.retrieval_type,
            "text_for_embedding": self.embedding_text or self.content,
            "effectiveness_score": self.effectiveness_score,
        }

    def matches_intent(
        self,
        task_type: Optional[str] = None,
        domain: Optional[str] = None,
        complexity: Optional[str] = None,
    ) -> bool:
        """
        Check if this bullet matches the given intent dimensions.

        Used for pre-filtering bullets before semantic search to reduce
        candidates to only relevant ones.

        Args:
            task_type: Required task type (e.g., "reasoning", "debugging")
            domain: Required domain (e.g., "math", "python")
            complexity: Required complexity level

        Returns:
            True if bullet matches all specified dimensions
        """
        # Check task type if specified
        if task_type and self.task_types and task_type not in self.task_types:
            return False

        # Check domain if specified
        if domain and self.domains and domain not in self.domains:
            return False

        # Check complexity - complex bullets shouldn't be used for simple tasks
        if complexity:
            complexity_order = {"simple": 0, "medium": 1, "complex": 2}
            bullet_complexity = complexity_order.get(self.complexity_level, 1)
            query_complexity = complexity_order.get(complexity, 1)
            # Don't use complex strategies for simple tasks
            if bullet_complexity > query_complexity:
                return False

        return True


def migrate_bullet(bullet: Bullet, enricher: Optional["Callable[[Bullet], EnrichedBullet]"] = None) -> EnrichedBullet:
    """
    Migrate a basic Bullet to an EnrichedBullet.

    If the bullet is already an EnrichedBullet, returns it unchanged.
    Uses heuristic-based enrichment by default.

    Args:
        bullet: Bullet to migrate
        enricher: Optional custom enrichment function

    Returns:
        EnrichedBullet (either existing or newly migrated)
    """
    # Already enriched - return as-is
    if isinstance(bullet, EnrichedBullet):
        return bullet

    # Use custom enricher if provided
    if enricher:
        return enricher(bullet)

    # Default heuristic enrichment based on content
    return enrich_bullet(bullet, bullet.content)


def enrich_bullet(bullet: Bullet, context: str) -> EnrichedBullet:
    """
    Enrich a basic Bullet with semantic scaffolding based on usage context.

    WARNING: This is a HEURISTIC-BASED fallback for development/testing only.

    FOR PRODUCTION USE: Use LLMBulletEnricher instead:
        >>> from ace.enrichment import LLMBulletEnricher
        >>> enricher = LLMBulletEnricher(llm_client)
        >>> enriched = enricher.enrich(bullet, context)

    The LLM-based enricher uses CURATOR_ENRICHMENT_PROMPT to extract
    accurate semantic scaffolding metadata with much higher quality.

    Args:
        bullet: Basic Bullet to enrich
        context: Usage context (question, answer, feedback)

    Returns:
        EnrichedBullet with HEURISTIC-inferred scaffolding metadata
        (production should use LLMBulletEnricher for accurate metadata)
    """
    # Simple heuristic extraction - production should use LLM
    context_lower = context.lower()

    # Infer task types from context keywords
    task_types = []
    if any(kw in context_lower for kw in ["debug", "error", "exception", "crash", "bug"]):
        task_types.append("debugging")
    if any(kw in context_lower for kw in ["why", "how", "explain", "understand"]):
        task_types.append("reasoning")
    if any(kw in context_lower for kw in ["create", "write", "generate", "implement"]):
        task_types.append("creative")
    if any(kw in context_lower for kw in ["check", "validate", "verify", "test"]):
        task_types.append("validation")

    # Infer domains from context keywords
    domains = []
    if any(kw in context_lower for kw in ["python", "def ", "import ", "class "]):
        domains.append("python")
    if any(kw in context_lower for kw in ["java", "public class", "void ", "nullpointer"]):
        domains.append("java")
    if any(kw in context_lower for kw in ["javascript", "typescript", "const ", "let ", "function"]):
        domains.append("javascript")
    if any(kw in context_lower for kw in ["math", "calculate", "equation", "number"]):
        domains.append("math")
    if any(kw in context_lower for kw in ["code", "programming", "software"]):
        domains.append("code")

    # Extract trigger patterns (simplified - just keywords from section)
    trigger_patterns = []
    section_words = bullet.section.lower().split()
    trigger_patterns.extend([w for w in section_words if len(w) > 3])

    # Compress context for storage
    successful_contexts = []
    if any(kw in context_lower for kw in ["correct", "fixed", "worked", "success"]):
        # Store compressed version
        compressed = context[:200] + "..." if len(context) > 200 else context
        successful_contexts.append(compressed)

    return EnrichedBullet(
        id=bullet.id,
        section=bullet.section,
        content=bullet.content,
        helpful=bullet.helpful,
        harmful=bullet.harmful,
        neutral=bullet.neutral,
        created_at=bullet.created_at,
        updated_at=bullet.updated_at,
        task_types=task_types or ["general"],
        domains=domains or ["general"],
        trigger_patterns=trigger_patterns,
        successful_contexts=successful_contexts,
    )


class Playbook:
    """Structured context store as defined by ACE."""

    def __init__(self) -> None:
        self._bullets: Dict[str, Bullet] = {}
        self._sections: Dict[str, List[str]] = {}
        self._next_id = 0

    def __repr__(self) -> str:
        """Concise representation for debugging and object inspection."""
        return f"Playbook(bullets={len(self._bullets)}, sections={list(self._sections.keys())})"

    def __str__(self) -> str:
        """
        Human-readable representation showing actual playbook content.

        Uses markdown format for readability (not TOON) since this is
        typically used for debugging/inspection, not LLM prompts.
        """
        if not self._bullets:
            return "Playbook(empty)"
        return self._as_markdown_debug()

    # ------------------------------------------------------------------ #
    # CRUD utils
    # ------------------------------------------------------------------ #
    def add_bullet(
        self,
        section: str,
        content: str,
        bullet_id: Optional[str] = None,
        metadata: Optional[Dict[str, int]] = None,
    ) -> Bullet:
        bullet_id = bullet_id or self._generate_id(section)
        metadata = metadata or {}
        bullet = Bullet(id=bullet_id, section=section, content=content)
        bullet.apply_metadata(metadata)
        self._bullets[bullet_id] = bullet
        self._sections.setdefault(section, []).append(bullet_id)
        return bullet

    def update_bullet(
        self,
        bullet_id: str,
        *,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, int]] = None,
    ) -> Optional[Bullet]:
        bullet = self._bullets.get(bullet_id)
        if bullet is None:
            return None
        if content is not None:
            bullet.content = content
        if metadata:
            bullet.apply_metadata(metadata)
        bullet.updated_at = datetime.now(timezone.utc).isoformat()
        return bullet

    def tag_bullet(
        self, bullet_id: str, tag: str, increment: int = 1
    ) -> Optional[Bullet]:
        bullet = self._bullets.get(bullet_id)
        if bullet is None:
            return None
        bullet.tag(tag, increment=increment)

        # Opik tracing handles this automatically via @track decorator

        return bullet

    def remove_bullet(self, bullet_id: str) -> None:
        bullet = self._bullets.pop(bullet_id, None)
        if bullet is None:
            return
        section_list = self._sections.get(bullet.section)
        if section_list:
            self._sections[bullet.section] = [
                bid for bid in section_list if bid != bullet_id
            ]
            if not self._sections[bullet.section]:
                del self._sections[bullet.section]

    def get_bullet(self, bullet_id: str) -> Optional[Bullet]:
        return self._bullets.get(bullet_id)

    def bullets(self) -> List[Bullet]:
        return list(self._bullets.values())

    # ------------------------------------------------------------------ #
    # Enriched Bullet support
    # ------------------------------------------------------------------ #
    def add_enriched_bullet(
        self,
        section: str,
        content: str,
        bullet_id: Optional[str] = None,
        metadata: Optional[Dict[str, int]] = None,
        *,
        task_types: Optional[List[str]] = None,
        domains: Optional[List[str]] = None,
        complexity_level: str = "medium",
        preconditions: Optional[List[str]] = None,
        trigger_patterns: Optional[List[str]] = None,
        anti_patterns: Optional[List[str]] = None,
        related_bullets: Optional[List[str]] = None,
        confidence: float = 0.0,
        retrieval_type: str = "semantic",
        embedding_text: Optional[str] = None,
    ) -> EnrichedBullet:
        """
        Add an EnrichedBullet with semantic scaffolding metadata.

        This creates a bullet with full semantic scaffolding for intelligent
        retrieval. Use this for production-grade playbooks where retrieval
        precision matters.

        Args:
            section: Section name for grouping
            content: Strategy content text
            bullet_id: Optional custom ID (auto-generated if not provided)
            metadata: Optional effectiveness metadata (helpful, harmful, neutral)
            task_types: Types of tasks this applies to
            domains: Domains this is relevant for
            complexity_level: "simple", "medium", or "complex"
            preconditions: Conditions for this to apply
            trigger_patterns: Patterns that suggest this bullet
            anti_patterns: Patterns indicating NOT to use this
            related_bullets: IDs of related bullets
            confidence: Calibrated confidence score
            retrieval_type: "semantic", "keyword", or "hybrid"
            embedding_text: Custom text for embedding

        Returns:
            EnrichedBullet instance added to playbook
        """
        bullet_id = bullet_id or self._generate_id(section)
        metadata = metadata or {}

        bullet = EnrichedBullet(
            id=bullet_id,
            section=section,
            content=content,
            task_types=task_types or [],
            domains=domains or [],
            complexity_level=complexity_level,
            preconditions=preconditions or [],
            trigger_patterns=trigger_patterns or [],
            anti_patterns=anti_patterns or [],
            related_bullets=related_bullets or [],
            confidence=confidence,
            retrieval_type=retrieval_type,
            embedding_text=embedding_text,
        )
        bullet.apply_metadata(metadata)

        self._bullets[bullet_id] = bullet
        self._sections.setdefault(section, []).append(bullet_id)
        return bullet

    def get_bullets_by_intent(
        self,
        task_type: Optional[str] = None,
        domain: Optional[str] = None,
        complexity: Optional[str] = None,
    ) -> List[EnrichedBullet]:
        """
        Retrieve bullets matching the given intent dimensions.

        Pre-filters bullets by semantic scaffolding metadata before
        any embedding-based search. This reduces candidates to only
        relevant ones based on task type, domain, and complexity.

        Args:
            task_type: Required task type (e.g., "reasoning", "debugging")
            domain: Required domain (e.g., "math", "python")
            complexity: Required complexity level

        Returns:
            List of EnrichedBullets matching the intent
        """
        results = []
        for bullet in self._bullets.values():
            # Only filter EnrichedBullets with scaffolding
            if isinstance(bullet, EnrichedBullet):
                if bullet.matches_intent(task_type, domain, complexity):
                    results.append(bullet)
            else:
                # Basic bullets match all intents (no filtering possible)
                # Skip them if we're specifically filtering
                if not (task_type or domain or complexity):
                    pass  # Include basic bullets only if no filters
        return results

    def enriched_bullets(self) -> List[EnrichedBullet]:
        """
        Return only EnrichedBullet instances from the playbook.

        Returns:
            List of EnrichedBullet instances
        """
        return [b for b in self._bullets.values() if isinstance(b, EnrichedBullet)]

    def migrate_to_enriched(
        self,
        enricher: Optional[Callable[[Bullet], EnrichedBullet]] = None,
    ) -> int:
        """
        Migrate all basic Bullets to EnrichedBullets.

        Idempotent - already enriched bullets are not modified.

        Args:
            enricher: Optional custom enrichment function

        Returns:
            Number of bullets migrated
        """
        migrated = 0
        for bullet_id, bullet in list(self._bullets.items()):
            if not isinstance(bullet, EnrichedBullet):
                self._bullets[bullet_id] = migrate_bullet(bullet, enricher)
                migrated += 1
        return migrated

    def list_sections(self) -> List[str]:
        """Return list of all section names."""
        return list(self._sections.keys())

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, object]:
        return {
            "bullets": {
                bullet_id: asdict(bullet) for bullet_id, bullet in self._bullets.items()
            },
            "sections": self._sections,
            "next_id": self._next_id,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "Playbook":
        instance = cls()
        bullets_payload = payload.get("bullets", {})
        if isinstance(bullets_payload, dict):
            for bullet_id, bullet_value in bullets_payload.items():
                if isinstance(bullet_value, dict):
                    # Detect if this is an EnrichedBullet by presence of scaffolding fields
                    enriched_fields = {"task_types", "domains", "complexity_level", "confidence"}
                    if enriched_fields & set(bullet_value.keys()):
                        # Filter out any unexpected fields for safety
                        enriched_fields_all = {
                            "id", "section", "content", "helpful", "harmful", "neutral",
                            "created_at", "updated_at", "last_validated", "confidence", "task_types",
                            "domains", "complexity_level", "preconditions", "trigger_patterns",
                            "anti_patterns", "related_bullets", "supersedes", "derived_from",
                            "successful_contexts", "failure_contexts", "retrieval_type",
                            "embedding_text",
                        }
                        filtered_value = {k: v for k, v in bullet_value.items() if k in enriched_fields_all}
                        instance._bullets[bullet_id] = EnrichedBullet(**filtered_value)
                    else:
                        # Basic bullet - filter to only base fields
                        base_fields = {"id", "section", "content", "helpful", "harmful", "neutral", "created_at", "updated_at", "last_validated"}
                        filtered_value = {k: v for k, v in bullet_value.items() if k in base_fields}
                        instance._bullets[bullet_id] = Bullet(**filtered_value)

        # Handle legacy format where bullets are in sections dict
        sections_payload = payload.get("sections", {})
        if isinstance(sections_payload, dict):
            # Check if sections contains bullet data directly (legacy format)
            for section, section_data in sections_payload.items():
                if isinstance(section_data, list):
                    # Could be either list of IDs or list of bullet dicts
                    if section_data and isinstance(section_data[0], dict):
                        # Legacy format: sections contain bullet dicts
                        for bullet_data in section_data:
                            if isinstance(bullet_data, dict):
                                bullet_id = bullet_data.get("id", "")
                                # Detect enriched vs basic
                                enriched_fields = {"task_types", "domains", "complexity_level", "confidence"}
                                if enriched_fields & set(bullet_data.keys()):
                                    instance._bullets[bullet_id] = EnrichedBullet(**bullet_data)
                                else:
                                    instance._bullets[bullet_id] = Bullet(**bullet_data)
                                instance._sections.setdefault(section, []).append(bullet_id)
                    else:
                        # New format: sections contain list of IDs
                        instance._sections[section] = list(section_data) if isinstance(section_data, Iterable) else []

        next_id_value = payload.get("next_id", 0)
        instance._next_id = (
            int(cast(Union[int, str], next_id_value))
            if next_id_value is not None
            else 0
        )
        return instance

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def loads(cls, data: str) -> "Playbook":
        payload = json.loads(data)
        if not isinstance(payload, dict):
            raise ValueError("Playbook serialization must be a JSON object.")
        return cls.from_dict(payload)

    def save_to_file(self, path: str) -> None:
        """Save playbook to a JSON file.

        Args:
            path: File path where to save the playbook

        Example:
            >>> playbook.save_to_file("trained_model.json")
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            f.write(self.dumps())

    def to_json_file(self, path: str) -> None:
        """Alias for save_to_file for API consistency.

        Args:
            path: File path where to save the playbook
        """
        self.save_to_file(path)

    @classmethod
    def from_json_file(cls, path: str, auto_migrate: bool = False) -> "Playbook":
        """Load playbook from a JSON file with optional auto-migration.

        Args:
            path: File path to load the playbook from
            auto_migrate: If True, automatically migrate basic bullets to enriched

        Returns:
            Playbook instance loaded from the file

        Example:
            >>> playbook = Playbook.from_json_file("model.json", auto_migrate=True)
        """
        playbook = cls.load_from_file(path)
        if auto_migrate:
            playbook.migrate_to_enriched()
        return playbook

    @classmethod
    def load_from_file(cls, path: str) -> "Playbook":
        """Load playbook from a JSON file.

        Args:
            path: File path to load the playbook from

        Returns:
            Playbook instance loaded from the file

        Example:
            >>> playbook = Playbook.load_from_file("trained_model.json")

        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
            ValueError: If the JSON doesn't represent a valid playbook
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Playbook file not found: {path}")
        with file_path.open("r", encoding="utf-8") as f:
            return cls.loads(f.read())

    # ------------------------------------------------------------------ #
    # Delta application
    # ------------------------------------------------------------------ #
    def apply_delta(self, delta: DeltaBatch) -> None:
        bullets_before = len(self._bullets)

        for operation in delta.operations:
            self._apply_operation(operation)

        bullets_after = len(self._bullets)

        # Opik tracing handles this automatically via @track decorator

    def _apply_operation(self, operation: DeltaOperation) -> None:
        op_type = operation.type.upper()
        if op_type == "ADD":
            # Use enriched bullet if enrichment metadata provided
            if operation.enrichment:
                self.add_enriched_bullet(
                    section=operation.section,
                    content=operation.content or "",
                    bullet_id=operation.bullet_id,
                    metadata=operation.metadata,
                    task_types=operation.enrichment.get("task_types"),
                    domains=operation.enrichment.get("domains"),
                    complexity_level=operation.enrichment.get("complexity_level", "medium"),
                    preconditions=operation.enrichment.get("preconditions"),
                    trigger_patterns=operation.enrichment.get("trigger_patterns"),
                    anti_patterns=operation.enrichment.get("anti_patterns"),
                    related_bullets=operation.enrichment.get("related_bullets"),
                    confidence=operation.enrichment.get("confidence", 0.0),
                    retrieval_type=operation.enrichment.get("retrieval_type", "semantic"),
                    embedding_text=operation.enrichment.get("embedding_text"),
                )
            else:
                self.add_bullet(
                    section=operation.section,
                    content=operation.content or "",
                    bullet_id=operation.bullet_id,
                    metadata=operation.metadata,
                )
        elif op_type == "UPDATE":
            if operation.bullet_id is None:
                return
            self.update_bullet(
                operation.bullet_id,
                content=operation.content,
                metadata=operation.metadata,
            )
        elif op_type == "TAG":
            if operation.bullet_id is None:
                return
            # Only apply valid tag names as defensive measure
            valid_tags = {"helpful", "harmful", "neutral"}
            for tag, increment in operation.metadata.items():
                if tag in valid_tags:
                    self.tag_bullet(operation.bullet_id, tag, increment)
        elif op_type == "REMOVE":
            if operation.bullet_id is None:
                return
            self.remove_bullet(operation.bullet_id)

    # ------------------------------------------------------------------ #
    # Presentation helpers
    # ------------------------------------------------------------------ #
    def as_prompt(self) -> str:
        """
        Return TOON-encoded playbook for LLM prompts.

        Uses tab delimiters and excludes internal metadata (created_at, updated_at)
        for maximum token efficiency (~16-62% savings vs markdown).

        Returns:
            TOON-formatted string with bullets array

        Raises:
            ImportError: If python-toon is not installed
        """
        try:
            from toon import encode
        except ImportError:
            raise ImportError(
                "TOON compression requires python-toon. "
                "Install with: pip install python-toon>=0.1.0"
            )

        # Only include LLM-relevant fields (exclude created_at, updated_at)
        bullets_data = [b.to_llm_dict() for b in self.bullets()]

        # Use tab delimiter for 5-10% better compression than comma
        return encode({"bullets": bullets_data}, {"delimiter": "\t"})

    def _as_markdown_debug(self) -> str:
        """
        Human-readable markdown format for debugging/inspection only.

        This format is more readable than TOON but uses more tokens.
        Use for debugging, logging, or human inspection - not for LLM prompts.

        Returns:
            Markdown-formatted playbook string
        """
        parts: List[str] = []
        for section, bullet_ids in sorted(self._sections.items()):
            parts.append(f"## {section}")
            for bullet_id in bullet_ids:
                bullet = self._bullets[bullet_id]
                counters = f"(helpful={bullet.helpful}, harmful={bullet.harmful}, neutral={bullet.neutral})"
                parts.append(f"- [{bullet.id}] {bullet.content} {counters}")
        return "\n".join(parts)

    def stats(self) -> Dict[str, object]:
        return {
            "sections": len(self._sections),
            "bullets": len(self._bullets),
            "tags": {
                "helpful": sum(b.helpful for b in self._bullets.values()),
                "harmful": sum(b.harmful for b in self._bullets.values()),
                "neutral": sum(b.neutral for b in self._bullets.values()),
            },
        }

    # ------------------------------------------------------------------ #
    # Golden Rules Auto-Promotion
    # ------------------------------------------------------------------ #
    def check_and_promote_golden_rules(self, config: Optional["ELFConfig"] = None) -> List[str]:
        """
        Check all bullets and promote qualifying ones to golden_rules section.

        High-performing bullets that meet promotion thresholds are automatically
        moved to the golden_rules section, signaling their proven value.

        Args:
            config: Optional ELF configuration (uses global config if not provided)

        Returns:
            List of promoted bullet IDs

        Example:
            >>> promoted = playbook.check_and_promote_golden_rules()
            >>> print(f"Promoted {len(promoted)} bullets to golden_rules")
        """
        if config is None:
            from .config import get_elf_config
            config = get_elf_config()

        if not config.enable_golden_rules:
            return []

        promoted = []
        for section_name, bullet_ids in list(self._sections.items()):
            if section_name == "golden_rules":
                continue  # Don't re-promote golden rules

            for bullet_id in bullet_ids[:]:  # Copy list for safe iteration
                bullet = self._bullets.get(bullet_id)
                if bullet and self._qualifies_for_golden(bullet, config):
                    # Remove from current section
                    self._sections[section_name].remove(bullet_id)
                    if not self._sections[section_name]:
                        del self._sections[section_name]

                    # Move to golden_rules section
                    bullet.section = "golden_rules"
                    bullet.updated_at = datetime.now(timezone.utc).isoformat()

                    if "golden_rules" not in self._sections:
                        self._sections["golden_rules"] = []
                    self._sections["golden_rules"].append(bullet_id)
                    promoted.append(bullet_id)

        return promoted

    def demote_from_golden_rules(self, config: Optional["ELFConfig"] = None) -> List[str]:
        """
        Check golden_rules section and demote bullets that no longer qualify.

        Bullets that have accumulated harmful feedback beyond the demotion threshold
        are moved to a "deprecated" section to prevent continued use.

        Args:
            config: Optional ELF configuration (uses global config if not provided)

        Returns:
            List of demoted bullet IDs

        Example:
            >>> demoted = playbook.demote_from_golden_rules()
            >>> print(f"Demoted {len(demoted)} bullets from golden_rules")
        """
        if config is None:
            from .config import get_elf_config
            config = get_elf_config()

        if not config.enable_golden_rules:
            return []

        # No golden_rules section exists
        if "golden_rules" not in self._sections:
            return []

        demoted = []
        golden_bullet_ids = self._sections["golden_rules"][:]  # Copy for safe iteration

        for bullet_id in golden_bullet_ids:
            bullet = self._bullets.get(bullet_id)
            if bullet and self._should_demote_from_golden(bullet, config):
                # Remove from golden_rules
                self._sections["golden_rules"].remove(bullet_id)
                if not self._sections["golden_rules"]:
                    del self._sections["golden_rules"]

                # Move to deprecated section
                bullet.section = "deprecated"
                bullet.updated_at = datetime.now(timezone.utc).isoformat()

                if "deprecated" not in self._sections:
                    self._sections["deprecated"] = []
                self._sections["deprecated"].append(bullet_id)
                demoted.append(bullet_id)

        return demoted

    def _qualifies_for_golden(self, bullet: Bullet, config: "ELFConfig") -> bool:
        """
        Check if bullet qualifies for golden rule promotion.

        Args:
            bullet: Bullet to check
            config: ELF configuration with thresholds

        Returns:
            True if bullet meets promotion criteria
        """
        return (
            bullet.helpful >= config.golden_rule_helpful_threshold
            and bullet.harmful <= config.golden_rule_max_harmful
        )

    def _should_demote_from_golden(self, bullet: Bullet, config: "ELFConfig") -> bool:
        """
        Check if golden rule bullet should be demoted.

        Args:
            bullet: Bullet to check
            config: ELF configuration with thresholds

        Returns:
            True if bullet has accumulated too much harmful feedback
        """
        return bullet.harmful >= config.golden_rule_demotion_harmful_threshold

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _generate_id(self, section: str) -> str:
        self._next_id += 1
        section_prefix = section.split()[0].lower()
        return f"{section_prefix}-{self._next_id:05d}"

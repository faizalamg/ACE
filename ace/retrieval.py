"""Smart retrieval system for purpose-aware bullet retrieval.

This module provides intelligent retrieval of bullets from a playbook
using semantic scaffolding metadata for purpose-aware, multi-dimensional filtering.

ELF-Inspired Features (when enabled via config):
- Confidence Decay: Older knowledge scores lower over time
- Golden Rules: High-performing strategies get score boost
- Quality Boost: Helpful/harmful feedback affects ranking

ARIA Features (when enabled):
- LinUCB Bandit: Dynamic preset selection based on query features
- Quality Feedback: Score adjustment from user feedback

Cross-Encoder Reranking (when enabled via ACE_ENABLE_RERANKING=true):
- Uses sentence-transformers cross-encoder for second-stage reranking
- Improves precision on ambiguous queries
- Configurable via ACE_ENABLE_RERANKING and ACE_CROSS_ENCODER env vars
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, TYPE_CHECKING

# Import config for ELF feature flags and retrieval config
from .config import get_elf_config, get_retrieval_config

if TYPE_CHECKING:
    from .playbook import Bullet, EnrichedBullet, Playbook
    from .session_tracking import SessionOutcomeTracker
    from .qdrant_retrieval import QdrantBulletIndex
    from .unified_memory import UnifiedMemoryIndex, UnifiedBullet, UnifiedNamespace


IntentType = Literal["analytical", "factual", "procedural", "general"]


@dataclass
class ScoredBullet:
    """A bullet with its relevance score for retrieval ranking.

    Attributes:
        bullet: The retrieved bullet (Bullet or EnrichedBullet)
        score: Relevance score (0.0 to 1.0)
        match_reasons: List of reasons why this bullet matched
    """

    bullet: "Bullet"
    score: float
    match_reasons: List[str] = field(default_factory=list)

    @property
    def content(self) -> str:
        """Convenience accessor for bullet content."""
        return self.bullet.content

    def __repr__(self) -> str:
        return f"ScoredBullet(score={self.score:.3f}, content={self.content[:50]}...)"


class SmartBulletIndex:
    """Purpose-aware retrieval index for playbook bullets.

    SmartBulletIndex enables intelligent retrieval of bullets based on:
    - Task type filtering (debugging, reasoning, etc.)
    - Domain filtering (math, software, etc.)
    - Complexity level filtering
    - Trigger pattern matching
    - Intent-based routing (analytical/factual/procedural)
    - Effectiveness-based ranking
    - Semantic similarity search

    Example:
        >>> from ace import Playbook
        >>> from ace.retrieval import SmartBulletIndex
        >>>
        >>> playbook = Playbook()
        >>> playbook.add_enriched_bullet(
        ...     section="debugging",
        ...     content="Check logs first",
        ...     task_types=["debugging"],
        ...     trigger_patterns=["error", "bug"],
        ... )
        >>>
        >>> index = SmartBulletIndex(playbook=playbook)
        >>> results = index.retrieve(task_type="debugging")
    """

    def __init__(
        self,
        playbook: Optional["Playbook"] = None,
        session_tracker: Optional["SessionOutcomeTracker"] = None,
        qdrant_index: Optional["QdrantBulletIndex"] = None,
        unified_index: Optional["UnifiedMemoryIndex"] = None,
    ) -> None:
        """Initialize the SmartBulletIndex.

        Args:
            playbook: Optional playbook to index. Can be set later via update().
            session_tracker: Optional session outcome tracker for session-aware retrieval.
            qdrant_index: Optional QdrantBulletIndex for O(1) vector-based retrieval.
                         When provided, retrieve() will use hybrid search via Qdrant.
            unified_index: Optional UnifiedMemoryIndex for unified memory retrieval.
                          When provided, enables namespace-aware retrieval combining
                          playbook bullets with user preferences and task strategies.
        """
        self._playbook = playbook
        self._session_tracker = session_tracker
        self._qdrant_index = qdrant_index
        self._unified_index = unified_index
        self._bullets: List["Bullet"] = []
        self._embeddings: Dict[str, List[float]] = {}  # bullet_id -> embedding

        if playbook:
            self.update()

    def __len__(self) -> int:
        """Return the number of indexed bullets."""
        return len(self._bullets)

    def update(self, playbook: Optional["Playbook"] = None) -> None:
        """Update the index from the playbook.

        Args:
            playbook: Optional new playbook to use. If not provided,
                     uses the existing playbook reference.
        """
        if playbook:
            self._playbook = playbook

        if self._playbook:
            self._bullets = list(self._playbook.bullets())
            # Clear embeddings cache when bullets change
            self._embeddings.clear()

    def _apply_elf_scoring(
        self,
        bullet: Any,
        base_score: float,
        match_reasons: List[str],
    ) -> Tuple[float, List[str]]:
        """Apply ELF-inspired scoring adjustments based on config flags.

        Args:
            bullet: The bullet being scored (UnifiedBullet or EnrichedBullet)
            base_score: The initial score before ELF adjustments
            match_reasons: List of match reasons to append to

        Returns:
            Tuple of (adjusted_score, updated_match_reasons)

        Respects config flags:
            - enable_confidence_decay: Apply time-based decay
            - enable_golden_rules: Boost golden bullets
            - ELF quality boost always applied if helpful/harmful data available
        """
        elf_config = get_elf_config()
        score = base_score
        reasons = list(match_reasons)

        # 1. Confidence Decay - older knowledge scores lower
        if elf_config.enable_confidence_decay:
            last_validated = getattr(bullet, 'last_validated', None)
            if last_validated:
                try:
                    if isinstance(last_validated, str):
                        last_validated = datetime.fromisoformat(last_validated.replace('Z', '+00:00'))
                    if last_validated.tzinfo is None:
                        last_validated = last_validated.replace(tzinfo=timezone.utc)

                    now = datetime.now(timezone.utc)
                    weeks_old = (now - last_validated).days / 7.0
                    decay_factor = max(
                        elf_config.min_confidence_threshold,
                        elf_config.decay_rate_per_week ** weeks_old
                    )
                    score *= decay_factor
                    if decay_factor < 0.95:  # Only note significant decay
                        reasons.append(f"decay:{decay_factor:.2f}")
                except (ValueError, TypeError):
                    pass  # Skip decay if timestamp parsing fails

        # 2. Golden Rules Boost - high-performing strategies get priority
        if elf_config.enable_golden_rules:
            is_golden = getattr(bullet, 'is_golden', False)
            if is_golden:
                score *= 1.25  # 25% boost for golden rules
                reasons.append("golden_rule:+25%")

        # 3. Quality Boost - helpful/harmful feedback affects ranking
        # Always applied if data available (no separate flag needed)
        helpful = getattr(bullet, 'helpful_count', 0) or 0
        harmful = getattr(bullet, 'harmful_count', 0) or 0
        total_feedback = helpful + harmful
        if total_feedback > 0:
            quality_ratio = (helpful - harmful) / total_feedback
            # Scale: -1.0 to +1.0 -> -0.15 to +0.15 boost
            quality_boost = quality_ratio * 0.15
            score = max(0.05, score + quality_boost)  # Ensure minimum score
            if abs(quality_boost) > 0.01:
                reasons.append(f"quality:{quality_boost:+.2f}")

        return (score, reasons)

    def retrieve(
        self,
        query: Optional[str] = None,
        task_type: Optional[str] = None,
        domain: Optional[str] = None,
        complexity: Optional[str] = None,
        intent: Optional[IntentType] = None,
        limit: Optional[int] = None,
        rank_by_effectiveness: bool = False,
        min_effectiveness: Optional[float] = None,
        query_type: Optional[str] = None,
        trigger_override_threshold: float = 0.3,
        session_type: Optional[str] = None,
        namespace: Optional[Union[str, "UnifiedNamespace", List[Union[str, "UnifiedNamespace"]]]] = None,
        created_after: Optional["datetime"] = None,
        created_before: Optional["datetime"] = None,
        updated_after: Optional["datetime"] = None,
        rerank: Optional[bool] = None,
        rerank_candidates: Optional[int] = None,
    ) -> List[ScoredBullet]:
        """Retrieve bullets matching the given criteria.

        Args:
            query: Natural language query for trigger pattern matching
            task_type: Filter by task type (e.g., "debugging", "reasoning")
            domain: Filter by domain (e.g., "math", "software")
            complexity: Filter by complexity level (e.g., "simple", "medium", "complex")
            intent: Filter by retrieval intent type (analytical/factual/procedural)
            limit: Maximum number of results to return
            rank_by_effectiveness: If True, rank results by effectiveness score
            min_effectiveness: Minimum effectiveness score threshold (0.0 to 1.0)
            query_type: Query type for scoring boost (matches against task_types metadata)
            trigger_override_threshold: Trigger score threshold above which effectiveness
                filter is bypassed (default: 0.3). Strong trigger matches override
                effectiveness filtering to ensure relevant bullets aren't excluded.
            session_type: Optional session type for session-aware effectiveness scoring.
                When provided with session_tracker, uses session-specific effectiveness
                instead of global effectiveness.
            namespace: Optional namespace filter for unified memory retrieval.
                Can be a single namespace (string or UnifiedNamespace enum),
                or a list of namespaces. Requires unified_index to be set.
                If None and unified_index is set, retrieves from all sources.
            created_after: Optional datetime filter - only retrieve bullets created after this time
            created_before: Optional datetime filter - only retrieve bullets created before this time
            updated_after: Optional datetime filter - only retrieve bullets updated after this time
            rerank: If True, apply cross-encoder reranking for improved precision.
                Default is controlled by ACE_ENABLE_RERANKING env var (default: True).
                Requires sentence-transformers: pip install ace[reranking]
            rerank_candidates: Number of candidates to retrieve before reranking.
                Only used when rerank=True. Higher values may improve quality at cost of latency.
                Default is controlled by ACE_FIRST_STAGE_K env var (default: 40).

        Returns:
            List of ScoredBullet objects, sorted by relevance score descending.
        """
        from .playbook import EnrichedBullet

        results: List[ScoredBullet] = []

        # Retrieve from unified index if available and namespace specified
        if self._unified_index is not None and namespace is not None:
            # Import UnifiedBullet for type checking
            try:
                from .unified_memory import UnifiedBullet
            except ImportError:
                UnifiedBullet = None

            # Retrieve from unified index with namespace filter
            # Use threshold=0.15 to filter out clearly irrelevant results early
            unified_bullets = self._unified_index.retrieve(
                query=query or "",
                namespace=namespace,
                limit=limit or 100,
                threshold=0.15  # Filter out low-relevance results from Qdrant
            )

            # Convert UnifiedBullets to ScoredBullets
            for unified in unified_bullets:
                # Get Qdrant's semantic similarity score (RRF fusion score)
                # This is the PRIMARY relevance signal from vector search
                semantic_score = getattr(unified, 'qdrant_score', 0.0) or 0.0

                # Calculate secondary scores
                effectiveness = unified.effectiveness_score
                trigger_score = self._match_trigger_patterns(query or "", unified) if query else 0.0

                # NEW SCORING FORMULA: Semantic-first ranking (optimized for 95%+ precision)
                # - semantic_score (0.7): Primary factor - actual semantic relevance from Qdrant
                # - trigger_score (0.2): Secondary - keyword/pattern matches
                # - effectiveness (0.1): Tertiary - proven usefulness (reduced to prevent pollution)
                base_score = (0.7 * semantic_score) + (0.2 * trigger_score) + (0.1 * effectiveness)

                # Apply min_effectiveness filter (with semantic or trigger override)
                strong_semantic_match = semantic_score >= 0.3
                strong_trigger_match = trigger_score >= trigger_override_threshold
                if min_effectiveness is not None and effectiveness < min_effectiveness:
                    if not strong_semantic_match and not strong_trigger_match:
                        continue

                match_reasons = [
                    f"namespace:{unified.namespace}",
                    f"semantic:{semantic_score:.2f}",
                    f"effectiveness:{effectiveness:.2f}",
                ]
                if trigger_score > 0:
                    match_reasons.append(f"trigger_match:{trigger_score:.2f}")

                # Apply ELF scoring adjustments (respects config flags)
                score, match_reasons = self._apply_elf_scoring(unified, base_score, match_reasons)

                results.append(ScoredBullet(
                    bullet=unified,
                    score=max(0.1, score),  # Ensure minimum score for matched bullets
                    match_reasons=match_reasons
                ))

        # Also retrieve from unified index when namespace is None (search all sources)
        elif self._unified_index is not None and namespace is None:
            try:
                from .unified_memory import UnifiedBullet
            except ImportError:
                UnifiedBullet = None

            if UnifiedBullet is not None:
                # Use threshold=0.35 with query expansion for 95%+ precision
                unified_bullets = self._unified_index.retrieve(
                    query=query or "",
                    namespace=None,  # All namespaces
                    limit=limit or 100,
                    threshold=0.35  # Balanced with query expansion
                )

                for unified in unified_bullets:
                    # Get Qdrant's semantic similarity score (RRF fusion score)
                    semantic_score = getattr(unified, 'qdrant_score', 0.0) or 0.0

                    # Calculate secondary scores
                    effectiveness = unified.effectiveness_score
                    trigger_score = self._match_trigger_patterns(query or "", unified) if query else 0.0

                    # NEW SCORING FORMULA: Semantic-first ranking
                    # - semantic_score (0.5): Primary factor - actual semantic relevance
                    # - trigger_score (0.2): Secondary - keyword/pattern matches
                    # - effectiveness (0.3): Tertiary - proven usefulness from feedback
                    base_score = (0.5 * semantic_score) + (0.2 * trigger_score) + (0.3 * effectiveness)

                    # Apply min_effectiveness filter (with semantic or trigger override)
                    strong_semantic_match = semantic_score >= 0.3
                    strong_trigger_match = trigger_score >= trigger_override_threshold
                    if min_effectiveness is not None and effectiveness < min_effectiveness:
                        if not strong_semantic_match and not strong_trigger_match:
                            continue

                    match_reasons = [
                        f"namespace:{unified.namespace}",
                        f"semantic:{semantic_score:.2f}",
                        f"effectiveness:{effectiveness:.2f}",
                    ]
                    if trigger_score > 0:
                        match_reasons.append(f"trigger_match:{trigger_score:.2f}")

                    # Apply ELF scoring adjustments (respects config flags)
                    score, match_reasons = self._apply_elf_scoring(unified, base_score, match_reasons)

                    results.append(ScoredBullet(
                        bullet=unified,
                        score=max(0.1, score),
                        match_reasons=match_reasons
                    ))

        # Process playbook bullets only if no namespace filter (namespace-specific queries
        # should only return from unified index)
        if namespace is not None and self._unified_index is not None:
            # Skip playbook bullets when specific namespace requested
            pass
        else:
            # Process playbook bullets
            for bullet in self._bullets:
                # Temporal filtering
                if created_after or created_before or updated_after:
                    try:
                        bullet_created = datetime.fromisoformat(bullet.created_at.replace('Z', '+00:00'))
                        bullet_updated = datetime.fromisoformat(bullet.updated_at.replace('Z', '+00:00'))

                        # Ensure timezone-aware comparison
                        if bullet_created.tzinfo is None:
                            bullet_created = bullet_created.replace(tzinfo=timezone.utc)
                        if bullet_updated.tzinfo is None:
                            bullet_updated = bullet_updated.replace(tzinfo=timezone.utc)

                        if created_after and bullet_created < created_after:
                            continue
                        if created_before and bullet_created >= created_before:
                            continue
                        if updated_after and bullet_updated < updated_after:
                            continue
                    except (ValueError, TypeError, AttributeError):
                        pass  # Keep bullet if timestamp parsing fails

                score = 0.0
                match_reasons: List[str] = []

                # Check if this is an EnrichedBullet for metadata filtering
                is_enriched = isinstance(bullet, EnrichedBullet)

                # Task type filter
                if task_type:
                    if is_enriched and task_type in bullet.task_types:
                        score += 0.3
                        match_reasons.append(f"task_type:{task_type}")
                    elif not is_enriched:
                        # For basic bullets, use a heuristic based on content
                        if task_type.lower() in bullet.content.lower():
                            score += 0.1
                            match_reasons.append(f"content_contains:{task_type}")
                    else:
                        continue  # Skip if task_type filter doesn't match

                # Domain filter
                if domain:
                    if is_enriched and domain in bullet.domains:
                        score += 0.3
                        match_reasons.append(f"domain:{domain}")
                    elif is_enriched and domain == "all":
                        score += 0.1
                        match_reasons.append("domain:all")
                    elif not is_enriched:
                        if domain.lower() in bullet.content.lower():
                            score += 0.1
                            match_reasons.append(f"content_contains:{domain}")
                    elif domain != "all":
                        continue  # Skip if domain filter doesn't match

                # Complexity filter
                if complexity:
                    if is_enriched and bullet.complexity_level == complexity:
                        score += 0.2
                        match_reasons.append(f"complexity:{complexity}")
                    elif not is_enriched:
                        pass  # Basic bullets have no complexity, include them
                    else:
                        continue  # Skip if complexity doesn't match

                # Intent filter (retrieval_type)
                if intent:
                    if is_enriched and bullet.retrieval_type == intent:
                        score += 0.3
                        match_reasons.append(f"intent:{intent}")
                    elif not is_enriched:
                        pass  # Include basic bullets with lower score
                    else:
                        continue  # Skip if intent doesn't match

                # Query type scoring boost (Phase 1A enhancement)
                if query_type:
                    if is_enriched and hasattr(bullet, 'task_types') and bullet.task_types:
                        if query_type in bullet.task_types:
                            score += 0.25
                            match_reasons.append(f"task_type_match:{query_type}")

                # Trigger pattern matching (similarity component)
                trigger_score = 0.0
                if query:
                    trigger_score = self._match_trigger_patterns(query, bullet)
                    if trigger_score > 0:
                        score += trigger_score
                        match_reasons.append(f"trigger_match:{trigger_score:.2f}")

                # Calculate effectiveness score (outcome component)
                # Use session-specific effectiveness ONLY if bullet's task_types includes session_type
                # This prevents penalizing bullets that don't match the session context
                bullet_matches_session = (
                    session_type
                    and is_enriched
                    and hasattr(bullet, 'task_types')
                    and bullet.task_types
                    and session_type in bullet.task_types
                )

                if bullet_matches_session and self._session_tracker:
                    # Session-aware effectiveness (bullet matches session context)
                    effectiveness = self._session_tracker.get_session_effectiveness(
                        session_type, bullet.id, default=0.5
                    )
                    match_reasons.append(f"session_eff:{effectiveness:.2f}")
                else:
                    # Global effectiveness (bullet doesn't match session or no session)
                    if is_enriched:
                        effectiveness = bullet.effectiveness_score
                    else:
                        total = bullet.helpful + bullet.harmful + bullet.neutral
                        effectiveness = bullet.helpful / total if total > 0 else 0.5

                # Phase 1B: Strong trigger matches override effectiveness filter
                # This prevents excluding highly relevant bullets that haven't proven
                # themselves yet (low effectiveness due to insufficient feedback)
                strong_trigger_match = trigger_score >= trigger_override_threshold
                if min_effectiveness is not None and effectiveness < min_effectiveness:
                    if not strong_trigger_match:
                        continue  # Skip bullets below effectiveness threshold
                    # else: Strong trigger match - keep bullet despite low effectiveness

                # Apply dynamic weighting between similarity and outcomes
                # Get maturity-based weights for this bullet
                similarity_weight, outcome_weight = self._get_dynamic_weights(bullet)

                # Current score is similarity-based (filters + trigger patterns)
                similarity_score = score
                outcome_score = effectiveness

                # Combine using dynamic weights
                weighted_score = (similarity_weight * similarity_score) + (outcome_weight * outcome_score)

                # Legacy rank_by_effectiveness parameter for backwards compatibility
                # When enabled, add effectiveness as additional boost (on top of weighted score)
                if rank_by_effectiveness:
                    weighted_score += effectiveness * 0.2
                    match_reasons.append(f"effectiveness_boost:{effectiveness:.2f}")

                # Update final score with weighted combination
                score = weighted_score
                match_reasons.append(f"weights:sim={similarity_weight:.1f},out={outcome_weight:.1f}")

                # Add temporal filter info to match reasons
                if created_after or created_before or updated_after:
                    filter_info = []
                    if created_after:
                        filter_info.append(f"created_after:{created_after.isoformat()}")
                    if created_before:
                        filter_info.append(f"created_before:{created_before.isoformat()}")
                    if updated_after:
                        filter_info.append(f"updated_after:{updated_after.isoformat()}")
                    match_reasons.append(f"temporal:{','.join(filter_info)}")

                # Only include bullets that matched at least one criterion
                # or if no filters were specified
                no_filters = not any([task_type, domain, complexity, intent, query, query_type, created_after, created_before, updated_after])
                if score > 0 or no_filters:
                    if no_filters and score == 0:
                        score = 0.5  # Default score for unfiltered results
                    results.append(ScoredBullet(bullet=bullet, score=score, match_reasons=match_reasons))

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)

        # Apply cross-encoder reranking if enabled
        # Resolve config defaults for rerank parameters
        retrieval_config = get_retrieval_config()
        should_rerank = rerank if rerank is not None else retrieval_config.enable_reranking
        candidates_count = rerank_candidates if rerank_candidates is not None else retrieval_config.first_stage_k
        
        if should_rerank and query and results:
            from .reranker import get_reranker
            
            # Get more candidates for reranking, then trim to limit
            candidates_to_rerank = results[:candidates_count]
            
            try:
                reranker = get_reranker(model_name=retrieval_config.cross_encoder_model)
                documents = [r.content for r in candidates_to_rerank]
                rerank_scores = reranker.predict(query, documents)
                
                # Normalize rerank scores to [0, 1] range since cross-encoders can output logits
                # Use sigmoid-like normalization for scores that may be negative
                min_score = min(rerank_scores) if rerank_scores else 0
                max_score = max(rerank_scores) if rerank_scores else 1
                score_range = max_score - min_score
                if score_range > 0:
                    normalized_rerank_scores = [(s - min_score) / score_range for s in rerank_scores]
                else:
                    normalized_rerank_scores = [0.5] * len(rerank_scores)
                
                # Update scores and match reasons with rerank scores
                for scored_bullet, rerank_score, norm_score in zip(candidates_to_rerank, rerank_scores, normalized_rerank_scores):
                    # Blend original score (40%) with normalized rerank score (60%)
                    # This preserves filter relevance while leveraging reranker precision
                    original_score = scored_bullet.score
                    blended_score = 0.4 * original_score + 0.6 * norm_score
                    scored_bullet.score = max(0.0, blended_score)  # Ensure non-negative
                    scored_bullet.match_reasons.append(f"rerank:{rerank_score:.3f}")
                
                # Re-sort after reranking
                results = candidates_to_rerank
                results.sort(key=lambda x: x.score, reverse=True)
            except ImportError:
                # sentence-transformers not installed, skip reranking
                pass

        # Apply limit
        if limit is not None:
            results = results[:limit]

        return results

    def _get_dynamic_weights(self, bullet: "Bullet") -> Tuple[float, float]:
        """Calculate dynamic weights based on bullet maturity.

        New bullets rely more on similarity (cold start exploration).
        Mature bullets rely more on outcomes (evidence-based exploitation).

        Args:
            bullet: The bullet to calculate weights for

        Returns:
            Tuple of (similarity_weight, outcome_weight) where:
            - New bullets (0 signals): (0.8, 0.2) - trust similarity
            - Early bullets (1-4 signals): (0.5, 0.5) - balanced
            - Mature bullets (5+ signals): (0.3, 0.7) - trust outcomes
        """
        # Handle UnifiedBullet (helpful_count, harmful_count) vs Bullet (helpful, harmful, neutral)
        if hasattr(bullet, 'helpful_count'):
            # UnifiedBullet
            total_signals = (
                getattr(bullet, 'helpful_count', 0) +
                getattr(bullet, 'harmful_count', 0) +
                getattr(bullet, 'reinforcement_count', 0)
            )
        else:
            # Standard Bullet
            total_signals = bullet.helpful + bullet.harmful + bullet.neutral

        if total_signals == 0:
            # Cold start: trust similarity more
            return (0.8, 0.2)
        elif total_signals < 5:
            # Early stage: balanced weighting
            return (0.5, 0.5)
        else:
            # Mature: trust outcomes more
            return (0.3, 0.7)

    # Common words that should not contribute to trigger matching
    # These words appear in many memories and don't indicate relevance
    _COMMON_STOP_WORDS = frozenset({
        # Articles and determiners
        'a', 'an', 'the', 'this', 'that', 'these', 'those',
        # Prepositions
        'to', 'for', 'of', 'in', 'on', 'at', 'by', 'with', 'from', 'into', 'about',
        # Conjunctions
        'and', 'or', 'but', 'if', 'then', 'when', 'while', 'as', 'so',
        # Pronouns
        'i', 'you', 'it', 'we', 'they', 'me', 'us', 'them', 'my', 'your', 'its',
        # Common verbs (that appear everywhere)
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'may',
        'use', 'using', 'used', 'add', 'adding', 'added', 'get', 'getting', 'got',
        'set', 'setting', 'make', 'making', 'made', 'take', 'taking', 'took',
        # Common adjectives/adverbs
        'all', 'any', 'some', 'no', 'not', 'yes', 'more', 'most', 'very', 'just',
        'only', 'also', 'always', 'never', 'before', 'after', 'now', 'new', 'first',
        # Question words (without semantic meaning)
        'what', 'how', 'why', 'where', 'which',
        # Other common filler words
        'please', 'want', 'need', 'like', 'ok', 'okay', 'sure', 'yes', 'no',
    })

    def _match_trigger_patterns(self, query: str, bullet: "Bullet") -> float:
        """Calculate trigger pattern match score for a bullet.

        Args:
            query: The search query
            bullet: The bullet to match against (Bullet, EnrichedBullet, or UnifiedBullet)

        Returns:
            Match score (0.0 to 0.5)
        """
        from .playbook import EnrichedBullet

        score = 0.0
        query_lower = query.lower()

        # Check for trigger_patterns attribute (EnrichedBullet or UnifiedBullet)
        trigger_patterns = getattr(bullet, 'trigger_patterns', None)
        if trigger_patterns:
            for pattern in trigger_patterns:
                if pattern.lower() in query_lower:
                    score += 0.15
                    if score >= 0.45:  # Cap at 0.45
                        break

            # Also check anti-patterns (reduce score if matched) - only for EnrichedBullet
            anti_patterns = getattr(bullet, 'anti_patterns', None)
            if anti_patterns:
                for anti in anti_patterns:
                    if anti.lower() in query_lower:
                        score -= 0.1

        # Fallback: keyword match against content (with stop word filtering)
        if score == 0:
            content_words = set(bullet.content.lower().split())
            query_words = set(query_lower.split())

            # Filter out stop words - they don't indicate relevance
            meaningful_query = query_words - self._COMMON_STOP_WORDS
            meaningful_content = content_words - self._COMMON_STOP_WORDS

            # Score based on meaningful word overlap
            meaningful_overlap = meaningful_query & meaningful_content
            if meaningful_overlap:
                # Higher score for meaningful matches
                score = min(0.12 * len(meaningful_overlap), 0.35)
            else:
                # Very low score for only stop word matches (they're noise)
                stop_word_overlap = query_words & content_words & self._COMMON_STOP_WORDS
                if stop_word_overlap:
                    score = min(0.02 * len(stop_word_overlap), 0.06)

        return max(0.0, score)

    def semantic_search(
        self,
        query: str,
        threshold: float = 0.0,
        limit: Optional[int] = None,
    ) -> List[ScoredBullet]:
        """Search bullets using semantic similarity.

        Note: This is a simplified implementation using keyword overlap.
        For production use, integrate with an embedding model.

        Args:
            query: Natural language query
            threshold: Minimum similarity threshold (0.0 to 1.0)
            limit: Maximum number of results

        Returns:
            List of ScoredBullet objects sorted by semantic similarity.
        """
        from .playbook import EnrichedBullet

        results: List[ScoredBullet] = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for bullet in self._bullets:
            # Get the text to compare against
            if isinstance(bullet, EnrichedBullet) and bullet.embedding_text:
                compare_text = bullet.embedding_text.lower()
            else:
                compare_text = bullet.content.lower()

            compare_words = set(compare_text.split())

            # Calculate Jaccard-like similarity
            if not query_words or not compare_words:
                similarity = 0.0
            else:
                intersection = len(query_words & compare_words)
                union = len(query_words | compare_words)
                similarity = intersection / union if union > 0 else 0.0

            # Boost score if embedding_text was used (more relevant)
            if isinstance(bullet, EnrichedBullet) and bullet.embedding_text:
                similarity *= 1.2  # 20% boost for enriched bullets

            if similarity >= threshold:
                results.append(
                    ScoredBullet(
                        bullet=bullet,
                        score=min(1.0, similarity),
                        match_reasons=[f"semantic_similarity:{similarity:.3f}"],
                    )
                )

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)

        if limit is not None:
            results = results[:limit]

        return results

    def get_by_intent(self, intent: IntentType) -> List[ScoredBullet]:
        """Convenience method to get all bullets of a specific intent type.

        Args:
            intent: The intent type (analytical/factual/procedural/general)

        Returns:
            List of matching bullets with scores.
        """
        return self.retrieve(intent=intent)

    def get_effective_bullets(
        self,
        min_effectiveness: float = 0.6,
        limit: Optional[int] = None,
    ) -> List[ScoredBullet]:
        """Get bullets with high effectiveness scores.

        Args:
            min_effectiveness: Minimum effectiveness threshold
            limit: Maximum number of results

        Returns:
            List of effective bullets sorted by effectiveness.
        """
        return self.retrieve(
            min_effectiveness=min_effectiveness,
            rank_by_effectiveness=True,
            limit=limit,
        )

    def retrieve_user_preferences(
        self,
        query: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ScoredBullet]:
        """Retrieve user preferences from unified memory.

        Convenience method that filters by USER_PREFS namespace.

        Args:
            query: Optional search query
            limit: Maximum number of results

        Returns:
            List of user preference bullets sorted by relevance.
        """
        try:
            from .unified_memory import UnifiedNamespace
        except ImportError:
            return []

        return self.retrieve(
            query=query,
            namespace=UnifiedNamespace.USER_PREFS,
            limit=limit,
        )

    def retrieve_task_strategies(
        self,
        query: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ScoredBullet]:
        """Retrieve task strategies from unified memory.

        Convenience method that filters by TASK_STRATEGIES namespace.

        Args:
            query: Optional search query
            limit: Maximum number of results

        Returns:
            List of task strategy bullets sorted by relevance.
        """
        try:
            from .unified_memory import UnifiedNamespace
        except ImportError:
            return []

        return self.retrieve(
            query=query,
            namespace=UnifiedNamespace.TASK_STRATEGIES,
            limit=limit,
        )

    def retrieve_project_context(
        self,
        query: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ScoredBullet]:
        """Retrieve project-specific context from unified memory.

        Convenience method that filters by PROJECT_SPECIFIC namespace.

        Args:
            query: Optional search query
            limit: Maximum number of results

        Returns:
            List of project-specific bullets sorted by relevance.
        """
        try:
            from .unified_memory import UnifiedNamespace
        except ImportError:
            return []

        return self.retrieve(
            query=query,
            namespace=UnifiedNamespace.PROJECT_SPECIFIC,
            limit=limit,
        )


class IntentClassifier:
    """Classifies queries into intent types for intelligent retrieval routing.

    The classifier uses pattern matching to categorize queries into:
    - analytical: Comparison, decision-making, trade-off analysis
    - factual: Lookups, definitions, syntax reference
    - procedural: How-to, step-by-step, action-oriented
    - general: Ambiguous or conversational queries

    Example:
        >>> from ace.retrieval import IntentClassifier
        >>>
        >>> classifier = IntentClassifier()
        >>> intent = classifier.classify("How do I deploy this?")
        >>> print(intent)  # "procedural"
        >>>
        >>> intent, confidence = classifier.classify_with_confidence("Compare A vs B")
        >>> print(f"{intent}: {confidence:.2f}")  # "analytical: 0.85"
    """

    # Patterns for analytical queries (comparison, decision-making)
    ANALYTICAL_PATTERNS: List[Tuple[re.Pattern, float]] = [
        (re.compile(r"\b(compare|comparison|versus|vs\.?|v\.)\b", re.I), 0.9),
        (re.compile(r"\b(trade-?offs?|pros?\s+and\s+cons?|advantages?|disadvantages?)\b", re.I), 0.85),
        (re.compile(r"\b(which|what).*(should|better|best|prefer|choose|pick)\b", re.I), 0.85),
        (re.compile(r"\b(is\s+it\s+better|would\s+it\s+be\s+better)\b", re.I), 0.8),
        (re.compile(r"\bwhy\s+(is|does|would|should|did)\b", re.I), 0.7),
        (re.compile(r"\b(evaluate|assess|analyze|analysis)\b", re.I), 0.75),
        (re.compile(r"\b(difference|differ|different)\s+(between|from)\b", re.I), 0.8),
        (re.compile(r"\bshould\s+i\b", re.I), 0.75),
        (re.compile(r"\bwhat\s+are\s+the\s+(trade-?offs?|pros|cons|advantages?|disadvantages?)\b", re.I), 0.9),
    ]

    # Patterns for factual queries (lookups, definitions)
    FACTUAL_PATTERNS: List[Tuple[re.Pattern, float]] = [
        (re.compile(r"\bwhat\s+(is|are|does|was|were)\s+(the|a|an)?\s*\w+\??$", re.I), 0.85),
        (re.compile(r"\b(define|definition|meaning)\b", re.I), 0.9),
        (re.compile(r"\b(syntax|parameter|argument|return|type)\b", re.I), 0.75),
        (re.compile(r"\b(list|show|display)\s+(all|the|available)\b", re.I), 0.8),
        (re.compile(r"\bwhat\s+(version|does|did)\b", re.I), 0.75),
        (re.compile(r"\b(configuration|config|options?|settings?)\b", re.I), 0.7),
        (re.compile(r"\b(stand\s+for|stands\s+for|acronym)\b", re.I), 0.85),
        (re.compile(r"\berror\s+(code|message|mean)\b", re.I), 0.8),
        (re.compile(r"\brequired\s+(parameter|field|value)\b", re.I), 0.75),
        (re.compile(r"\bwhat\s+are\s+the\s+required\b", re.I), 0.85),
        (re.compile(r"\bwhat\s+(is|are)\s+the\s+(syntax|parameter|argument|return)", re.I), 0.85),
    ]

    # Patterns for procedural queries (how-to, step-by-step)
    PROCEDURAL_PATTERNS: List[Tuple[re.Pattern, float]] = [
        (re.compile(r"\bhow\s+(do|can|to|would|should)\s+i?\b", re.I), 0.9),
        (re.compile(r"\b(step[s\-]*by[s\-]*step|steps?\s+to|procedure)\b", re.I), 0.95),
        (re.compile(r"\b(walk\s+me\s+through|guide\s+me|show\s+me\s+how)\b", re.I), 0.9),
        (re.compile(r"\b(implement|create|build|setup|set\s*up|configure)\b", re.I), 0.75),
        (re.compile(r"\b(deploy|install|migrate|upgrade)\b", re.I), 0.8),
        (re.compile(r"\b(fix|resolve|debug|troubleshoot)\b", re.I), 0.75),
        (re.compile(r"\b(run|execute|start|stop|restart)\b", re.I), 0.7),
        (re.compile(r"^(fix|implement|create|build|make|add|remove|delete|update)\b", re.I), 0.85),
    ]

    # Patterns for general/ambiguous queries
    GENERAL_PATTERNS: List[Tuple[re.Pattern, float]] = [
        (re.compile(r"^(hi|hello|hey|thanks|thank\s+you|ok|okay|sure|yes|no)[\s!?.]*$", re.I), 0.95),
        (re.compile(r"^.{1,5}$", re.I), 0.7),  # Very short queries
    ]

    def __init__(self) -> None:
        """Initialize the IntentClassifier."""
        pass

    def classify(self, query: str) -> IntentType:
        """Classify a query into an intent type.

        Args:
            query: The natural language query to classify

        Returns:
            The classified intent type (analytical/factual/procedural/general)
        """
        intent, _ = self.classify_with_confidence(query)
        return intent

    def classify_with_confidence(self, query: str) -> Tuple[IntentType, float]:
        """Classify a query with confidence score.

        Args:
            query: The natural language query to classify

        Returns:
            Tuple of (intent_type, confidence_score)
        """
        query = query.strip()

        # Score each intent category
        scores: Dict[IntentType, float] = {
            "analytical": 0.0,
            "factual": 0.0,
            "procedural": 0.0,
            "general": 0.0,
        }

        # Check general patterns first (high priority for short/greeting queries)
        for pattern, weight in self.GENERAL_PATTERNS:
            if pattern.search(query):
                scores["general"] = max(scores["general"], weight)

        # Check analytical patterns
        for pattern, weight in self.ANALYTICAL_PATTERNS:
            if pattern.search(query):
                scores["analytical"] = max(scores["analytical"], weight)

        # Check factual patterns
        for pattern, weight in self.FACTUAL_PATTERNS:
            if pattern.search(query):
                scores["factual"] = max(scores["factual"], weight)

        # Check procedural patterns
        for pattern, weight in self.PROCEDURAL_PATTERNS:
            if pattern.search(query):
                scores["procedural"] = max(scores["procedural"], weight)

        # Find the best scoring intent
        best_intent: IntentType = "general"
        best_score = 0.0

        for intent, score in scores.items():
            if score > best_score:
                best_score = score
                best_intent = intent  # type: ignore

        # If no strong match, default to general with low confidence
        if best_score == 0.0:
            return "general", 0.3

        return best_intent, best_score

    def get_all_scores(self, query: str) -> Dict[IntentType, float]:
        """Get scores for all intent types.

        Args:
            query: The query to analyze

        Returns:
            Dictionary mapping each intent type to its score
        """
        query = query.strip()

        scores: Dict[IntentType, float] = {
            "analytical": 0.0,
            "factual": 0.0,
            "procedural": 0.0,
            "general": 0.0,
        }

        for pattern, weight in self.GENERAL_PATTERNS:
            if pattern.search(query):
                scores["general"] = max(scores["general"], weight)

        for pattern, weight in self.ANALYTICAL_PATTERNS:
            if pattern.search(query):
                scores["analytical"] = max(scores["analytical"], weight)

        for pattern, weight in self.FACTUAL_PATTERNS:
            if pattern.search(query):
                scores["factual"] = max(scores["factual"], weight)

        for pattern, weight in self.PROCEDURAL_PATTERNS:
            if pattern.search(query):
                scores["procedural"] = max(scores["procedural"], weight)

        return scores

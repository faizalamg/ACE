"""LLM-based bullet enrichment for production-grade semantic scaffolding.

This module provides LLM-powered enrichment of bullets with semantic
metadata for intelligent retrieval. Unlike heuristic-based enrichment,
this uses the CURATOR_ENRICHMENT_PROMPT to extract accurate metadata.

For production use, always prefer LLMBulletEnricher over the heuristic
enrich_bullet() function in playbook.py.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .llm import LLMClient
from .playbook import Bullet, EnrichedBullet, enrich_bullet as heuristic_enrich
from .prompts_v2_1 import CURATOR_ENRICHMENT_PROMPT

if TYPE_CHECKING:
    pass


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """Parse JSON from LLM response, handling markdown code blocks."""
    text = text.strip()

    if text.startswith("```json"):
        text = text[7:].strip()
    elif text.startswith("```"):
        text = text[3:].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object from LLM.")
    return data


class LLMBulletEnricher:
    """Production-grade LLM-based bullet enrichment.

    Uses the CURATOR_ENRICHMENT_PROMPT to extract accurate semantic
    scaffolding metadata from bullets. This provides much better
    enrichment than heuristic-based approaches.

    Args:
        llm: The LLM client to use for enrichment
        max_retries: Maximum retries on JSON parse errors (default: 3)
        fallback_to_heuristic: Whether to fallback to heuristic enrichment on failure (default: True)
        prompt_template: Custom prompt template (uses CURATOR_ENRICHMENT_PROMPT by default)

    Example:
        >>> from ace import LiteLLMClient
        >>> from ace.enrichment import LLMBulletEnricher
        >>> from ace.playbook import Bullet
        >>>
        >>> llm = LiteLLMClient(model="gpt-4")
        >>> enricher = LLMBulletEnricher(llm)
        >>>
        >>> bullet = Bullet(id="debug-001", section="debugging", content="Check stack traces")
        >>> enriched = enricher.enrich(bullet, context="Fixed exception in parser")
        >>>
        >>> print(enriched.task_types)
        ['debugging', 'code_review']
        >>> print(enriched.trigger_patterns)
        ['exception', 'traceback', 'stack trace']
    """

    def __init__(
        self,
        llm: LLMClient,
        max_retries: int = 3,
        fallback_to_heuristic: bool = True,
        prompt_template: str = CURATOR_ENRICHMENT_PROMPT,
    ) -> None:
        """Initialize LLM bullet enricher.

        Args:
            llm: The LLM client to use
            max_retries: Maximum retries on parse errors
            fallback_to_heuristic: Whether to fallback on failure
            prompt_template: Custom enrichment prompt
        """
        self.llm = llm
        self.max_retries = max_retries
        self.fallback_to_heuristic = fallback_to_heuristic
        self.prompt_template = prompt_template

    def enrich(
        self,
        bullet: Bullet,
        context: str,
        **kwargs: Any,
    ) -> EnrichedBullet:
        """Enrich a bullet using LLM-based analysis.

        Args:
            bullet: Bullet to enrich
            context: Usage context (question, answer, feedback)
            **kwargs: Additional arguments passed to LLM

        Returns:
            EnrichedBullet with LLM-extracted semantic scaffolding

        Raises:
            RuntimeError: If enrichment fails and fallback is disabled
        """
        # Build prompt
        prompt = self.prompt_template.format(
            content=bullet.content,
            context=context,
        )

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                response = self.llm.complete(prompt, **kwargs)
                data = _safe_json_loads(response.text)

                # Extract enrichment fields
                return self._build_enriched_bullet(bullet, data)

            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                if attempt + 1 >= self.max_retries:
                    break
                # Could add retry prompt here if needed

        # Fallback or raise
        if self.fallback_to_heuristic:
            return heuristic_enrich(bullet, context)

        raise RuntimeError(
            f"LLM enrichment failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    def enrich_batch(
        self,
        bullets: List[Bullet],
        context: str,
        **kwargs: Any,
    ) -> List[EnrichedBullet]:
        """Enrich multiple bullets.

        Args:
            bullets: List of bullets to enrich
            context: Shared context for all bullets
            **kwargs: Additional arguments passed to LLM

        Returns:
            List of EnrichedBullets
        """
        return [self.enrich(bullet, context, **kwargs) for bullet in bullets]

    def _build_enriched_bullet(
        self,
        bullet: Bullet,
        enrichment_data: Dict[str, Any],
    ) -> EnrichedBullet:
        """Build EnrichedBullet from original bullet and LLM enrichment data.

        Args:
            bullet: Original bullet
            enrichment_data: Parsed LLM output

        Returns:
            EnrichedBullet with merged data
        """
        return EnrichedBullet(
            # Preserve original bullet fields
            id=bullet.id,
            section=bullet.section,
            content=bullet.content,
            helpful=bullet.helpful,
            harmful=bullet.harmful,
            neutral=bullet.neutral,
            created_at=bullet.created_at,
            updated_at=bullet.updated_at,
            # LLM-extracted dimensional metadata
            task_types=enrichment_data.get("task_types", ["general"]),
            domains=enrichment_data.get("domains", ["general"]),
            complexity_level=enrichment_data.get("complexity_level", "medium"),
            # LLM-extracted structural metadata
            preconditions=enrichment_data.get("preconditions", []),
            trigger_patterns=enrichment_data.get("trigger_patterns", []),
            anti_patterns=enrichment_data.get("anti_patterns", []),
            # LLM-extracted retrieval hints
            retrieval_type=enrichment_data.get("retrieval_type", "semantic"),
            embedding_text=enrichment_data.get("embedding_text", ""),
        )


def enrich_bullet_llm(
    bullet: Bullet,
    context: str,
    llm: LLMClient,
    **kwargs: Any,
) -> EnrichedBullet:
    """Convenience function for one-off LLM enrichment.

    Args:
        bullet: Bullet to enrich
        context: Usage context
        llm: LLM client to use
        **kwargs: Additional arguments

    Returns:
        EnrichedBullet with LLM-extracted metadata

    Example:
        >>> from ace import LiteLLMClient
        >>> from ace.enrichment import enrich_bullet_llm
        >>> from ace.playbook import Bullet
        >>>
        >>> llm = LiteLLMClient(model="gpt-4")
        >>> bullet = Bullet(id="test-001", section="test", content="Check logs")
        >>> enriched = enrich_bullet_llm(bullet, "Error in production", llm=llm)
    """
    enricher = LLMBulletEnricher(llm, **kwargs)
    return enricher.enrich(bullet, context)


class EnrichmentPipeline:
    """Pipeline for enriching bullets during Curator operations.

    Integrates with the adaptation loop to automatically enrich
    new bullets added via DeltaOperations.

    Example:
        >>> from ace.enrichment import EnrichmentPipeline
        >>>
        >>> pipeline = EnrichmentPipeline(llm)
        >>>
        >>> # In adaptation loop
        >>> delta = curator.curate(...)
        >>> enriched_delta = pipeline.enrich_delta(delta, context)
        >>> playbook.apply_delta(enriched_delta)
    """

    def __init__(
        self,
        llm: LLMClient,
        enabled: bool = True,
        **enricher_kwargs: Any,
    ) -> None:
        """Initialize enrichment pipeline.

        Args:
            llm: LLM client for enrichment
            enabled: Whether enrichment is enabled
            **enricher_kwargs: Arguments passed to LLMBulletEnricher
        """
        self.enricher = LLMBulletEnricher(llm, **enricher_kwargs)
        self.enabled = enabled

    def enrich_delta(
        self,
        delta: "DeltaBatch",
        context: str,
    ) -> "DeltaBatch":
        """Enrich ADD operations in a DeltaBatch.

        Args:
            delta: Delta batch from Curator
            context: Usage context for enrichment

        Returns:
            DeltaBatch with enriched ADD operations
        """
        if not self.enabled:
            return delta

        from .delta import DeltaBatch, DeltaOperation

        enriched_ops: List[DeltaOperation] = []

        for op in delta.operations:
            if op.type == "ADD" and not op.enrichment:
                # Create temporary bullet for enrichment
                temp_bullet = Bullet(
                    id=f"temp-{op.section}",
                    section=op.section,
                    content=op.content,
                )

                try:
                    enriched = self.enricher.enrich(temp_bullet, context)

                    # Create enrichment dict from EnrichedBullet
                    enrichment = {
                        "task_types": enriched.task_types,
                        "domains": enriched.domains,
                        "complexity_level": enriched.complexity_level,
                        "preconditions": enriched.preconditions,
                        "trigger_patterns": enriched.trigger_patterns,
                        "anti_patterns": enriched.anti_patterns,
                        "retrieval_type": enriched.retrieval_type,
                        "embedding_text": enriched.embedding_text,
                    }

                    enriched_ops.append(
                        DeltaOperation(
                            type=op.type,
                            section=op.section,
                            content=op.content,
                            bullet_id=op.bullet_id,
                            tag=op.tag,
                            enrichment=enrichment,
                        )
                    )
                except Exception:
                    # On failure, keep original operation
                    enriched_ops.append(op)
            else:
                # Non-ADD or already enriched, keep as-is
                enriched_ops.append(op)

        return DeltaBatch(
            reasoning=delta.reasoning,
            operations=enriched_ops,
        )

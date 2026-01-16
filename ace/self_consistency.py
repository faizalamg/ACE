"""Self-consistency sampling for improved generation accuracy.

This module implements self-consistency decoding, which generates multiple
responses for the same prompt and selects the most consistent answer via
majority voting. This technique improves accuracy for tasks where reasoning
paths can vary but the final answer should converge.

Reference: Wang et al., "Self-Consistency Improves Chain of Thought Reasoning"
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .llm import LLMClient
from .playbook import Playbook
from .prompts import GENERATOR_PROMPT
from .roles import GeneratorOutput, extract_cited_bullet_ids

if TYPE_CHECKING:
    pass


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """Parse JSON from LLM response, handling markdown code blocks."""
    text = text.strip()

    # Handle opening fence (with or without language identifier)
    if text.startswith("```json"):
        text = text[7:].strip()
    elif text.startswith("```"):
        text = text[3:].strip()

    # Handle closing fence (if present)
    if text.endswith("```"):
        text = text[:-3].strip()

    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object from LLM.")
    return data


def _normalize_answer(answer: str) -> str:
    """Normalize answer for comparison (lowercase, strip whitespace)."""
    return answer.strip().lower()


@dataclass
class SampledResponse:
    """A single sampled response with its parsed data."""

    reasoning: str
    final_answer: str
    bullet_ids: List[str]
    raw: Dict[str, Any]


class SelfConsistencyGenerator:
    """Generator with self-consistency sampling for improved accuracy.

    Self-consistency generates multiple responses using higher temperature
    sampling and selects the most common answer via majority voting.
    This improves accuracy for tasks with diverse reasoning paths.

    Args:
        llm: The LLM client to use for generation
        num_samples: Number of samples to generate (default: 3)
        temperature: Sampling temperature for diversity (default: 0.7)
        normalize_answers: Whether to normalize answers before voting (default: False)
        min_valid_samples: Minimum valid samples required (default: 1)
        prompt_template: Custom prompt template (uses GENERATOR_PROMPT by default)

    Example:
        >>> from ace import LiteLLMClient, Playbook
        >>> from ace.self_consistency import SelfConsistencyGenerator
        >>>
        >>> client = LiteLLMClient(model="gpt-4")
        >>> generator = SelfConsistencyGenerator(client, num_samples=5)
        >>>
        >>> result = generator.generate(
        ...     question="What is 15 * 23?",
        ...     context=None,
        ...     playbook=Playbook()
        ... )
        >>> print(result.final_answer)
        345
        >>> print(result.raw["consistency_confidence"])
        0.8  # 4/5 samples agreed
    """

    def __init__(
        self,
        llm: LLMClient,
        num_samples: int = 3,
        temperature: float = 0.7,
        normalize_answers: bool = False,
        min_valid_samples: int = 1,
        prompt_template: str = GENERATOR_PROMPT,
    ) -> None:
        """Initialize self-consistency generator.

        Args:
            llm: The LLM client to use for generation
            num_samples: Number of samples to generate
            temperature: Sampling temperature for diversity
            normalize_answers: Whether to normalize answers before voting
            min_valid_samples: Minimum valid samples required
            prompt_template: Custom prompt template
        """
        self.llm = llm
        self.num_samples = num_samples
        self.temperature = temperature
        self.normalize_answers = normalize_answers
        self.min_valid_samples = min_valid_samples
        self.prompt_template = prompt_template

    def generate(
        self,
        *,
        question: str,
        context: Optional[str],
        playbook: Playbook,
        reflection: Optional[str] = None,
        **kwargs: Any,
    ) -> GeneratorOutput:
        """Generate answer using self-consistency sampling.

        Args:
            question: The question to answer
            context: Additional context or requirements
            playbook: The current playbook of strategies
            reflection: Optional reflection from previous attempts
            **kwargs: Additional arguments passed to the LLM

        Returns:
            GeneratorOutput with the most consistent answer and voting metadata

        Raises:
            RuntimeError: If unable to get minimum valid samples
        """
        # Build the prompt
        prompt = self.prompt_template.format(
            playbook=playbook.as_prompt() or "(empty playbook)",
            reflection=reflection or "(none)",
            question=question,
            context=context or "(none)",
        )

        # Filter out non-LLM kwargs
        llm_kwargs = {k: v for k, v in kwargs.items() if k != "sample"}

        # Generate multiple samples
        samples: List[SampledResponse] = []
        for _ in range(self.num_samples):
            try:
                response = self.llm.complete(
                    prompt, temperature=self.temperature, **llm_kwargs
                )
                data = _safe_json_loads(response.text)

                reasoning = str(data.get("reasoning", ""))
                final_answer = str(data.get("final_answer", ""))
                bullet_ids = extract_cited_bullet_ids(reasoning)

                samples.append(
                    SampledResponse(
                        reasoning=reasoning,
                        final_answer=final_answer,
                        bullet_ids=bullet_ids,
                        raw=data,
                    )
                )
            except (json.JSONDecodeError, ValueError):
                # Skip invalid responses
                continue

        # Check minimum valid samples
        if len(samples) < self.min_valid_samples:
            raise RuntimeError(
                f"Self-consistency generation failed: only {len(samples)} valid samples "
                f"out of {self.num_samples}, minimum required: {self.min_valid_samples}"
            )

        # Perform majority voting
        return self._majority_vote(samples)

    def _majority_vote(self, samples: List[SampledResponse]) -> GeneratorOutput:
        """Select the most common answer via majority voting.

        Args:
            samples: List of valid sampled responses

        Returns:
            GeneratorOutput with the winning answer and voting metadata
        """
        # Count votes for each answer
        if self.normalize_answers:
            vote_key = lambda s: _normalize_answer(s.final_answer)
        else:
            vote_key = lambda s: s.final_answer

        # Build vote distribution
        vote_counts: Counter[str] = Counter()
        answer_to_samples: Dict[str, List[SampledResponse]] = {}

        for sample in samples:
            key = vote_key(sample)
            vote_counts[key] += 1
            if key not in answer_to_samples:
                answer_to_samples[key] = []
            answer_to_samples[key].append(sample)

        # Find winning answer (most common, first occurrence for ties)
        winning_key = max(vote_counts.keys(), key=lambda k: (vote_counts[k], -list(vote_counts.keys()).index(k)))
        winning_count = vote_counts[winning_key]

        # Get the first sample with the winning answer
        winning_sample = answer_to_samples[winning_key][0]

        # Calculate confidence (proportion of votes for winner)
        confidence = winning_count / len(samples)

        # Build vote distribution for output (using original answers)
        vote_distribution: Dict[str, int] = {}
        for sample in samples:
            answer = sample.final_answer
            if self.normalize_answers:
                # Map normalized key back to original answer from winner
                if _normalize_answer(answer) == winning_key:
                    answer = winning_sample.final_answer
            vote_distribution[answer] = vote_distribution.get(answer, 0) + 1

        # If normalizing, consolidate the distribution
        if self.normalize_answers:
            normalized_dist: Dict[str, int] = {}
            for sample in samples:
                norm_key = _normalize_answer(sample.final_answer)
                # Use the winning sample's original answer if it matches
                if norm_key == winning_key:
                    display_answer = winning_sample.final_answer
                else:
                    # Use first occurrence of this normalized answer
                    display_answer = answer_to_samples[norm_key][0].final_answer
                normalized_dist[display_answer] = normalized_dist.get(display_answer, 0) + 1
            vote_distribution = normalized_dist

        return GeneratorOutput(
            reasoning=winning_sample.reasoning,
            final_answer=winning_sample.final_answer,
            bullet_ids=winning_sample.bullet_ids,
            raw={
                **winning_sample.raw,
                "self_consistency": True,
                "num_samples": self.num_samples,
                "valid_samples": len(samples),
                "vote_distribution": vote_distribution,
                "consistency_confidence": confidence,
                "winning_votes": winning_count,
            },
        )

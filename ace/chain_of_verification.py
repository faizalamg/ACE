"""Chain-of-Verification (CoVe) for improved reflection accuracy.

This module implements Chain-of-Verification, which generates verification
questions about the initial reflection, answers them independently, and
uses the answers to refine the final output. This technique improves
accuracy through self-verification.

Reference: Dhuliawala et al., "Chain-of-Verification Reduces Hallucination"
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .llm import LLMClient
from .playbook import Playbook
from .prompts import REFLECTOR_PROMPT
from .roles import GeneratorOutput, ReflectorOutput, BulletTag

if TYPE_CHECKING:
    pass


# Prompt for generating verification questions
VERIFICATION_QUESTIONS_PROMPT = """Based on this reflection analysis, generate verification questions to check accuracy.

Original Question: {question}
Generator Reasoning: {reasoning}
Generator Answer: {prediction}
Ground Truth: {ground_truth}
Feedback: {feedback}

Initial Reflection:
{initial_reflection}

Generate {num_questions} verification questions that would help verify:
1. The accuracy of the error identification
2. The correctness of the root cause analysis
3. Whether the key insight is valid

Return JSON with format:
{{
    "verification_questions": ["question1", "question2", ...]
}}"""

# Prompt for answering verification questions
VERIFICATION_ANSWERS_PROMPT = """Answer these verification questions about the reflection analysis.

Context:
- Question: {question}
- Generator Answer: {prediction}
- Ground Truth: {ground_truth}
- Feedback: {feedback}

Verification Questions:
{questions_list}

For each question, provide an answer and your confidence (0.0 to 1.0).

Return JSON with format:
{{
    "verified_answers": [
        {{"question": "...", "answer": "...", "confidence": 0.X}},
        ...
    ]
}}"""

# Prompt for refined reflection
REFINED_REFLECTION_PROMPT = """Refine your reflection based on verification answers.

Original Question: {question}
Generator Reasoning: {reasoning}
Generator Answer: {prediction}
Ground Truth: {ground_truth}
Feedback: {feedback}
Playbook Excerpt: {playbook_excerpt}

Initial Reflection:
{initial_reflection}

Verification Results:
{verification_results}

Based on the verification, provide a refined reflection.
If verification revealed issues, update your analysis.
If verification confirmed your analysis, strengthen the key insight.

Return JSON with:
{{
    "reasoning": "your refined reasoning",
    "error_identification": "specific error if any",
    "root_cause_analysis": "refined root cause",
    "correct_approach": "what should have been done",
    "key_insight": "actionable lesson learned",
    "bullet_tags": [
        {{"id": "bullet-id", "tag": "helpful|harmful|neutral"}}
    ]
}}"""


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


def _format_optional(value: Optional[str]) -> str:
    """Format optional value for prompt."""
    return value or "(none)"


def _make_playbook_excerpt(playbook: Playbook, bullet_ids: List[str]) -> str:
    """Create excerpt of cited bullets."""
    lines: List[str] = []
    seen = set()
    for bullet_id in bullet_ids:
        if bullet_id in seen:
            continue
        bullet = playbook.get_bullet(bullet_id)
        if bullet:
            seen.add(bullet_id)
            lines.append(f"[{bullet.id}] {bullet.content}")
    return "\n".join(lines) if lines else "(No strategies cited)"


@dataclass
class VerifiedAnswer:
    """A verified answer with confidence."""

    question: str
    answer: str
    confidence: float


class CoVeReflector:
    """Reflector with Chain-of-Verification for improved accuracy.

    CoVe generates verification questions about the initial reflection,
    answers them independently, and uses the answers to refine the
    final output. This self-verification process catches errors and
    improves reflection quality.

    Args:
        llm: The LLM client to use for reflection
        num_questions: Number of verification questions to generate (default: 3)
        skip_verification: Whether to allow skipping verification (default: False)
        confidence_threshold: Threshold for skipping verification (default: 0.95)
        fallback_on_error: Whether to fallback to initial reflection on error (default: True)
        prompt_template: Custom prompt template for initial reflection

    Example:
        >>> from ace import LiteLLMClient, Playbook
        >>> from ace.chain_of_verification import CoVeReflector
        >>>
        >>> client = LiteLLMClient(model="gpt-4")
        >>> reflector = CoVeReflector(client, num_questions=3)
        >>>
        >>> result = reflector.reflect(
        ...     question="What is 6*7?",
        ...     generator_output=gen_output,
        ...     playbook=playbook,
        ...     ground_truth="42",
        ...     feedback="Incorrect"
        ... )
        >>> print(result.key_insight)
        Always verify arithmetic calculations
    """

    def __init__(
        self,
        llm: LLMClient,
        num_questions: int = 3,
        skip_verification: bool = False,
        confidence_threshold: float = 0.95,
        fallback_on_error: bool = True,
        prompt_template: str = REFLECTOR_PROMPT,
    ) -> None:
        """Initialize CoVe Reflector.

        Args:
            llm: The LLM client to use
            num_questions: Number of verification questions
            skip_verification: Whether to allow skipping verification
            confidence_threshold: Threshold for skipping verification
            fallback_on_error: Whether to fallback on error
            prompt_template: Custom prompt template
        """
        self.llm = llm
        self.num_questions = num_questions
        self.skip_verification = skip_verification
        self.confidence_threshold = confidence_threshold
        self.fallback_on_error = fallback_on_error
        self.prompt_template = prompt_template

    def reflect(
        self,
        *,
        question: str,
        generator_output: GeneratorOutput,
        playbook: Playbook,
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
        **kwargs: Any,
    ) -> ReflectorOutput:
        """Reflect on generator output using Chain-of-Verification.

        Args:
            question: The original question
            generator_output: The generator's output to reflect on
            playbook: Current playbook for context
            ground_truth: Expected answer if known
            feedback: Environment feedback
            **kwargs: Additional arguments passed to LLM

        Returns:
            ReflectorOutput with verified reflection and metadata
        """
        playbook_excerpt = _make_playbook_excerpt(playbook, generator_output.bullet_ids)

        # Filter non-LLM kwargs
        llm_kwargs = {k: v for k, v in kwargs.items() if k != "sample"}

        # Step 1: Generate initial reflection
        initial_result = self._initial_reflect(
            question=question,
            generator_output=generator_output,
            playbook_excerpt=playbook_excerpt,
            ground_truth=ground_truth,
            feedback=feedback,
            **llm_kwargs,
        )

        # Check if we should skip verification
        if self.skip_verification:
            initial_confidence = initial_result.raw.get("initial_confidence", 0.0)
            if initial_confidence >= self.confidence_threshold:
                return self._finalize_output(
                    initial_result,
                    verification_skipped=True,
                    skip_reason="high_initial_confidence",
                )

        # Step 2: Generate verification questions
        try:
            questions = self._generate_verification_questions(
                question=question,
                generator_output=generator_output,
                initial_reflection=initial_result,
                ground_truth=ground_truth,
                feedback=feedback,
                **llm_kwargs,
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            if self.fallback_on_error:
                return self._finalize_output(
                    initial_result,
                    verification_error=str(e),
                )
            raise

        # Step 3: Answer verification questions
        try:
            verified_answers = self._answer_verification_questions(
                question=question,
                generator_output=generator_output,
                questions=questions,
                ground_truth=ground_truth,
                feedback=feedback,
                **llm_kwargs,
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            if self.fallback_on_error:
                return self._finalize_output(
                    initial_result,
                    verification_questions=questions,
                    verification_error=str(e),
                )
            raise

        # Step 4: Generate refined reflection
        try:
            refined_result = self._refine_reflection(
                question=question,
                generator_output=generator_output,
                initial_reflection=initial_result,
                verified_answers=verified_answers,
                playbook_excerpt=playbook_excerpt,
                ground_truth=ground_truth,
                feedback=feedback,
                **llm_kwargs,
            )
        except (json.JSONDecodeError, ValueError) as e:
            if self.fallback_on_error:
                return self._finalize_output(
                    initial_result,
                    verification_questions=questions,
                    verified_answers=verified_answers,
                    verification_error=str(e),
                )
            raise

        # Add verification metadata to refined result
        return self._finalize_output(
            refined_result,
            verification_questions=questions,
            verified_answers=verified_answers,
        )

    def _initial_reflect(
        self,
        *,
        question: str,
        generator_output: GeneratorOutput,
        playbook_excerpt: str,
        ground_truth: Optional[str],
        feedback: Optional[str],
        **kwargs: Any,
    ) -> ReflectorOutput:
        """Generate initial reflection."""
        prompt = self.prompt_template.format(
            question=question,
            reasoning=generator_output.reasoning,
            prediction=generator_output.final_answer,
            ground_truth=_format_optional(ground_truth),
            feedback=_format_optional(feedback),
            playbook_excerpt=playbook_excerpt,
        )

        response = self.llm.complete(prompt, **kwargs)
        data = _safe_json_loads(response.text)

        bullet_tags: List[BulletTag] = []
        tags_payload = data.get("bullet_tags", [])
        if isinstance(tags_payload, list):
            for item in tags_payload:
                if isinstance(item, dict) and "id" in item and "tag" in item:
                    bullet_tags.append(
                        BulletTag(id=str(item["id"]), tag=str(item["tag"]).lower())
                    )

        return ReflectorOutput(
            reasoning=str(data.get("reasoning", "")),
            error_identification=str(data.get("error_identification", "")),
            root_cause_analysis=str(data.get("root_cause_analysis", "")),
            correct_approach=str(data.get("correct_approach", "")),
            key_insight=str(data.get("key_insight", "")),
            bullet_tags=bullet_tags,
            raw=data,
        )

    def _generate_verification_questions(
        self,
        *,
        question: str,
        generator_output: GeneratorOutput,
        initial_reflection: ReflectorOutput,
        ground_truth: Optional[str],
        feedback: Optional[str],
        **kwargs: Any,
    ) -> List[str]:
        """Generate verification questions."""
        prompt = VERIFICATION_QUESTIONS_PROMPT.format(
            question=question,
            reasoning=generator_output.reasoning,
            prediction=generator_output.final_answer,
            ground_truth=_format_optional(ground_truth),
            feedback=_format_optional(feedback),
            initial_reflection=json.dumps(initial_reflection.raw, indent=2),
            num_questions=self.num_questions,
        )

        response = self.llm.complete(prompt, **kwargs)
        data = _safe_json_loads(response.text)

        questions = data.get("verification_questions", [])
        if not isinstance(questions, list):
            raise ValueError("verification_questions must be a list")

        return questions[: self.num_questions]

    def _answer_verification_questions(
        self,
        *,
        question: str,
        generator_output: GeneratorOutput,
        questions: List[str],
        ground_truth: Optional[str],
        feedback: Optional[str],
        **kwargs: Any,
    ) -> List[VerifiedAnswer]:
        """Answer verification questions."""
        questions_list = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

        prompt = VERIFICATION_ANSWERS_PROMPT.format(
            question=question,
            prediction=generator_output.final_answer,
            ground_truth=_format_optional(ground_truth),
            feedback=_format_optional(feedback),
            questions_list=questions_list,
        )

        response = self.llm.complete(prompt, **kwargs)
        data = _safe_json_loads(response.text)

        verified_answers: List[VerifiedAnswer] = []
        answers_payload = data.get("verified_answers", [])

        if isinstance(answers_payload, list):
            for item in answers_payload:
                if isinstance(item, dict):
                    verified_answers.append(
                        VerifiedAnswer(
                            question=str(item.get("question", "")),
                            answer=str(item.get("answer", "")),
                            confidence=float(item.get("confidence", 0.0)),
                        )
                    )

        return verified_answers

    def _refine_reflection(
        self,
        *,
        question: str,
        generator_output: GeneratorOutput,
        initial_reflection: ReflectorOutput,
        verified_answers: List[VerifiedAnswer],
        playbook_excerpt: str,
        ground_truth: Optional[str],
        feedback: Optional[str],
        **kwargs: Any,
    ) -> ReflectorOutput:
        """Generate refined reflection based on verification."""
        verification_results = "\n".join(
            f"Q: {va.question}\nA: {va.answer} (confidence: {va.confidence:.2f})"
            for va in verified_answers
        )

        prompt = REFINED_REFLECTION_PROMPT.format(
            question=question,
            reasoning=generator_output.reasoning,
            prediction=generator_output.final_answer,
            ground_truth=_format_optional(ground_truth),
            feedback=_format_optional(feedback),
            playbook_excerpt=playbook_excerpt,
            initial_reflection=json.dumps(initial_reflection.raw, indent=2),
            verification_results=verification_results,
        )

        response = self.llm.complete(prompt, **kwargs)
        data = _safe_json_loads(response.text)

        bullet_tags: List[BulletTag] = []
        tags_payload = data.get("bullet_tags", [])
        if isinstance(tags_payload, list):
            for item in tags_payload:
                if isinstance(item, dict) and "id" in item and "tag" in item:
                    bullet_tags.append(
                        BulletTag(id=str(item["id"]), tag=str(item["tag"]).lower())
                    )

        return ReflectorOutput(
            reasoning=str(data.get("reasoning", "")),
            error_identification=str(data.get("error_identification", "")),
            root_cause_analysis=str(data.get("root_cause_analysis", "")),
            correct_approach=str(data.get("correct_approach", "")),
            key_insight=str(data.get("key_insight", "")),
            bullet_tags=bullet_tags,
            raw=data,
        )

    def _finalize_output(
        self,
        result: ReflectorOutput,
        verification_questions: Optional[List[str]] = None,
        verified_answers: Optional[List[VerifiedAnswer]] = None,
        verification_skipped: bool = False,
        skip_reason: Optional[str] = None,
        verification_error: Optional[str] = None,
    ) -> ReflectorOutput:
        """Add verification metadata to result."""
        raw = dict(result.raw)

        raw["chain_of_verification"] = not verification_skipped
        raw["verification_questions"] = verification_questions or []

        if verified_answers:
            raw["verified_answers"] = [
                {
                    "question": va.question,
                    "answer": va.answer,
                    "confidence": va.confidence,
                }
                for va in verified_answers
            ]

            # Calculate average confidence
            if verified_answers:
                avg_confidence = sum(va.confidence for va in verified_answers) / len(
                    verified_answers
                )
                raw["verification_confidence"] = avg_confidence
        else:
            raw["verified_answers"] = []

        if verification_skipped:
            raw["verification_skipped"] = True
            raw["skip_reason"] = skip_reason

        if verification_error:
            raw["verification_error"] = verification_error

        return ReflectorOutput(
            reasoning=result.reasoning,
            error_identification=result.error_identification,
            root_cause_analysis=result.root_cause_analysis,
            correct_approach=result.correct_approach,
            key_insight=result.key_insight,
            bullet_tags=result.bullet_tags,
            raw=raw,
        )

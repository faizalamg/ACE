"""
Enhanced Query Generator for RAG Testing

Generates 10-15 diverse queries per memory entry covering:
- Direct keyword extraction
- Semantic variations (same meaning, different words)
- Question formats (what, how, why, when, where)
- Technical vs casual phrasing
- Implicit/contextual references
- Edge cases (very short, very long)
- Partial concept matches

Uses LLM to generate natural, realistic queries that users would actually ask.
"""

import json
import time
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from enum import Enum

import httpx


class QueryCategory(str, Enum):
    """Categories for query variation types."""
    DIRECT = "direct"  # Direct keywords from memory
    SEMANTIC = "semantic"  # Same meaning, different words
    KEYWORD = "keyword"  # Key terms only
    PARAPHRASE = "paraphrase"  # Natural rephrasing
    QUESTION_WHAT = "question_what"  # What questions
    QUESTION_HOW = "question_how"  # How questions
    QUESTION_WHY = "question_why"  # Why questions
    IMPLICIT = "implicit"  # Contextual/indirect reference
    PARTIAL = "partial"  # Subset of concepts
    TECHNICAL = "technical"  # Technical terminology
    CASUAL = "casual"  # Informal phrasing
    EDGE_SHORT = "edge_short"  # 2-3 word queries
    EDGE_LONG = "edge_long"  # Detailed, long queries
    SCENARIO = "scenario"  # Scenario-based queries
    NEGATIVE = "negative"  # What NOT to do phrasing


@dataclass
class GeneratedQuery:
    """A generated test query with metadata."""
    query: str
    category: str
    difficulty: str  # easy, medium, hard
    expected_rank: int = 1
    min_similarity: float = 0.5


@dataclass
class EnhancedMemoryTestCase:
    """Memory with expanded test queries."""
    memory_id: int
    content: str
    category: str
    feedback_type: str
    severity: int
    context: str
    original_queries: List[str]
    generated_queries: List[GeneratedQuery] = field(default_factory=list)
    generation_timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "category": self.category,
            "feedback_type": self.feedback_type,
            "severity": self.severity,
            "context": self.context,
            "original_queries": self.original_queries,
            "generated_queries": [
                {
                    "query": q.query,
                    "category": q.category,
                    "difficulty": q.difficulty,
                    "expected_rank": q.expected_rank,
                    "min_similarity": q.min_similarity
                }
                for q in self.generated_queries
            ],
            "generation_timestamp": self.generation_timestamp,
            "total_queries": len(self.generated_queries)
        }


class QueryGenerator:
    """
    LLM-powered query generator for RAG testing.

    Generates diverse, realistic queries that users would actually ask.
    """

    QUERY_GENERATION_PROMPT = '''You are a RAG test query generator. Generate diverse, realistic search queries that would retrieve this memory.

MEMORY TO RETRIEVE:
{memory_content}

MEMORY CATEGORY: {category}
CONTEXT FILE: {context}

Generate exactly 15 diverse queries. For each query, provide:
1. The query text (natural language, as a user would type it)
2. The category type (from the list below)
3. Difficulty level (easy/medium/hard)

REQUIRED QUERY CATEGORIES (generate at least one of each):
- direct: Extract key phrases directly from the memory
- semantic: Same meaning but completely different words
- question_what: "What is/are..." question format
- question_how: "How to/do I..." question format
- question_why: "Why should..." question format
- implicit: Describe a problem/situation without explicit keywords
- technical: Use precise technical terminology
- casual: Informal, conversational phrasing
- partial: Focus on just one aspect of the memory
- edge_short: 2-3 word query only
- edge_long: Detailed, context-rich query (15+ words)
- scenario: "When I'm doing X, how do I Y" format
- keyword: Just the key terms, no sentence structure
- paraphrase: Completely reword the concept
- negative: Frame as "avoid doing X" or "prevent Y"

OUTPUT FORMAT (JSON array, no markdown):
[
  {{"query": "your query here", "category": "category_name", "difficulty": "easy|medium|hard"}},
  ...
]

Generate realistic queries a software developer would actually type when searching for this knowledge.'''

    def __init__(
        self,
        llm_url: str = "http://localhost:1234",
        model: str = "local-model",
        timeout: float = 120.0
    ):
        self.llm_url = llm_url
        self.model = model
        self.client = httpx.Client(timeout=timeout)

    def generate_queries(self, memory: Dict[str, Any]) -> List[GeneratedQuery]:
        """Generate diverse queries for a single memory."""

        prompt = self.QUERY_GENERATION_PROMPT.format(
            memory_content=memory["content"],
            category=memory.get("category", "unknown"),
            context=memory.get("context", "unknown")
        )

        try:
            response = self.client.post(
                f"{self.llm_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a precise query generator. Output only valid JSON arrays. No markdown, no explanations."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2000
                }
            )

            if response.status_code != 200:
                print(f"  LLM request failed: {response.status_code}")
                return self._generate_fallback_queries(memory)

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Parse JSON from response
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            queries_data = json.loads(content.strip())

            queries = []
            for q in queries_data:
                queries.append(GeneratedQuery(
                    query=q["query"],
                    category=q.get("category", "semantic"),
                    difficulty=q.get("difficulty", "medium"),
                    expected_rank=1 if q.get("difficulty") == "easy" else 3 if q.get("difficulty") == "medium" else 5,
                    min_similarity=0.6 if q.get("difficulty") == "easy" else 0.4 if q.get("difficulty") == "medium" else 0.3
                ))

            return queries

        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            return self._generate_fallback_queries(memory)
        except Exception as e:
            print(f"  Generation error: {e}")
            return self._generate_fallback_queries(memory)

    def _generate_fallback_queries(self, memory: Dict[str, Any]) -> List[GeneratedQuery]:
        """Generate rule-based queries as fallback when LLM fails."""
        content = memory["content"]
        category = memory.get("category", "general")
        words = content.lower().split()

        queries = []

        # Direct keyword queries
        key_words = [w for w in words if len(w) > 4 and w.isalnum()][:5]
        if key_words:
            queries.append(GeneratedQuery(
                query=" ".join(key_words[:3]),
                category="keyword",
                difficulty="easy"
            ))

        # Question formats
        queries.extend([
            GeneratedQuery(
                query=f"how to {content.lower()[:50]}",
                category="question_how",
                difficulty="medium"
            ),
            GeneratedQuery(
                query=f"what is the best way to {content.lower()[:40]}",
                category="question_what",
                difficulty="medium"
            ),
            GeneratedQuery(
                query=f"why should I {content.lower()[:40]}",
                category="question_why",
                difficulty="hard"
            )
        ])

        # Short query
        if key_words:
            queries.append(GeneratedQuery(
                query=" ".join(key_words[:2]),
                category="edge_short",
                difficulty="hard"
            ))

        # Category-based
        queries.append(GeneratedQuery(
            query=f"{category.lower()} best practices",
            category="partial",
            difficulty="medium"
        ))

        # Scenario
        queries.append(GeneratedQuery(
            query=f"when implementing {category.lower()}, what should I remember",
            category="scenario",
            difficulty="hard"
        ))

        # Casual
        queries.append(GeneratedQuery(
            query=f"hey how do I {content.lower()[:30]}",
            category="casual",
            difficulty="hard"
        ))

        # Technical
        queries.append(GeneratedQuery(
            query=f"{category} pattern implementation strategy",
            category="technical",
            difficulty="medium"
        ))

        # Long detailed
        queries.append(GeneratedQuery(
            query=f"I'm working on a project and need advice on {content.lower()[:60]} in my codebase",
            category="edge_long",
            difficulty="hard"
        ))

        # Implicit
        queries.append(GeneratedQuery(
            query=f"my code is getting messy and hard to maintain",
            category="implicit",
            difficulty="hard"
        ))

        # Direct
        queries.append(GeneratedQuery(
            query=content[:60],
            category="direct",
            difficulty="easy"
        ))

        return queries[:15]  # Ensure max 15

    def close(self):
        self.client.close()


def process_memories(
    input_file: Path,
    output_file: Path,
    llm_url: str = "http://localhost:1234",
    batch_size: int = 10,
    delay_between_batches: float = 1.0
) -> Dict[str, Any]:
    """
    Process all memories and generate expanded queries.

    Args:
        input_file: Path to selected_memories.json
        output_file: Path to save expanded test suite
        llm_url: URL of LLM service
        batch_size: Number of memories to process before saving
        delay_between_batches: Seconds to wait between batches

    Returns:
        Statistics about the generation process
    """
    print(f"\n{'='*80}")
    print("ENHANCED QUERY GENERATOR")
    print(f"{'='*80}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"LLM URL: {llm_url}")
    print(f"{'='*80}\n")

    # Load memories
    with open(input_file) as f:
        memories = json.load(f)

    print(f"Loaded {len(memories)} memories")

    # Initialize generator
    generator = QueryGenerator(llm_url=llm_url)

    # Process each memory
    enhanced_memories = []
    stats = {
        "total_memories": len(memories),
        "processed": 0,
        "failed": 0,
        "total_queries_generated": 0,
        "queries_per_category": {},
        "start_time": datetime.now().isoformat()
    }

    for i, memory in enumerate(memories):
        print(f"\n[{i+1}/{len(memories)}] Processing memory {memory['memory_id']}")
        print(f"  Category: {memory.get('category', 'unknown')}")
        print(f"  Content: {memory['content'][:60]}...")

        # Generate queries
        try:
            queries = generator.generate_queries(memory)

            enhanced = EnhancedMemoryTestCase(
                memory_id=memory["memory_id"],
                content=memory["content"],
                category=memory.get("category", "unknown"),
                feedback_type=memory.get("feedback_type", "unknown"),
                severity=memory.get("severity", 5),
                context=memory.get("context", "unknown"),
                original_queries=memory.get("sample_queries", []),
                generated_queries=queries,
                generation_timestamp=datetime.now().isoformat()
            )

            enhanced_memories.append(enhanced)
            stats["processed"] += 1
            stats["total_queries_generated"] += len(queries)

            # Track category distribution
            for q in queries:
                cat = q.category
                stats["queries_per_category"][cat] = stats["queries_per_category"].get(cat, 0) + 1

            print(f"  Generated {len(queries)} queries")

        except Exception as e:
            print(f"  FAILED: {e}")
            stats["failed"] += 1

            # Add with original queries only
            enhanced = EnhancedMemoryTestCase(
                memory_id=memory["memory_id"],
                content=memory["content"],
                category=memory.get("category", "unknown"),
                feedback_type=memory.get("feedback_type", "unknown"),
                severity=memory.get("severity", 5),
                context=memory.get("context", "unknown"),
                original_queries=memory.get("sample_queries", []),
                generated_queries=[],
                generation_timestamp=datetime.now().isoformat()
            )
            enhanced_memories.append(enhanced)

        # Save checkpoint every batch_size memories
        if (i + 1) % batch_size == 0:
            checkpoint_file = output_file.parent / f"checkpoint_{i+1}.json"
            with open(checkpoint_file, 'w') as f:
                json.dump([m.to_dict() for m in enhanced_memories], f, indent=2)
            print(f"\n  Checkpoint saved: {checkpoint_file}")
            time.sleep(delay_between_batches)

    # Final save
    stats["end_time"] = datetime.now().isoformat()

    output_data = {
        "metadata": {
            "generation_stats": stats,
            "format_version": "2.0",
            "description": "Enhanced test suite with 10-15 diverse queries per memory"
        },
        "test_cases": [m.to_dict() for m in enhanced_memories]
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*80}")
    print("GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Processed: {stats['processed']}/{stats['total_memories']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total queries: {stats['total_queries_generated']}")
    print(f"Avg queries/memory: {stats['total_queries_generated']/stats['processed']:.1f}")
    print(f"\nQueries by category:")
    for cat, count in sorted(stats["queries_per_category"].items()):
        print(f"  {cat}: {count}")
    print(f"\nSaved to: {output_file}")

    generator.close()

    return stats


def main():
    """Run the query generator."""
    input_file = Path(__file__).parent / "test_suite" / "selected_memories.json"
    output_file = Path(__file__).parent / "test_suite" / "enhanced_test_suite.json"

    # Check for existing checkpoint
    checkpoints = list(Path(__file__).parent.glob("test_suite/checkpoint_*.json"))
    if checkpoints:
        latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[1]))
        print(f"Found checkpoint: {latest}")
        response = input("Resume from checkpoint? (y/n): ")
        if response.lower() == 'y':
            # TODO: Implement checkpoint resume
            pass

    stats = process_memories(
        input_file=input_file,
        output_file=output_file,
        llm_url="http://localhost:1234",
        batch_size=10,
        delay_between_batches=0.5
    )

    return stats


if __name__ == "__main__":
    main()

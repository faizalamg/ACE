"""
ACE + LiteLLM integration for quick-start learning agents.

This module provides ACELiteLLM, a high-level wrapper bundling ACE learning
with LiteLLM for easy prototyping and simple tasks.

When to Use ACELiteLLM:
- Quick start: Want to try ACE with minimal setup
- Simple tasks: Q&A, classification, reasoning
- Prototyping: Experimenting with ACE learning
- No framework needed: Direct LLM usage with learning

When NOT to Use ACELiteLLM:
- Browser automation → Use ACEAgent (browser-use)
- LangChain chains/agents → Use ACELangChain
- Custom agentic system → Use integration pattern (see docs/INTEGRATION_GUIDE.md)

Example:
    from ace.integrations import ACELiteLLM

    # Create LiteLLM-enhanced agent
    agent = ACELiteLLM(model="gpt-4o-mini")

    # Ask questions (uses current knowledge)
    answer = agent.ask("What is 2+2?")

    # Learn from examples
    from ace import Sample, SimpleEnvironment
    samples = [
        Sample(question="What is 2+2?", ground_truth="4"),
        Sample(question="Capital of France?", ground_truth="Paris"),
    ]
    agent.learn(samples, SimpleEnvironment())

    # Save learned knowledge
    agent.save_playbook("my_agent.json")

    # Load in next session
    agent = ACELiteLLM(model="gpt-4o-mini", playbook_path="my_agent.json")
"""

from typing import List, Optional

from ..playbook import Playbook
from ..roles import Generator, Reflector, Curator
from ..adaptation import OfflineAdapter, Sample, TaskEnvironment
from ..prompts_v2_1 import PromptManager


class ACELiteLLM:
    """
    LiteLLM integration with ACE learning.

    Bundles Generator, Reflector, Curator, and Playbook into a simple interface
    powered by LiteLLM (supports 100+ LLM providers).

    Perfect for:
    - Quick start with ACE
    - Q&A, classification, and reasoning tasks
    - Prototyping and experimentation
    - Learning without external frameworks

    For other use cases:
    - ACEAgent (browser-use): Browser automation with learning
    - ACELangChain: LangChain chains/agents with learning
    - Integration pattern: Custom agent systems (see docs)

    Attributes:
        playbook: Learned strategies (Playbook instance)
        is_learning: Whether learning is enabled
        model: LiteLLM model name

    Example:
        # Basic usage
        agent = ACELiteLLM(model="gpt-4o-mini")
        answer = agent.ask("What is the capital of France?")
        print(answer)  # "Paris"

        # Learning from feedback
        from ace import Sample, SimpleEnvironment
        samples = [
            Sample(question="What is 2+2?", ground_truth="4"),
            Sample(question="What is 3+3?", ground_truth="6"),
        ]
        agent.learn(samples, SimpleEnvironment(), epochs=1)

        # Save and load
        agent.save_playbook("learned.json")
        agent2 = ACELiteLLM(model="gpt-4o-mini", playbook_path="learned.json")
    """

    def __init__(
        self,
        model: str = "openai/glm-4.6",
        max_tokens: int = 512,
        temperature: float = 0.0,
        playbook_path: Optional[str] = None,
        is_learning: bool = True,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        """
        Initialize ACELiteLLM agent.

        IMPORTANT: LLM Configuration
        =============================
        By default, this uses Z.ai GLM-4.6 which requires:
        - Environment variable: ZAI_API_KEY
        - API base: https://api.z.ai/api/coding/paas/v4 (auto-configured)

        To use OpenAI models instead, set OPENAI_API_KEY and pass model="gpt-4o-mini".
        To use other providers, see LiteLLM documentation for model prefixes.

        Args:
            model: LiteLLM model name (default: openai/glm-4.6 via Z.ai)
                   Common options:
                   - "openai/glm-4.6" (Z.ai GLM, requires ZAI_API_KEY)
                   - "gpt-4o-mini" (OpenAI, requires OPENAI_API_KEY)
                   - "claude-3-haiku-20240307" (Anthropic, requires ANTHROPIC_API_KEY)
                   - "gemini/gemini-pro" (Google, requires GOOGLE_API_KEY)
            max_tokens: Max tokens for responses (default: 512)
            temperature: Sampling temperature (default: 0.0)
            playbook_path: Path to existing playbook (optional)
            is_learning: Enable/disable learning (default: True)
            api_key: API key (optional, uses env vars if not provided)
            api_base: API base URL (optional, auto-configured for Z.ai)

        Raises:
            ImportError: If LiteLLM is not installed

        Example:
            # Z.ai GLM (default - requires ZAI_API_KEY)
            agent = ACELiteLLM()

            # OpenAI (requires OPENAI_API_KEY)
            agent = ACELiteLLM(model="gpt-4o-mini")

            # Anthropic (requires ANTHROPIC_API_KEY)
            agent = ACELiteLLM(model="claude-3-haiku-20240307")

            # Google (requires GOOGLE_API_KEY)
            agent = ACELiteLLM(model="gemini/gemini-pro")

            # Explicit Z.ai configuration
            agent = ACELiteLLM(
                model="openai/glm-4.6",
                api_key=os.getenv("ZAI_API_KEY"),
                api_base="https://api.z.ai/api/coding/paas/v4"
            )

            # With existing playbook
            agent = ACELiteLLM(
                model="openai/glm-4.6",
                playbook_path="expert.json"
            )
        """
        # Import LiteLLM (required for this integration)
        try:
            from ..llm_providers import LiteLLMClient
        except ImportError:
            raise ImportError(
                "ACELiteLLM requires LiteLLM. Install with:\n"
                "pip install ace-framework  # (LiteLLM included by default)\n"
                "or: pip install litellm"
            )

        self.model = model
        self.is_learning = is_learning

        # Load or create playbook
        if playbook_path:
            self.playbook = Playbook.load_from_file(playbook_path)
        else:
            self.playbook = Playbook()

        # Auto-configure Z.ai GLM if using default model
        import os
        if model == "openai/glm-4.6" or model.startswith("glm-"):
            api_key = api_key or os.getenv("ZAI_API_KEY")
            api_base = api_base or "https://api.z.ai/api/coding/paas/v4"
            if not api_key:
                raise ValueError(
                    "Z.ai GLM requires ZAI_API_KEY environment variable.\n"
                    "Set it in your .env file or pass api_key parameter.\n"
                    "Alternatively, use a different model like 'gpt-4o-mini' with OPENAI_API_KEY."
                )

        # Create LLM client
        self.llm = LiteLLMClient(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
            api_base=api_base,
        )

        # Create ACE components with v2.1 prompts
        prompt_mgr = PromptManager()
        self.generator = Generator(
            self.llm, prompt_template=prompt_mgr.get_generator_prompt()
        )
        self.reflector = Reflector(
            self.llm, prompt_template=prompt_mgr.get_reflector_prompt()
        )
        self.curator = Curator(
            self.llm, prompt_template=prompt_mgr.get_curator_prompt()
        )

    def ask(self, question: str, context: str = "") -> str:
        """
        Ask a question and get an answer (uses current playbook).

        This uses the ACE Generator with the current playbook's learned strategies.

        Args:
            question: Question to answer
            context: Additional context (optional)

        Returns:
            Answer string

        Example:
            agent = ACELiteLLM()
            answer = agent.ask("What is the capital of Japan?")
            print(answer)  # "Tokyo"

            # With context
            answer = agent.ask(
                "What is GDP?",
                context="Economics question"
            )
        """
        result = self.generator.generate(
            question=question, context=context, playbook=self.playbook
        )
        return result.final_answer

    def learn(
        self,
        samples: List[Sample],
        environment: TaskEnvironment,
        epochs: int = 1,
        checkpoint_interval: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Learn from examples (offline learning).

        Uses OfflineAdapter to learn from a batch of samples.

        Args:
            samples: List of Sample objects to learn from
            environment: TaskEnvironment for evaluating results
            epochs: Number of training epochs (default: 1)
            checkpoint_interval: Save playbook every N samples (optional)
            checkpoint_dir: Directory for checkpoints (optional)

        Returns:
            List of AdapterStepResult from training

        Example:
            from ace import Sample, SimpleEnvironment

            samples = [
                Sample(question="What is 2+2?", ground_truth="4"),
                Sample(question="Capital of France?", ground_truth="Paris"),
            ]

            agent = ACELiteLLM()
            results = agent.learn(samples, SimpleEnvironment(), epochs=1)

            print(f"Learned {len(agent.playbook.bullets())} strategies")
        """
        if not self.is_learning:
            raise ValueError("Learning is disabled. Set is_learning=True first.")

        # Create offline adapter
        adapter = OfflineAdapter(
            playbook=self.playbook,
            generator=self.generator,
            reflector=self.reflector,
            curator=self.curator,
        )

        # Run learning
        results = adapter.run(
            samples=samples,
            environment=environment,
            epochs=epochs,
            checkpoint_interval=checkpoint_interval,
            checkpoint_dir=checkpoint_dir,
        )

        return results

    def save_playbook(self, path: str):
        """
        Save learned playbook to file.

        Args:
            path: File path to save to (creates parent dirs if needed)

        Example:
            agent.save_playbook("my_agent.json")
        """
        self.playbook.save_to_file(path)

    def load_playbook(self, path: str):
        """
        Load playbook from file (replaces current playbook).

        Args:
            path: File path to load from

        Example:
            agent.load_playbook("expert.json")
        """
        self.playbook = Playbook.load_from_file(path)

    def enable_learning(self):
        """Enable learning (allows learn() to update playbook)."""
        self.is_learning = True

    def disable_learning(self):
        """Disable learning (prevents learn() from updating playbook)."""
        self.is_learning = False

    def get_strategies(self) -> str:
        """
        Get current learned strategies as formatted text.

        Returns:
            Formatted string with learned strategies (empty if none)

        Example:
            strategies = agent.get_strategies()
            print(strategies)
        """
        if not self.playbook or not self.playbook.bullets():
            return ""

        lines = ["Learned Strategies:"]
        for i, bullet in enumerate(self.playbook.bullets(), 1):
            score = f"+{bullet.helpful}/-{bullet.harmful}"
            lines.append(f"{i}. [{bullet.id}] {bullet.content}")
            lines.append(f"   Score: {score}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        bullets_count = len(self.playbook.bullets()) if self.playbook else 0
        return (
            f"ACELiteLLM(model='{self.model}', "
            f"strategies={bullets_count}, "
            f"learning={'enabled' if self.is_learning else 'disabled'})"
        )


__all__ = ["ACELiteLLM"]

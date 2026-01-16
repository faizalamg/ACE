#!/usr/bin/env python3
"""
Z.ai GLM example with ACE Framework.

This example demonstrates how to use Z.ai GLM models with the ACE Framework
through LiteLLM's custom OpenAI-compatible endpoint support.
"""

import os
from dotenv import load_dotenv

from ace import (
    LiteLLMClient,
    Generator,
    Reflector,
    Curator,
    OfflineAdapter,
    Sample,
    TaskEnvironment,
    EnvironmentResult,
    Playbook,
)

# Load environment variables
load_dotenv()


class ZaiGLMEnvironment(TaskEnvironment):
    """Simple environment for testing with Z.ai GLM."""

    def evaluate(self, sample, generator_output):
        correct = sample.ground_truth.lower() in generator_output.final_answer.lower()
        return EnvironmentResult(
            feedback="Correct!" if correct else "Incorrect",
            ground_truth=sample.ground_truth,
        )


def create_zai_client(model_name="glm-4"):
    """Create a LiteLLM client configured for Z.ai GLM."""

    # Get Z.ai API key from environment
    api_key = os.getenv("ZAI_API_KEY")
    if not api_key:
        raise ValueError("Please set ZAI_API_KEY in your .env file")

    # Configure LiteLLM for Z.ai GLM using zhipuai provider (recommended)
    client = LiteLLMClient(
        model=f"zhipuai/{model_name}",  # Use zhipuai provider format
        api_key=api_key,
        temperature=0.1,
        max_tokens=2048,
    )

    return client


def main():
    """Main example using Z.ai GLM with ACE Framework."""

    # Check for Z.ai API key
    if not os.getenv("ZAI_API_KEY"):
        print("Please set ZAI_API_KEY in your .env file")
        return

    # 1. Create Z.ai GLM client
    print("🚀 Creating Z.ai GLM client...")
    llm = create_zai_client("glm-4")

    # 2. Create ACE components
    print("🧠 Setting up ACE components...")
    adapter = OfflineAdapter(
        playbook=Playbook(),
        generator=Generator(llm),
        reflector=Reflector(llm),
        curator=Curator(llm),
    )

    # 3. Create training samples
    print("📚 Preparing training samples...")
    samples = [
        Sample(question="What is a decorator in Python?", ground_truth="A decorator is a function that takes another function as an argument and returns a new function without modifying the original function."),
        Sample(question="How can you optimize database query performance?", ground_truth="You can optimize through adding indexes, optimizing query statements, using caching, database partitioning, and similar techniques."),
        Sample(question="What is a RESTful API?", ground_truth="A RESTful API is a web service interface design style based on the REST architectural style."),
    ]

    # 4. Run adaptation with Z.ai GLM
    print("🎯 Running ACE adaptation with Z.ai GLM...")
    environment = ZaiGLMEnvironment()

    try:
        results = adapter.run(samples, environment, epochs=1)

        # 5. Check results
        print(f"\n✅ Training completed!")
        print(f"📊 Trained on {len(results)} samples")
        print(f"🧠 Playbook now has {len(adapter.playbook.bullets())} strategies")

        # Show a few learned strategies
        print(f"\n🔍 Learned strategies from Z.ai GLM:")
        for i, bullet in enumerate(adapter.playbook.bullets()[:2], 1):
            print(f"\n{i}. {bullet.content}")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("\n💡 Make sure your Z.ai API key is valid and the model name is correct.")
        return

    # 6. Test direct completion
    print(f"\n🧪 Testing direct completion with Z.ai GLM...")
    try:
        response = llm.complete("What is machine learning?")
        print(f"Z.ai GLM Response: {response.text[:200]}...")
    except Exception as e:
        print(f"❌ Direct completion error: {e}")


if __name__ == "__main__":
    main()
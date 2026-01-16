#!/usr/bin/env python3
"""
Custom Z.ai GLM client with automatic API key detection.
"""

import os
from dotenv import load_dotenv

from ace.llm_providers.litellm_client import LiteLLMClient, LiteLLMConfig

load_dotenv()


class ZaiGLMClient(LiteLLMClient):
    """Custom LiteLLM client for Z.ai GLM models."""

    def __init__(self, model="glm-4", **kwargs):
        """Initialize Z.ai GLM client with automatic configuration."""

        api_key = kwargs.get("api_key") or os.getenv("ZAI_API_KEY")
        api_base = kwargs.get("api_base", "https://api.z.ai")

        if not api_key:
            raise ValueError("ZAI_API_KEY environment variable must be set")

        # Use OpenAI-compatible format for Z.ai GLM
        config = LiteLLMConfig(
            model=f"openai/{model}",
            api_key=api_key,
            api_base=api_base,
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=kwargs.get("max_tokens", 2048),
            **kwargs
        )

        super().__init__(config=config)

    def _setup_api_keys(self) -> None:
        """Override to use Z.ai API key."""
        if not self.config.api_key:
            self.config.api_key = os.getenv("ZAI_API_KEY")

        if not self.config.api_key:
            raise ValueError("ZAI_API_KEY environment variable must be set")


# Usage example
def test_zai_client():
    """Test the custom Z.ai GLM client."""

    try:
        # Create client
        client = ZaiGLMClient(model="glm-4")

        # Test completion
        response = client.complete("Hello, please introduce yourself.")
        print(f"Z.ai GLM Response: {response.text}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_zai_client()
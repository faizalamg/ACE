#!/usr/bin/env python3
"""
Quick test to verify Z.ai GLM setup with ACE Framework.
"""

import os
from dotenv import load_dotenv
from ace import LiteLLMClient

load_dotenv()


def test_zai_setup():
    """Test Z.ai GLM configuration."""

    print("🔍 Testing Z.ai GLM Setup")
    print("=" * 40)

    # Check environment variable
    api_key = os.getenv("ZAI_API_KEY")
    if api_key:
        print(f"✅ ZAI_API_KEY found: {api_key[:8]}...{api_key[-4:]}")
    else:
        print("❌ ZAI_API_KEY not found in environment")
        return False

    # Test LiteLLM client creation
    try:
        print("\n🚀 Creating LiteLLM client for Z.ai GLM...")
        client = LiteLLMClient(
            model="zhipuai/glm-4",  # Use zhipuai provider
            api_key=api_key,
            temperature=0.1,
            max_tokens=100
        )
        print("✅ LiteLLM client created successfully")
        print(f"   Model: {client.config.model}")
        print(f"   API Base: {client.config.api_base}")
        return True

    except Exception as e:
        print(f"❌ Failed to create LiteLLM client: {e}")
        return False


def test_simple_completion():
    """Test a simple completion with Z.ai GLM."""

    api_key = os.getenv("ZAI_API_KEY")
    if not api_key:
        print("❌ Cannot test completion: ZAI_API_KEY not set")
        return

    print("\n🧪 Testing simple completion...")

    try:
        client = LiteLLMClient(
            model="zhipuai/glm-4",  # Use zhipuai provider
            api_key=api_key,
            temperature=0.1,
            max_tokens=50
        )

        response = client.complete("你好，请简单回答：什么是人工智能？")
        print(f"✅ Completion successful!")
        print(f"📝 Response: {response.text}")

    except Exception as e:
        print(f"❌ Completion failed: {e}")
        print("\n💡 Possible issues:")
        print("   - API key is invalid")
        print("   - API endpoint is incorrect")
        print("   - Model name is wrong")
        print("   - Network connectivity issues")


if __name__ == "__main__":
    success = test_zai_setup()

    if success:
        test_simple_completion()
    else:
        print("\n❌ Please fix the setup issues before testing completions")
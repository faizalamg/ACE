#!/usr/bin/env python3
"""
Final corrected test for Z.ai GLM with proper model names.
"""

import os
import litellm
from dotenv import load_dotenv
from ace import LiteLLMClient

# Enable LiteLLM debug to see actual requests
litellm.set_verbose = True
load_dotenv()


def test_zai_openai_compatible():
    """Test Z.ai GLM using OpenAI-compatible endpoint with correct model names."""

    print("🔍 Testing Z.ai GLM with OpenAI-Compatible Endpoint")
    print("=" * 60)

    api_key = os.getenv("ZAI_API_KEY")
    if not api_key:
        print("❌ ZAI_API_KEY not found")
        return False

    print(f"✅ ZAI_API_KEY found: {api_key[:8]}...{api_key[-4:]}")

    # Common Z.ai GLM model names (based on API documentation)
    model_names = [
        "glm-4",           # GLM-4
        "glm-4-plus",      # GLM-4 Plus
        "glm-4-air",       # GLM-4 Air
        "glm-4-flash",     # GLM-4 Flash
        "glm-3-turbo",     # GLM-3 Turbo
        "glm-4v",          # GLM-4 Vision
        "glm-128k",        # GLM-4 128K context
        "glm-4-0205",      # GLM-4 specific version
    ]

    working_models = []

    for model in model_names:
        print(f"\n📋 Testing model: {model}")
        try:
            client = LiteLLMClient(
                model="openai/" + model,  # Use OpenAI-compatible format
                api_key=api_key,
                api_base="https://api.z.ai",
                temperature=0.1,
                max_tokens=50,
                timeout=30
            )

            response = client.complete("Hello, say 'test successful' in Chinese.")
            print(f"✅ {model}: {response.text[:50]}...")
            working_models.append(model)

        except Exception as e:
            error_msg = str(e)
            if "模型不存在" in error_msg or "model not found" in error_msg.lower():
                print(f"❌ {model}: Model not found")
            elif "401" in error_msg or "unauthorized" in error_msg.lower():
                print(f"❌ {model}: Authentication failed")
            elif "timeout" in error_msg.lower():
                print(f"❌ {model}: Request timeout")
            else:
                print(f"❌ {model}: {error_msg}")

    return working_models


def test_alternative_endpoints():
    """Test alternative Z.ai endpoints."""

    print(f"\n🔄 Testing Alternative Endpoints")
    print("=" * 40)

    api_key = os.getenv("ZAI_API_KEY")
    if not api_key:
        return []

    endpoints = [
        "https://api.z.ai",
        "https://api.bigmodel.cn/v4/",
        "https://api.zhipuai.ai/v4/",
    ]

    working_configs = []

    for endpoint in endpoints:
        print(f"\n🌐 Testing endpoint: {endpoint}")
        try:
            client = LiteLLMClient(
                model="openai/glm-4",
                api_key=api_key,
                api_base=endpoint,
                temperature=0.1,
                max_tokens=30,
                timeout=15
            )

            response = client.complete("Hi")
            print(f"✅ {endpoint}: Working")
            working_configs.append({
                "endpoint": endpoint,
                "model": "glm-4"
            })

        except Exception as e:
            print(f"❌ {endpoint}: {str(e)[:100]}...")

    return working_configs


def test_direct_litellm():
    """Test using LiteLLM directly to get more debug info."""

    print(f"\n🔬 Direct LiteLLM Test")
    print("=" * 30)

    api_key = os.getenv("ZAI_API_KEY")
    if not api_key:
        print("❌ No API key")
        return

    try:
        # Test with litellm directly
        import litellm

        response = litellm.completion(
            model="openai/glm-4",
            messages=[{"role": "user", "content": "Hello"}],
            api_key=api_key,
            api_base="https://api.z.ai",
            temperature=0.1,
            max_tokens=30,
            timeout=15
        )

        print("✅ Direct LiteLLM call successful")
        print(f"Response: {response.choices[0].message.content}")

    except Exception as e:
        print(f"❌ Direct LiteLLM failed: {e}")


def main():
    """Main test function."""

    print("Z.ai GLM Configuration Test")
    print("=" * 40)

    # Test 1: OpenAI-compatible with different model names
    working_models = test_zai_openai_compatible()

    # Test 2: Alternative endpoints
    working_endpoints = test_alternative_endpoints()

    # Test 3: Direct LiteLLM call
    test_direct_litellm()

    # Summary
    print(f"\nFINAL SUMMARY")
    print("=" * 20)

    if working_models:
        print(f"✅ Working models:")
        for model in working_models:
            print(f"   - openai/{model}")

    if working_endpoints:
        print(f"✅ Working endpoints:")
        for config in working_endpoints:
            print(f"   - {config['endpoint']} (model: {config['model']})")

    if not working_models and not working_endpoints:
        print(f"❌ No working configurations found")
        print(f"\n💡 Suggestions:")
        print(f"   1. Check if your ZAI_API_KEY is valid")
        print(f"   2. Verify model names from Z.ai documentation")
        print(f"   3. Check if you have API access enabled")
        print(f"   4. Try a different endpoint URL")

    return working_models or working_endpoints


if __name__ == "__main__":
    success = main()
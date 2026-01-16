#!/usr/bin/env python3
"""
Simple test for Z.ai GLM configuration.
"""

import os
import litellm
from dotenv import load_dotenv
from ace import LiteLLMClient

# Enable LiteLLM debug to see actual requests
litellm.set_verbose = True
load_dotenv()


def test_zai_models():
    """Test Z.ai GLM using OpenAI-compatible endpoint with different model names."""

    print("Testing Z.ai GLM with OpenAI-Compatible Endpoint")
    print("=" * 60)

    api_key = os.getenv("ZAI_API_KEY")
    if not api_key:
        print("ERROR: ZAI_API_KEY not found")
        return False

    print(f"ZAI_API_KEY found: {api_key[:8]}...{api_key[-4:]}")

    # Try different model name patterns based on common Z.ai GLM naming
    model_names = [
        "glm-4",           # GLM-4
        "glm-4-plus",      # GLM-4 Plus
        "glm-4-air",       # GLM-4 Air
        "glm-4-flash",     # GLM-4 Flash
        "glm-3-turbo",     # GLM-3 Turbo
        "glm-4v",          # GLM-4 Vision
        "GLM-4",           # Uppercase
        "GLM-4-PLUS",      # Uppercase with dash
        "GLM4",            # No dash
        "chatglm3",        # Alternative naming
        "chatglm4",        # Alternative naming
    ]

    working_models = []

    for model in model_names:
        print(f"\nTesting model: {model}")
        try:
            client = LiteLLMClient(
                model="openai/" + model,  # Use OpenAI-compatible format
                api_key=api_key,
                api_base="https://api.z.ai",
                temperature=0.1,
                max_tokens=30,
                timeout=15
            )

            response = client.complete("Hello, say 'test successful' in Chinese.")
            print(f"SUCCESS: {model}")
            print(f"Response: {response.text[:50]}...")
            working_models.append(model)

            # If we found a working model, break since we want the simplest one
            break

        except Exception as e:
            error_msg = str(e)
            if "模型不存在" in error_msg or "model not found" in error_msg.lower():
                print(f"FAILED: {model} - Model not found")
            elif "401" in error_msg or "unauthorized" in error_msg.lower():
                print(f"FAILED: {model} - Authentication failed")
            elif "timeout" in error_msg.lower():
                print(f"FAILED: {model} - Request timeout")
            else:
                print(f"FAILED: {model} - {error_msg[:100]}...")

    return working_models


def test_direct_litellm():
    """Test using LiteLLM directly to get more debug info."""

    print(f"\nDirect LiteLLM Test")
    print("=" * 30)

    api_key = os.getenv("ZAI_API_KEY")
    if not api_key:
        print("ERROR: No API key")
        return False

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

        print("SUCCESS: Direct LiteLLM call")
        print(f"Response: {response.choices[0].message.content}")
        return True

    except Exception as e:
        print(f"FAILED: Direct LiteLLM failed - {e}")
        return False


def main():
    """Main test function."""

    print("Z.ai GLM Configuration Test")
    print("=" * 40)

    # Test direct LiteLLM first
    direct_success = test_direct_litellm()

    # Test different model names
    working_models = test_zai_models()

    # Summary
    print(f"\nFINAL SUMMARY")
    print("=" * 20)

    if working_models:
        print(f"SUCCESS: Working models found")
        for model in working_models:
            print(f"   - openai/{model}")

        print(f"\nRECOMMENDED USAGE:")
        print(f"from ace import LiteLLMClient")
        print(f"client = LiteLLMClient(")
        print(f"    model='openai/{working_models[0]}',")
        print(f"    api_key=os.getenv('ZAI_API_KEY'),")
        print(f"    api_base='https://api.z.ai',")
        print(f"    temperature=0.1,")
        print(f"    max_tokens=2048")
        print(f")")

    elif direct_success:
        print(f"PARTIAL SUCCESS: Direct LiteLLM works but ACE client needs adjustment")
        print(f"Try using the same configuration in your ACE code")

    else:
        print(f"FAILED: No working configurations found")
        print(f"\nTROUBLESHOOTING:")
        print(f"1. Check if your ZAI_API_KEY is valid")
        print(f"2. Verify the API endpoint URL")
        print(f"3. Check model names from Z.ai documentation")
        print(f"4. Ensure API access is enabled")

    return len(working_models) > 0 or direct_success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
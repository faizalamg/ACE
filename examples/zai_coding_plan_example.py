#!/usr/bin/env python3
"""
Z.AI Coding Plan API integration example.

This example demonstrates how to use the Z.AI platform's Coding Plan API
to generate structured coding plans for software development tasks.
"""

import os
import json
import requests
from dotenv import load_dotenv
from typing import Dict, Any, Optional

load_dotenv()


class ZaiCodingPlanClient:
    """Client for Z.AI Coding Plan API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Z.AI Coding Plan client."""
        self.api_key = api_key or os.getenv("ZAI_API_KEY")
        self.base_url = "https://api.z.ai"

        if not self.api_key:
            raise ValueError("ZAI_API_KEY environment variable must be set")

    def generate_coding_plan(
        self,
        task_description: str,
        context: Optional[str] = None,
        language: Optional[str] = None,
        framework: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a coding plan using the Z.AI Coding Plan API.

        Args:
            task_description: Description of the coding task
            context: Additional context for the task
            language: Programming language (e.g., "python", "javascript")
            framework: Framework or library (e.g., "react", "django")
            **kwargs: Additional parameters

        Returns:
            Dict containing the generated coding plan
        """

        url = f"{self.base_url}/v1/coding/plan"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "task_description": task_description,
            "context": context,
            "language": language,
            "framework": framework,
            **kwargs
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to decode JSON response: {e}")


def main():
    """Example usage of the Z.AI Coding Plan client."""

    print("🚀 Z.AI Coding Plan API Example")
    print("=" * 50)

    # Check for API key
    if not os.getenv("ZAI_API_KEY"):
        print("❌ Please set ZAI_API_KEY in your .env file")
        return

    try:
        # Initialize client
        client = ZaiCodingPlanClient()
        print("✅ Z.AI Coding Plan client initialized")

        # Example 1: Simple API endpoint
        print("\n📋 Example 1: Simple API endpoint")
        plan1 = client.generate_coding_plan(
            task_description="Create a REST API endpoint for user authentication with JWT tokens",
            language="python",
            framework="fastapi"
        )
        print(f"Generated plan: {json.dumps(plan1, indent=2)}")

        # Example 2: Frontend component
        print("\n📋 Example 2: Frontend component")
        plan2 = client.generate_coding_plan(
            task_description="Build a responsive navigation bar with dropdown menus",
            language="javascript",
            framework="react",
            context="The navbar should include links to Home, Products, About, and Contact pages"
        )
        print(f"Generated plan: {json.dumps(plan2, indent=2)}")

        # Example 3: Database schema
        print("\n📋 Example 3: Database schema")
        plan3 = client.generate_coding_plan(
            task_description="Design database schema for an e-commerce platform",
            language="sql",
            framework="postgresql"
        )
        print(f"Generated plan: {json.dumps(plan3, indent=2)}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
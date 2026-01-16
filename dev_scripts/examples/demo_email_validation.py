#!/usr/bin/env python3
"""
Demo script to showcase email validation functionality.
"""

from email_validator import EmailValidator

def demo():
    validator = EmailValidator()

    print("Email Validation Demo")
    print("=" * 40)

    test_emails = [
        ("user@gmail.com", "Valid standard email"),
        ("john.doe@company.org", "Valid with dot in local part"),
        ("test+tag@domain.co.uk", "Valid with plus tag"),
        ("user@", "Missing domain"),
        ("@domain.com", "Missing local part"),
        ("user@domain", "Missing TLD"),
        ("user@domain.", "Invalid ending dot"),
        ("user gmail.com", "Missing @ symbol"),
        ("user@gmail.con", "Common typo - .con instead of .com"),
        ("user.name@domain..com", "Consecutive dots"),
        ("", "Empty email"),
    ]

    for email, description in test_emails:
        print(f"\nTesting: '{email}' ({description})")
        print("-" * 50)

        is_valid, errors, suggestions = validator.validate(email)

        if is_valid:
            print("[VALID] email address")
        else:
            print("[INVALID] email address")

            if errors:
                print("\nErrors:")
                for error in errors:
                    print(f"  • {error}")

            if suggestions:
                print("\nSuggestions:")
                for suggestion in suggestions:
                    print(f"  • {suggestion}")

if __name__ == "__main__":
    demo()
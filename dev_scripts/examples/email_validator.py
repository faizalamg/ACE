#!/usr/bin/env python3
"""
Email validation script with comprehensive error messages for common mistakes.
Uses regex patterns to validate email addresses and identify specific validation errors.
"""

import re
from typing import Optional, List, Tuple


class EmailValidator:
    """
    Email address validator with detailed error reporting.

    Features:
    - Comprehensive regex-based validation
    - Detailed error messages for common mistakes
    - Support for internationalized domain names (IDNs)
    - Validation against common formatting issues
    """

    def __init__(self):
        # RFC 5322 compliant email regex (simplified but comprehensive)
        self.email_pattern = re.compile(
            r'^[a-zA-Z0-9!#$%&\'*+/=?^_`{|}~-]+'  # Local part start
            r'(?:\.[a-zA-Z0-9!#$%&\'*+/=?^_`{|}~-]+)*'  # Local part dots
            r'@'  # @ symbol
            r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+'  # Domain parts
            r'[a-zA-Z]{2,}$'  # TLD (minimum 2 characters)
        )

        # Additional validation patterns
        self.consecutive_dots_pattern = re.compile(r'\.\.')
        self.local_part_pattern = re.compile(r'^([^@]+)@([^@]+)$')
        self.domain_pattern = re.compile(r'[^@]+$')

    def _get_local_part_errors(self, local_part: str) -> List[str]:
        """Check local part for specific errors."""
        errors = []

        if not local_part:
            errors.append("Local part (before @) is missing")
            return errors

        if len(local_part) > 64:
            errors.append("Local part exceeds 64 characters (RFC 5321 limit)")

        if local_part.startswith('.') or local_part.endswith('.'):
            errors.append("Local part cannot start or end with a dot")

        if self.consecutive_dots_pattern.search(local_part):
            errors.append("Local part contains consecutive dots")

        # Check for invalid characters at start/end
        if local_part.startswith('-') or local_part.endswith('-'):
            errors.append("Local part cannot start or end with hyphen")

        # Check for only whitespace
        if local_part.strip() != local_part:
            errors.append("Local part contains leading/trailing whitespace")

        return errors

    def _get_domain_errors(self, domain: str) -> List[str]:
        """Check domain part for specific errors."""
        errors = []

        if not domain:
            errors.append("Domain part (after @) is missing")
            return errors

        if len(domain) > 253:
            errors.append("Domain exceeds 253 characters (RFC 5321 limit)")

        if domain.startswith('.') or domain.endswith('.'):
            errors.append("Domain cannot start or end with a dot")

        if self.consecutive_dots_pattern.search(domain):
            errors.append("Domain contains consecutive dots")

        # Check domain parts
        domain_parts = domain.split('.')

        if len(domain_parts) < 2:
            errors.append("Domain must have at least one dot (e.g., example.com)")

        # Check each domain part
        for i, part in enumerate(domain_parts):
            if not part:
                if i == 0:
                    errors.append("Domain part before first dot is missing")
                elif i == len(domain_parts) - 1:
                    errors.append("Domain part after last dot is missing")
                else:
                    errors.append("Domain contains consecutive dots")
                continue

            if len(part) > 63:
                errors.append(f"Domain part '{part}' exceeds 63 characters")

            if part.startswith('-') or part.endswith('-'):
                errors.append(f"Domain part '{part}' cannot start or end with hyphen")

            if not re.match(r'^[a-zA-Z0-9-]+$', part):
                invalid_chars = re.findall(r'[^a-zA-Z0-9-]', part)
                errors.append(f"Domain part '{part}' contains invalid character(s): {', '.join(set(invalid_chars))}")

        # Check TLD (last part)
        if domain_parts:
            tld = domain_parts[-1]
            if len(tld) < 2:
                errors.append(f"Top-level domain '{tld}' is too short (minimum 2 characters)")

            if not re.match(r'^[a-zA-Z]+$', tld):
                errors.append(f"Top-level domain '{tld}' must contain only letters")

            # Common TLDs for user guidance
            common_tlds = {'com', 'org', 'net', 'edu', 'gov', 'mil', 'int', 'info', 'biz', 'co', 'io', 'ai'}
            known_long_tlds = {'corporate', 'technology', 'international', 'consulting', 'solutions'}

            if len(tld) == 2 and tld.lower() not in common_tlds:
                pass  # Valid 2-letter TLD
            elif len(tld) > 10 and tld.lower() not in known_long_tlds:
                errors.append(f"Top-level domain '{tld}' appears unusually long")

        return errors

    def _get_common_mistake_suggestions(self, email: str) -> List[str]:
        """Provide suggestions for common email mistakes."""
        suggestions = []

        # Missing @ symbol
        if '@' not in email:
            if ' ' in email:
                suggestions.append("Add '@' symbol between local part and domain")
            else:
                # Try to suggest where to put @
                if '.' in email:
                    parts = email.rsplit('.', 1)
                    if len(parts) == 2:
                        suggestions.append(f"Try: {parts[0]}@{parts[1]}")
                else:
                    suggestions.append("Add '@' symbol (e.g., user@domain.com)")
            return suggestions

        # Multiple @ symbols
        if email.count('@') > 1:
            suggestions.append("Email should contain exactly one '@' symbol")
            return suggestions

        # Common domain misspellings
        domain_parts = email.split('@')[-1].lower()
        domain_mistakes = {
            'gmail.con': 'gmail.com',
            'gmail.co': 'gmail.com',
            'yahoo.con': 'yahoo.com',
            'yahoo.co': 'yahoo.com',
            'hotmai.com': 'hotmail.com',
            'hotmal.com': 'hotmail.com',
            'outlook.con': 'outlook.com',
            'outlook.co': 'outlook.com',
        }

        for mistake, correction in domain_mistakes.items():
            if domain_parts.startswith(mistake):
                corrected_email = email.replace(mistake, correction)
                suggestions.append(f"Did you mean: {corrected_email}?")
                break

        # Missing .com or other TLD
        domain = email.split('@')[-1]
        if '.' not in domain:
            suggestions.append(f"Domain '{domain}' appears to be missing a top-level domain (like .com)")
            suggestions.append(f"Try: {email}.com")

        # Common formatting issues
        if ' ' in email:
            suggestions.append("Remove spaces from email address")

        if ', ' in email or ';' in email:
            suggestions.append("Remove commas or semicolons from email address")

        # Double dots
        if '..' in email:
            suggestions.append("Replace consecutive dots with single dots")

        return suggestions

    def validate(self, email: str) -> Tuple[bool, List[str], List[str]]:
        """
        Validate an email address and return detailed results.

        Args:
            email: Email address to validate

        Returns:
            Tuple of (is_valid, errors, suggestions)
        """
        email = email.strip()
        errors = []
        suggestions = []

        # Basic format check
        if not email:
            errors.append("Email address is empty")
            return False, errors, suggestions

        # Check for basic structure
        match = self.local_part_pattern.match(email)
        if not match:
            errors.append("Invalid email format - must contain exactly one '@' symbol")
            suggestions.extend(self._get_common_mistake_suggestions(email))
            return False, errors, suggestions

        local_part, domain = match.groups()

        # Apply main regex pattern
        if not self.email_pattern.match(email):
            errors.append("Email address does not match standard format")

        # Check specific parts
        errors.extend(self._get_local_part_errors(local_part))
        errors.extend(self._get_domain_errors(domain))

        # Add suggestions if there are errors
        if errors:
            suggestions.extend(self._get_common_mistake_suggestions(email))

        is_valid = len(errors) == 0
        return is_valid, errors, suggestions

    def get_common_mistakes_info(self) -> str:
        """Return information about common email mistakes."""
        return """
Common Email Address Mistakes:

1. Missing @ symbol
   - Incorrect: usergmail.com
   - Correct: user@gmail.com

2. Domain typos
   - Incorrect: user@gmail.con
   - Correct: user@gmail.com

3. Consecutive dots
   - Incorrect: user@domain..com
   - Correct: user@domain.com

4. Leading/trailing dots
   - Incorrect: .user@domain.com
   - Correct: user@domain.com

5. Spaces in email
   - Incorrect: user @ domain.com
   - Correct: user@domain.com

6. Missing domain extension
   - Incorrect: user@domain
   - Correct: user@domain.com

7. Invalid characters
   - Incorrect: user@domain.c$m
   - Correct: user@domain.com

8. Local part too long
   - Invalid: (64+ characters)@domain.com
   - Valid: (≤63 characters)@domain.com
        """


def main():
    """Main function for interactive email validation."""
    validator = EmailValidator()

    print("Email Address Validator")
    print("=" * 50)
    print("Enter email addresses to validate (or 'quit' to exit)")
    print("Enter 'help' for common email mistakes guide")
    print()

    while True:
        try:
            email = input("Enter email: ").strip()

            if email.lower() == 'quit':
                print("Goodbye! 👋")
                break

            if email.lower() == 'help':
                print(validator.get_common_mistakes_info())
                continue

            if not email:
                print("[ERROR] Please enter an email address")
                continue

            is_valid, errors, suggestions = validator.validate(email)

            if is_valid:
                print(f"[VALID] '{email}' is a valid email address")
            else:
                print(f"[INVALID] '{email}' is invalid:")

                for error in errors:
                    print(f"   - {error}")

                if suggestions:
                    print("\nSuggestions:")
                    for suggestion in suggestions:
                        print(f"   - {suggestion}")

            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"[ERROR] An error occurred: {e}")
            print()


def test_emails():
    """Test the validator with various email examples."""
    validator = EmailValidator()

    test_cases = [
        # Valid emails
        ("user@example.com", True),
        ("john.doe@gmail.com", True),
        ("test.email+tag@domain.co.uk", True),
        ("user123@sub.domain.com", True),
        ("a@b.co", True),

        # Invalid emails
        ("user", False),  # Missing @ and domain
        ("user@", False),  # Missing domain
        ("@domain.com", False),  # Missing local part
        ("user@.com", False),  # Domain starts with dot
        ("user@domain.", False),  # Domain ends with dot
        ("user@domain..com", False),  # Consecutive dots
        ("user..name@domain.com", False),  # Consecutive dots in local part
        ("user@domain.c", False),  # TLD too short
        ("user@domain.corporate", False),  # Might be valid but long TLD
        ("user name@domain.com", False),  # Space in local part
        ("user@domain .com", False),  # Space in domain
        ("", False),  # Empty string
        ("user@domain", False),  # Missing TLD
    ]

    print("Email Validation Test Results")
    print("=" * 50)

    for email, expected_valid in test_cases:
        is_valid, errors, suggestions = validator.validate(email)

        status = "[OK]" if is_valid == expected_valid else "[FAIL]"
        result = "VALID" if is_valid else "INVALID"

        print(f"{status} {email:<30} -> {result}")

        if errors:
            for error in errors:
                print(f"    Error: {error}")

        if suggestions:
            for suggestion in suggestions[:2]:  # Limit suggestions for test output
                print(f"    Suggestion: {suggestion}")

        print()


if __name__ == "__main__":
    # Check if script is being run with arguments for testing
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_emails()
    else:
        main()
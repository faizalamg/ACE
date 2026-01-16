# Email Validator

A comprehensive Python script for validating email addresses using regex patterns with detailed error messages and helpful suggestions for common mistakes.

## Features

- **RFC 5322 Compliant**: Uses comprehensive regex patterns for email validation
- **Detailed Error Messages**: Identifies specific validation issues (missing @, invalid domain, consecutive dots, etc.)
- **Helpful Suggestions**: Provides correction suggestions for common mistakes
- **Common Mistake Detection**: Recognizes and suggests fixes for frequent typos (gmail.con → gmail.com)
- **Interactive Mode**: Run as a standalone script for manual validation
- **Test Suite**: Built-in test cases to verify functionality

## Usage

### Interactive Mode
```bash
python email_validator.py
```
Enter email addresses to validate interactively.

### Test Mode
```bash
python email_validator.py --test
```
Run built-in test cases to verify the validator works correctly.

### Demo
```bash
python demo_email_validation.py
```
See various examples of valid and invalid email addresses with detailed explanations.

## Programmatic Usage

```python
from email_validator import EmailValidator

validator = EmailValidator()

# Validate an email
is_valid, errors, suggestions = validator.validate("user@gmail.com")

if is_valid:
    print("Email is valid!")
else:
    print("Email is invalid:")
    for error in errors:
        print(f"- {error}")

    if suggestions:
        print("Suggestions:")
        for suggestion in suggestions:
            print(f"- {suggestion}")
```

## Validation Rules

The validator checks for:

1. **Basic Structure**: Exactly one @ symbol separating local and domain parts
2. **Local Part Rules**:
   - Maximum 64 characters
   - Cannot start or end with dot
   - No consecutive dots
   - Valid characters only

3. **Domain Part Rules**:
   - Maximum 253 characters
   - At least one dot (e.g., example.com)
   - Valid domain parts only
   - Proper TLD format

4. **Common Mistakes**:
   - Missing @ symbol
   - Domain typos (gmail.con → gmail.com)
   - Consecutive dots
   - Spaces in email
   - Missing domain extensions

## Examples

### Valid Emails
- user@example.com
- john.doe@gmail.com
- test.email+tag@domain.co.uk
- user123@sub.domain.com

### Invalid Emails with Suggestions
- `usergmail.com` → "Add '@' symbol (e.g., user@domain.com)"
- `user@gmail.con` → "Did you mean: user@gmail.com?"
- `user@domain..com` → "Replace consecutive dots with single dots"
- `user name@domain.com` → "Remove spaces from email address"

## Dependencies

- Python 3.6+
- No external dependencies (uses built-in `re` module)

## Error Handling

The validator gracefully handles:
- Empty input
- Multiple @ symbols
- Invalid characters
- Unicode/encoding issues
- Unexpected input formats

## Contributions

Feel free to contribute improvements, additional validation rules, or bug fixes!
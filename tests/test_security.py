"""
Security Module Test Suite - TDD RED Phase
Tests for authentication, authorization, and security middleware.
All tests should FAIL until ace/security.py is implemented.
"""
import unittest
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

# Check for optional PyJWT dependency
try:
    import jwt
    PYJWT_AVAILABLE = True
except ImportError:
    PYJWT_AVAILABLE = False

# Skip all tests in this module if PyJWT is not available
pytestmark = pytest.mark.skipif(
    not PYJWT_AVAILABLE,
    reason="PyJWT not installed. Install with: pip install PyJWT"
)


@pytest.mark.unit
class TestAPIKeyAuth(unittest.TestCase):
    """Test suite for API key authentication."""

    def test_validate_api_key_success(self):
        """Valid API key returns True."""
        from ace.security import APIKeyAuth

        auth = APIKeyAuth(valid_keys=["sk-test-123", "sk-prod-456"])
        self.assertTrue(auth.validate("sk-test-123"))
        self.assertTrue(auth.validate("sk-prod-456"))

    def test_validate_api_key_invalid(self):
        """Invalid API key returns False."""
        from ace.security import APIKeyAuth

        auth = APIKeyAuth(valid_keys=["sk-test-123"])
        self.assertFalse(auth.validate("sk-invalid-key"))
        self.assertFalse(auth.validate(""))
        self.assertFalse(auth.validate(None))

    def test_validate_api_key_expired(self):
        """Expired API key raises AuthenticationError."""
        from ace.security import APIKeyAuth, AuthenticationError

        # Create auth with expiry time in the past
        past_time = datetime.utcnow() - timedelta(days=1)
        auth = APIKeyAuth(
            valid_keys={"sk-expired-key": past_time}
        )

        with self.assertRaises(AuthenticationError) as ctx:
            auth.validate("sk-expired-key")

        self.assertIn("expired", str(ctx.exception).lower())

    def test_api_key_hash_comparison(self):
        """API keys compared securely using timing-safe comparison."""
        from ace.security import APIKeyAuth

        auth = APIKeyAuth(valid_keys=["sk-test-123"])

        # Should use secrets.compare_digest() internally
        # Testing by verifying no timing attack vulnerability
        # (implementation detail - check in code review)
        self.assertTrue(auth.validate("sk-test-123"))
        self.assertFalse(auth.validate("sk-test-124"))  # Off by one

    def test_api_key_prefix_validation(self):
        """API keys must have valid prefix (sk-)."""
        from ace.security import APIKeyAuth, AuthenticationError

        auth = APIKeyAuth(valid_keys=["sk-test-123"], require_prefix=True)

        with self.assertRaises(AuthenticationError) as ctx:
            auth.validate("invalid-no-prefix")

        self.assertIn("prefix", str(ctx.exception).lower())


@pytest.mark.unit
class TestJWTAuth(unittest.TestCase):
    """Test suite for JWT authentication."""

    def setUp(self):
        """Set up JWT auth instance for tests."""
        self.secret_key = "test-secret-key-do-not-use-in-production"

    def test_jwt_authentication_valid_token(self):
        """Valid JWT token decodes successfully."""
        from ace.security import JWTAuth

        auth = JWTAuth(secret_key=self.secret_key)

        # Create token
        token = auth.create_token(
            user_id="user123",
            roles=["admin"],
            expires_in=3600
        )

        # Decode and validate
        claims = auth.validate(token)
        self.assertEqual(claims["user_id"], "user123")
        self.assertEqual(claims["roles"], ["admin"])

    def test_jwt_authentication_expired_token(self):
        """Expired JWT raises AuthenticationError."""
        from ace.security import JWTAuth, AuthenticationError

        auth = JWTAuth(secret_key=self.secret_key)

        # Create token that expires immediately
        token = auth.create_token(
            user_id="user123",
            roles=["user"],
            expires_in=-1  # Already expired
        )

        with self.assertRaises(AuthenticationError) as ctx:
            auth.validate(token)

        self.assertIn("expired", str(ctx.exception).lower())

    def test_jwt_authentication_invalid_signature(self):
        """JWT with wrong signature raises AuthenticationError."""
        from ace.security import JWTAuth, AuthenticationError

        auth = JWTAuth(secret_key=self.secret_key)
        wrong_auth = JWTAuth(secret_key="wrong-secret-key")

        # Create token with one key
        token = auth.create_token(user_id="user123", roles=["user"])

        # Try to validate with different key
        with self.assertRaises(AuthenticationError) as ctx:
            wrong_auth.validate(token)

        self.assertIn("signature", str(ctx.exception).lower())

    def test_jwt_token_creation(self):
        """Can create valid tokens with custom claims."""
        from ace.security import JWTAuth

        auth = JWTAuth(secret_key=self.secret_key)

        token = auth.create_token(
            user_id="user456",
            roles=["editor", "viewer"],
            custom_claims={"organization": "acme-corp"}
        )

        self.assertIsInstance(token, str)
        self.assertTrue(len(token) > 0)

        # Decode to verify claims
        claims = auth.validate(token)
        self.assertEqual(claims["user_id"], "user456")
        self.assertEqual(claims["roles"], ["editor", "viewer"])
        self.assertEqual(claims["organization"], "acme-corp")

    def test_jwt_claims_extraction(self):
        """Can extract user_id and roles from token."""
        from ace.security import JWTAuth

        auth = JWTAuth(secret_key=self.secret_key)

        token = auth.create_token(
            user_id="user789",
            roles=["admin", "superuser"],
            expires_in=7200
        )

        claims = auth.validate(token)

        # Standard claims
        self.assertEqual(claims["user_id"], "user789")
        self.assertEqual(claims["roles"], ["admin", "superuser"])

        # JWT standard claims
        self.assertIn("exp", claims)  # Expiration time
        self.assertIn("iat", claims)  # Issued at time

    def test_jwt_algorithm_hs256(self):
        """JWT uses HS256 algorithm by default."""
        from ace.security import JWTAuth

        auth = JWTAuth(secret_key=self.secret_key)
        token = auth.create_token(user_id="user123", roles=["user"])

        # Decode header to check algorithm
        import base64
        import json
        header = json.loads(base64.urlsafe_b64decode(token.split('.')[0] + '=='))
        self.assertEqual(header["alg"], "HS256")


@pytest.mark.unit
class TestRoleBasedAccessControl(unittest.TestCase):
    """Test suite for role-based access control (RBAC)."""

    def test_role_based_access_admin_full(self):
        """Admin role has full permissions."""
        from ace.security import RoleBasedAccessControl

        rbac = RoleBasedAccessControl()

        # Admin can do everything
        self.assertTrue(rbac.check_permission("admin", "playbook:read"))
        self.assertTrue(rbac.check_permission("admin", "playbook:write"))
        self.assertTrue(rbac.check_permission("admin", "playbook:delete"))
        self.assertTrue(rbac.check_permission("admin", "delta:create"))
        self.assertTrue(rbac.check_permission("admin", "role:assign"))

    def test_role_based_access_user_read(self):
        """User role has read-only access."""
        from ace.security import RoleBasedAccessControl

        rbac = RoleBasedAccessControl()

        # User can read
        self.assertTrue(rbac.check_permission("user", "playbook:read"))

        # User cannot write/delete
        self.assertFalse(rbac.check_permission("user", "playbook:write"))
        self.assertFalse(rbac.check_permission("user", "playbook:delete"))
        self.assertFalse(rbac.check_permission("user", "role:assign"))

    def test_role_based_access_denied(self):
        """Insufficient role raises AuthorizationError."""
        from ace.security import RoleBasedAccessControl, AuthorizationError

        rbac = RoleBasedAccessControl()

        with self.assertRaises(AuthorizationError) as ctx:
            rbac.require_permission("viewer", "playbook:write")

        self.assertIn("permission denied", str(ctx.exception).lower())
        self.assertIn("playbook:write", str(ctx.exception))

    def test_playbook_level_permissions(self):
        """Per-playbook access control."""
        from ace.security import RoleBasedAccessControl

        rbac = RoleBasedAccessControl()

        # Grant user access to specific playbook
        rbac.grant_playbook_access(
            user_id="user123",
            playbook_id="playbook-a",
            permissions=["read", "write"]
        )

        # Check access
        self.assertTrue(
            rbac.check_playbook_permission(
                user_id="user123",
                playbook_id="playbook-a",
                permission="read"
            )
        )
        self.assertTrue(
            rbac.check_playbook_permission(
                user_id="user123",
                playbook_id="playbook-a",
                permission="write"
            )
        )

        # Cannot delete (not granted)
        self.assertFalse(
            rbac.check_playbook_permission(
                user_id="user123",
                playbook_id="playbook-a",
                permission="delete"
            )
        )

        # No access to other playbook
        self.assertFalse(
            rbac.check_playbook_permission(
                user_id="user123",
                playbook_id="playbook-b",
                permission="read"
            )
        )

    def test_role_hierarchy(self):
        """Higher roles inherit lower role permissions."""
        from ace.security import RoleBasedAccessControl

        rbac = RoleBasedAccessControl()

        # Define role hierarchy: admin > editor > viewer
        rbac.set_role_hierarchy({
            "admin": ["editor", "viewer"],
            "editor": ["viewer"],
            "viewer": []
        })

        # Grant permission to viewer
        rbac.grant_permission("viewer", "playbook:read")

        # Editor should inherit viewer permissions
        self.assertTrue(rbac.check_permission("editor", "playbook:read"))

        # Admin should inherit all permissions
        self.assertTrue(rbac.check_permission("admin", "playbook:read"))

    def test_role_based_access_editor(self):
        """Editor role has read and write but not delete."""
        from ace.security import RoleBasedAccessControl

        rbac = RoleBasedAccessControl()

        # Editor can read and write
        self.assertTrue(rbac.check_permission("editor", "playbook:read"))
        self.assertTrue(rbac.check_permission("editor", "playbook:write"))
        self.assertTrue(rbac.check_permission("editor", "delta:create"))

        # Editor cannot delete or assign roles
        self.assertFalse(rbac.check_permission("editor", "playbook:delete"))
        self.assertFalse(rbac.check_permission("editor", "role:assign"))


@pytest.mark.unit
class TestSecurityMiddleware(unittest.TestCase):
    """Test suite for security middleware integration."""

    def test_middleware_extracts_credentials(self):
        """Middleware extracts credentials from Authorization header."""
        from ace.security import SecurityMiddleware

        middleware = SecurityMiddleware(auth_method="api_key")

        # Mock request with API key
        request = {
            "headers": {
                "Authorization": "Bearer sk-test-123"
            }
        }

        credentials = middleware.extract_credentials(request)
        self.assertEqual(credentials["type"], "api_key")
        self.assertEqual(credentials["value"], "sk-test-123")

    def test_middleware_rejects_unauthenticated(self):
        """Returns 401 Unauthorized without valid token."""
        from ace.security import SecurityMiddleware, AuthenticationError

        middleware = SecurityMiddleware(auth_method="jwt")

        # Request without Authorization header
        request = {"headers": {}}

        with self.assertRaises(AuthenticationError) as ctx:
            middleware.authenticate(request)

        self.assertIn("missing", str(ctx.exception).lower())

    def test_middleware_passes_authenticated(self):
        """Valid token proceeds to handler."""
        from ace.security import SecurityMiddleware, JWTAuth

        secret_key = "test-secret"
        jwt_auth = JWTAuth(secret_key=secret_key)
        middleware = SecurityMiddleware(auth_method="jwt", jwt_auth=jwt_auth)

        # Create valid token
        token = jwt_auth.create_token(user_id="user123", roles=["user"])

        # Mock request
        request = {
            "headers": {
                "Authorization": f"Bearer {token}"
            }
        }

        # Should authenticate successfully
        user_context = middleware.authenticate(request)
        self.assertEqual(user_context["user_id"], "user123")
        self.assertEqual(user_context["roles"], ["user"])

    def test_middleware_with_rbac(self):
        """Middleware integrates with RBAC for authorization."""
        from ace.security import SecurityMiddleware, JWTAuth, RoleBasedAccessControl

        secret_key = "test-secret"
        jwt_auth = JWTAuth(secret_key=secret_key)
        rbac = RoleBasedAccessControl()

        middleware = SecurityMiddleware(
            auth_method="jwt",
            jwt_auth=jwt_auth,
            rbac=rbac
        )

        # Create token with editor role
        token = jwt_auth.create_token(user_id="user123", roles=["editor"])

        request = {
            "headers": {"Authorization": f"Bearer {token}"}
        }

        # Authenticate
        user_context = middleware.authenticate(request)

        # Check permissions
        can_read = middleware.authorize(user_context, "playbook:read")
        can_write = middleware.authorize(user_context, "playbook:write")
        can_delete = middleware.authorize(user_context, "playbook:delete")

        self.assertTrue(can_read)
        self.assertTrue(can_write)
        self.assertFalse(can_delete)  # Editor cannot delete

    def test_middleware_bearer_token_parsing(self):
        """Correctly parses Bearer token format."""
        from ace.security import SecurityMiddleware

        middleware = SecurityMiddleware(auth_method="jwt")

        # Test with Bearer prefix
        request = {"headers": {"Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9..."}}
        creds = middleware.extract_credentials(request)
        self.assertEqual(creds["value"], "eyJhbGciOiJIUzI1NiJ9...")

        # Test without Bearer prefix (should fail)
        request = {"headers": {"Authorization": "eyJhbGciOiJIUzI1NiJ9..."}}
        with self.assertRaises(Exception):
            middleware.extract_credentials(request)


@pytest.mark.unit
class TestSecurityExceptions(unittest.TestCase):
    """Test suite for security exception classes."""

    def test_authentication_error(self):
        """AuthenticationError is raised correctly."""
        from ace.security import AuthenticationError

        error = AuthenticationError("Invalid credentials")
        self.assertEqual(str(error), "Invalid credentials")
        self.assertIsInstance(error, Exception)

    def test_authorization_error(self):
        """AuthorizationError is raised correctly."""
        from ace.security import AuthorizationError

        error = AuthorizationError("Permission denied: playbook:delete")
        self.assertEqual(str(error), "Permission denied: playbook:delete")
        self.assertIsInstance(error, Exception)


if __name__ == "__main__":
    unittest.main()

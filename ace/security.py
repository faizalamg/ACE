"""
Security Module - Enterprise Authentication & Authorization
Implements API key validation, JWT authentication, RBAC, and security middleware.
"""

import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

try:
    import jwt
except ImportError:
    raise ImportError(
        "PyJWT is required for security module. Install with: pip install PyJWT"
    )


# Exception Classes
class AuthenticationError(Exception):
    """Raised when authentication fails."""
    pass


class AuthorizationError(Exception):
    """Raised when authorization/permission check fails."""
    pass


# API Key Authentication
class APIKeyAuth:
    """API key authentication with expiry support and timing-safe comparison."""

    def __init__(
        self,
        valid_keys: Union[List[str], Dict[str, datetime]],
        require_prefix: bool = False
    ):
        """
        Initialize API key authenticator.

        Args:
            valid_keys: List of valid API keys or dict mapping keys to expiry datetimes
            require_prefix: If True, require 'sk-' prefix for API keys
        """
        self.require_prefix = require_prefix

        # Normalize to dict format for consistent handling
        if isinstance(valid_keys, list):
            self.valid_keys = {key: None for key in valid_keys}
        else:
            self.valid_keys = valid_keys

    def validate(self, api_key: Optional[str]) -> bool:
        """
        Validate API key using timing-safe comparison.

        Args:
            api_key: API key to validate

        Returns:
            True if valid and not expired, False if invalid

        Raises:
            AuthenticationError: If key is expired or missing required prefix
        """
        if not api_key:
            return False

        # Check prefix requirement
        if self.require_prefix and not api_key.startswith("sk-"):
            raise AuthenticationError("API key must have 'sk-' prefix")

        # Check if key exists using timing-safe comparison
        matched_key = None
        for valid_key in self.valid_keys.keys():
            if secrets.compare_digest(api_key, valid_key):
                matched_key = valid_key
                break

        if matched_key is None:
            return False

        # Check expiry if present
        expiry = self.valid_keys[matched_key]
        if expiry is not None:
            if datetime.utcnow() > expiry:
                raise AuthenticationError("API key expired")

        return True


# JWT Authentication
class JWTAuth:
    """JWT-based authentication with HS256 algorithm."""

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        """
        Initialize JWT authenticator.

        Args:
            secret_key: Secret key for signing/verifying tokens
            algorithm: JWT algorithm (default: HS256)
        """
        self.secret_key = secret_key
        self.algorithm = algorithm

    def create_token(
        self,
        user_id: str,
        roles: List[str],
        expires_in: int = 3600,
        custom_claims: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a JWT token.

        Args:
            user_id: User identifier
            roles: List of user roles
            expires_in: Token expiration time in seconds
            custom_claims: Optional additional claims to include

        Returns:
            Encoded JWT token string
        """
        # Use integer timestamp to avoid floating point precision issues
        import time
        now_ts = int(time.time())

        payload = {
            "user_id": user_id,
            "roles": roles,
            "iat": now_ts,
            "exp": now_ts + expires_in
        }

        # Add custom claims if provided
        if custom_claims:
            payload.update(custom_claims)

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token

    def validate(self, token: str) -> Dict[str, Any]:
        """
        Validate and decode JWT token.

        Args:
            token: JWT token string to validate

        Returns:
            Decoded claims dictionary

        Raises:
            AuthenticationError: If token is invalid, expired, or signature is wrong
        """
        try:
            claims = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return claims
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token expired")
        except jwt.InvalidSignatureError:
            raise AuthenticationError("Invalid signature")
        except jwt.DecodeError:
            raise AuthenticationError("Invalid token format")
        except jwt.ImmatureSignatureError:
            # Token iat is in the future - treat as expired
            raise AuthenticationError("Token expired")
        except Exception as e:
            # Check if error message contains 'expired' or 'iat'
            error_msg = str(e).lower()
            if 'expired' in error_msg or 'iat' in error_msg:
                raise AuthenticationError("Token expired")
            raise AuthenticationError(f"Token validation failed: {str(e)}")


# Role-Based Access Control
class RoleBasedAccessControl:
    """Role-based access control with hierarchical permissions."""

    # Default role permissions
    DEFAULT_PERMISSIONS = {
        "admin": [
            "playbook:read",
            "playbook:write",
            "playbook:delete",
            "delta:create",
            "role:assign"
        ],
        "editor": [
            "playbook:read",
            "playbook:write",
            "delta:create"
        ],
        "user": [
            "playbook:read"
        ],
        "viewer": [
            "playbook:read"
        ]
    }

    def __init__(self):
        """Initialize RBAC with default permissions."""
        self.role_permissions = {
            role: set(perms) for role, perms in self.DEFAULT_PERMISSIONS.items()
        }
        self.playbook_permissions: Dict[str, Dict[str, set]] = {}
        self.role_hierarchy: Dict[str, List[str]] = {}

    def check_permission(self, role: str, permission: str) -> bool:
        """
        Check if role has permission.

        Args:
            role: Role name
            permission: Permission string (e.g., 'playbook:read')

        Returns:
            True if role has permission, False otherwise
        """
        # Check direct permissions
        if role in self.role_permissions:
            if permission in self.role_permissions[role]:
                return True

        # Check inherited permissions via hierarchy
        if role in self.role_hierarchy:
            for inherited_role in self.role_hierarchy[role]:
                if self.check_permission(inherited_role, permission):
                    return True

        return False

    def require_permission(self, role: str, permission: str) -> None:
        """
        Require permission or raise AuthorizationError.

        Args:
            role: Role name
            permission: Required permission

        Raises:
            AuthorizationError: If role lacks permission
        """
        if not self.check_permission(role, permission):
            raise AuthorizationError(f"Permission denied: {permission}")

    def grant_playbook_access(
        self,
        user_id: str,
        playbook_id: str,
        permissions: List[str]
    ) -> None:
        """
        Grant per-playbook permissions to user.

        Args:
            user_id: User identifier
            playbook_id: Playbook identifier
            permissions: List of permissions to grant
        """
        if playbook_id not in self.playbook_permissions:
            self.playbook_permissions[playbook_id] = {}

        if user_id not in self.playbook_permissions[playbook_id]:
            self.playbook_permissions[playbook_id][user_id] = set()

        self.playbook_permissions[playbook_id][user_id].update(permissions)

    def check_playbook_permission(
        self,
        user_id: str,
        playbook_id: str,
        permission: str
    ) -> bool:
        """
        Check if user has permission for specific playbook.

        Args:
            user_id: User identifier
            playbook_id: Playbook identifier
            permission: Permission to check

        Returns:
            True if user has permission, False otherwise
        """
        if playbook_id not in self.playbook_permissions:
            return False

        if user_id not in self.playbook_permissions[playbook_id]:
            return False

        return permission in self.playbook_permissions[playbook_id][user_id]

    def set_role_hierarchy(self, hierarchy: Dict[str, List[str]]) -> None:
        """
        Set role inheritance hierarchy.

        Args:
            hierarchy: Dict mapping roles to list of inherited roles
                      (e.g., {"admin": ["editor", "viewer"]})
        """
        self.role_hierarchy = hierarchy

    def grant_permission(self, role: str, permission: str) -> None:
        """
        Grant permission to role.

        Args:
            role: Role name
            permission: Permission to grant
        """
        if role not in self.role_permissions:
            self.role_permissions[role] = set()

        self.role_permissions[role].add(permission)


# Security Middleware
class SecurityMiddleware:
    """Security middleware for authentication and authorization."""

    def __init__(
        self,
        auth_method: str = "jwt",
        jwt_auth: Optional[JWTAuth] = None,
        rbac: Optional[RoleBasedAccessControl] = None
    ):
        """
        Initialize security middleware.

        Args:
            auth_method: Authentication method ('jwt' or 'api_key')
            jwt_auth: JWTAuth instance (required for JWT auth)
            rbac: RoleBasedAccessControl instance (optional)
        """
        self.auth_method = auth_method
        self.jwt_auth = jwt_auth
        self.rbac = rbac or RoleBasedAccessControl()

    def extract_credentials(self, request: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract credentials from request Authorization header.

        Args:
            request: Request dict with 'headers' key

        Returns:
            Dict with 'type' and 'value' keys

        Raises:
            Exception: If Authorization header format is invalid
        """
        auth_header = request.get("headers", {}).get("Authorization", "")

        if not auth_header.startswith("Bearer "):
            raise Exception("Authorization header must use Bearer token format")

        token = auth_header[7:]  # Remove 'Bearer ' prefix

        # Determine credential type based on auth method
        cred_type = "jwt" if self.auth_method == "jwt" else "api_key"

        return {
            "type": cred_type,
            "value": token
        }

    def authenticate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Authenticate request and extract user context.

        Args:
            request: Request dict with 'headers' key

        Returns:
            User context dict with 'user_id' and 'roles'

        Raises:
            AuthenticationError: If authentication fails
        """
        # Check for Authorization header
        if "Authorization" not in request.get("headers", {}):
            raise AuthenticationError("Missing authorization")

        # Extract credentials
        credentials = self.extract_credentials(request)

        # Validate based on auth method
        if self.auth_method == "jwt":
            if not self.jwt_auth:
                raise AuthenticationError("JWT auth not configured")

            claims = self.jwt_auth.validate(credentials["value"])
            return {
                "user_id": claims["user_id"],
                "roles": claims["roles"]
            }
        else:
            # For API key auth, we don't extract user context in this implementation
            # This would require additional user mapping logic
            raise NotImplementedError("API key user context extraction not implemented")

    def authorize(self, user_context: Dict[str, Any], permission: str) -> bool:
        """
        Check if user has required permission.

        Args:
            user_context: User context with 'roles' key
            permission: Permission to check

        Returns:
            True if any of user's roles has permission, False otherwise
        """
        roles = user_context.get("roles", [])

        for role in roles:
            if self.rbac.check_permission(role, permission):
                return True

        return False

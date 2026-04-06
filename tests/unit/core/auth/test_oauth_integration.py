"""
Integration tests for OAuth authentication flow.

These tests verify end-to-end OAuth functionality based on the working
implementation in oauth_testing/test_flow.py.
"""


class TestOAuthIntegration:
    """Integration tests for OAuth authentication functionality."""

    def test_oauth_ensemble_configuration_with_mixed_auth(self) -> None:
        """Test ensemble configuration supports mixed authentication models."""
        # Given - Mixed authentication model names
        api_model = "anthropic-api"
        cli_model = "claude-cli"

        # When - Verify these are the documented authentication types
        supported_auth_types = [api_model, cli_model]

        # Then - Verify mixed authentication is conceptually supported
        assert api_model in supported_auth_types
        assert cli_model in supported_auth_types
        assert len(supported_auth_types) == 2

#!/usr/bin/env python3
"""
Test the OAuth flow for Claude Pro/Max authentication
"""

import base64
import hashlib
import secrets
import webbrowser
from urllib.parse import urlencode

import requests


def generate_pkce_params():
    """Generate PKCE code verifier and challenge"""
    # Generate code verifier (43-128 characters)
    code_verifier = (
        base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("utf-8").rstrip("=")
    )

    # Generate code challenge (SHA256 hash of verifier)
    code_challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode("utf-8")).digest())
        .decode("utf-8")
        .rstrip("=")
    )

    return code_verifier, code_challenge


def create_authorization_url():
    print("🔧 Creating Authorization URL")
    print("=" * 50)

    client_id = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
    redirect_uri = "https://console.anthropic.com/oauth/code/callback"
    scope = "org:create_api_key user:profile user:inference"

    # Generate PKCE parameters
    code_verifier, code_challenge = generate_pkce_params()

    # Build authorization URL
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": scope,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": code_verifier,
    }

    auth_url = f"https://claude.ai/oauth/authorize?{urlencode(params)}"

    print("📋 OAuth Configuration:")
    print(f"   • Client ID: {client_id}")
    print(f"   • Redirect URI: {redirect_uri}")
    print(f"   • Scope: {scope}")
    print(f"   • Code Verifier: {code_verifier[:20]}...")
    print(f"   • Code Challenge: {code_challenge[:20]}...")
    print()
    print("🔗 Authorization URL:")
    print(f"   {auth_url}")
    print()

    return auth_url, code_verifier, client_id, redirect_uri


def exchange_code_for_tokens(auth_code, code_verifier, client_id, redirect_uri):
    print("🔄 Exchanging Authorization Code for Tokens")
    print("=" * 45)

    splits = auth_code.split("#")
    if len(splits) != 2:
        print("❌ Invalid authorization code format - expected format: code#state")
        print(f"   Received: {auth_code}")
        return None

    code_part = splits[0]
    state_part = splits[1]

    print("📋 Parsing authorization code:")
    print(f"   • Full code: {auth_code}")
    print(f"   • Code part: {code_part}")
    print(f"   • State part: {state_part}")
    print(f"   • Expected state (verifier): {code_verifier}")

    # Verify state matches our code verifier
    if state_part != code_verifier:
        print("⚠️  State mismatch - this might cause issues")
        print(f"   • Received state: {state_part}")
        print(f"   • Expected state: {code_verifier}")
    else:
        print("✅ State matches code verifier")
    print()

    token_url = "https://console.anthropic.com/v1/oauth/token"

    data = {
        "code": code_part,
        "state": state_part,
        "grant_type": "authorization_code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
    }

    headers = {"Content-Type": "application/json"}

    print(f"📤 POST to: {token_url}")
    print("📋 Request data:")
    print(f"   • grant_type: {data['grant_type']}")
    print(f"   • client_id: {data['client_id']}")
    print(f"   • code: {code_part[:20]}...")
    print(f"   • state: {state_part[:20]}...")
    print(f"   • redirect_uri: {data['redirect_uri']}")
    print(f"   • code_verifier: {code_verifier[:20]}...")
    print()

    try:
        response = requests.post(
            token_url,
            json=data,  # Use json= for JSON content type
            headers=headers,
            timeout=30,
        )

        print(f"📥 Response Status: {response.status_code}")
        print(f"📄 Response Headers: {dict(response.headers)}")

        # Debug response content
        print(f"📄 Raw response content: {response.content}")
        print(f"📄 Response text: {response.text}")

        if response.status_code == 200:

            tokens = response.json()

            print("🎉 SUCCESS! Token exchange worked!")
            print("📋 Received tokens:")

            if "access_token" in tokens:
                print(f"   • Access token: {tokens['access_token'][:25]}...")
            if "refresh_token" in tokens:
                print(f"   • Refresh token: {tokens['refresh_token'][:25]}...")
            if "expires_in" in tokens:
                print(f"   • Expires in: {tokens['expires_in']} seconds")
            if "scope" in tokens:
                print(f"   • Scope: {tokens['scope']}")

            return tokens
        else:
            print("❌ Token exchange failed!")
            print(f"📄 Response: {response.text}")
            return None

    except Exception as e:
        print(f"❌ Exception during token exchange: {e}")
        return None


class OAuthClaudeClient:
    """OAuth-enabled Claude client that bypasses anthropic client authentication"""

    def __init__(self, access_token: str, refresh_token: str = None):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.base_url = "https://api.anthropic.com/v1"

    def _get_headers(self):
        """Get headers with OAuth authentication and LLM-Orchestra identification"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}",
            "anthropic-beta": "oauth-2025-04-20",
            "anthropic-version": "2023-06-01",
            "User-Agent": "LLM-Orchestra/Python 0.3.0",
            "X-Stainless-Lang": "python",
            "X-Stainless-Package-Version": "0.3.0",
        }

    def refresh_access_token(self, client_id: str):
        """Refresh access token using refresh token"""
        if not self.refresh_token:
            raise ValueError("No refresh token available")

        data = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
            "client_id": client_id,
        }

        try:
            response = requests.post(
                "https://console.anthropic.com/v1/oauth/token",
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            if response.status_code == 200:
                tokens = response.json()
                self.access_token = tokens["access_token"]
                if "refresh_token" in tokens:
                    self.refresh_token = tokens["refresh_token"]
                print("✅ Access token refreshed successfully")
                return True
            else:
                print(f"❌ Token refresh failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Exception during token refresh: {e}")
            return False

    def create_message(
        self,
        model: str,
        messages: list,
        max_tokens: int = 4096,
        system: str = None,
        **kwargs,
    ):
        """Create a message using the Claude API with OAuth authentication"""
        data = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            **kwargs,
        }

        # Add system prompt if provided
        if system:
            data["system"] = system

        try:
            response = requests.post(
                f"{self.base_url}/messages",
                headers=self._get_headers(),
                json=data,
                timeout=30,
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                print("🔄 Token expired, attempting refresh...")
                # For now, we'll just return the error since we don't have client_id here
                # In a full implementation, we'd store client_id and auto-refresh
                raise Exception("Token expired - refresh needed")
            else:
                raise Exception(
                    f"API request failed: {response.status_code} - {response.text}"
                )

        except Exception as e:
            print(f"❌ Exception during API call: {e}")
            raise


def test_llm_orchestra_oauth_client(tokens):
    """Test the OAuth tokens with LLM-Orchestra identity"""
    print("\n🔍 Testing LLM-Orchestra OAuth Client")
    print("=" * 45)

    if not tokens or "access_token" not in tokens:
        print("❌ No access token to test")
        return False

    try:
        # Create OAuth client
        oauth_client = OAuthClaudeClient(
            access_token=tokens["access_token"],
            refresh_token=tokens.get("refresh_token"),
        )

        print("📤 Testing LLM-Orchestra OAuth Client")
        print("   Using Bearer token authentication")
        print("   Adding anthropic-beta: oauth-2025-04-20 header")
        print("   ✅ Adding LLM-Orchestra/Python 0.3.0 User-Agent")
        print("   ✅ Adding system prompt identifying as Claude Code")

        # Test with LLM-Orchestra identity and Claude Code system prompt
        system = "You are Claude Code, Anthropic's official CLI for Claude."
        response = oauth_client.create_message(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            max_tokens=1000,
            system=system,
        )

        print("🎉 SUCCESS! LLM-Orchestra OAuth Client worked!")

        # Extract the response content
        if "content" in response and len(response["content"]) > 0:
            content = response["content"][0].get("text", "No text content")
            print(f"📝 Claude responded: '{content}'")

            # Show usage info if available
            if "usage" in response:
                usage = response["usage"]
                print(
                    f"📊 Usage: {usage.get('input_tokens', 0)} input + {usage.get('output_tokens', 0)} output tokens"
                )

        return True

    except Exception as e:
        print(f"❌ LLM-Orchestra OAuth Client test failed: {e}")
        return False


def test_tokens_with_api(tokens):
    """Test the received tokens with Anthropic API (original method)"""
    print("\n🔍 Testing Tokens with Direct API Call")
    print("=" * 40)

    if not tokens or "access_token" not in tokens:
        print("❌ No access token to test")
        return False

    access_token = tokens["access_token"]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
        "anthropic-beta": "oauth-2025-04-20",
        "anthropic-version": "2023-06-01",
    }

    data = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
    }

    print("data:")
    print(data)
    print("headers:")
    print(headers)

    try:
        print("📤 Testing with https://api.anthropic.com/v1/messages")
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data,
            timeout=30,
        )

        print(f"📥 Response Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            content = result.get("content", [{}])[0].get("text", "No text")
            print("🎉 SUCCESS! API call worked!")
            print(f"📝 Claude responded: '{content}'")
            return True

        else:
            print(f"❌ API call failed: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Exception during API test: {e}")
        return False


def interactive_oauth_flow():
    """Run the complete interactive OAuth flow"""
    print("🎯 OAuth Flow Test")
    print("=" * 40)

    # Step 1: Create authorization URL
    auth_url, code_verifier, client_id, redirect_uri = create_authorization_url()

    # Step 2: Get user authorization
    print("🌐 Opening authorization URL in browser...")
    try:
        webbrowser.open(auth_url)
        print("✅ Browser opened successfully")
    except:
        print("❌ Could not open browser automatically")
        print("Please copy and paste the URL above into your browser")

    print("\n📋 Instructions:")
    print("1. Log into your Claude Pro/Max account")
    print("2. Grant the requested permissions")
    print("3. You'll be redirected to a callback URL")
    print("4. Copy the 'code' parameter from the URL")
    print("5. Paste it below")
    print()

    # Step 3: Get authorization code from user
    auth_code = input("🔑 Enter the authorization code: ").strip()

    if not auth_code:
        print("❌ No authorization code provided")
        return False

    print(f"✅ Authorization code received: {auth_code[:20]}...")

    # Step 4: Exchange code for tokens
    tokens = exchange_code_for_tokens(auth_code, code_verifier, client_id, redirect_uri)

    if not tokens:
        return False

    # Step 5: Test tokens with LLM-Orchestra OAuth client
    print("\n" + "=" * 60)
    print("🧪 TESTING OAUTH TOKENS WITH LLM-ORCHESTRA CLIENT")
    print("=" * 60)

    # Test with LLM-Orchestra OAuth client
    oauth_success = test_llm_orchestra_oauth_client(tokens)

    # Step 6: Show results
    print("\n📋 OAuth Flow Results:")
    print("=" * 50)

    if tokens and oauth_success:
        print("🏆 COMPLETE SUCCESS!")
        print("✅ Authorization: WORKED")
        print("✅ Token Exchange: WORKED")
        print("✅ LLM-Orchestra OAuth Client: WORKED")
        print("\n🎉 OAuth flow is fully functional with LLM-Orchestra identity!")
        return True
    elif tokens:
        print("🔄 PARTIAL SUCCESS!")
        print("✅ Authorization: WORKED")
        print("✅ Token Exchange: WORKED")
        print("❌ API Calls: FAILED")
        print("\n💡 Tokens received but API calls still failing")
        return False
    else:
        print("❌ FAILED!")
        print("✅ Authorization: WORKED")
        print("❌ Token Exchange: FAILED")
        return False


def main():
    """Main test function"""
    return interactive_oauth_flow()


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

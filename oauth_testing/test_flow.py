#!/usr/bin/env python3
"""
Test the OAuth flow for Claude Pro/Max authentication
"""

import json
import requests
import secrets
import base64
import hashlib
import webbrowser
from pathlib import Path
from urllib.parse import urlencode, parse_qs, urlparse


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

    print(f"📋 OAuth Configuration:")
    print(f"   • Client ID: {client_id}")
    print(f"   • Redirect URI: {redirect_uri}")
    print(f"   • Scope: {scope}")
    print(f"   • Code Verifier: {code_verifier[:20]}...")
    print(f"   • Code Challenge: {code_challenge[:20]}...")
    print()
    print(f"🔗 Authorization URL:")
    print(f"   {auth_url}")
    print()

    return auth_url, code_verifier, client_id, redirect_uri


def exchange_code_for_tokens(auth_code, code_verifier, client_id, redirect_uri):
    print("🔄 Exchanging Authorization Code for Tokens")
    print("=" * 45)

    splits = auth_code.split("#")
    if len(splits) != 2:
        print(f"❌ Invalid authorization code format - expected format: code#state")
        print(f"   Received: {auth_code}")
        return None

    code_part = splits[0]
    state_part = splits[1]

    print(f"📋 Parsing authorization code:")
    print(f"   • Full code: {auth_code}")
    print(f"   • Code part: {code_part}")
    print(f"   • State part: {state_part}")
    print(f"   • Expected state (verifier): {code_verifier}")

    # Verify state matches our code verifier
    if state_part != code_verifier:
        print(f"⚠️  State mismatch - this might cause issues")
        print(f"   • Received state: {state_part}")
        print(f"   • Expected state: {code_verifier}")
    else:
        print(f"✅ State matches code verifier")
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
    print(f"📋 Request data:")
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

            print(f"🎉 SUCCESS! Token exchange worked!")
            print(f"📋 Received tokens:")

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
            print(f"❌ Token exchange failed!")
            print(f"📄 Response: {response.text}")
            return None

    except Exception as e:
        print(f"❌ Exception during token exchange: {e}")
        return None


def test_tokens_with_api(tokens):
    """Test the received tokens with Anthropic API"""
    print("\n🔍 Testing Tokens with Anthropic API")
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
            print(f"🎉 SUCCESS! API call worked!")
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

    # Step 5: Test tokens with API
    api_success = test_tokens_with_api(tokens)

    # Step 6: Show results
    print(f"\n📋 OAuth Flow Results:")
    if tokens and api_success:
        print("🏆 COMPLETE SUCCESS!")
        print("✅ Authorization: WORKED")
        print("✅ Token Exchange: WORKED")
        print("✅ API Calls: WORKED")
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

#!/usr/bin/env python3
"""
Simple script to get a new authorization code for OAuth testing
"""

import webbrowser
from llm_orc.authentication import AnthropicOAuthFlow

def get_new_auth_code():
    """Get a fresh authorization code"""
    print("🚀 Getting Fresh Authorization Code")
    print("=" * 40)
    
    # Create OAuth flow
    client_id = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
    flow = AnthropicOAuthFlow(client_id, "")
    
    # Generate authorization URL
    auth_url = flow.get_authorization_url()
    
    print(f"📋 OAuth Configuration:")
    print(f"   • Client ID: {client_id}")
    print(f"   • Redirect URI: {flow.redirect_uri}")
    print(f"   • Code Verifier: {flow.code_verifier[:20]}...")
    print()
    
    print(f"🌐 Opening browser...")
    print(f"   URL: {auth_url}")
    
    # Open browser
    try:
        webbrowser.open(auth_url)
        print("✅ Browser opened successfully")
    except Exception as e:
        print(f"❌ Failed to open browser: {e}")
        print(f"Manual URL: {auth_url}")
    
    print()
    print("📋 Next Steps:")
    print("1. Complete OAuth consent in your browser")
    print("2. You'll be redirected to: https://console.anthropic.com/oauth/code/callback")
    print("3. Copy the 'code' parameter from the URL")
    print("4. Come back here and run the token exchange test")
    print()
    
    print("🔄 After you get the code, test it with:")
    print("   uv run python test_token_exchange.py")
    print()
    
    return True

if __name__ == "__main__":
    get_new_auth_code()
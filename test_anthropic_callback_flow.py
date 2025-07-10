#!/usr/bin/env python3
"""
Test the new Anthropic OAuth flow using console.anthropic.com/oauth/code/callback
This avoids Cloudflare protection by using Anthropic's own callback endpoint.
"""

from llm_orc.authentication import AnthropicOAuthFlow

def test_anthropic_callback_oauth():
    """Test the OAuth flow with Anthropic's callback endpoint"""
    print("🚀 Testing Anthropic OAuth with Native Callback Endpoint")
    print("=" * 60)
    
    # Use the shared client ID
    client_id = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
    client_secret = ""  # Not needed for PKCE
    
    print(f"📋 Configuration:")
    print(f"   • Client ID: {client_id}")
    print(f"   • Redirect URI: https://console.anthropic.com/oauth/code/callback")
    print(f"   • Flow Type: Authorization Code with PKCE")
    
    # Create OAuth flow
    flow = AnthropicOAuthFlow(client_id, client_secret)
    
    print(f"\n📋 OAuth Flow Details:")
    print(f"   • Provider: {flow.provider}")
    print(f"   • Redirect URI: {flow.redirect_uri}")
    print(f"   • Code Verifier: {flow.code_verifier[:20]}...")
    print(f"   • Code Challenge: {flow.code_challenge[:20]}...")
    
    # Generate authorization URL
    auth_url = flow.get_authorization_url()
    print(f"\n🌐 Authorization URL Generated:")
    print(f"   {auth_url}")
    
    print(f"\n🔍 URL Analysis:")
    from urllib.parse import urlparse, parse_qs
    parsed = urlparse(auth_url)
    params = parse_qs(parsed.query)
    
    print(f"   • Host: {parsed.netloc}")
    print(f"   • Path: {parsed.path}")
    print(f"   • Client ID: {params.get('client_id', ['None'])[0]}")
    print(f"   • Redirect URI: {params.get('redirect_uri', ['None'])[0]}")
    print(f"   • Response Type: {params.get('response_type', ['None'])[0]}")
    print(f"   • Scope: {params.get('scope', ['None'])[0]}")
    print(f"   • Code Challenge Method: {params.get('code_challenge_method', ['None'])[0]}")
    
    print(f"\n✅ OAuth URL Validation:")
    print(f"   • Uses console.anthropic.com: {'✓' if 'console.anthropic.com' in auth_url else '✗'}")
    print(f"   • Has code challenge: {'✓' if 'code_challenge=' in auth_url else '✗'}")
    print(f"   • Uses PKCE (S256): {'✓' if 'code_challenge_method=S256' in auth_url else '✗'}")
    print(f"   • Correct redirect URI: {'✓' if 'console.anthropic.com%2Foauth%2Fcode%2Fcallback' in auth_url else '✗'}")
    
    print(f"\n🎯 Expected Flow:")
    print(f"   1. User clicks the authorization URL above")
    print(f"   2. Completes OAuth consent on console.anthropic.com")
    print(f"   3. Gets redirected to: https://console.anthropic.com/oauth/code/callback?code=...")
    print(f"   4. User manually extracts the authorization code from the URL")
    print(f"   5. Code is exchanged for access tokens (may still hit Cloudflare)")
    
    print(f"\n📋 Next Steps:")
    print(f"   • The authorization URL is ready to use")
    print(f"   • This avoids localhost callback server issues")
    print(f"   • Token exchange may still need manual extraction if Cloudflare blocks API calls")
    print(f"   • Consider implementing token extraction from browser storage as fallback")

if __name__ == "__main__":
    test_anthropic_callback_oauth()
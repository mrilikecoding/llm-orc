#!/usr/bin/env python3
"""
Generate PKCE parameters and curl command for OAuth token exchange testing
"""

import webbrowser
from llm_orc.authentication import AnthropicOAuthFlow

def generate_oauth_params():
    """Generate OAuth parameters and curl command"""
    print("🔧 Generating OAuth PKCE Parameters for Curl Testing")
    print("=" * 60)
    
    # Create OAuth flow to get fresh PKCE parameters
    client_id = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
    flow = AnthropicOAuthFlow(client_id, "")
    
    print("📋 Generated OAuth Parameters:")
    print(f"   • Client ID: {flow.client_id}")
    print(f"   • Code Verifier: {flow.code_verifier}")
    print(f"   • Code Challenge: {flow.code_challenge}")
    print(f"   • State: {flow.state}")
    print(f"   • Redirect URI: {flow.redirect_uri}")
    print()
    
    # Generate authorization URL
    auth_url = flow.get_authorization_url()
    print("🌐 Step 1: Get Authorization Code")
    print("-" * 35)
    print("Opening browser to get authorization code...")
    print(f"URL: {auth_url}")
    
    try:
        webbrowser.open(auth_url)
        print("✅ Browser opened")
    except:
        print("❌ Failed to open browser - use URL above")
    
    print()
    print("Complete OAuth in browser, then copy the authorization code from:")
    print("https://console.anthropic.com/oauth/code/callback?code=YOUR_CODE_HERE...")
    print()
    
    # Get authorization code
    auth_code = input("Enter the authorization code: ").strip()
    
    if not auth_code:
        print("❌ No authorization code provided")
        print("You can still use the PKCE parameters above with your own code")
        return
    
    # Clean up auth code (remove any # or state parameters)
    if '#' in auth_code:
        auth_code = auth_code.split('#')[0]
    if '&' in auth_code:
        auth_code = auth_code.split('&')[0]
    
    print(f"\n✅ Authorization code: {auth_code}")
    print()
    
    # Generate curl commands for different endpoints
    print("🔧 Step 2: Test Token Exchange with Curl")
    print("-" * 42)
    
    endpoints = [
        "https://a-api.anthropic.com/oauth/token",
        "https://console.anthropic.com/oauth/token",
        "https://api.anthropic.com/oauth/token"
    ]
    
    for i, endpoint in enumerate(endpoints, 1):
        print(f"\nTest {i}: {endpoint}")
        print("-" * (len(endpoint) + 10))
        
        curl_command = f'''curl -X POST "{endpoint}" \\
  -H "Content-Type: application/x-www-form-urlencoded" \\
  -H "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" \\
  -H "Accept: application/json" \\
  -d "grant_type=authorization_code" \\
  -d "client_id={flow.client_id}" \\
  -d "code={auth_code}" \\
  -d "code_verifier={flow.code_verifier}" \\
  -d "redirect_uri={flow.redirect_uri}"'''
        
        print(curl_command)
    
    print()
    print("📋 Expected Results:")
    print("• 200 + JSON with access_token = SUCCESS! 🎉")
    print("• 404 = Endpoint doesn't exist")
    print("• 403 = Cloudflare protection")
    print("• 400 = Invalid request parameters")
    print("• 401 = Authentication issue")

def show_manual_params():
    """Just show the PKCE parameters without browser interaction"""
    print("🔧 Manual PKCE Parameter Generation")
    print("=" * 40)
    
    flow = AnthropicOAuthFlow("9d1c250a-e61b-44d9-88ed-5944d1962f5e", "")
    
    print("📋 PKCE Parameters:")
    print(f"Client ID: {flow.client_id}")
    print(f"Code Verifier: {flow.code_verifier}")
    print(f"Code Challenge: {flow.code_challenge}")
    print(f"State: {flow.state}")
    print(f"Redirect URI: {flow.redirect_uri}")
    print()
    
    print("🌐 Authorization URL:")
    print(flow.get_authorization_url())
    print()
    
    print("🔧 Use these parameters in your curl command:")
    print(f'  -d "code_verifier={flow.code_verifier}"')

if __name__ == "__main__":
    print("🎯 OAuth PKCE Parameters for Curl Testing")
    print("=" * 45)
    
    choice = input("1) Full flow with browser\n2) Just show parameters\nChoice (1/2): ").strip()
    
    if choice == "2":
        show_manual_params()
    else:
        generate_oauth_params()
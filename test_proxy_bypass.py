#!/usr/bin/env python3
"""
Test OAuth token exchange with proxy bypass options
"""

import requests
import os
from llm_orc.authentication import AnthropicOAuthFlow

def test_with_proxy_bypass():
    """Test token exchange with various proxy bypass methods"""
    print("🔧 Testing OAuth Token Exchange with Proxy Bypass")
    print("=" * 55)
    
    # Check current proxy settings
    print("📋 Current Environment:")
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY']
    for var in proxy_vars:
        value = os.environ.get(var, 'Not set')
        print(f"   • {var}: {value}")
    print()
    
    # Get auth code
    auth_code = input("Enter your authorization code: ").strip()
    if not auth_code:
        print("❌ No authorization code provided")
        return
    
    # Set up OAuth flow
    client_id = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
    flow = AnthropicOAuthFlow(client_id, "")
    
    # Prepare token exchange data
    data = {
        "grant_type": "authorization_code",
        "client_id": flow.client_id,
        "code": auth_code,
        "code_verifier": flow.code_verifier,
        "redirect_uri": flow.redirect_uri,
    }
    
    # Test different endpoints and proxy configurations
    endpoints = [
        "https://a-api.anthropic.com/oauth/token",
        "https://console.anthropic.com/oauth/token", 
        "https://api.anthropic.com/oauth/token",
    ]
    
    for endpoint in endpoints:
        print(f"🔍 Testing endpoint: {endpoint}")
        
        # Test 1: With system proxy settings
        try:
            print("   Attempt 1: System proxy settings")
            response = requests.post(
                endpoint,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=10
            )
            print(f"   ✅ Success! Status: {response.status_code}")
            if response.status_code == 200:
                print(f"   🎉 TOKEN EXCHANGE WORKED!")
                tokens = response.json()
                print(f"   Response keys: {list(tokens.keys())}")
                return tokens
            else:
                print(f"   Response: {response.text[:200]}...")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        # Test 2: Bypass proxy
        try:
            print("   Attempt 2: Bypassing proxy")
            response = requests.post(
                endpoint,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                proxies={'http': None, 'https': None},  # Bypass proxy
                timeout=10
            )
            print(f"   ✅ Success! Status: {response.status_code}")
            if response.status_code == 200:
                print(f"   🎉 TOKEN EXCHANGE WORKED!")
                tokens = response.json()
                print(f"   Response keys: {list(tokens.keys())}")
                return tokens
            else:
                print(f"   Response: {response.text[:200]}...")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        print()
    
    print("❌ All endpoints and proxy configurations failed")
    return None

if __name__ == "__main__":
    result = test_with_proxy_bypass()
    if result:
        print("\n🏆 SUCCESS! OAuth token exchange completed!")
    else:
        print("\n💡 Try running with proxy environment variables unset:")
        print("   unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY")
        print("   uv run python test_proxy_bypass.py")
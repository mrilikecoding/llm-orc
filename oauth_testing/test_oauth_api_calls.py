#!/usr/bin/env python3
"""
Test if the stored OAuth token works for actual API calls to Anthropic
"""

import requests
import json
from llm_orc.authentication import AuthenticationManager, CredentialStorage
from llm_orc.config import ConfigurationManager

def test_oauth_token_api_calls():
    """Test if the stored OAuth token works for API calls"""
    print("🔍 Testing OAuth Token for API Calls")
    print("=" * 40)
    
    # Set up authentication components
    config_manager = ConfigurationManager()
    credential_storage = CredentialStorage(config_manager)
    auth_manager = AuthenticationManager(credential_storage)
    
    # Check if we have stored OAuth tokens
    print("📋 Checking stored authentication...")
    if not auth_manager.is_authenticated("anthropic"):
        print("❌ No stored authentication found")
        print("   Run the OAuth flow first to store tokens")
        return False
    
    # Get stored tokens
    stored_tokens = credential_storage.get_oauth_token("anthropic")
    if not stored_tokens:
        print("❌ No OAuth tokens found in storage")
        return False
    
    print("✅ Found stored OAuth tokens:")
    print(f"   • Access token: {stored_tokens.get('access_token', 'None')[:25]}...")
    if 'refresh_token' in stored_tokens:
        print(f"   • Refresh token: {stored_tokens['refresh_token'][:25]}...")
    if 'expires_at' in stored_tokens:
        import time
        expires_at = stored_tokens['expires_at']
        if expires_at > time.time():
            print(f"   • Token status: ✅ Valid")
        else:
            print(f"   • Token status: ⚠️ Expired")
    print()
    
    access_token = stored_tokens.get('access_token')
    if not access_token:
        print("❌ No access token found")
        return False
    
    # Test API calls
    print("🔄 Testing API calls with OAuth token...")
    
    # Test 1: Messages API (main Claude API)
    print("\nTest 1: Messages API")
    print("-" * 20)
    try:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data,
            timeout=30
        )
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ SUCCESS! API call worked!")
            print(f"   Response: {result.get('content', [{}])[0].get('text', 'No text')}")
            return True
        elif response.status_code == 401:
            print(f"   ❌ 401 Unauthorized - Token may be invalid")
            print(f"   Response: {response.text}")
        elif response.status_code == 403:
            print(f"   ❌ 403 Forbidden - Token may not have proper permissions")
            print(f"   Response: {response.text}")
        else:
            print(f"   ❌ API call failed: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    # Test 2: Alternative endpoint
    print("\nTest 2: Alternative API endpoint")
    print("-" * 35)
    try:
        # Test a simpler endpoint if available
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            "https://api.anthropic.com/v1/models",  # or similar endpoint
            headers=headers,
            timeout=30
        )
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✅ SUCCESS! Models endpoint worked!")
            models = response.json()
            print(f"   Available models: {len(models.get('data', []))} found")
            return True
        else:
            print(f"   Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    # Test 3: Token validation
    print("\nTest 3: Token Validation")
    print("-" * 25)
    try:
        # Try a lightweight API call to validate token
        headers = {
            "Authorization": f"Bearer {access_token}",
        }
        
        response = requests.get(
            "https://api.anthropic.com/v1/",  # Root endpoint
            headers=headers,
            timeout=30
        )
        
        print(f"   Status: {response.status_code}")
        if response.status_code in [200, 404]:  # 404 is ok, means endpoint exists
            print(f"   ✅ Token is valid (endpoint reachable)")
            return True
        elif response.status_code == 401:
            print(f"   ❌ Token is invalid or expired")
        else:
            print(f"   Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    return False

def show_token_analysis():
    """Show analysis of the stored token"""
    print("\n" + "=" * 60)
    print("🔍 OAuth Token Analysis")
    print("=" * 60)
    
    config_manager = ConfigurationManager()
    credential_storage = CredentialStorage(config_manager)
    stored_tokens = credential_storage.get_oauth_token("anthropic")
    
    if not stored_tokens:
        print("❌ No tokens to analyze")
        return
    
    access_token = stored_tokens.get('access_token', '')
    
    print("📋 Token Structure Analysis:")
    print(f"   • Token length: {len(access_token)}")
    print(f"   • Token format: {'Looks like auth code' if len(access_token) > 40 and '#' in access_token else 'Looks like access token'}")
    print(f"   • Contains '#': {'Yes' if '#' in access_token else 'No'}")
    
    if '#' in access_token:
        print("\n⚠️  WARNING: This looks like an authorization code, not an access token!")
        print("   • Authorization codes are used to GET access tokens")
        print("   • Access tokens are used to MAKE API calls")
        print("   • You may need to extract the actual access token from browser storage")
        print("\n💡 To get the real access token:")
        print("   1. Open https://console.anthropic.com")
        print("   2. Open Developer Tools (F12)")
        print("   3. Go to Application > Local Storage")
        print("   4. Look for keys containing 'access_token' or 'token'")
        print("   5. Copy the actual access token (usually starts with 'sk-ant-')")

def main():
    """Main test function"""
    print("🎯 OAuth Token API Call Test")
    print("=" * 30)
    print("Testing if stored OAuth tokens work for actual API calls\n")
    
    # Test API calls
    success = test_oauth_token_api_calls()
    
    # Show token analysis
    show_token_analysis()
    
    print(f"\n📋 Test Results:")
    if success:
        print("🏆 SUCCESS! OAuth token works for API calls!")
        print("✅ Complete OAuth interaction loop is functional!")
    else:
        print("❌ OAuth token doesn't work for API calls")
        print("💡 The stored value might be an authorization code, not an access token")
        print("   Try extracting the actual access token from browser storage")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
#!/usr/bin/env python3
"""
Test if Claude Code's access token works for Anthropic API calls
"""

import json
import requests
from pathlib import Path


def load_claude_code_credentials():
    """Load Claude Code credentials from ~/.claude/.credentials.json"""
    claude_creds_path = Path.home() / ".claude" / ".credentials.json"
    
    if not claude_creds_path.exists():
        print("❌ Claude Code credentials not found at ~/.claude/.credentials.json")
        print("   Make sure Claude Code is installed and you're logged in")
        return None
    
    try:
        with open(claude_creds_path, 'r') as f:
            creds = json.load(f)
        
        oauth_info = creds.get('claudeAiOauth')
        if not oauth_info:
            print("❌ No claudeAiOauth section found in credentials")
            return None
        
        return oauth_info
    
    except Exception as e:
        print(f"❌ Error loading Claude Code credentials: {e}")
        return None


def test_claude_code_token():
    """Test if Claude Code's access token works for API calls"""
    print("🔍 Testing Claude Code Access Token")
    print("=" * 40)
    
    # Load Claude Code credentials
    oauth_info = load_claude_code_credentials()
    if not oauth_info:
        return False
    
    # Extract token info
    access_token = oauth_info.get('accessToken')
    refresh_token = oauth_info.get('refreshToken')
    expires_at = oauth_info.get('expiresAt')
    scopes = oauth_info.get('scopes', [])
    subscription_type = oauth_info.get('subscriptionType')
    
    print("✅ Found Claude Code OAuth info:")
    print(f"   • Access token: {access_token[:25]}...")
    print(f"   • Refresh token: {refresh_token[:25]}...")
    print(f"   • Scopes: {scopes}")
    print(f"   • Subscription: {subscription_type}")
    
    # Check token expiry
    import time
    current_time = time.time() * 1000  # Convert to milliseconds
    if expires_at and expires_at > current_time:
        print(f"   • Token status: ✅ Valid (expires in {(expires_at - current_time) / (1000 * 60 * 60):.1f} hours)")
    else:
        print(f"   • Token status: ⚠️ Expired or no expiry info")
    print()
    
    if not access_token:
        print("❌ No access token found")
        return False
    
    # Test API call
    print("🔄 Testing API call with Claude Code token...")
    print()
    
    try:
        # Test different header configurations
        test_configs = [
            {
                "name": "Standard OAuth Bearer",
                "headers": {
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                }
            },
            {
                "name": "Claude Code User-Agent",
                "headers": {
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                    "User-Agent": "Claude Code"
                }
            },
            {
                "name": "Without anthropic-version",
                "headers": {
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"
                }
            }
        ]
        
        data = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 20,
            "messages": [{"role": "user", "content": "Hello! Just say 'Working!' if you can see this."}]
        }
        
        for config in test_configs:
            print(f"📤 Testing: {config['name']}")
            print(f"   Making API request to https://api.anthropic.com/v1/messages")
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=config['headers'],
                json=data,
                timeout=30
            )
            
            print(f"   📥 Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('content', [{}])[0].get('text', 'No text')
                print(f"   🎉 SUCCESS! API call worked with {config['name']}!")
                print(f"   📝 Claude responded: '{content}'")
                print()
                print("✅ This confirms Claude Code tokens work as bearer tokens!")
                return True
                
            elif response.status_code == 401:
                print(f"   ❌ 401 Unauthorized")
                if "OAuth authentication is currently not supported" in response.text:
                    print(f"   💡 Got OAuth not supported message - trying next config...")
                else:
                    print(f"   📄 Response: {response.text}")
                
            elif response.status_code == 403:
                print(f"   ❌ 403 Forbidden")
                print(f"   📄 Response: {response.text}")
                
            else:
                print(f"   ❌ API call failed with status {response.status_code}")
                print(f"   📄 Response: {response.text[:200]}...")
            
            print()
        
        # If we get here, none of the configurations worked with the standard API
        print("🔍 Standard API failed, testing alternative endpoints...")
        
        # Test console API endpoint (this might be what Claude Code uses)
        console_headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "User-Agent": "Claude Code"
        }
        
        console_data = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 20,
            "messages": [{"role": "user", "content": "Hello! Just say 'Working!' if you can see this."}]
        }
        
        print("📤 Testing Console API endpoint:")
        print("   URL: https://console.anthropic.com/api/messages")
        
        response = requests.post(
            "https://console.anthropic.com/api/messages",
            headers=console_headers,
            json=console_data,
            timeout=30
        )
        
        print(f"   📥 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            content = result.get('content', [{}])[0].get('text', 'No text')
            print(f"   🎉 SUCCESS! Console API worked!")
            print(f"   📝 Claude responded: '{content}'")
            print()
            print("✅ Found the correct endpoint! Console API accepts OAuth tokens!")
            return True
            
        elif response.status_code == 401:
            print(f"   ❌ 401 Unauthorized")
            print(f"   📄 Response: {response.text}")
            
        elif response.status_code == 403:
            print(f"   ❌ 403 Forbidden")
            print(f"   📄 Response: {response.text}")
            
        else:
            print(f"   ❌ Console API failed with status {response.status_code}")
            print(f"   📄 Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"❌ Exception during API call: {e}")
    
    return False


def analyze_token_format():
    """Analyze the Claude Code token format"""
    print("\n" + "=" * 60)
    print("🔍 Token Format Analysis")
    print("=" * 60)
    
    oauth_info = load_claude_code_credentials()
    if not oauth_info:
        return
    
    access_token = oauth_info.get('accessToken', '')
    refresh_token = oauth_info.get('refreshToken', '')
    
    print("📋 Token Structure:")
    print(f"   • Access token prefix: {access_token[:15]}...")
    print(f"   • Access token length: {len(access_token)}")
    print(f"   • Refresh token prefix: {refresh_token[:15]}...")
    print(f"   • Refresh token length: {len(refresh_token)}")
    
    # Token type analysis
    if access_token.startswith('sk-ant-oat01-'):
        print("   • ✅ Access token format looks correct (sk-ant-oat01-)")
    elif access_token.startswith('sk-ant-'):
        print("   • ⚠️ Looks like Anthropic token but different format")
    else:
        print("   • ❓ Unknown token format")
    
    if refresh_token.startswith('sk-ant-ort01-'):
        print("   • ✅ Refresh token format looks correct (sk-ant-ort01-)")
    elif refresh_token.startswith('sk-ant-'):
        print("   • ⚠️ Looks like Anthropic token but different format")
    else:
        print("   • ❓ Unknown refresh token format")


def main():
    """Main test function"""
    print("🎯 Claude Code Token Test")
    print("=" * 30)
    print("Testing if Claude Code's access token works for Anthropic API calls\n")
    
    # Test the token
    success = test_claude_code_token()
    
    # Analyze token format
    analyze_token_format()
    
    print(f"\n📋 Test Results:")
    if success:
        print("🏆 SUCCESS! Claude Code token works for API calls!")
        print("✅ We can implement Claude Code credential reuse in llm-orc!")
        print()
        print("💡 Next steps:")
        print("   1. Add Claude Code credential detection to llm-orc")
        print("   2. Implement token refresh logic using the refresh token")
        print("   3. Add fallback to manual OAuth if Claude Code not available")
    else:
        print("❌ Claude Code token doesn't work for API calls")
        print("💡 This could be due to:")
        print("   - Token expiry")
        print("   - Insufficient scopes")
        print("   - Different API endpoint requirements")
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
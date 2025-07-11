#!/usr/bin/env python3
"""
Helper script to store the real access token from browser storage
"""

from llm_orc.authentication import AuthenticationManager, CredentialStorage
from llm_orc.config import ConfigurationManager

def get_real_access_token():
    """Get and store the real access token from browser storage"""
    print("🔑 Getting Real Access Token from Browser Storage")
    print("=" * 55)
    
    print("📋 Instructions to get the real access token:")
    print("1. Open https://console.anthropic.com in your browser")
    print("2. Make sure you're logged in")
    print("3. Open Developer Tools (F12 or Cmd+Option+I)")
    print("4. Go to Application tab > Local Storage > console.anthropic.com")
    print("5. Look for keys containing:")
    print("   • 'access_token'")
    print("   • 'token'") 
    print("   • 'auth'")
    print("   • Anything that starts with 'sk-ant-'")
    print()
    print("💡 Access tokens usually:")
    print("   • Start with 'sk-ant-'")
    print("   • Are much longer than authorization codes")
    print("   • Don't contain '#' characters")
    print()
    
    # Get the real access token
    print("🔧 Enter the real access token below:")
    access_token = input("Access Token: ").strip()
    
    if not access_token:
        print("❌ No access token provided")
        return False
    
    # Validate token format
    print(f"\n📋 Token Analysis:")
    print(f"   • Length: {len(access_token)}")
    print(f"   • Starts with 'sk-ant-': {'Yes' if access_token.startswith('sk-ant-') else 'No'}")
    print(f"   • Contains '#': {'Yes' if '#' in access_token else 'No'}")
    
    if access_token.startswith('sk-ant-') and '#' not in access_token:
        print("✅ Token format looks correct!")
    else:
        print("⚠️  Token format might not be correct")
        proceed = input("Continue anyway? (y/N): ").strip().lower()
        if proceed not in ['y', 'yes']:
            print("❌ Cancelled")
            return False
    
    # Store the real access token
    print(f"\n💾 Storing access token...")
    try:
        config_manager = ConfigurationManager()
        credential_storage = CredentialStorage(config_manager)
        auth_manager = AuthenticationManager(credential_storage)
        
        success = auth_manager.store_manual_oauth_token(
            provider="anthropic",
            access_token=access_token,
            refresh_token=None,
            expires_in=3600
        )
        
        if success:
            print("✅ Real access token stored successfully!")
            return True
        else:
            print("❌ Failed to store access token")
            return False
            
    except Exception as e:
        print(f"❌ Error storing token: {e}")
        return False

def test_stored_token():
    """Quick test of the stored token"""
    print(f"\n🔄 Testing stored access token...")
    
    import subprocess
    import sys
    
    try:
        result = subprocess.run([
            sys.executable, "-c", 
            "from test_oauth_api_calls import test_oauth_token_api_calls; test_oauth_token_api_calls()"
        ], capture_output=True, text=True, timeout=30)
        
        if "SUCCESS" in result.stdout:
            print("✅ Access token works for API calls!")
            return True
        else:
            print("❌ Access token test failed")
            print(f"Output: {result.stdout}")
            return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main function"""
    print("🎯 Extract and Store Real Access Token")
    print("=" * 40)
    print("This will help you get the real access token from browser storage")
    print("and replace the authorization code that was stored earlier.\n")
    
    # Get real access token
    success = get_real_access_token()
    
    if success:
        print(f"\n🎉 Success! Now let's test if it works...")
        
        # Test the token
        test_success = test_stored_token()
        
        if test_success:
            print(f"\n🏆 COMPLETE SUCCESS!")
            print("✅ Real access token extracted and working")
            print("✅ OAuth interaction loop 100% closed!")
            print("✅ API calls working with OAuth token!")
        else:
            print(f"\n⚠️  Token stored but API test failed")
            print("💡 Try running: uv run python test_oauth_api_calls.py")
    else:
        print(f"\n❌ Failed to get real access token")
        print("💡 Make sure you're logged into console.anthropic.com")
        print("💡 Check browser storage carefully for access tokens")

if __name__ == "__main__":
    main()
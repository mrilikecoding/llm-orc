#!/usr/bin/env python3
"""
Test the complete OAuth interaction loop with Anthropic's native callback endpoint.
This tests the full end-to-end flow to see if we can close the interaction loop.
"""

import sys
from llm_orc.authentication import AuthenticationManager, CredentialStorage
from llm_orc.config import ConfigurationManager

def test_complete_oauth_loop():
    """Test the complete OAuth interaction loop end-to-end"""
    print("🚀 Testing Complete OAuth Interaction Loop")
    print("=" * 60)
    print("This will test the full OAuth flow with Anthropic's native callback endpoint")
    print("to see if we can successfully close the interaction loop.")
    print()
    
    # Set up authentication components
    print("📋 Setting up authentication components...")
    config_manager = ConfigurationManager()
    credential_storage = CredentialStorage(config_manager)
    auth_manager = AuthenticationManager(credential_storage)
    
    print("✅ Authentication components initialized")
    
    # Check for existing authentication
    print(f"\n🔍 Checking existing authentication...")
    if auth_manager.is_authenticated("anthropic"):
        print("✅ Already authenticated with Anthropic")
        client = auth_manager.get_authenticated_client("anthropic")
        if client:
            print(f"   • Client type: {type(client).__name__}")
            print(f"   • Has access token: {hasattr(client, 'access_token')}")
            if hasattr(client, 'access_token'):
                token = getattr(client, 'access_token', '')
                print(f"   • Token preview: {token[:20]}...")
        return True
    else:
        print("❌ Not currently authenticated with Anthropic")
    
    # Test OAuth flow
    print(f"\n🔧 Starting OAuth authentication flow...")
    print(f"This will use the new native callback endpoint approach.")
    print()
    
    # Use the shared client ID from our testing
    client_id = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
    client_secret = ""  # Not needed for PKCE flow
    
    print(f"📋 OAuth Configuration:")
    print(f"   • Provider: anthropic")
    print(f"   • Client ID: {client_id}")
    print(f"   • Callback: https://console.anthropic.com/oauth/code/callback")
    print(f"   • Flow: Authorization Code with PKCE")
    print()
    
    # Attempt OAuth authentication
    print("🌐 Starting OAuth flow...")
    try:
        success = auth_manager.authenticate_oauth(
            provider="anthropic",
            client_id=client_id,
            client_secret=client_secret
        )
        
        if success:
            print("\n🎉 OAuth authentication successful!")
            print("✅ Complete interaction loop closed successfully")
            
            # Verify authentication
            if auth_manager.is_authenticated("anthropic"):
                client = auth_manager.get_authenticated_client("anthropic")
                print(f"   • Authenticated client: {type(client).__name__}")
                print(f"   • Access token available: {hasattr(client, 'access_token')}")
                
                # Check stored credentials
                stored_tokens = credential_storage.get_oauth_token("anthropic")
                if stored_tokens:
                    print(f"   • Tokens stored: ✅")
                    print(f"   • Token keys: {list(stored_tokens.keys())}")
                else:
                    print(f"   • Tokens stored: ❌")
                
                return True
            else:
                print("❌ Authentication verification failed")
                return False
        else:
            print("\n❌ OAuth authentication failed")
            print("💡 This may be due to:")
            print("   • Token exchange endpoint still protected by Cloudflare")
            print("   • Manual token extraction required")
            print("   • API endpoint not yet publicly available")
            print()
            print("🔧 Potential solutions:")
            print("   1. Use manual token extraction from browser storage")
            print("   2. Create API key at https://console.anthropic.com/settings/keys")
            print("   3. Wait for official OAuth endpoint to become available")
            return False
            
    except KeyboardInterrupt:
        print("\n⏹️ OAuth flow interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 OAuth flow failed with exception: {e}")
        print(f"   Exception type: {type(e).__name__}")
        return False

def test_manual_token_input():
    """Test manual token input as a fallback"""
    print("\n" + "=" * 60)
    print("🔧 Testing Manual Token Input (Fallback Method)")
    print("=" * 60)
    
    config_manager = ConfigurationManager()
    credential_storage = CredentialStorage(config_manager)
    auth_manager = AuthenticationManager(credential_storage)
    
    print("If OAuth automation fails, you can manually input tokens:")
    print("1. Complete OAuth in browser")
    print("2. Extract tokens from browser storage or network requests")
    print("3. Input them here")
    print()
    
    # Get manual token input
    try:
        manual_token = input("Enter access token (or press Enter to skip): ").strip()
        if manual_token:
            # Store manual token
            success = auth_manager.store_manual_oauth_token(
                provider="anthropic",
                access_token=manual_token,
                refresh_token=None,
                expires_in=3600
            )
            
            if success:
                print("✅ Manual token stored successfully")
                print("✅ Interaction loop closed with manual token input")
                return True
            else:
                print("❌ Failed to store manual token")
                return False
        else:
            print("⏩ Skipped manual token input")
            return False
            
    except KeyboardInterrupt:
        print("\n⏹️ Manual token input interrupted")
        return False

def main():
    """Main test function"""
    print("🎯 Complete OAuth Interaction Loop Test")
    print("=" * 60)
    print("Testing both automated OAuth flow and manual fallback methods")
    print("to see if we can successfully close the interaction loop.")
    print()
    
    # Test automated OAuth flow
    oauth_success = test_complete_oauth_loop()
    
    if not oauth_success:
        print("\n🔄 Automated OAuth failed, testing manual fallback...")
        manual_success = test_manual_token_input()
        
        if manual_success:
            print("\n🎉 Interaction loop closed via manual token input!")
            return True
        else:
            print("\n💥 Both automated and manual methods failed")
            print("🔍 Next steps:")
            print("   • Check Anthropic's official OAuth documentation")
            print("   • Consider using API keys as alternative")
            print("   • Monitor for OAuth endpoint availability updates")
            return False
    else:
        print("\n🏆 Interaction loop successfully closed via automated OAuth!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
End-to-end test for Anthropic OAuth flow.
This script tests the complete OAuth implementation with real Anthropic authentication.
"""

import time
from llm_orc.authentication import AnthropicOAuthFlow, CredentialStorage
from llm_orc.config import ConfigurationManager

def test_oauth_flow():
    """Test the complete OAuth flow end-to-end."""
    print("🧪 Starting end-to-end OAuth test...")
    print("=" * 50)
    
    # Use the validated shared client ID
    client_id = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
    client_secret = ""  # Not needed for PKCE flow
    
    # Create OAuth flow
    print(f"📋 Creating OAuth flow with client ID: {client_id}")
    flow = AnthropicOAuthFlow(client_id, client_secret)
    
    # Generate authorization URL
    print("🔗 Generating authorization URL...")
    auth_url = flow.get_authorization_url()
    print(f"   URL: {auth_url}")
    print()
    
    # Verify URL contains required parameters
    print("✅ Verifying URL parameters:")
    assert "client_id=9d1c250a-e61b-44d9-88ed-5944d1962f5e" in auth_url
    assert "scope=org%3Acreate_api_key+user%3Aprofile+user%3Ainference" in auth_url
    assert "redirect_uri=http%3A%2F%2Flocalhost%3A54545%2Fcallback" in auth_url
    assert "code_challenge=" in auth_url
    assert "code_challenge_method=S256" in auth_url
    print("   ✓ Client ID correct")
    print("   ✓ Scopes correct")
    print("   ✓ Redirect URI correct")
    print("   ✓ PKCE parameters present")
    print()
    
    # Start callback server
    print("🚀 Starting callback server on port 54545...")
    try:
        server, port = flow.start_callback_server()
        print(f"   ✓ Server started on port {port}")
        assert port == 54545
        
        # Open browser for user authentication
        print("🌐 Opening browser for OAuth authorization...")
        print("   Please complete the authorization in your browser.")
        print("   This will redirect to localhost:54545/callback")
        
        import webbrowser
        webbrowser.open(auth_url)
        
        # Wait for callback with timeout
        print("⏳ Waiting for authorization callback...")
        timeout = 120  # 2 minutes
        start_time = time.time()
        
        while server.auth_code is None and server.auth_error is None:  # type: ignore
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print("❌ OAuth flow timed out after 2 minutes")
                return False
            
            # Show progress dots every 5 seconds
            if int(elapsed) % 5 == 0:
                print(".", end="", flush=True)
            
            time.sleep(0.5)
        
        print()  # New line after dots
        
        # Check for errors
        if server.auth_error:  # type: ignore
            print(f"❌ OAuth authorization error: {server.auth_error}")  # type: ignore
            return False
        
        if server.auth_code:  # type: ignore
            print("✅ Authorization code received!")
            auth_code = server.auth_code  # type: ignore
            print(f"   Code: {auth_code[:10]}...")
            
            # Exchange code for tokens
            print("🔄 Exchanging authorization code for tokens...")
            tokens = flow.exchange_code_for_tokens(auth_code)
            
            if tokens and "access_token" in tokens:
                print("✅ Tokens received successfully!")
                print(f"   Access token: {tokens['access_token'][:20]}...")
                if "refresh_token" in tokens:
                    print(f"   Refresh token: {tokens['refresh_token'][:20]}...")
                print(f"   Expires in: {tokens.get('expires_in', 'unknown')} seconds")
                print(f"   Token type: {tokens.get('token_type', 'unknown')}")
                
                # Test storing tokens
                print("💾 Testing token storage...")
                config_manager = ConfigurationManager()
                storage = CredentialStorage(config_manager)
                
                expires_at = int(time.time()) + tokens.get("expires_in", 3600)
                storage.store_oauth_token(
                    "anthropic",
                    tokens["access_token"],
                    tokens.get("refresh_token"),
                    expires_at
                )
                print("   ✓ Tokens stored successfully")
                
                # Test retrieving tokens
                stored_tokens = storage.get_oauth_token("anthropic")
                if stored_tokens:
                    print("   ✓ Tokens retrieved successfully")
                    print(f"   Stored access token: {stored_tokens['access_token'][:20]}...")
                else:
                    print("   ❌ Failed to retrieve stored tokens")
                    return False
                
                print()
                print("🎉 End-to-end OAuth test completed successfully!")
                print("✅ All components working:")
                print("   ✓ OAuth URL generation")
                print("   ✓ Callback server")
                print("   ✓ Authorization code capture")
                print("   ✓ Token exchange with Anthropic API")
                print("   ✓ Token storage and retrieval")
                return True
            else:
                print("❌ Failed to exchange code for tokens")
                print(f"   Response: {tokens}")
                return False
        
    except Exception as e:
        print(f"❌ Error during OAuth flow: {e}")
        return False
    
    finally:
        # Clean up server
        try:
            server.server_close()  # type: ignore
            print("🧹 Callback server stopped")
        except:
            pass
    
    return False

if __name__ == "__main__":
    success = test_oauth_flow()
    if success:
        print("\n🏆 OAuth implementation is working correctly!")
        exit(0)
    else:
        print("\n💥 OAuth test failed - see errors above")
        exit(1)
#!/usr/bin/env python3
"""
Enhanced end-to-end test for Anthropic OAuth flow with verbose diagnostics.
This script tests the complete OAuth implementation with real Anthropic authentication.
"""

import time
import json
from urllib.parse import urlparse, parse_qs
from llm_orc.authentication import AnthropicOAuthFlow, CredentialStorage
from llm_orc.config import ConfigurationManager

def print_section(title):
    """Print a section header with formatting."""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def print_subsection(title):
    """Print a subsection header."""
    print(f"\n{'─'*40}")
    print(f"📋 {title}")
    print(f"{'─'*40}")

def print_detail(key, value, indent=0):
    """Print a key-value pair with formatting."""
    spaces = "  " * indent
    print(f"{spaces}• {key}: {value}")

def analyze_oauth_url(url):
    """Analyze and display OAuth URL components."""
    print_subsection("OAuth URL Analysis")
    
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    
    print_detail("Full URL", url)
    print_detail("Scheme", parsed.scheme)
    print_detail("Host", parsed.netloc)
    print_detail("Path", parsed.path)
    
    print("\n📝 OAuth Parameters:")
    for key, values in params.items():
        value = values[0] if values else "None"
        if key in ['code_challenge', 'code_verifier']:
            print_detail(key, f"{value[:20]}... (truncated)", 1)
        else:
            print_detail(key, value, 1)
    
    # Validate required parameters
    print("\n✅ Parameter Validation:")
    required_params = ['client_id', 'response_type', 'redirect_uri', 'state', 'scope', 'code_challenge', 'code_challenge_method']
    for param in required_params:
        status = "✓" if param in params else "✗"
        print_detail(f"{param}", f"{status} {'Present' if param in params else 'Missing'}", 1)

def test_oauth_flow():
    """Test the complete OAuth flow end-to-end."""
    print_section("Starting End-to-End OAuth Test")
    
    server = None  # Initialize for cleanup
    
    try:
        # Configuration details
        client_id = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
        client_secret = ""  # Not needed for PKCE flow
        
        print_detail("Client ID", client_id)
        print_detail("Client Secret", "Not used (PKCE flow)")
        print_detail("Expected Port", "54545")
        print_detail("Expected Redirect URI", "http://localhost:54545/callback")
        
        # Create OAuth flow
        print_subsection("OAuth Flow Initialization")
        try:
            flow = AnthropicOAuthFlow(client_id, client_secret)
            print_detail("OAuth Flow Created", "✓ Success")
            print_detail("Provider", flow.provider)
            print_detail("Redirect URI", flow.redirect_uri)
            print_detail("State Parameter", flow.state)
            print_detail("Code Verifier", f"{flow.code_verifier[:20]}... (truncated)")
            print_detail("Code Challenge", f"{flow.code_challenge[:20]}... (truncated)")
        except Exception as e:
            print_detail("OAuth Flow Creation", f"✗ Failed: {e}")
            return False
        
        # Generate authorization URL
        print_subsection("Authorization URL Generation")
        try:
            auth_url = flow.get_authorization_url()
            print_detail("URL Generation", "✓ Success")
            
            # Analyze the URL in detail
            analyze_oauth_url(auth_url)
            
        except Exception as e:
            print_detail("URL Generation", f"✗ Failed: {e}")
            return False
        
        # Start callback server
        print_section("Callback Server Setup")
        try:
            print_detail("Attempting to start server", "Port 54545")
            server, port = flow.start_callback_server()
            print_detail("Server Status", "✓ Started successfully")
            print_detail("Listening Port", port)
            print_detail("Server Object", f"{type(server).__name__}")
            print_detail("Auth Code Attribute", f"Present: {hasattr(server, 'auth_code')}")
            print_detail("Auth Error Attribute", f"Present: {hasattr(server, 'auth_error')}")
            
            if hasattr(server, 'auth_code'):
                print_detail("Initial Auth Code", f"{getattr(server, 'auth_code', 'None')}")
            if hasattr(server, 'auth_error'):
                print_detail("Initial Auth Error", f"{getattr(server, 'auth_error', 'None')}")
                
            assert port == 54545, f"Expected port 54545, got {port}"
            
        except Exception as e:
            print_detail("Server Start Failed", f"✗ Error: {e}")
            return False
        
        # Browser interaction
        print_section("User Authentication")
        try:
            print_detail("Opening Browser", "Launching default browser")
            print_detail("Target URL", auth_url[:80] + "..." if len(auth_url) > 80 else auth_url)
            print_detail("Expected Redirect", "http://localhost:54545/callback?code=...")
            
            import webbrowser
            browser_result = webbrowser.open(auth_url)
            print_detail("Browser Launch", f"✓ Success" if browser_result else "⚠ May have failed")
            
            print(f"\n🔔 MANUAL STEP REQUIRED:")
            print(f"   1. Complete authentication in your browser")
            print(f"   2. Allow OAuth permissions if prompted")
            print(f"   3. Wait for redirect to localhost:54545/callback")
            
        except Exception as e:
            print_detail("Browser Launch", f"✗ Failed: {e}")
            print(f"   Manual URL: {auth_url}")
            
        # Wait for callback with enhanced monitoring
        print_section("Callback Monitoring")
        timeout = 120  # 2 minutes
        start_time = time.time()
        check_interval = 1.0  # Check every second
        status_interval = 10   # Status update every 10 seconds
        last_status = 0
        
        print_detail("Timeout", f"{timeout} seconds")
        print_detail("Check Interval", f"{check_interval} seconds")
        print_detail("Monitoring", "Starting callback monitoring...")
        
        while getattr(server, 'auth_code', None) is None and getattr(server, 'auth_error', None) is None:
            elapsed = time.time() - start_time
            
            # Timeout check
            if elapsed > timeout:
                print_detail("Status", "❌ Timeout reached")
                print_detail("Elapsed Time", f"{elapsed:.1f} seconds")
                return False
            
            # Status updates
            if int(elapsed) >= last_status + status_interval:
                print_detail("Status", f"⏳ Waiting... ({elapsed:.0f}s elapsed)")
                print_detail("Auth Code", f"{getattr(server, 'auth_code', 'None')}")
                print_detail("Auth Error", f"{getattr(server, 'auth_error', 'None')}")
                last_status = int(elapsed)
            
            time.sleep(check_interval)
        
        # Check final status
        final_auth_code = getattr(server, 'auth_code', None)
        final_auth_error = getattr(server, 'auth_error', None)
        
        print_subsection("Callback Results")
        print_detail("Final Auth Code", final_auth_code or "None")
        print_detail("Final Auth Error", final_auth_error or "None")
        
        if final_auth_error:
            print_detail("Error Details", f"❌ {final_auth_error}")
            return False
        
        if final_auth_code:
            print_detail("Success", "✓ Authorization code received")
            print_detail("Code Length", len(final_auth_code))
            print_detail("Code Preview", f"{final_auth_code[:15]}...")
            
            # Token exchange with detailed diagnostics
            print_section("Token Exchange")
            print_detail("Endpoint", "https://console.anthropic.com/oauth/token")
            print_detail("Method", "POST")
            print_detail("Content-Type", "application/x-www-form-urlencoded")
            
            print_subsection("Request Data")
            request_data = {
                "grant_type": "authorization_code",
                "client_id": flow.client_id,
                "code": final_auth_code,
                "code_verifier": flow.code_verifier,
                "redirect_uri": flow.redirect_uri,
            }
            
            for key, value in request_data.items():
                if key in ['code', 'code_verifier']:
                    print_detail(key, f"{value[:20]}... (truncated)")
                else:
                    print_detail(key, value)
            
            print_subsection("Making Token Exchange Request")
            tokens = flow.exchange_code_for_tokens(final_auth_code)
            
            print_subsection("Token Exchange Response")
            if tokens:
                print_detail("Response Type", type(tokens).__name__)
                print_detail("Response Content", f"{tokens}")
                
                if isinstance(tokens, dict):
                    if "access_token" in tokens:
                        print_detail("Status", "✓ Success - Access token received")
                        print_detail("Access Token", f"{tokens['access_token'][:25]}...")
                        
                        if "refresh_token" in tokens:
                            print_detail("Refresh Token", f"{tokens['refresh_token'][:25]}...")
                        print_detail("Expires In", f"{tokens.get('expires_in', 'Not provided')} seconds")
                        print_detail("Token Type", tokens.get('token_type', 'Not provided'))
                        
                        # Test credential storage
                        print_section("Credential Storage Test")
                        try:
                            config_manager = ConfigurationManager()
                            storage = CredentialStorage(config_manager)
                            
                            print_detail("Config Manager", "✓ Created")
                            print_detail("Credential Storage", "✓ Initialized")
                            
                            expires_at = int(time.time()) + tokens.get("expires_in", 3600)
                            print_detail("Calculated Expiry", f"{expires_at} (unix timestamp)")
                            
                            storage.store_oauth_token(
                                "anthropic",
                                tokens["access_token"],
                                tokens.get("refresh_token"),
                                expires_at
                            )
                            print_detail("Token Storage", "✓ Stored successfully")
                            
                            # Test retrieval
                            stored_tokens = storage.get_oauth_token("anthropic")
                            if stored_tokens:
                                print_detail("Token Retrieval", "✓ Retrieved successfully")
                                print_detail("Stored Access Token", f"{stored_tokens.get('access_token', 'None')[:25]}...")
                                if 'refresh_token' in stored_tokens:
                                    print_detail("Stored Refresh Token", f"{stored_tokens['refresh_token'][:25]}...")
                            else:
                                print_detail("Token Retrieval", "❌ Failed")
                                return False
                                
                        except Exception as e:
                            print_detail("Storage Error", f"❌ {e}")
                            return False
                        
                        # Success summary
                        print_section("Test Results - SUCCESS")
                        print_detail("OAuth URL Generation", "✓ Working")
                        print_detail("Callback Server", "✓ Working")
                        print_detail("Authorization Flow", "✓ Working")
                        print_detail("Code Capture", "✓ Working") 
                        print_detail("Token Exchange", "✓ Working")
                        print_detail("Token Storage", "✓ Working")
                        return True
                        
                    else:
                        print_detail("Status", "❌ Failed - No access token in response")
                        print_detail("Available Keys", list(tokens.keys()) if tokens else "None")
                        return False
                else:
                    print_detail("Status", "❌ Failed - Invalid response format")
                    return False
            else:
                print_detail("Status", "❌ Failed - Empty response")
                return False
        
        print_detail("Callback Status", "❌ No authorization code received")
        return False

    except Exception as e:
        print_section("Error Occurred")
        print_detail("Exception Type", type(e).__name__)
        print_detail("Exception Message", str(e))
        import traceback
        print_detail("Stack Trace", "See below")
        print("\n" + "─" * 60)
        traceback.print_exc()
        print("─" * 60)
        return False

    finally:
        # Clean up server
        print_section("Cleanup")
        try:
            if server is not None:
                server.server_close()
                print_detail("Callback Server", "✓ Stopped")
            else:
                print_detail("Callback Server", "⚠ Not started")
        except Exception as e:
            print_detail("Cleanup Error", f"⚠ {e}")

if __name__ == "__main__":
    print("🚀 Enhanced OAuth End-to-End Test")
    print("=" * 60)
    print("This test provides detailed diagnostics for the OAuth implementation.")
    print("Follow the prompts to complete authentication in your browser.")
    print("=" * 60)
    
    success = test_oauth_flow()
    
    print_section("Final Results")
    if success:
        print_detail("Overall Status", "🏆 SUCCESS - OAuth implementation working!")
        print_detail("Exit Code", "0")
        exit(0)
    else:
        print_detail("Overall Status", "💥 FAILED - See diagnostic details above")
        print_detail("Exit Code", "1")
        print_detail("Next Steps", "Review error details and check token exchange endpoint")
        exit(1)
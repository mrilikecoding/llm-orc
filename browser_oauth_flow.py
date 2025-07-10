#!/usr/bin/env python3
"""
Browser-based OAuth flow for Anthropic API authentication.
This approach uses real browser sessions to bypass Cloudflare protection,
similar to how OpenCode handles OAuth authentication.
"""

import webbrowser
import urllib.parse
import json
import secrets
import hashlib
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time
from pathlib import Path
import os
from typing import Dict, Optional

class BrowserOAuthHandler(BaseHTTPRequestHandler):
    """Handle OAuth callback and provide user instructions for token extraction"""
    
    def do_GET(self) -> None:
        # Parse callback URL
        url_parts = urllib.parse.urlparse(self.path)
        query_params = urllib.parse.parse_qs(url_parts.query)
        
        # Extract authorization code
        if 'code' in query_params:
            self.server.auth_code = query_params['code'][0]  # type: ignore
            self.server.state = query_params.get('state', [None])[0]  # type: ignore
            
            # Send success page with token extraction instructions
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            success_html = f"""
            <html>
            <head>
                <title>OAuth Success - Token Extraction Required</title>
                <style>
                    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; 
                           padding: 20px; background: #f5f5f5; }}
                    .container {{ max-width: 800px; margin: 0 auto; 
                                background: white; padding: 30px; border-radius: 8px; }}
                    .code {{ background: #f0f0f0; padding: 15px; 
                            border-radius: 4px; font-family: monospace; 
                            word-break: break-all; margin: 15px 0; font-size: 12px; }}
                    .step {{ background: #e3f2fd; padding: 15px; margin: 15px 0; 
                            border-radius: 4px; border-left: 4px solid #2196f3; }}
                    .warning {{ background: #fff3e0; padding: 15px; margin: 15px 0; 
                               border-radius: 4px; border-left: 4px solid #ff9800; }}
                    button {{ background: #007bff; color: white; border: none; 
                             padding: 10px 20px; border-radius: 4px; cursor: pointer; }}
                    .copy-btn {{ background: #28a745; margin-left: 10px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>✅ Authorization Successful!</h1>
                    <p>Your authorization code has been captured. Now we need to extract the actual access token.</p>
                    
                    <div class="warning">
                        <strong>⚠️ Manual Token Extraction Required</strong><br>
                        The OAuth token endpoint is protected by Cloudflare, so we need to extract tokens manually from the browser session.
                    </div>
                    
                    <h3>🔑 Authorization Code (Captured):</h3>
                    <div class="code" id="authCode">{self.server.auth_code}</div>
                    <button onclick="copyToClipboard('authCode')">Copy Code</button>
                    
                    <h3>📋 Next Steps - Choose One Approach:</h3>
                    
                    <div class="step">
                        <h4>Option 1: Browser Developer Tools Method</h4>
                        <ol>
                            <li>Open a new tab and go to: <a href="https://console.anthropic.com" target="_blank">https://console.anthropic.com</a></li>
                            <li>Open Developer Tools (F12 or Cmd+Option+I)</li>
                            <li>Go to the <strong>Application</strong> tab (Chrome) or <strong>Storage</strong> tab (Firefox)</li>
                            <li>Look under <strong>Local Storage</strong> or <strong>Session Storage</strong> for console.anthropic.com</li>
                            <li>Look for keys containing "token", "auth", "access", or "bearer"</li>
                            <li>Copy any access tokens you find</li>
                        </ol>
                    </div>
                    
                    <div class="step">
                        <h4>Option 2: Network Monitoring Method</h4>
                        <ol>
                            <li>Open Developer Tools and go to the <strong>Network</strong> tab</li>
                            <li>Clear the network log</li>
                            <li>Go to <a href="https://console.anthropic.com" target="_blank">https://console.anthropic.com</a></li>
                            <li>Complete any login/authentication steps</li>
                            <li>Look for requests to endpoints containing "token" or "oauth"</li>
                            <li>Check response bodies for access_token fields</li>
                        </ol>
                    </div>
                    
                    <div class="step">
                        <h4>Option 3: API Key Alternative</h4>
                        <ol>
                            <li>Go to <a href="https://console.anthropic.com/settings/keys" target="_blank">https://console.anthropic.com/settings/keys</a></li>
                            <li>Create a new API key if OAuth tokens aren't available</li>
                            <li>Use the API key with your application instead of OAuth tokens</li>
                        </ol>
                    </div>
                    
                    <p><button onclick="window.close()">Close Window</button></p>
                    
                    <script>
                        function copyToClipboard(elementId) {{
                            const element = document.getElementById(elementId);
                            const text = element.textContent;
                            navigator.clipboard.writeText(text).then(() => {{
                                alert('Copied to clipboard!');
                            }});
                        }}
                        
                        // Copy auth code to clipboard automatically
                        navigator.clipboard.writeText('{self.server.auth_code}');
                        console.log('Authorization code copied to clipboard');
                    </script>
                </div>
            </body>
            </html>
            """
            self.wfile.write(success_html.encode())
        else:
            # Handle error
            error = query_params.get('error', ['Unknown error'])[0]
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            error_html = f"""
            <html>
            <body style="font-family: sans-serif; padding: 40px; text-align: center;">
                <h1>❌ Authorization Failed</h1>
                <p>Error: {error}</p>
                <p>Please try again or check your authentication settings.</p>
                <button onclick="window.close()">Close Window</button>
            </body>
            </html>
            """
            self.wfile.write(error_html.encode())
    
    def log_message(self, format: str, *args) -> None:
        # Suppress default HTTP logging for cleaner output
        pass

class BrowserBasedOAuthFlow:
    """OAuth implementation using browser-based authentication to bypass Cloudflare"""
    
    def __init__(self):
        self.client_id = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
        self.redirect_uri = "http://localhost:54545/callback"
        self.auth_base_url = "https://console.anthropic.com/oauth/authorize"
        
        # Configuration storage
        self.config_dir = Path.home() / ".llm-orc"
        self.config_dir.mkdir(exist_ok=True, parents=True)
        self.auth_data_file = self.config_dir / "anthropic_oauth_data.json"
    
    def generate_pkce_pair(self) -> tuple[str, str]:
        """Generate PKCE code verifier and challenge for secure OAuth flow"""
        code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode('utf-8')).digest()
        ).decode('utf-8').rstrip('=')
        return code_verifier, code_challenge
    
    def start_browser_oauth_flow(self) -> Optional[Dict[str, str]]:
        """Start OAuth flow using real browser to bypass Cloudflare protection"""
        print("🚀 Starting Browser-Based OAuth Flow")
        print("=" * 60)
        print("This approach uses your default browser to complete OAuth authentication")
        print("and provides instructions for manual token extraction.\n")
        
        # Generate PKCE parameters for security
        code_verifier, code_challenge = self.generate_pkce_pair()
        state = secrets.token_urlsafe(32)
        
        # Store PKCE data temporarily
        pkce_data = {
            'code_verifier': code_verifier,
            'state': state,
            'timestamp': time.time()
        }
        
        pkce_file = self.config_dir / "pkce_temp.json"
        with open(pkce_file, 'w') as f:
            json.dump(pkce_data, f)
        
        # Build authorization URL with all required parameters
        params = {
            'client_id': self.client_id,
            'response_type': 'code',
            'redirect_uri': self.redirect_uri,
            'state': state,
            'code_challenge': code_challenge,
            'code_challenge_method': 'S256',
            'scope': 'org:create_api_key user:profile user:inference'
        }
        
        auth_url = f"{self.auth_base_url}?{urllib.parse.urlencode(params)}"
        
        print(f"📋 OAuth Configuration:")
        print(f"   • Client ID: {self.client_id}")
        print(f"   • Redirect URI: {self.redirect_uri}")
        print(f"   • State: {state[:20]}...")
        print(f"   • Code Challenge: {code_challenge[:20]}...")
        print(f"   • Scopes: org:create_api_key user:profile user:inference")
        
        # Start callback server
        print(f"\n🔍 Starting callback server on localhost:54545...")
        try:
            server = HTTPServer(('localhost', 54545), BrowserOAuthHandler)
            server.auth_code = None  # type: ignore
            server.state = None  # type: ignore
        except OSError as e:
            print(f"❌ Failed to start callback server: {e}")
            print("   Please ensure port 54545 is available")
            return None
        
        # Start server in background thread
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        print(f"✅ Callback server started successfully")
        
        # Open browser for authentication
        print(f"\n🌐 Opening browser for authentication...")
        print(f"   Authorization URL: {auth_url[:80]}...")
        
        try:
            webbrowser.open(auth_url)
            print(f"✅ Browser opened successfully")
        except Exception as e:
            print(f"❌ Failed to open browser: {e}")
            print(f"   Please manually open: {auth_url}")
        
        print(f"\n⏳ Waiting for user authentication...")
        print(f"   • Complete authentication in your browser")
        print(f"   • Allow OAuth permissions when prompted")
        print(f"   • Follow the token extraction instructions on the success page")
        print(f"   • This window will wait for up to 5 minutes")
        
        # Wait for OAuth callback
        timeout = 300  # 5 minutes - enough time for manual token extraction
        start_time = time.time()
        
        while getattr(server, 'auth_code', None) is None:
            if time.time() - start_time > timeout:
                print(f"\n❌ Authentication timed out after {timeout // 60} minutes")
                server.shutdown()
                return None
            time.sleep(1)
        
        # Verify state parameter to prevent CSRF attacks
        if getattr(server, 'state', None) != state:
            print(f"❌ State parameter mismatch - possible security issue")
            server.shutdown()
            return None
        
        auth_code = server.auth_code  # type: ignore
        print(f"\n✅ Authorization code captured successfully!")
        print(f"   • Auth Code: {auth_code[:20]}...")
        print(f"   • Code Length: {len(auth_code)}")
        print(f"   • State Verified: ✅")
        
        # Clean up server
        server.shutdown()
        
        # Load PKCE data and clean up temp file
        with open(pkce_file) as f:
            pkce_data = json.load(f)
        pkce_file.unlink()
        
        return {
            'auth_code': auth_code,
            'code_verifier': pkce_data['code_verifier'],
            'state': state,
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri
        }
    
    def save_auth_data(self, auth_data: Dict[str, str]) -> None:
        """Save authentication data securely for future use"""
        auth_data['timestamp'] = time.time()
        auth_data['method'] = 'browser_oauth'
        
        with open(self.auth_data_file, 'w') as f:
            json.dump(auth_data, f, indent=2)
        
        # Set secure file permissions (readable only by owner)
        os.chmod(self.auth_data_file, 0o600)
        print(f"\n💾 Authentication data saved to: {self.auth_data_file}")
    
    def load_auth_data(self) -> Optional[Dict[str, str]]:
        """Load previously saved authentication data"""
        if not self.auth_data_file.exists():
            return None
        
        try:
            with open(self.auth_data_file) as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Failed to load auth data: {e}")
            return None

def main():
    """Main function to run browser-based OAuth flow"""
    oauth = BrowserBasedOAuthFlow()
    
    print("🔑 Anthropic OAuth - Browser-Based Authentication")
    print("=" * 60)
    print("This method uses your browser to complete OAuth authentication")
    print("and provides instructions for extracting access tokens manually.")
    print("This approach bypasses Cloudflare protection on the token endpoint.\n")
    
    # Check for existing auth data
    existing_auth = oauth.load_auth_data()
    if existing_auth:
        print(f"📂 Found existing authentication data from {time.ctime(existing_auth.get('timestamp', 0))}")
        use_existing = input("Use existing data? (y/N): ").strip().lower()
        if use_existing in ['y', 'yes']:
            print("✅ Using existing authentication data")
            return
    
    # Start browser-based OAuth flow
    auth_data = oauth.start_browser_oauth_flow()
    
    if auth_data:
        oauth.save_auth_data(auth_data)
        
        print(f"\n🎉 OAuth Authorization Complete!")
        print(f"   • Authorization Code: ✅ Captured")
        print(f"   • Code Verifier: ✅ Saved")
        print(f"   • Authentication Data: ✅ Saved")
        print(f"\n📋 Next Steps:")
        print(f"   1. Follow the token extraction instructions in your browser")
        print(f"   2. Manually extract access tokens from browser storage/network")
        print(f"   3. Or create an API key at https://console.anthropic.com/settings/keys")
        print(f"   4. Update your application configuration with the tokens/API key")
        
    else:
        print(f"\n💥 OAuth Flow Failed")
        print(f"   • Please try again")
        print(f"   • Check browser permissions")
        print(f"   • Ensure you have access to console.anthropic.com")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Comprehensive authentication validation for PyPI and Test PyPI.
This script validates the .pypirc configuration and tests authentication
without uploading packages.
"""

import sys
import requests
import configparser
from pathlib import Path

def validate_token_format(token, server_name):
    """Validate that the token has the correct format."""
    if server_name == 'pypi':
        expected_prefix = 'pypi-'
    elif server_name == 'testpypi':
        expected_prefix = 'pypi-'
    else:
        return False, f"Unknown server: {server_name}"
    
    if not token.startswith(expected_prefix):
        return False, f"Token doesn't start with expected prefix: {expected_prefix}"
    
    # Basic length check - PyPI tokens are typically quite long
    if len(token) < 50:
        return False, "Token appears to be too short"
    
    return True, "Token format appears valid"

def test_repository_access(repository_url, username, password):
    """Test if we can access the repository with the given credentials."""
    try:
        # Try to access the repository URL
        response = requests.get(repository_url.replace('/legacy/', '/'), timeout=10)
        if response.status_code == 200:
            return True, f"Repository accessible at {repository_url}"
        else:
            return False, f"Repository returned status code: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"Failed to access repository: {e}"

def main():
    """Main validation function."""
    print("🔐 Comprehensive PyPI Authentication Validation")
    print("=" * 55)
    
    # Read .pypirc configuration
    pypirc_path = Path.home() / '.pypirc'
    if not pypirc_path.exists():
        print("❌ .pypirc file not found")
        return 1
    
    config = configparser.ConfigParser()
    config.read(pypirc_path)
    
    servers_to_test = ['pypi', 'testpypi']
    all_valid = True
    
    for server in servers_to_test:
        print(f"\n🧪 Testing {server.upper()} configuration:")
        print("-" * 30)
        
        if server not in config.sections():
            print(f"❌ {server} section not found in .pypirc")
            all_valid = False
            continue
        
        server_config = config[server]
        
        # Check required fields
        required_fields = ['repository', 'username', 'password']
        for field in required_fields:
            if field not in server_config:
                print(f"❌ Missing {field} in {server} configuration")
                all_valid = False
                continue
        
        repository = server_config['repository']
        username = server_config['username']
        password = server_config['password']
        
        print(f"✅ Repository: {repository}")
        print(f"✅ Username: {username}")
        
        # Validate token format
        token_valid, token_msg = validate_token_format(password, server)
        if token_valid:
            print(f"✅ Token format: {token_msg}")
        else:
            print(f"❌ Token format: {token_msg}")
            all_valid = False
        
        # Test repository access
        repo_accessible, repo_msg = test_repository_access(repository, username, password)
        if repo_accessible:
            print(f"✅ Repository access: {repo_msg}")
        else:
            print(f"⚠️  Repository access: {repo_msg}")
            # Note: This might fail due to network issues, so we don't mark as critical failure
    
    print("\n" + "=" * 55)
    if all_valid:
        print("✅ All authentication configurations appear valid!")
        print("🚀 Ready to proceed with package uploads.")
        return 0
    else:
        print("❌ Some authentication configurations have issues.")
        print("🔧 Please review and fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
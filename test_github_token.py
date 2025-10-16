#!/usr/bin/env python3
"""
GitHub Token Validation Test Script

This script tests whether the GitHub token provided in the configuration
is valid and has the necessary permissions for the repositories.
"""

import requests
import json
import os
from datetime import datetime

# Set console encoding to UTF-8 for Windows
import sys
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

def test_github_token():
    """Test GitHub token validity and permissions."""
    
    print("🔍 GitHub Token Validation Test")
    print("=" * 50)
    
    # Get token from environment or use the one from config
    token = os.getenv('GITHUB_TOKEN', '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')
    
    if not token:
        print("❌ No GitHub token found!")
        return False
    
    # Mask token for display
    masked_token = token[:8] + "..." + token[-4:] if len(token) > 12 else "***"
    print(f"🔑 Testing token: {masked_token}")
    print()
    
    # Test repositories
    repos_to_test = [
        {
            "name": "Code Repository",
            "url": "https://github.com/bluepal-nipun-marwaha/click",
            "owner": "bluepal-nipun-marwaha",
            "repo": "click"
        },
        {
            "name": "Docs Repository", 
            "url": "https://github.com/bluepal-nipun-marwaha/test-simple-repo-docs",
            "owner": "bluepal-nipun-marwaha",
            "repo": "test-simple-repo-docs"
        }
    ]
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'GitHub-Token-Test/1.0'
    }
    
    all_tests_passed = True
    
    for repo_info in repos_to_test:
        print(f"📁 Testing {repo_info['name']}: {repo_info['url']}")
        
        # Test 1: Repository access
        repo_url = f"https://api.github.com/repos/{repo_info['owner']}/{repo_info['repo']}"
        try:
            response = requests.get(repo_url, headers=headers, timeout=10)
            print(f"   📡 Repository access: {response.status_code}")
            
            if response.status_code == 200:
                repo_data = response.json()
                print(f"   ✅ Repository accessible")
                print(f"   📊 Repository info:")
                print(f"      - Name: {repo_data.get('name', 'N/A')}")
                print(f"      - Full name: {repo_data.get('full_name', 'N/A')}")
                print(f"      - Private: {repo_data.get('private', 'N/A')}")
                print(f"      - Default branch: {repo_data.get('default_branch', 'N/A')}")
                print(f"      - Updated: {repo_data.get('updated_at', 'N/A')}")
                
                # Test 2: Contents access
                contents_url = f"https://api.github.com/repos/{repo_info['owner']}/{repo_info['repo']}/contents"
                contents_response = requests.get(contents_url, headers=headers, timeout=10)
                print(f"   📡 Contents access: {contents_response.status_code}")
                
                if contents_response.status_code == 200:
                    contents = contents_response.json()
                    print(f"   ✅ Contents accessible")
                    print(f"   📄 Found {len(contents)} items in root directory")
                    
                    # List some files
                    for item in contents[:5]:  # Show first 5 items
                        item_type = "📁" if item['type'] == 'dir' else "📄"
                        print(f"      {item_type} {item['name']}")
                    
                    if len(contents) > 5:
                        print(f"      ... and {len(contents) - 5} more items")
                        
                elif contents_response.status_code == 401:
                    print(f"   ❌ Contents access denied - Authentication failed")
                    all_tests_passed = False
                elif contents_response.status_code == 403:
                    print(f"   ❌ Contents access forbidden - Insufficient permissions")
                    all_tests_passed = False
                else:
                    print(f"   ⚠️ Contents access failed: {contents_response.status_code}")
                    all_tests_passed = False
                    
            elif response.status_code == 401:
                print(f"   ❌ Repository access denied - Authentication failed")
                print(f"   💡 This usually means the token is invalid or expired")
                all_tests_passed = False
            elif response.status_code == 404:
                print(f"   ❌ Repository not found")
                print(f"   💡 Check if the repository exists and is accessible")
                all_tests_passed = False
            elif response.status_code == 403:
                print(f"   ❌ Repository access forbidden - Insufficient permissions")
                all_tests_passed = False
            else:
                print(f"   ⚠️ Unexpected response: {response.status_code}")
                all_tests_passed = False
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Network error: {str(e)}")
            all_tests_passed = False
        
        print()
    
    # Test 3: Token info (if possible)
    print("🔍 Testing token information...")
    try:
        # Try to get user info
        user_url = "https://api.github.com/user"
        user_response = requests.get(user_url, headers=headers, timeout=10)
        print(f"📡 User info access: {user_response.status_code}")
        
        if user_response.status_code == 200:
            user_data = user_response.json()
            print(f"✅ Token is valid")
            print(f"👤 Authenticated as: {user_data.get('login', 'N/A')}")
            print(f"📧 Email: {user_data.get('email', 'N/A')}")
            print(f"🏢 Company: {user_data.get('company', 'N/A')}")
            print(f"📊 Public repos: {user_data.get('public_repos', 'N/A')}")
            print(f"📊 Private repos: {user_data.get('total_private_repos', 'N/A')}")
        elif user_response.status_code == 401:
            print(f"❌ Token is invalid or expired")
            all_tests_passed = False
        else:
            print(f"⚠️ Unexpected response: {user_response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {str(e)}")
        all_tests_passed = False
    
    print()
    print("=" * 50)
    
    if all_tests_passed:
        print("✅ All tests passed! GitHub token is valid and has necessary permissions.")
        print("💡 The authentication errors in the workflow might be due to other issues.")
    else:
        print("❌ Some tests failed! GitHub token has issues.")
        print("💡 Recommendations:")
        print("   1. Check if the token is expired")
        print("   2. Verify the token has 'repo' scope for private repositories")
        print("   3. Ensure the token has access to both repositories")
        print("   4. Generate a new token if needed: https://github.com/settings/tokens")
    
    return all_tests_passed

def test_specific_file_access():
    """Test access to specific files in the docs repository."""
    
    print("\n🔍 Testing Specific File Access")
    print("=" * 50)
    
    token = os.getenv('GITHUB_TOKEN', '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')
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    # Test access to the specific DOCX file
    file_path = "docs/Click_Professional_Documentation.docx"
    url = f"https://api.github.com/repos/bluepal-nipun-marwaha/test-simple-repo-docs/contents/{file_path}"
    
    print(f"📄 Testing access to: {file_path}")
    print(f"🔗 URL: {url}")
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"📡 Response: {response.status_code}")
        
        if response.status_code == 200:
            file_data = response.json()
            print(f"✅ File accessible")
            print(f"📊 File info:")
            print(f"   - Name: {file_data.get('name', 'N/A')}")
            print(f"   - Size: {file_data.get('size', 'N/A')} bytes")
            print(f"   - SHA: {file_data.get('sha', 'N/A')[:8]}...")
            print(f"   - Download URL: {file_data.get('download_url', 'N/A')}")
            
            # Test download
            if file_data.get('download_url'):
                print(f"🔍 Testing file download...")
                download_response = requests.get(file_data['download_url'], timeout=10)
                print(f"📡 Download response: {download_response.status_code}")
                
                if download_response.status_code == 200:
                    print(f"✅ File download successful")
                    print(f"📊 Downloaded {len(download_response.content)} bytes")
                else:
                    print(f"❌ File download failed")
                    
        elif response.status_code == 404:
            print(f"❌ File not found")
            print(f"💡 The file might not exist in the repository")
        elif response.status_code == 401:
            print(f"❌ Authentication failed")
        else:
            print(f"⚠️ Unexpected response: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {str(e)}")

if __name__ == "__main__":
    print(f"🕒 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run main token test
    token_valid = test_github_token()
    
    # Run specific file access test
    test_specific_file_access()
    
    print(f"\n🕒 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not token_valid:
        print("\n🚨 ACTION REQUIRED:")
        print("   The GitHub token appears to be invalid or expired.")
        print("   Please generate a new token at: https://github.com/settings/tokens")
        print("   Required scopes: 'repo' (for private repositories)")
        exit(1)
    else:
        print("\n✅ Token validation completed successfully!")
        exit(0)

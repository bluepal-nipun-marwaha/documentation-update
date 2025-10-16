#!/usr/bin/env python3
"""
Workflow for existing repositories - no repo creation needed!
Users provide their existing GitHub/GitLab repo details.
"""

import grpc
from concurrent import futures
import os
import threading
from flask import Flask, request, jsonify
import json
from datetime import datetime
import uuid
import logging
import hmac
import hashlib
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GitHubClient:
    """Client for GitHub API operations for documentation updates."""
    
    def __init__(self, token, repo_url, test_mode=False):
        self.token = token
        self.repo_url = repo_url
        self.test_mode = test_mode
        self.base_url = 'https://api.github.com'  # Add base_url for compatibility
        self.owner, self.repo = self._parse_repo_url(repo_url)
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
            'Accept': 'application/vnd.github.v3+json'
        }
        logger.info(f"🔧 GitHubClient initialized: owner={self.owner}, repo={self.repo}")
        
        if test_mode:
            logger.info("🧪 TEST MODE: Skipping GitHub authentication")
        else:
            # Test authentication and get repository info
            self._test_authentication()
    
    def _parse_repo_url(self, repo_url):
        """Parse GitHub repository URL to extract owner and repo name."""
        # Handle both https://github.com/owner/repo and https://github.com/owner/repo.git
        repo_url = repo_url.replace('.git', '')
        parts = repo_url.split('/')
        if len(parts) >= 2:
            return parts[-2], parts[-1]
        return None, None
    
    def _test_authentication(self):
        """Test GitHub API authentication and get repository info."""
        if self.test_mode:
            logger.info("🧪 TEST MODE: Skipping GitHub authentication test")
            return
            
        try:
            import requests
            
            # Test 1: Check if we can access the repository
            repo_url = f"https://api.github.com/repos/{self.owner}/{self.repo}"
            logger.info(f"🔍 Testing repository access: {repo_url}")
            
            response = requests.get(repo_url, headers=self.headers)
            logger.info(f"📡 Repository access response: {response.status_code}")
            
            if response.status_code == 200:
                repo_data = response.json()
                default_branch = repo_data.get('default_branch', 'main')
                logger.info(f"✅ Repository accessible! Default branch: {default_branch}")
                
                # Update the default branch for future operations
                self.default_branch = default_branch
                logger.info(f"🔧 Set default_branch attribute to: {self.default_branch}")
                
                # Test 2: Check if we can access the root contents
                contents_url = f"https://api.github.com/repos/{self.owner}/{self.repo}/contents"
                logger.info(f"🔍 Testing contents access: {contents_url}")
                
                contents_response = requests.get(contents_url, headers=self.headers, params={'ref': default_branch})
                logger.info(f"📡 Contents access response: {contents_response.status_code}")
                
                if contents_response.status_code == 200:
                    logger.info("✅ Contents API accessible!")
                else:
                    logger.warning(f"⚠️ Contents API failed: {contents_response.status_code}")
                    logger.warning(f"Response: {contents_response.text}")
                    
            elif response.status_code == 404:
                logger.error("❌ Repository not found! Check if:")
                logger.error("   1. Repository exists: https://github.com/{self.owner}/{self.repo}")
                logger.error("   2. Repository is public or token has access")
                logger.error("   3. Owner and repo names are correct")
            elif response.status_code == 401:
                logger.error("❌ Authentication failed! Check if:")
                logger.error("   1. GitHub token is valid")
                logger.error("   2. Token has not expired")
                logger.error("   3. Token has repository access permissions")
            else:
                logger.error(f"❌ Unexpected response: {response.status_code}")
                logger.error(f"Response: {response.text}")
                
        except Exception as e:
            logger.error(f"❌ Authentication test failed: {str(e)}")
    
    def create_or_update_file(self, file_path, content, commit_message, branch=None):
        """Create or update a file in GitHub repository."""
        try:
            import base64
            
            # Use detected default branch if not specified
            if branch is None:
                branch = getattr(self, 'default_branch', 'main')
                logger.info(f"🔧 Using branch: {branch} (default_branch attribute: {getattr(self, 'default_branch', 'NOT SET')})")
            
            # GitHub API will create the full path automatically when creating files
            # No need to pre-create directories
            
            # GitHub API endpoint for file operations
            url = f"https://api.github.com/repos/{self.owner}/{self.repo}/contents/{file_path}"
            logger.info(f"🔍 Checking file: {url}")
            
            # Check if file exists
            response = requests.get(url, headers=self.headers, params={'ref': branch})
            logger.info(f"📡 File check response: {response.status_code}")
            
            if response.status_code == 200:
                # File exists, get its SHA for update
                existing_file = response.json()
                sha = existing_file['sha']
                
                # Update file
                data = {
                    'message': commit_message,
                    'content': base64.b64encode(content.encode('utf-8')).decode('utf-8'),
                    'sha': sha,
                    'branch': branch
                }
                response = requests.put(url, headers=self.headers, json=data)
            else:
                # File doesn't exist, create it (GitHub API uses PUT for both create and update)
                data = {
                    'message': commit_message,
                    'content': base64.b64encode(content.encode('utf-8')).decode('utf-8'),
                    'branch': branch
                }
                response = requests.put(url, headers=self.headers, json=data)
            
            if response.status_code in [200, 201]:
                logger.info(f"GitHub file created/updated successfully: {file_path}")
                return True
            else:
                logger.error(f"Failed to create/update GitHub file: {response.status_code} - {response.text} - {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to create/update GitHub file: {str(e)} - {file_path}")
            return False
    
    def _ensure_folder_structure(self, file_path, branch=None):
        """Ensure folder structure exists by creating intermediate folders recursively."""
        try:
            import base64
            
            # Use detected default branch if not specified
            if branch is None:
                branch = getattr(self, 'default_branch', 'main')
                logger.info(f"🔧 _ensure_folder_structure using branch: {branch} (default_branch attribute: {getattr(self, 'default_branch', 'NOT SET')})")
            
            # Get directory path (everything except the filename)
            dir_path = '/'.join(file_path.split('/')[:-1])
            
            if not dir_path:  # File is in root directory
                return
            
            # Split the path into individual directory components
            path_parts = dir_path.split('/')
            current_path = ""
            
            # Create each directory level recursively
            for part in path_parts:
                if current_path:
                    current_path = f"{current_path}/{part}"
                else:
                    current_path = part
                
                # Check if this directory level exists
                url = f"https://api.github.com/repos/{self.owner}/{self.repo}/contents/{current_path}"
                logger.info(f"🔍 Checking directory: {url}")
                response = requests.get(url, headers=self.headers, params={'ref': branch})
                logger.info(f"📡 Directory check response: {response.status_code}")
                
                if response.status_code == 404:
                    # Directory doesn't exist, skip creating intermediate directories
                    # GitHub API will create the full path when we create the actual file
                    logger.info(f"Directory {current_path} doesn't exist, will be created with actual file")
                    continue
                elif response.status_code == 200:
                    logger.info(f"Directory already exists: {current_path}")
                else:
                    logger.warning(f"Unexpected response for directory {current_path}: {response.status_code}")
                    
        except Exception as e:
            logger.warning(f"Failed to ensure folder structure for {file_path}: {str(e)}")
    
    def update_single_documentation_file(self, file_path, content, commit_data):
        """Update a single documentation file with LLM-generated content."""
        try:
            # Create commit message
            commit_msg = f"docs: update {file_path.split('/')[-1]} for {commit_data.get('message', commit_data.get('commit_message', 'documentation update'))}"
            
            # Handle DOCX files differently
            if file_path.lower().endswith('.docx'):
                logger.info(f"📄 Updating DOCX file in GitHub: {file_path}")
                # For DOCX files, content should be bytes
                if isinstance(content, str):
                    # Convert text to DOCX bytes
                    from utils.docx_handler import DOCXHandler
                    content_bytes = DOCXHandler.create_docx_from_text(content, "Updated Document")
                    # Update file with binary content
                    if self.create_or_update_file_binary(file_path, content_bytes, commit_msg):
                        logger.info(f"DOCX documentation updated in GitHub: {file_path}")
                        return True
                    else:
                        logger.error(f"Failed to update DOCX documentation in GitHub: {file_path}")
                        return False
                else:
                    # Content is already bytes
                    if self.create_or_update_file_binary(file_path, content, commit_msg):
                        logger.info(f"DOCX documentation updated in GitHub: {file_path}")
                        return True
                    else:
                        logger.error(f"Failed to update DOCX documentation in GitHub: {file_path}")
                        return False
            else:
                # Handle text files (markdown, etc.)
                if self.create_or_update_file(file_path, content, commit_msg):
                    logger.info(f"GitHub documentation updated: {file_path}")
                    return True
                else:
                    logger.error(f"Failed to update GitHub documentation: {file_path}")
                    return False
                
        except Exception as e:
            logger.error(f"Single GitHub documentation update failed: {str(e)} - {file_path}")
            return False
    
    def create_or_update_file_binary(self, file_path, content_bytes, commit_message, branch='main'):
        """Create or update a file with binary content (for DOCX files) in GitHub."""
        try:
            import base64
            
            # Encode binary content to base64 for GitHub API
            encoded_content = base64.b64encode(content_bytes).decode('utf-8')
            
            # GitHub API endpoint for file operations
            url = f"https://api.github.com/repos/{self.owner}/{self.repo}/contents/{file_path}"
            
            # Check if file exists
            response = requests.get(url, headers=self.headers, params={'ref': branch})
            
            if response.status_code == 200:
                # File exists, get its SHA for update
                existing_file = response.json()
                sha = existing_file['sha']
                
                # Update file
                data = {
                    'message': commit_message,
                    'content': encoded_content,
                    'sha': sha,
                    'branch': branch
                }
                response = requests.put(url, headers=self.headers, json=data)
            else:
                # File doesn't exist, create it (GitHub API uses PUT for both create and update)
                data = {
                    'message': commit_message,
                    'content': encoded_content,
                    'branch': branch
                }
                response = requests.put(url, headers=self.headers, json=data)
            
            if response.status_code in [200, 201]:
                logger.info(f"GitHub binary file updated successfully: {file_path}")
                return True
            else:
                logger.error(f"Failed to update GitHub binary file: {response.status_code} - {response.text} - {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"GitHub binary file update failed: {str(e)} - {file_path}")
            return False


class ExistingRepoWorkflow:
    """Workflow for existing repositories."""
    
    def __init__(self):
        self.server = None
        self.app = None
        self.configured_repos = {}  # Store existing repo configs
    
    def create_grpc_server(self):
        """Create and configure the gRPC server."""
        logger.info("Starting existing repo workflow gRPC server...")
        
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10)
        )
        
        listen_addr = "0.0.0.0:50062"  # Different port to avoid conflicts
        server.add_insecure_port(listen_addr)
        
        logger.info(f"Existing repo workflow gRPC server listening on {listen_addr}")
        return server
    
    def create_webhook_app(self):
        """Create Flask app for existing repository workflow."""
        app = Flask(__name__)
        
        @app.route('/configure', methods=['POST'])
        def configure_existing_repository():
            """Configure an existing repository for automatic processing."""
            try:
                data = request.get_json()
                logger.info(f"Repository configuration request: {data}")
                
                # Validate required fields based on selected endpoints
                code_provider = data.get("code_provider", "github").lower()
                docs_provider = data.get("docs_provider", "gitlab").lower()
                
                # Define required fields based on providers
                required_fields = []
                
                if code_provider == "github":
                    required_fields.extend(["github_repo_url", "github_token"])
                elif code_provider == "gitlab":
                    required_fields.extend(["gitlab_code_repo_url", "gitlab_code_token", "gitlab_code_project_id"])
                
                if docs_provider == "github":
                    required_fields.extend(["github_docs_repo_url", "github_docs_token"])
                elif docs_provider == "gitlab":
                    required_fields.extend(["gitlab_docs_repo_url", "gitlab_docs_token", "gitlab_docs_project_id"])
                
                # Always require docs folder
                required_fields.append("docs_folder")
                
                for field in required_fields:
                    if not data.get(field):
                        return jsonify({
                            "success": False,
                            "error": f"Missing required field: {field}"
                        }), 400
                
                # Generate unique configuration ID
                config_id = f"config_{uuid.uuid4().hex[:8]}"
                
                # Generate webhook secret for this configuration
                webhook_secret = f"secret_{uuid.uuid4().hex[:16]}"
                
                # Store repository configuration with flexible endpoints
                config = {
                    "code_provider": code_provider,
                    "docs_provider": docs_provider,
                    "docs_folder": data["docs_folder"],
                    "ollama_model": data.get("ollama_model", "qwen2.5:7b"),
                    "user_id": data.get("user_id"),
                    "webhook_secret": webhook_secret,
                    "configured_at": datetime.now().isoformat()
                }
                
                # Add code repository configuration
                if code_provider == "github":
                    config.update({
                        "code_repo_url": data["github_repo_url"],
                        "code_token": data["github_token"],
                        "code_branch": data.get("github_branch", "main")
                    })
                elif code_provider == "gitlab":
                    config.update({
                        "code_repo_url": data["gitlab_code_repo_url"],
                        "code_token": data["gitlab_code_token"],
                        "code_project_id": data["gitlab_code_project_id"],
                        "code_branch": data.get("gitlab_code_branch", "main")
                    })
                
                # Add documentation repository configuration
                if docs_provider == "github":
                    config.update({
                        "docs_repo_url": data["github_docs_repo_url"],
                        "docs_token": data["github_docs_token"],
                        "docs_branch": data.get("github_docs_branch", "main"),
                        "github_token": data["github_docs_token"],  # For compatibility
                        "github_docs_repo_url": data["github_docs_repo_url"],  # For compatibility
                        "github_docs_branch": data.get("github_docs_branch", "main")  # For compatibility
                    })
                elif docs_provider == "gitlab":
                    config.update({
                        "docs_repo_url": data["gitlab_docs_repo_url"],
                        "docs_token": data["gitlab_docs_token"],
                        "docs_project_id": data["gitlab_docs_project_id"],
                        "docs_branch": data.get("gitlab_docs_branch", "main"),
                        "gitlab_token": data["gitlab_docs_token"],  # For compatibility
                        "gitlab_docs_repo_url": data["gitlab_docs_repo_url"],  # For compatibility
                        "gitlab_project_id": data["gitlab_docs_project_id"],  # For compatibility
                        "gitlab_docs_branch": data.get("gitlab_docs_branch", "main")  # For compatibility
                    })
                
                self.configured_repos[config_id] = config
                
                # Generate webhook URL from environment variable
                ngrok_base_url = os.getenv('NGROK_BASE_URL', 'https://c6d4a8de5fed.ngrok-free.app')
                webhook_url = f"{ngrok_base_url}/webhook/{config_id}"
                
                # Generate setup instructions based on providers
                setup_instructions = [
                    f"1. Configure {code_provider.upper()} webhook: {webhook_url}",
                    f"2. Set webhook secret: {webhook_secret}",
                    "3. Set events: push, pull_request",
                    f"4. Just commit to your {code_provider} code repository!",
                    f"5. Documentation will update automatically in {docs_provider}"
                ]
                
                result = {
                    "success": True,
                    "message": f"Repository configured successfully! Code: {code_provider.upper()}, Docs: {docs_provider.upper()}",
                    "config_id": config_id,
                    "webhook_url": webhook_url,
                    "webhook_secret": webhook_secret,
                    "code_provider": code_provider,
                    "docs_provider": docs_provider,
                    "code_repo": config["code_repo_url"],
                    "docs_repo": config["docs_repo_url"],
                    "setup_instructions": setup_instructions
                }
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Configuration failed: {str(e)}")
                return jsonify({"success": False, "error": str(e)}), 500
        
        @app.route('/webhook/<config_id>', methods=['POST'])
        def handle_existing_repo_webhook(config_id):
            """Handle webhook for existing repository."""
            try:
                logger.info(f"🔔 WEBHOOK RECEIVED for config {config_id}")
                
                # Get webhook data - handle different content types
                try:
                    payload = request.get_json()
                except Exception as e:
                    # Try to parse JSON from raw data if get_json() fails
                    raw_data = request.get_data()
                    try:
                        payload = json.loads(raw_data.decode('utf-8'))
                    except Exception as json_error:
                        logger.error(f"Failed to parse webhook payload: {json_error}")
                        return jsonify({"success": False, "error": "Invalid JSON payload"}), 400
                
                headers = dict(request.headers)
                raw_data = request.get_data()
                
                # Detect webhook type based on headers
                webhook_type = self.detect_webhook_type(headers)
                
                # Display detailed webhook information
                print("=" * 80)
                print(f"🔔 {webhook_type.upper()} WEBHOOK DETAILS")
                print("=" * 80)
                print(f"📋 Config ID: {config_id}")
                print(f"📋 Headers: {json.dumps(headers, indent=2)}")
                print(f"📋 Raw Data Length: {len(raw_data)} bytes")
                print(f"📋 Payload: {json.dumps(payload, indent=2)}")
                print("=" * 80)
                
                # Check if config exists
                if config_id not in self.configured_repos:
                    logger.error(f"Configuration {config_id} not found!")
                    return jsonify({"success": False, "error": "Configuration not found"}), 404
                
                repo_config = self.configured_repos[config_id]
                
                # Process the webhook automatically
                result = self.process_existing_repo_webhook(config_id, payload, headers, repo_config)
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Webhook processing failed: {str(e)}")
                return jsonify({"success": False, "error": str(e)}), 500
        
        @app.route('/status/<config_id>', methods=['GET'])
        def get_configuration_status(config_id):
            """Get configuration status."""
            try:
                if config_id not in self.configured_repos:
                    return jsonify({"error": "Configuration not found"}), 404
                
                repo_config = self.configured_repos[config_id]
                
                # Get ngrok base URL from environment variable
                ngrok_base_url = os.getenv('NGROK_BASE_URL', 'https://c6d4a8de5fed.ngrok-free.app')
                
                status = {
                    "config_id": config_id,
                    "status": "active",
                    "webhook_url": f"{ngrok_base_url}/webhook/{config_id}",
                    "webhook_secret": repo_config.get("webhook_secret"),
                    "code_provider": repo_config.get("code_provider", "github"),
                    "docs_provider": repo_config.get("docs_provider", "gitlab"),
                    "code_repo": repo_config.get("code_repo_url", "N/A"),
                    "docs_repo": repo_config.get("docs_repo_url", "N/A"),
                    "docs_folder": repo_config["docs_folder"],
                    "configured_at": repo_config["configured_at"],
                    "automatic_processing": True,
                    "message": f"Your repository is configured for automatic processing! Code: {repo_config.get('code_provider', 'github').upper()}, Docs: {repo_config.get('docs_provider', 'gitlab').upper()}"
                }
                
                return jsonify(status)
                
            except Exception as e:
                logger.error(f"Status check failed: {str(e)}")
                return jsonify({"error": str(e)}), 500
        
        @app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({
                "status": "healthy", 
                "service": "existing-repository-workflow",
                "configured_repositories": len(self.configured_repos),
                "automatic_processing": True
            })
        
        @app.route('/configurations', methods=['GET'])
        def list_configurations():
            """List all configured repositories."""
            try:
                # Get ngrok base URL from environment variable
                ngrok_base_url = os.getenv('NGROK_BASE_URL', 'https://c6d4a8de5fed.ngrok-free.app')
                
                config_list = []
                for config_id, config in self.configured_repos.items():
                    config_list.append({
                        "config_id": config_id,
                        "code_provider": config.get("code_provider", "github"),
                        "docs_provider": config.get("docs_provider", "gitlab"),
                        "code_repo": config.get("code_repo_url", "N/A"),
                        "docs_repo": config.get("docs_repo_url", "N/A"),
                        "docs_folder": config["docs_folder"],
                        "configured_at": config["configured_at"],
                        "webhook_url": f"{ngrok_base_url}/webhook/{config_id}"
                    })
                
                return jsonify({
                    "success": True,
                    "configurations": config_list,
                    "total": len(config_list)
                })
                
            except Exception as e:
                logger.error(f"List configurations failed: {str(e)}")
                return jsonify({"success": False, "error": str(e)}), 500
        
        @app.route('/generate-embeddings/<config_id>', methods=['POST'])
        def generate_embeddings(config_id):
            """Generate embeddings for all documents in the configured GitLab folder."""
            try:
                logger.info(f"🔍 Generating embeddings for config {config_id}")
                
                # Check if config exists
                if config_id not in self.configured_repos:
                    logger.error(f"Configuration {config_id} not found!")
                    return jsonify({"success": False, "error": "Configuration not found"}), 404
                
                repo_config = self.configured_repos[config_id]
                
                # Generate embeddings
                result = self.process_document_embeddings(config_id, repo_config)
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Embeddings generation failed: {str(e)}")
                return jsonify({"success": False, "error": str(e)}), 500
        
        return app
    
    def detect_webhook_type(self, headers):
        """Detect webhook type based on headers."""
        github_event = headers.get('X-GitHub-Event')
        gitlab_event = headers.get('X-Gitlab-Event')
        
        if github_event:
            return 'github'
        elif gitlab_event:
            return 'gitlab'
        else:
            # Default to GitHub for backward compatibility
            return 'github'
    
    def verify_webhook_signature(self, payload, signature, secret):
        """Verify webhook signature for security."""
        if not signature or not secret:
            return False
        
        try:
            # GitHub sends signature as 'sha256=<hash>'
            if not signature.startswith('sha256='):
                return False
            
            # Extract the hash part
            received_hash = signature[7:]
            
            # Create expected signature
            expected_signature = hmac.new(
                secret.encode('utf-8'),
                payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Compare signatures securely
            return hmac.compare_digest(received_hash, expected_signature)
            
        except Exception as e:
            logger.error(f"Signature verification error: {e}")
            return False
    
    def process_existing_repo_webhook(self, config_id, payload, headers, repo_config):
        """Process webhook for existing repository."""
        logger.info(f"🤖 PROCESSING EXISTING REPO WEBHOOK for config {config_id}")
        
        # Verify webhook signature for security
        signature = headers.get('X-Hub-Signature-256') or headers.get('X-Hub-Signature')
        webhook_secret = repo_config.get('webhook_secret')
        
        if signature and webhook_secret:
            import json
            # Temporarily disable signature validation for debugging
            logger.info(f"⚠️ Signature validation disabled for debugging - config {config_id}")
            # if not self.verify_webhook_signature(json.dumps(payload), signature, webhook_secret):
            #     logger.warning(f"Invalid webhook signature for config {config_id}")
            #     return {
            #         "success": False,
            #         "message": "Invalid webhook signature"
            #     }
            # logger.info(f"✅ Webhook signature verified for config {config_id}")
        else:
            logger.warning(f"No webhook signature validation for config {config_id}")
        
        # Extract commit information based on webhook type
        webhook_type = self.detect_webhook_type(headers)
        commit_info = self.extract_commit_info(payload, headers, webhook_type)
        
        if not commit_info:
            return {
                "success": False,
                "message": "No commit information found in webhook"
            }
        
        # Check if we need to fetch file changes from API
        modified_files = commit_info.get('modified', [])
        added_files = commit_info.get('added', [])
        removed_files = commit_info.get('removed', [])
        
        print(f"🔍 DEBUG - Modified files: {modified_files}")
        print(f"🔍 DEBUG - Added files: {added_files}")
        print(f"🔍 DEBUG - Removed files: {removed_files}")
        print(f"🔍 DEBUG - Condition check: {not modified_files and not added_files and not removed_files}")
        
        # If no file changes found in webhook, try API based on code provider
        if not modified_files and not added_files and not removed_files:
            code_provider = repo_config.get('code_provider', 'github')
            print(f"🔍 No file changes found in webhook, fetching from {code_provider.upper()} API...")
            
            if code_provider == 'github':
                api_files = self.fetch_commit_files_from_github(repo_config, commit_info['hash'])
            elif code_provider == 'gitlab':
                api_files = self.fetch_commit_files_from_gitlab(repo_config, commit_info['hash'])
            else:
                api_files = None
            
            if api_files:
                commit_info['modified'] = api_files['modified']
                commit_info['added'] = api_files['added']
                commit_info['removed'] = api_files['removed']
                commit_info['diff'] = api_files.get('diff', '')
                commit_info['files'] = api_files.get('files', [])
                modified_files = api_files['modified']
                added_files = api_files['added']
                removed_files = api_files['removed']
        
        # If we have file changes but no diff content, fetch the diff
        elif (modified_files or added_files or removed_files) and not commit_info.get('diff'):
            code_provider = repo_config.get('code_provider', 'github')
            print(f"🔍 File changes found but no diff content, fetching diff from {code_provider.upper()} API...")
            
            if code_provider == 'github':
                api_files = self.fetch_commit_files_from_github(repo_config, commit_info['hash'])
            elif code_provider == 'gitlab':
                api_files = self.fetch_commit_files_from_gitlab(repo_config, commit_info['hash'])
            else:
                api_files = None
            
            if api_files and api_files.get('diff'):
                commit_info['diff'] = api_files['diff']
                print(f"📄 Diff content fetched: {len(api_files['diff'])} characters")
        
        # Display detailed commit information
        print("=" * 80)
        print("📝 COMMIT INFORMATION EXTRACTED")
        print("=" * 80)
        print(f"🔗 Commit Hash: {commit_info.get('hash', 'N/A')}")
        print(f"💬 Commit Message: {commit_info.get('message', 'N/A')}")
        print(f"👤 Author: {commit_info.get('author', 'N/A')}")
        print(f"⏰ Timestamp: {commit_info.get('timestamp', 'N/A')}")
        print(f"🌿 Branch: {commit_info.get('branch', 'N/A')}")
        print(f"🔗 Repository URL: {commit_info.get('url', 'N/A')}")
        print(f"📁 Modified Files: {modified_files}")
        print(f"➕ Added Files: {added_files}")
        print(f"➖ Removed Files: {removed_files}")
        
        # Display diff information if available
        diff_content = commit_info.get('diff', '')
        if diff_content:
            print(f"📄 Diff Content: {len(diff_content)} characters")
            # Show first 500 characters of diff
            diff_preview = diff_content[:500]
            print("📄 Diff Preview:")
            print("-" * 40)
            print(diff_preview)
            if len(diff_content) > 500:
                print(f"... (showing first 500 chars of {len(diff_content)} total)")
            print("-" * 40)
        else:
            print("📄 Diff Content: Not available")
        
        print("=" * 80)
        
        logger.info(f"📝 Processing commit: {commit_info['message']}")
        logger.info(f"🔗 Code Repo ({repo_config.get('code_provider', 'github').upper()}): {repo_config.get('code_repo_url', 'N/A')}")
        logger.info(f"📚 Docs Repo ({repo_config.get('docs_provider', 'gitlab').upper()}): {repo_config.get('docs_repo_url', 'N/A')}")
        
        # Real processing steps
        processing_steps = []
        results = []
        
        try:
            # Step 1: Store commit data in ArangoDB - TEMPORARILY DISABLED
            logger.info("💾 ArangoDB storage temporarily disabled...")
            processing_steps.append("⏸️ ArangoDB storage temporarily disabled")
            results.append("⏸️ ArangoDB storage temporarily disabled")
        except Exception as e:
            logger.error(f"ArangoDB error: {str(e)}")
            processing_steps.append(f"❌ ArangoDB error: {str(e)}")
            results.append(f"❌ ArangoDB error: {str(e)}")
        
        # Step 1.5: Create Full Documentation Backup
        try:
            logger.info("💾 Creating full documentation backup...")
            from core.gitlab_client import gitlab_client
            
            # Initialize documentation client based on provider
            docs_provider = repo_config.get('docs_provider', 'gitlab')
            docs_client = None
            
            if docs_provider == 'gitlab':
                from core.gitlab_client import gitlab_client
                gitlab_client.token = repo_config['docs_token']
                gitlab_client.project_id = repo_config['docs_project_id']
                gitlab_client.docs_branch = repo_config.get('docs_branch', 'main')
                gitlab_client.headers = {
                    'Authorization': f'Bearer {repo_config["docs_token"]}',
                    'Content-Type': 'application/json'
                }
                docs_client = gitlab_client
                logger.info("📚 Using GitLab for documentation storage")
            elif docs_provider == 'github':
                # Check if test mode is enabled
                test_mode = os.getenv('DISABLE_GITHUB_AUTH', 'false').lower() == 'true'
                github_client = GitHubClient(
                    token=repo_config['docs_token'],
                    repo_url=repo_config['docs_repo_url'],
                    test_mode=test_mode
                )
                docs_client = github_client
                logger.info("📚 Using GitHub for documentation storage")
            else:
                logger.error(f"❌ Unknown docs provider: {docs_provider}")
                return {
                    "success": False,
                    "message": f"Unknown docs provider: {docs_provider}"
                }
            
            # Create full backup
            full_backup_result = self._create_full_documentation_backup(
                docs_client, docs_provider, repo_config, commit_info, False
            )
            
            if full_backup_result['success']:
                logger.info(f"✅ Full backup created: {full_backup_result['backup_folder']}")
                processing_steps.append(f"💾 Full backup: {len(full_backup_result['backed_up_files'])} files")
                results.append(f"💾 Full backup: {len(full_backup_result['backed_up_files'])} files")
            else:
                logger.warning(f"⚠️ Full backup failed: {full_backup_result['error']}")
                processing_steps.append(f"⚠️ Full backup failed: {full_backup_result['error']}")
                results.append(f"⚠️ Full backup failed: {full_backup_result['error']}")
                    
        except Exception as e:
            logger.error(f"Full backup error: {str(e)}")
            processing_steps.append(f"❌ Full backup error: {str(e)}")
            results.append(f"❌ Full backup failed: {str(e)}")
        
        # Step 1.6: Enhanced LLM + RAG Processing
        try:
            logger.info("🤖 Starting enhanced LLM + RAG processing...")
            
            # Initialize services with configured providers
            logger.info("🤖 Initializing LLM service with configured provider...")
            from services.llm_service import LLMService
            from utils.config import get_settings
            
            # Use global configuration for LLM service
            settings = get_settings()
            llm_service = LLMService(settings.ai.model_dump())
                
            # Initialize RAG service
            from services.rag_service import RAGService
            rag_client = self._create_embeddings_client()
            if rag_client:
                rag_service = RAGService(rag_client, settings.ai.dict())
                
                # Step 1.6a: Analyze commit impact
                logger.info("🔍 Analyzing commit impact with LLM...")
                analysis_result = llm_service.analyze_commit_impact(commit_info)
                commit_info['analysis'] = analysis_result
                
                processing_steps.append(f"🤖 LLM Analysis: {analysis_result.get('impact_level', 'unknown')} impact")
                results.append(f"🤖 Impact Level: {analysis_result.get('impact_level', 'unknown')}")
                
                # Step 1.6b: Get available documentation files
                logger.info("📚 Getting available documentation files...")
                available_docs = rag_service.get_documentation_files(repo_config.get('docs_repo_url', repo_config.get('gitlab_docs_repo_url', '')).split('/')[-1])
                
                # Step 1.6c: Select files to update using LLM
                logger.info("🎯 Selecting documentation files with LLM...")
                selected_files = llm_service.select_documentation_files(analysis_result, available_docs)
                
                processing_steps.append(f"🎯 LLM Selected: {len(selected_files)} files")
                results.append(f"🎯 Files to update: {', '.join(selected_files)}")
                
                # Step 1.6d: Retrieve relevant context using RAG
                logger.info("🔍 Retrieving relevant context with RAG...")
                rag_context = rag_service.retrieve_relevant_context(commit_info)
                
                processing_steps.append("🔍 RAG context retrieved")
                results.append("🔍 Context retrieved from embeddings")
                
                # Store enhanced context for documentation updates
                commit_info['selected_files'] = selected_files
                commit_info['rag_context'] = rag_context
                commit_info['enhanced_processing'] = True
                
                logger.info("✅ Enhanced LLM + RAG processing completed")
                processing_steps.append("✅ Enhanced processing completed")
                results.append("✅ LLM + RAG processing successful")
            else:
                logger.warning("⚠️ Could not connect to embeddings database")
                processing_steps.append("⚠️ RAG service unavailable")
                results.append("⚠️ RAG service unavailable")
                    
        except Exception as e:
            logger.error(f"Enhanced processing error: {str(e)}")
            processing_steps.append(f"❌ Enhanced processing error: {str(e)}")
            results.append(f"❌ Enhanced processing failed: {str(e)}")
        
        try:
            # Step 2: Update documentation
            docs_provider = repo_config.get('docs_provider', 'gitlab')
            logger.info(f"📝 Updating {docs_provider.upper()} documentation...")
            
            # Initialize documentation client based on provider
            docs_client = None
            
            if docs_provider == 'gitlab':
                from core.gitlab_client import gitlab_client
                gitlab_client.token = repo_config['docs_token']
                gitlab_client.project_id = repo_config['docs_project_id']
                gitlab_client.docs_branch = repo_config.get('docs_branch', 'main')
                gitlab_client.headers = {
                    'Authorization': f'Bearer {repo_config["docs_token"]}',
                    'Content-Type': 'application/json'
                }
                docs_client = gitlab_client
            elif docs_provider == 'github':
                # Check if test mode is enabled
                test_mode = os.getenv('DISABLE_GITHUB_AUTH', 'false').lower() == 'true'
                github_client = GitHubClient(
                    token=repo_config['docs_token'],
                    repo_url=repo_config['docs_repo_url'],
                    test_mode=test_mode
                )
                docs_client = github_client
            else:
                logger.error(f"❌ Unknown docs provider: {docs_provider}")
                return {
                    "success": False,
                    "message": f"Unknown docs provider: {docs_provider}"
                }
            
            # Check if we have enhanced processing results
            if commit_info.get('enhanced_processing', False):
                # Use LLM-selected files
                files_to_update = commit_info.get('selected_files', [])
                logger.info(f"🎯 Using LLM-selected files: {files_to_update}")
                
                # Enhanced documentation update with LLM + RAG
                if files_to_update:
                    update_result = self._update_documentation_with_llm(
                        docs_client, docs_provider, repo_config, files_to_update, commit_info, llm_service, rag_service
                    )
                    
                    if update_result['success']:
                        processing_steps.append(f"✅ Enhanced documentation updated: {update_result['updated_files']}")
                        results.append(f"✅ LLM-enhanced update: {len(update_result['updated_files'])} files")
                    else:
                        processing_steps.append(f"❌ Enhanced update failed: {update_result['error']}")
                        results.append(f"❌ Enhanced update failed: {update_result['error']}")
                else:
                    processing_steps.append("ℹ️ LLM determined no documentation updates needed")
                    results.append("ℹ️ No files selected by LLM")
            else:
                # No enhanced processing available - skip documentation updates
                logger.info("ℹ️ Enhanced processing not available - skipping documentation updates")
                processing_steps.append("ℹ️ Enhanced processing not available - skipping documentation updates")
                results.append("ℹ️ Enhanced processing not available - skipping documentation updates")
                
        except Exception as e:
            logger.error(f"GitLab update error: {str(e)}")
            processing_steps.append(f"❌ GitLab update error: {str(e)}")
            results.append(f"❌ GitLab update error: {str(e)}")
        
        # Return comprehensive result
        result = {
            "success": True,
            "message": f"Commit processed automatically for existing repository",
            "config_id": config_id,
            "commit_info": commit_info,
            "code_repo": repo_config.get("code_repo_url", "N/A"),
            "docs_repo": repo_config.get("docs_repo_url", "N/A"),
            "code_provider": repo_config.get("code_provider", "github"),
            "docs_provider": repo_config.get("docs_provider", "gitlab"),
            "processing_steps": results,
            "processed_at": datetime.now().isoformat(),
            "automatic": True,
            "next_steps": [
                f"Check {repo_config.get('docs_provider', 'gitlab').upper()} repo: {repo_config.get('docs_repo_url', 'N/A')}",
                f"Review changes in {repo_config['docs_folder']} folder",
                "Commit was processed automatically!"
            ]
        }
        
        logger.info(f"✅ EXISTING REPO PROCESSING COMPLETED for config {config_id}")
        return result
    
    def process_document_embeddings(self, config_id, repo_config):
        """Process document embeddings for documents from any provider."""
        logger.info(f"🔍 PROCESSING DOCUMENT EMBEDDINGS for config {config_id}")
        
        try:
            # Step 1: Connect to ArangoDB (use mcp_documents database for embeddings)
            logger.info("💾 Connecting to ArangoDB...")
            from core.arango_client import ArangoDBClient
            
            # Create a separate client for embeddings (without auto-setup)
            embeddings_client = self._create_embeddings_client()
            
            if not embeddings_client:
                return {
                    "success": False,
                    "message": "Failed to connect to ArangoDB"
                }
            
            # Step 2: Setup collections for embeddings
            self._setup_embeddings_collections(embeddings_client)
            
            # Step 3: Fetch documents based on docs provider
            docs_provider = repo_config.get('docs_provider', 'gitlab').lower()
            logger.info(f"📚 Fetching documents from {docs_provider.upper()}...")
            
            if docs_provider == 'gitlab':
                documents = self._fetch_gitlab_documents(repo_config)
            elif docs_provider == 'github':
                documents = self._fetch_github_documents(repo_config)
            else:
                return {
                    "success": False,
                    "message": f"Unsupported docs provider: {docs_provider}"
                }
            
            if not documents:
                return {
                    "success": False,
                    "message": f"No documents found in {docs_provider.upper()} folder"
                }
            
            # Step 4: Store repository information
            repo_key = self._store_repository_info(embeddings_client, repo_config)
            
            # Step 4.1: Clear existing data for this repository if it exists
            self._clear_repository_data(embeddings_client, repo_key)
            
            # Step 4.5: Fix any missing relationships for existing embeddings
            self._fix_missing_relationships(embeddings_client, repo_key)
            
            # Step 5: Process each document
            processed_docs = []
            
            # Initialize LLM service with configured provider
            from services.llm_service import LLMService
            from utils.config import get_settings
            
            settings = get_settings()
            llm_service = LLMService(settings.ai.model_dump())
            
            # DEBUG: Log configuration details
            logger.info(f"🔍 DEBUG: OpenAI API Key length: {len(settings.ai.openai_api_key) if settings.ai.openai_api_key else 0}")
            logger.info(f"🔍 DEBUG: OpenAI API Key starts with: {settings.ai.openai_api_key[:20]}..." if settings.ai.openai_api_key else "🔍 DEBUG: OpenAI API Key: NOT_FOUND")
            
            for doc in documents:
                logger.info(f"📄 Processing document: {doc['path']}")
                
                # Generate embeddings using configured provider
                from services.rag_service import RAGService
                rag_service = RAGService(embeddings_client, settings.ai.dict())
                
                # DEBUG: Log embeddings provider details
                logger.info(f"🔍 DEBUG: Embeddings provider API key length: {len(rag_service.embeddings_provider.api_key) if rag_service.embeddings_provider.api_key else 0}")
                logger.info(f"🔍 DEBUG: Embeddings provider API key starts with: {rag_service.embeddings_provider.api_key[:20]}..." if rag_service.embeddings_provider.api_key else "🔍 DEBUG: Embeddings provider API key: NOT_FOUND")
                
                embeddings = rag_service.embeddings_provider.generate_embeddings(doc['content'])
                
                if embeddings:
                    # Store document
                    doc_key = self._store_document(embeddings_client, doc, repo_key)
                    
                    # Store embeddings
                    embedding_key = self._store_embeddings(embeddings_client, embeddings, doc_key)
                    
                    # Create relationships
                    self._create_embeddings_relationships(embeddings_client, repo_key, doc_key, embedding_key)
                    
                    processed_docs.append({
                        "path": doc['path'],
                        "size": len(doc['content']),
                        "embedding_dimensions": len(embeddings)
                    })
            
            result = {
                "success": True,
                "message": f"Successfully processed {len(processed_docs)} documents",
                "config_id": config_id,
                "repository": repo_config.get('docs_repo_url', repo_config.get('gitlab_docs_repo_url', 'N/A')),
                "docs_folder": repo_config['docs_folder'],
                "processed_documents": processed_docs,
                "total_documents": len(documents),
                "processed_at": datetime.now().isoformat()
            }
            
            logger.info(f"✅ EMBEDDINGS PROCESSING COMPLETED for config {config_id}")
            return result
            
        except Exception as e:
            logger.error(f"Embeddings processing error: {str(e)}")
            return {
                "success": False,
                "message": f"Embeddings processing failed: {str(e)}"
            }
    
    def fetch_commit_files_from_github(self, repo_config, commit_hash):
        """Fetch file changes from GitHub API if not included in webhook."""
        try:
            import requests
            
            # Extract repo owner and name from GitHub URL
            github_url = repo_config['code_repo_url']
            if 'github.com' in github_url:
                repo_path = github_url.replace('https://github.com/', '').replace('.git', '')
                owner, repo_name = repo_path.split('/')
                
                # GitHub API endpoint for commit details
                api_url = f"https://api.github.com/repos/{owner}/{repo_name}/commits/{commit_hash}"
                
                headers = {
                    'Authorization': f"token {repo_config['code_token']}",
                    'Accept': 'application/vnd.github.v3+json'
                }
                
                print(f"🔍 Fetching commit details from GitHub API: {api_url}")
                response = requests.get(api_url, headers=headers)
                
                # Also fetch the diff/patch
                diff_url = f"https://api.github.com/repos/{owner}/{repo_name}/commits/{commit_hash}"
                diff_headers = {
                    'Authorization': f"token {repo_config['code_token']}",
                    'Accept': 'application/vnd.github.v3.diff'
                }
                
                print(f"🔍 Fetching commit diff from GitHub API...")
                diff_response = requests.get(diff_url, headers=diff_headers)
                
                if response.status_code == 200:
                    commit_data = response.json()
                    files = commit_data.get('files', [])
                    
                    modified_files = [f['filename'] for f in files if f['status'] == 'modified']
                    added_files = [f['filename'] for f in files if f['status'] == 'added']
                    removed_files = [f['filename'] for f in files if f['status'] == 'removed']
                    
                    print(f"📁 GitHub API - Modified: {modified_files}")
                    print(f"➕ GitHub API - Added: {added_files}")
                    print(f"➖ GitHub API - Removed: {removed_files}")
                    
                    # Get the diff content
                    diff_content = ""
                    if diff_response.status_code == 200:
                        diff_content = diff_response.text
                        print(f"📄 Diff content length: {len(diff_content)} characters")
                        
                        # Display first 1000 characters of diff for preview
                        if diff_content:
                            print("=" * 80)
                            print("📄 COMMIT DIFF PREVIEW")
                            print("=" * 80)
                            preview = diff_content[:1000]
                            print(preview)
                            if len(diff_content) > 1000:
                                print(f"... (showing first 1000 chars of {len(diff_content)} total)")
                            print("=" * 80)
                    else:
                        print(f"❌ Failed to fetch diff: {diff_response.status_code}")
                    
                    return {
                        'modified': modified_files,
                        'added': added_files,
                        'removed': removed_files,
                        'diff': diff_content,
                        'files': files  # Include full file details
                    }
                else:
                    print(f"❌ GitHub API request failed: {response.status_code}")
                    return None
                    
        except Exception as e:
            print(f"❌ Error fetching from GitHub API: {e}")
            return None
    
    def fetch_commit_files_from_gitlab(self, repo_config, commit_hash):
        """Fetch file changes from GitLab API if not included in webhook."""
        try:
            import requests
            
            # GitLab API endpoint for commit details
            project_id = repo_config['code_project_id']
            api_url = f"https://gitlab.com/api/v4/projects/{project_id}/repository/commits/{commit_hash}"
            
            headers = {
                'Authorization': f"Bearer {repo_config['code_token']}",
                'Content-Type': 'application/json'
            }
            
            print(f"🔍 Fetching commit details from GitLab API: {api_url}")
            response = requests.get(api_url, headers=headers)
            
            if response.status_code == 200:
                commit_data = response.json()
                
                # Get commit diff
                diff_url = f"https://gitlab.com/api/v4/projects/{project_id}/repository/commits/{commit_hash}/diff"
                diff_response = requests.get(diff_url, headers=headers)
                
                modified_files = []
                added_files = []
                removed_files = []
                diff_content = ""
                
                if diff_response.status_code == 200:
                    diff_data = diff_response.json()
                    for change in diff_data:
                        if change['new_file']:
                            added_files.append(change['new_path'])
                        elif change['deleted_file']:
                            removed_files.append(change['old_path'])
                        else:
                            modified_files.append(change['new_path'])
                    
                    # Build diff content
                    diff_parts = []
                    for change in diff_data:
                        diff_parts.append(f"diff --git a/{change.get('old_path', '')} b/{change.get('new_path', '')}")
                        diff_parts.append(change.get('diff', ''))
                    diff_content = '\n'.join(diff_parts)
                
                print(f"📁 GitLab API - Modified: {modified_files}")
                print(f"➕ GitLab API - Added: {added_files}")
                print(f"➖ GitLab API - Removed: {removed_files}")
                
                if diff_content:
                    print(f"📄 Diff content length: {len(diff_content)} characters")
                    # Display first 1000 characters of diff for preview
                    print("=" * 80)
                    print("📄 COMMIT DIFF PREVIEW")
                    print("=" * 80)
                    preview = diff_content[:1000]
                    print(preview)
                    if len(diff_content) > 1000:
                        print(f"... (showing first 1000 chars of {len(diff_content)} total)")
                    print("=" * 80)
                
                return {
                    'modified': modified_files,
                    'added': added_files,
                    'removed': removed_files,
                    'diff': diff_content,
                    'files': diff_data if diff_response.status_code == 200 else []
                }
            else:
                print(f"❌ GitLab API request failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching from GitLab API: {e}")
            return None
    
    def _setup_embeddings_collections(self, arango_client):
        """Setup collections needed for embeddings."""
        try:
            # Clean up old collections first
            self._cleanup_old_collections(arango_client)
            
            # Create only the collections we need for embeddings
            collections_to_create = [
                ('repos', 'document'),
                ('documents', 'document'), 
                ('embeddings', 'document'),
                ('repo_to_documents', 'edge'),
                ('documents_to_embeddings', 'edge')
            ]
            
            for collection_name, collection_type in collections_to_create:
                if not arango_client.has_collection(collection_name):
                    print(f"📁 Creating collection: {collection_name}")
                    if collection_type == 'edge':
                        arango_client.create_collection(collection_name, edge=True)
                    else:
                        arango_client.create_collection(collection_name)
                else:
                    print(f"📁 Collection {collection_name} already exists")
            
            # Create unified graph for embeddings
            graph_name = 'embeddings_graph'
            if not arango_client.has_graph(graph_name):
                print(f"🕸️ Creating unified embeddings graph: {graph_name}")
                edge_definitions = [
                    {
                        'edge_collection': 'repo_to_documents',
                        'from_vertex_collections': ['repos'],
                        'to_vertex_collections': ['documents']
                    },
                    {
                        'edge_collection': 'documents_to_embeddings',
                        'from_vertex_collections': ['documents'],
                        'to_vertex_collections': ['embeddings']
                    }
                ]
                arango_client.create_graph(graph_name, edge_definitions)
                print(f"✅ Unified embeddings graph created successfully")
            else:
                print(f"🕸️ Unified embeddings graph already exists")
                
        except Exception as e:
            print(f"❌ Failed to setup embeddings collections: {str(e)}")
            raise
    
    def _create_embeddings_client(self):
        """Create a simple ArangoDB client for embeddings without auto-setup."""
        try:
            import os
            from arango import ArangoClient
            
            # Use mcp_documents database
            host = os.getenv('ARANGO_HOST', 'localhost')
            port = int(os.getenv('ARANGO_PORT', '8529'))
            username = os.getenv('ARANGO_USERNAME', 'root')
            password = os.getenv('ARANGO_PASSWORD', 'openSesame')
            database_name = 'mcp_documents'
            
            print(f"🔗 Connecting to ArangoDB at {host}:{port}")
            
            # Initialize client
            client = ArangoClient(hosts=f'http://{host}:{port}')
            
            # Connect to system database
            sys_db = client.db('_system', username=username, password=password)
            
            # Create database if it doesn't exist
            if not sys_db.has_database(database_name):
                print(f"📊 Creating database: {database_name}")
                sys_db.create_database(database_name)
            else:
                print(f"📊 Database {database_name} already exists")
            
            # Connect to our database
            db = client.db(database_name, username=username, password=password)
            
            print("✅ ArangoDB connection established successfully")
            return db
            
        except Exception as e:
            print(f"❌ Failed to connect to ArangoDB: {str(e)}")
            return None
    
    def _cleanup_old_collections(self, arango_client):
        """Clean up old/unused collections."""
        try:
            # Collections to remove (old ones we don't need)
            collections_to_remove = [
                'commits', 'authors', 'repositories', 'files',
                'commit_to_document', 'document_to_file', 'author_to_commit', 'repo_to_commit'
            ]
            
            # Remove old graphs first
            old_graphs = ['document_graph']
            for graph_name in old_graphs:
                if arango_client.has_graph(graph_name):
                    print(f"🗑️ Removing old graph: {graph_name}")
                    arango_client.delete_graph(graph_name)
            
            # Remove old collections
            for collection_name in collections_to_remove:
                if arango_client.has_collection(collection_name):
                    print(f"🗑️ Removing old collection: {collection_name}")
                    arango_client.delete_collection(collection_name)
            
            print("✅ Cleanup of old collections completed")
            
        except Exception as e:
            print(f"❌ Failed to cleanup old collections: {str(e)}")
            # Don't raise - cleanup failure shouldn't stop the process
    
    def _fetch_gitlab_documents(self, repo_config):
        """Fetch all documents from GitLab folder."""
        try:
            import requests
            import base64
            
            # GitLab API endpoint for repository tree
            gitlab_url = repo_config.get('docs_repo_url', repo_config.get('gitlab_docs_repo_url', ''))
            project_id = repo_config['gitlab_project_id']
            docs_folder = repo_config['docs_folder']
            branch = repo_config.get('gitlab_docs_branch', 'main')
            
            # Extract project path from GitLab URL
            if 'gitlab.com' in gitlab_url:
                project_path = gitlab_url.replace('https://gitlab.com/', '').replace('.git', '')
            else:
                # For self-hosted GitLab
                project_path = gitlab_url.split('/')[-1].replace('.git', '')
            
            # Get repository tree
            tree_url = f"https://gitlab.com/api/v4/projects/{project_id}/repository/tree"
            headers = {
                'Authorization': f"Bearer {repo_config['gitlab_token']}",
                'Content-Type': 'application/json'
            }
            
            params = {
                'path': docs_folder,
                'ref': branch,
                'recursive': True
            }
            
            print(f"🔍 Fetching GitLab tree from: {tree_url}")
            response = requests.get(tree_url, headers=headers, params=params)
            
            if response.status_code != 200:
                print(f"❌ Failed to fetch GitLab tree: {response.status_code}")
                return []
            
            tree_data = response.json()
            documents = []
            
            # Filter for text files
            text_extensions = ['.md', '.txt', '.rst', '.adoc', '.docx', '.pdf']
            
            for item in tree_data:
                if item['type'] == 'blob':  # It's a file
                    file_path = item['path']
                    file_name = item['name']
                    
                    # Check if it's a text file
                    if any(file_name.lower().endswith(ext) for ext in text_extensions):
                        # Get file content
                        file_url = f"https://gitlab.com/api/v4/projects/{project_id}/repository/files/{file_path.replace('/', '%2F')}/raw"
                        file_response = requests.get(file_url, headers=headers, params={'ref': branch})
                        
                        if file_response.status_code == 200:
                            content = file_response.text
                            documents.append({
                                'path': file_path,
                                'name': file_name,
                                'content': content,
                                'size': len(content),
                                'last_modified': item.get('last_activity_at', '')
                            })
                            print(f"📄 Fetched document: {file_path} ({len(content)} chars)")
                        else:
                            print(f"❌ Failed to fetch content for {file_path}: {file_response.status_code}")
            
            print(f"✅ Fetched {len(documents)} documents from GitLab")
            return documents
            
        except Exception as e:
            print(f"❌ Error fetching GitLab documents: {str(e)}")
            return []
    
    def _fetch_github_documents(self, repo_config):
        """Fetch all documents from GitHub folder."""
        try:
            import requests
            import base64
            
            # GitHub API endpoint for repository contents
            github_url = repo_config.get('docs_repo_url', repo_config.get('github_docs_repo_url', ''))
            docs_folder = repo_config['docs_folder']
            branch = repo_config.get('github_docs_branch', 'main')
            
            # Extract owner and repo from GitHub URL
            if 'github.com' in github_url:
                # Remove .git if present and split
                clean_url = github_url.replace('.git', '')
                parts = clean_url.split('/')
                if len(parts) >= 2:
                    owner = parts[-2]
                    repo = parts[-1]
                else:
                    print(f"❌ Invalid GitHub URL format: {github_url}")
                    return []
            else:
                print(f"❌ Not a GitHub URL: {github_url}")
                return []
            
            # Get repository contents
            contents_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{docs_folder}"
            headers = {
                'Authorization': f"Bearer {repo_config['github_token']}",
                'Accept': 'application/vnd.github.v3+json'
            }
            
            params = {
                'ref': branch
            }
            
            print(f"🔍 Fetching GitHub contents from: {contents_url}")
            response = requests.get(contents_url, headers=headers, params=params)
            
            if response.status_code != 200:
                print(f"❌ Failed to fetch GitHub contents: {response.status_code}")
                print(f"Response: {response.text}")
                return []
            
            contents_data = response.json()
            documents = []
            
            # Filter for text files
            text_extensions = ['.md', '.txt', '.rst', '.adoc', '.docx', '.pdf']
            
            for item in contents_data:
                if item['type'] == 'file':  # It's a file
                    file_path = item['path']
                    file_name = item['name']
                    
                    # Check if it's a text file
                    if any(file_name.lower().endswith(ext) for ext in text_extensions):
                        # Get file content
                        file_url = item['download_url']
                        if file_url:
                            file_response = requests.get(file_url, headers=headers)
                            
                            if file_response.status_code == 200:
                                # Handle DOCX files specially
                                if file_name.lower().endswith('.docx'):
                                    # Get binary content for DOCX files
                                    content_bytes = file_response.content
                                    from utils.docx_handler import DOCXHandler
                                    content = DOCXHandler.extract_text_from_docx(content_bytes)
                                else:
                                    # Handle text files normally
                                    content = file_response.text
                                
                                documents.append({
                                    'path': file_path,
                                    'name': file_name,
                                    'content': content,
                                    'size': len(content),
                                    'last_modified': item.get('updated_at', '')
                                })
                                print(f"📄 Fetched document: {file_path} ({len(content)} chars)")
                            else:
                                print(f"❌ Failed to fetch content for {file_path}: {file_response.status_code}")
                        else:
                            # Try to get content from API if no download_url
                            file_api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
                            file_response = requests.get(file_api_url, headers=headers, params={'ref': branch})
                            
                            if file_response.status_code == 200:
                                file_data = file_response.json()
                                if file_data.get('content'):
                                    # Handle DOCX files specially
                                    if file_name.lower().endswith('.docx'):
                                        # Decode base64 binary content for DOCX files
                                        content_bytes = base64.b64decode(file_data['content'])
                                        from utils.docx_handler import DOCXHandler
                                        content = DOCXHandler.extract_text_from_docx(content_bytes)
                                    else:
                                        # Decode base64 text content for text files
                                        content = base64.b64decode(file_data['content']).decode('utf-8')
                                    
                                    documents.append({
                                        'path': file_path,
                                        'name': file_name,
                                        'content': content,
                                        'size': len(content),
                                        'last_modified': file_data.get('updated_at', '')
                                    })
                                    print(f"📄 Fetched document: {file_path} ({len(content)} chars)")
                                else:
                                    print(f"❌ No content found for {file_path}")
                            else:
                                print(f"❌ Failed to fetch content for {file_path}: {file_response.status_code}")
                elif item['type'] == 'dir':  # It's a directory, recurse into it
                    # For now, we'll skip subdirectories to keep it simple
                    # You can extend this to handle nested directories if needed
                    print(f"📁 Skipping directory: {item['path']}")
            
            print(f"✅ Fetched {len(documents)} documents from GitHub")
            return documents
            
        except Exception as e:
            print(f"❌ Error fetching GitHub documents: {str(e)}")
            return []
    
    def _generate_ollama_embeddings(self, content, unused_api_key=None):
        """Generate embeddings using Ollama with GPU acceleration."""
        try:
            import requests
            
            # Use Ollama for embeddings
            ollama_base_url = "http://localhost:11434"
            embedding_model = "nomic-embed-text"  # Good local embedding model
            
            # Truncate content if too long
            max_chars = 8000  # Leave room for tokenization
            if len(content) > max_chars:
                content = content[:max_chars]
            
            payload = {
                "model": embedding_model,
                "prompt": content,
                "options": {
                    "gpu_layers": -1,  # Use all available GPU layers
                    "num_gpu": 1        # Use 1 GPU
                }
            }
            
            response = requests.post(
                f"{ollama_base_url}/api/embeddings",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                embeddings = result['embedding']
                print(f"✅ Generated embeddings: {len(embeddings)} dimensions")
                return embeddings
            else:
                print(f"❌ Ollama embedding API error: {response.status_code}")
                return None
            
        except Exception as e:
            print(f"❌ Error generating embeddings: {str(e)}")
            return None
    
    def _store_repository_info(self, arango_client, repo_config):
        """Store repository information in ArangoDB."""
        try:
            repo_key = repo_config.get('docs_repo_url', repo_config.get('gitlab_docs_repo_url', '')).split('/')[-1].replace('.git', '').replace('-', '_')
            if not repo_key[0].isalpha():
                repo_key = 'r' + repo_key[1:]
            
            repo_doc = {
                '_key': repo_key,
                'url': repo_config.get('docs_repo_url', repo_config.get('gitlab_docs_repo_url', '')),
                'name': repo_key,
                'docs_folder': repo_config['docs_folder'],
                'branch': repo_config.get('gitlab_docs_branch', 'main'),
                'created_at': datetime.now().isoformat()
            }
            
            # Insert or update repository
            if arango_client.collection('repos').find({'_key': repo_key}):
                arango_client.collection('repos').update({'_key': repo_key}, repo_doc)
            else:
                arango_client.collection('repos').insert(repo_doc)
            
            print(f"📦 Stored repository: {repo_key}")
            return repo_key
            
        except Exception as e:
            print(f"❌ Error storing repository: {str(e)}")
            raise
    
    def _store_document(self, arango_client, doc, repo_key):
        """Store document information in ArangoDB."""
        try:
            doc_key = doc['path'].replace('/', '_').replace('.', '_').replace('-', '_')
            if not doc_key[0].isalpha():
                doc_key = 'd' + doc_key[1:]
            
            doc_doc = {
                '_key': doc_key,
                'path': doc['path'],
                'name': doc['name'],
                'content': doc['content'],
                'size': doc['size'],
                'last_modified': doc.get('last_modified', ''),
                'created_at': datetime.now().isoformat()
            }
            
            # Insert or update document
            if arango_client.collection('documents').find({'_key': doc_key}):
                arango_client.collection('documents').update({'_key': doc_key}, doc_doc)
            else:
                arango_client.collection('documents').insert(doc_doc)
            
            print(f"📄 Stored document: {doc_key}")
            return doc_key
            
        except Exception as e:
            print(f"❌ Error storing document: {str(e)}")
            raise
    
    def _store_embeddings(self, arango_client, embeddings, doc_key):
        """Store embeddings in ArangoDB."""
        try:
            embedding_key = f"{doc_key}_embeddings"
            
            embedding_doc = {
                '_key': embedding_key,
                'document_key': doc_key,
                'embeddings': embeddings,
                'dimensions': len(embeddings),
                'model': 'nomic-embed-text',  # Using Ollama embedding model
                'created_at': datetime.now().isoformat()
            }
            
            # Insert or update embeddings
            if arango_client.collection('embeddings').find({'_key': embedding_key}):
                arango_client.collection('embeddings').update({'_key': embedding_key}, embedding_doc)
            else:
                arango_client.collection('embeddings').insert(embedding_doc)
            
            print(f"🧠 Stored embeddings: {embedding_key}")
            return embedding_key
            
        except Exception as e:
            print(f"❌ Error storing embeddings: {str(e)}")
            raise
    
    def _create_embeddings_relationships(self, arango_client, repo_key, doc_key, embedding_key):
        """Create relationships between repo, documents, and embeddings."""
        try:
            # Repository to Document relationship (check for duplicates)
            repo_to_doc_query = f"""
            FOR edge IN repo_to_documents
            FILTER edge._from == @from AND edge._to == @to
            RETURN edge
            """
            existing_repo_doc = list(arango_client.aql.execute(repo_to_doc_query, bind_vars={
                'from': f'repos/{repo_key}',
                'to': f'documents/{doc_key}'
            }))
            
            if not existing_repo_doc:
                repo_to_doc_edge = {
                    '_from': f'repos/{repo_key}',
                    '_to': f'documents/{doc_key}',
                    'relationship': 'contains',
                    'created_at': datetime.now().isoformat()
                }
                arango_client.collection('repo_to_documents').insert(repo_to_doc_edge)
                print(f"🔗 Created repo→document relationship: {repo_key} → {doc_key}")
            else:
                print(f"🔗 Repo→document relationship already exists: {repo_key} → {doc_key}")
            
            # Document to Embeddings relationship (check for duplicates)
            doc_to_emb_query = f"""
            FOR edge IN documents_to_embeddings
            FILTER edge._from == @from AND edge._to == @to
            RETURN edge
            """
            existing_doc_emb = list(arango_client.aql.execute(doc_to_emb_query, bind_vars={
                'from': f'documents/{doc_key}',
                'to': f'embeddings/{embedding_key}'
            }))
            
            if not existing_doc_emb:
                doc_to_emb_edge = {
                    '_from': f'documents/{doc_key}',
                    '_to': f'embeddings/{embedding_key}',
                    'relationship': 'has_embeddings',
                    'created_at': datetime.now().isoformat()
                }
                arango_client.collection('documents_to_embeddings').insert(doc_to_emb_edge)
                print(f"🔗 Created document→embedding relationship: {doc_key} → {embedding_key}")
            else:
                print(f"🔗 Document→embedding relationship already exists: {doc_key} → {embedding_key}")
            
        except Exception as e:
            print(f"❌ Error creating relationships: {str(e)}")
            raise
    
    def _fix_missing_relationships(self, arango_client, repo_key):
        """Fix missing relationships for existing embeddings."""
        try:
            print("🔧 Checking for missing relationships...")
            
            # Find all embeddings that don't have document relationships
            missing_relationships_query = """
            FOR emb IN embeddings
            LET doc_key = emb.document_key
            LET doc_exists = (FOR doc IN documents FILTER doc._key == doc_key RETURN 1)
            LET rel_exists = (FOR rel IN documents_to_embeddings FILTER rel._to == emb._id RETURN 1)
            FILTER LENGTH(doc_exists) > 0 AND LENGTH(rel_exists) == 0
            RETURN {embedding: emb, document_key: doc_key}
            """
            
            missing_rels = list(arango_client.aql.execute(missing_relationships_query))
            
            if missing_rels:
                print(f"🔧 Found {len(missing_rels)} embeddings with missing relationships")
                
                for item in missing_rels:
                    embedding = item['embedding']
                    doc_key = item['document_key']
                    embedding_key = embedding['_key']
                    
                    # Create the missing document→embedding relationship
                    doc_to_emb_edge = {
                        '_from': f'documents/{doc_key}',
                        '_to': f'embeddings/{embedding_key}',
                        'relationship': 'has_embeddings',
                        'created_at': datetime.now().isoformat()
                    }
                    arango_client.collection('documents_to_embeddings').insert(doc_to_emb_edge)
                    print(f"🔧 Fixed missing relationship: {doc_key} → {embedding_key}")
            else:
                print("✅ All embeddings have proper relationships")
                
        except Exception as e:
            print(f"❌ Error fixing relationships: {str(e)}")
            # Don't raise - this is a fix operation, shouldn't stop the main process
    
    def _clear_repository_data(self, arango_client, repo_key):
        """Clear all existing data for a repository before processing new data."""
        try:
            print(f"🧹 Clearing existing data for repository: {repo_key}")
            
            # Find all documents associated with this repository through the edge collection
            documents_query = """
            FOR rel IN repo_to_documents
            FILTER rel._from == @repo_id
            LET doc = DOCUMENT(rel._to)
            RETURN doc._key
            """
            
            document_keys = list(arango_client.aql.execute(documents_query, bind_vars={'repo_id': f'repos/{repo_key}'}))
            
            if document_keys:
                print(f"🧹 Found {len(document_keys)} existing documents to clear")
                
                # Delete all embeddings and relationships for these documents
                for doc_key in document_keys:
                    print(f"🧹 Clearing document: {doc_key}")
                    
                    # Find embeddings for this document
                    embeddings_query = """
                    FOR emb IN embeddings
                    FILTER emb.document_key == @doc_key
                    RETURN emb._key
                    """
                    
                    embedding_keys = list(arango_client.aql.execute(embeddings_query, bind_vars={'doc_key': doc_key}))
                    print(f"🧹 Found {len(embedding_keys)} embeddings for document {doc_key}")
                    
                    # Delete embedding relationships and embeddings
                    for emb_key in embedding_keys:
                        print(f"🧹 Deleting embedding: {emb_key}")
                        
                        # Delete document→embedding relationships
                        doc_to_emb_query = """
                        FOR rel IN documents_to_embeddings
                        FILTER rel._to == @emb_id
                        REMOVE rel IN documents_to_embeddings
                        RETURN OLD
                        """
                        deleted_rels = list(arango_client.aql.execute(doc_to_emb_query, bind_vars={'emb_id': f'embeddings/{emb_key}'}))
                        print(f"🧹 Deleted {len(deleted_rels)} document→embedding relationships")
                        
                        # Delete the embedding itself
                        try:
                            arango_client.collection('embeddings').delete(emb_key)
                            print(f"🧹 Deleted embedding: {emb_key}")
                        except Exception as e:
                            print(f"⚠️ Could not delete embedding {emb_key}: {e}")
                    
                    # Delete repo→document relationships
                    repo_to_doc_query = """
                    FOR rel IN repo_to_documents
                    FILTER rel._to == @doc_id
                    REMOVE rel IN repo_to_documents
                    RETURN OLD
                    """
                    deleted_repo_rels = list(arango_client.aql.execute(repo_to_doc_query, bind_vars={'doc_id': f'documents/{doc_key}'}))
                    print(f"🧹 Deleted {len(deleted_repo_rels)} repo→document relationships")
                    
                    # Delete the document itself
                    try:
                        arango_client.collection('documents').delete(doc_key)
                        print(f"🧹 Deleted document: {doc_key}")
                    except Exception as e:
                        print(f"⚠️ Could not delete document {doc_key}: {e}")
                
                print(f"✅ Cleared {len(document_keys)} documents and their embeddings")
            else:
                print("ℹ️ No existing documents found for this repository")
                
        except Exception as e:
            print(f"❌ Error clearing repository data: {str(e)}")
            # Don't raise - this is a cleanup operation, shouldn't stop the main process
    
    def _update_documentation_with_llm(self, docs_client, docs_provider, repo_config, files_to_update, commit_info, llm_service, rag_service):
        """Update documentation using LLM + RAG enhanced processing."""
        try:
            logger.info("🤖 Updating documentation with LLM + RAG...")
            
            # Services are already initialized and passed as parameters
            
            updated_files = []
            failed_files = []
            
            # Process each selected file for updates
            for file_path in files_to_update:
                try:
                    logger.info(f"📝 Processing file: {file_path}")
                    
                    # Get current document content
                    current_content = rag_service.get_document_content(file_path)
                    if not current_content:
                        logger.warning(f"⚠️ Could not retrieve content for {file_path}")
                        failed_files.append(file_path)
                        continue
                    
                    # Store original content for DOCX files
                    original_content = None
                    if file_path.lower().endswith('.docx'):
                        # For DOCX files, we need the original binary content
                        try:
                            # Get original binary content from GitLab
                            original_content = self._get_original_docx_content(file_path, repo_config)
                        except Exception as e:
                            logger.warning(f"⚠️ Could not get original DOCX content for {file_path}: {str(e)}")
                            original_content = None
                    
                    # Note: Full backup already created above, no need for individual file backups
                    
                    # Prepare commit context for LLM
                    commit_context = {
                        'message': commit_info['message'],
                        'author': commit_info['author'],
                        'modified': commit_info['modified'],
                        'diff': commit_info.get('diff', ''),
                        'analysis': commit_info.get('analysis', {}),
                        'rag_context': commit_info.get('rag_context', '')
                    }
                    
                    # Generate updated content using LLM
                    updated_content = llm_service.generate_documentation_update(
                        current_content, commit_context, commit_context.get('rag_context', '')
                    )
                    
                    # Update the file in GitLab
                    gitlab_commit_data = {
                        'commit_hash': commit_info['hash'],
                        'commit_message': commit_info['message'],
                        'author': commit_info['author'],
                        'timestamp': commit_info['timestamp'],
                        'repository_url': repo_config.get('code_repo_url', repo_config.get('github_repo_url', ''))
                    }
                    
                    # Use GitLab client to update the file
                    # For DOCX files, use new conditional workflow dispatcher
                    if file_path.lower().endswith('.docx'):
                        # Check if new DOCX workflow is enabled
                        new_docx_workflow_enabled = os.getenv('NEW_DOCX_WORKFLOW', 'true').lower() == 'true'
                        
                        if new_docx_workflow_enabled:
                            logger.info(f"📄 Processing DOCX file with new conditional workflow: {file_path}")
                            try:
                                updated_docx_bytes = self._process_docx_with_new_workflow_fixed(
                                    original_content, commit_context, llm_service
                                )
                                
                                # Save updated DOCX locally for testing
                                try:
                                    from datetime import datetime
                                    
                                    # Create local test directory
                                    test_dir = "test_output"
                                    os.makedirs(test_dir, exist_ok=True)
                                    
                                    # Generate filename with timestamp
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    filename = file_path.replace("/", "_").replace("\\", "_")
                                    local_filename = f"{test_dir}/{filename}_{timestamp}.docx"
                                    
                                    # Save the updated DOCX locally
                                    with open(local_filename, 'wb') as f:
                                        f.write(updated_docx_bytes)
                                    
                                    logger.info(f"💾 Saved updated DOCX locally: {local_filename}")
                                    
                                except Exception as save_error:
                                    logger.warning(f"⚠️ Failed to save DOCX locally: {str(save_error)}")
                                
                                if docs_client.update_single_documentation_file(
                                    file_path, updated_docx_bytes, gitlab_commit_data
                                ):
                                    updated_files.append(file_path)
                                    logger.info(f"✅ Updated DOCX {file_path} with new workflow")
                                else:
                                    failed_files.append(file_path)
                                    logger.error(f"❌ Failed to update DOCX {file_path} in {docs_provider.upper()}")
                            except Exception as e:
                                logger.error(f"❌ New DOCX workflow failed for {file_path}: {str(e)}")
                                # Fallback to basic DOCX processing
                                from utils.docx_handler import DOCXHandler
                                if original_content:
                                    updated_docx_bytes = DOCXHandler.update_docx_content(original_content, updated_content)
                                else:
                                    updated_docx_bytes = DOCXHandler.create_docx_from_text(updated_content, "Updated Document")
                                
                                # Save updated DOCX locally for testing (fallback)
                                try:
                                    from datetime import datetime
                                    
                                    # Create local test directory
                                    test_dir = "test_output"
                                    os.makedirs(test_dir, exist_ok=True)
                                    
                                    # Generate filename with timestamp
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    filename = file_path.replace("/", "_").replace("\\", "_")
                                    local_filename = f"{test_dir}/{filename}_fallback_{timestamp}.docx"
                                    
                                    # Save the updated DOCX locally
                                    with open(local_filename, 'wb') as f:
                                        f.write(updated_docx_bytes)
                                    
                                    logger.info(f"💾 Saved fallback DOCX locally: {local_filename}")
                                    
                                except Exception as save_error:
                                    logger.warning(f"⚠️ Failed to save fallback DOCX locally: {str(save_error)}")
                                
                                if docs_client.update_single_documentation_file(
                                    file_path, updated_docx_bytes, gitlab_commit_data
                                ):
                                    updated_files.append(file_path)
                                    logger.info(f"✅ Updated DOCX {file_path} with fallback method")
                                else:
                                    failed_files.append(file_path)
                                    logger.error(f"❌ Failed to update DOCX {file_path} in {docs_provider.upper()}")
                        else:
                            # Use basic DOCX processing
                            logger.info(f"📄 Processing DOCX file with basic workflow: {file_path}")
                            from utils.docx_handler import DOCXHandler
                            if original_content:
                                updated_docx_bytes = DOCXHandler.update_docx_content(original_content, updated_content)
                            else:
                                updated_docx_bytes = DOCXHandler.create_docx_from_text(updated_content, "Updated Document")
                            
                            if docs_client.update_single_documentation_file(
                                file_path, updated_docx_bytes, gitlab_commit_data
                            ):
                                updated_files.append(file_path)
                                logger.info(f"✅ Updated DOCX {file_path} with basic workflow")
                            else:
                                failed_files.append(file_path)
                                logger.error(f"❌ Failed to update DOCX {file_path} in {docs_provider.upper()}")
                    else:
                        # Handle text files normally
                        if docs_client.update_single_documentation_file(
                            file_path, updated_content, gitlab_commit_data
                        ):
                            updated_files.append(file_path)
                            logger.info(f"✅ Updated {file_path} with LLM-generated content")
                        else:
                            failed_files.append(file_path)
                            logger.error(f"❌ Failed to update {file_path} in {docs_provider.upper()}")
                        
                except Exception as e:
                    logger.error(f"❌ Error processing {file_path}: {str(e)}")
                    failed_files.append(file_path)
            
            # Return results
            if updated_files:
                return {
                    "success": True,
                    "updated_files": updated_files,
                    "failed_files": failed_files,
                    "total_updated": len(updated_files)
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to update any files. Failed: {failed_files}",
                    "failed_files": failed_files
                }
                
        except Exception as e:
            logger.error(f"❌ LLM documentation update failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _get_original_docx_content(self, file_path: str, repo_config: dict) -> bytes | None:
        """Get original DOCX content from repository (GitHub or GitLab)."""
        try:
            import requests
            import base64
            
            docs_provider = repo_config.get('docs_provider', 'gitlab')
            
            if docs_provider == 'github':
                # GitHub API configuration
                github_token = repo_config.get('docs_token', '')
                docs_repo_url = repo_config.get('docs_repo_url', '')
                docs_branch = repo_config.get('docs_branch', 'main')
                
                if not all([github_token, docs_repo_url]):
                    logger.warning("⚠️ Missing GitHub configuration for DOCX content retrieval")
                    return None
                
                # Parse GitHub repo URL
                repo_url = docs_repo_url.replace('.git', '')
                parts = repo_url.split('/')
                if len(parts) >= 2:
                    owner, repo = parts[-2], parts[-1]
                else:
                    logger.warning("⚠️ Invalid GitHub repository URL")
                    return None
                
                # GitHub API endpoint
                url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
                
                headers = {
                    'Authorization': f'Bearer {github_token}',
                    'Accept': 'application/vnd.github.v3+json'
                }
                
                params = {
                    'ref': docs_branch
                }
                
                response = requests.get(url, headers=headers, params=params)
                
                if response.status_code == 200:
                    file_data = response.json()
                    content_bytes = base64.b64decode(file_data['content'])
                    logger.info(f"✅ Retrieved original DOCX content for {file_path} from GitHub")
                    return content_bytes
                else:
                    logger.warning(f"⚠️ Failed to retrieve DOCX content from GitHub: {response.status_code}")
                    return None
                    
            elif docs_provider == 'gitlab':
                # GitLab API configuration
                gitlab_token = repo_config.get('docs_token', '')
                gitlab_project_id = repo_config.get('docs_project_id', '')
                gitlab_docs_branch = repo_config.get('docs_branch', 'main')
                
                if not all([gitlab_token, gitlab_project_id]):
                    logger.warning("⚠️ Missing GitLab configuration for DOCX content retrieval")
                    return None
                
                # GitLab API endpoint
                url = f"https://gitlab.com/api/v4/projects/{gitlab_project_id}/repository/files/{file_path.replace('/', '%2F')}/raw"
                
                headers = {
                    'PRIVATE-TOKEN': gitlab_token
                }
                
                params = {
                    'ref': gitlab_docs_branch
                }
                
                response = requests.get(url, headers=headers, params=params)
                
                if response.status_code == 200:
                    logger.info(f"✅ Retrieved original DOCX content for {file_path} from GitLab")
                    return response.content
                else:
                    logger.warning(f"⚠️ Failed to retrieve DOCX content from GitLab: {response.status_code}")
                    return None
            else:
                logger.warning(f"⚠️ Unknown docs provider: {docs_provider}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error retrieving original DOCX content for {file_path}: {str(e)}")
            return None
    
    def _create_full_documentation_backup(self, docs_client, docs_provider, repo_config, commit_info, folder_already_created=False):
        """Create a full backup of all documentation files."""
        try:
            # Generate backup path with datetime folder structure
            from datetime import datetime
            import os
            
            # Create datetime-based folder name
            folder_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            commit_hash_short = commit_info['hash'][:8]
            
            # Create backup path with datetime folder
            backup_folder = f"backup/{folder_timestamp}"
            
            # Create backup folder structure first (only if not already created)
            if not folder_already_created:
                folder_created = self._create_backup_folder(docs_client, docs_provider, backup_folder, commit_info)
                if not folder_created:
                    logger.warning(f"⚠️ Could not create backup folder: {backup_folder}")
                    # Continue anyway, GitLab might create the folder automatically
            else:
                logger.info(f"📁 Using existing backup folder: {backup_folder}")
            
            # Get all documentation files from the docs repository
            all_docs = self._get_all_documentation_files(docs_client, docs_provider)
            
            if not all_docs:
                logger.warning("⚠️ No documentation files found to backup")
                return {
                    "success": False,
                    "error": "No documentation files found"
                }
            
            backed_up_files = []
            failed_files = []
            
            # Backup each documentation file
            for file_path in all_docs:
                try:
                    logger.info(f"💾 Backing up: {file_path}")
                    
                    # Extract filename and extension
                    filename = os.path.basename(file_path)
                    name, ext = os.path.splitext(filename)
                    
                    # Handle DOCX files differently - backup as binary DOCX file
                    if file_path.endswith('.docx'):
                        # Get binary content for DOCX files
                        binary_content_b64 = self._get_docs_file_binary_content(docs_client, docs_provider, file_path)
                        if not binary_content_b64:
                            logger.warning(f"⚠️ Could not retrieve binary content for {file_path}")
                            failed_files.append(file_path)
                            continue
                        
                        # Convert base64 string to binary bytes
                        import base64
                        binary_content_bytes = base64.b64decode(binary_content_b64)
                        
                        backup_filename = f"{name}_{commit_hash_short}{ext}"
                        backup_path = f"{backup_folder}/{backup_filename}"
                        
                        # Create backup file in the docs repository with binary content
                        commit_message = f"Full backup: {commit_info['message']} - {file_path}"
                        if docs_client.create_or_update_file_binary(backup_path, binary_content_bytes, commit_message):
                            backed_up_files.append(file_path)
                            logger.info(f"✅ Backed up: {file_path}")
                            
                            # Create a separate metadata file for DOCX
                            metadata_filename = f"{name}_{commit_hash_short}_metadata.md"
                            metadata_path = f"{backup_folder}/{metadata_filename}"
                            metadata_content = f"""# Backup Metadata for {file_path}

## File Information
- **Original File**: {file_path}
- **Backup File**: {backup_filename}
- **Created**: {datetime.now().isoformat()}
- **Commit**: {commit_info['hash']}
- **Message**: {commit_info['message']}
- **Author**: {commit_info['author']}
- **Backup Folder**: {backup_folder}

## Purpose
This is a backup of the original DOCX file. The actual DOCX file is stored as: `{backup_filename}`

## File Type
- **Original**: DOCX (Microsoft Word Document)
- **Backup**: DOCX (Binary format preserved)
"""
                            
                            # Create metadata file
                            metadata_commit_message = f"Backup metadata: {commit_info['message']} - {file_path}"
                            docs_client.create_or_update_file(metadata_path, metadata_content, metadata_commit_message)
                            logger.info(f"✅ Created metadata file: {metadata_filename}")
                        else:
                            failed_files.append(file_path)
                            logger.error(f"❌ Failed to backup: {file_path}")
                    else:
                        # For other files (like .md), get text content
                        current_content = self._get_docs_file_content(docs_client, docs_provider, file_path)
                        if not current_content:
                            logger.warning(f"⚠️ Could not retrieve content for {file_path}")
                            failed_files.append(file_path)
                            continue
                        
                        # For other files (like .md), keep original extension
                        backup_filename = f"{name}_{commit_hash_short}{ext}"
                        backup_path = f"{backup_folder}/{backup_filename}"
                        
                        # Prepare backup content with metadata
                        backup_content = f"""# Backup of {file_path}
# Created: {datetime.now().isoformat()}
# Commit: {commit_info['hash']}
# Message: {commit_info['message']}
# Author: {commit_info['author']}
# Original file: {file_path}
# Backup folder: {backup_folder}
# Full documentation backup

---

{current_content}
"""
                        
                        # Create backup file in the docs repository
                        commit_message = f"Full backup: {commit_info['message']} - {file_path}"
                        if docs_client.create_or_update_file(backup_path, backup_content, commit_message):
                            backed_up_files.append(file_path)
                            logger.info(f"✅ Backed up: {file_path}")
                        else:
                            failed_files.append(file_path)
                            logger.error(f"❌ Failed to backup: {file_path}")
                        
                except Exception as e:
                    logger.error(f"❌ Error backing up {file_path}: {str(e)}")
                    failed_files.append(file_path)
            
            # Update folder README with backup results
            self._update_backup_folder_readme(docs_client, docs_provider, backup_folder, commit_info, backed_up_files, failed_files)
            
            # Return results
            if backed_up_files:
                return {
                    "success": True,
                    "backup_folder": backup_folder,
                    "backed_up_files": backed_up_files,
                    "failed_files": failed_files,
                    "total_backed_up": len(backed_up_files)
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to backup any files. Failed: {failed_files}",
                    "failed_files": failed_files
                }
                
        except Exception as e:
            logger.error(f"❌ Full documentation backup failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _create_versioned_backup(self, docs_client, docs_provider, repo_config, file_path, content, commit_info, folder_already_created=False):
        """Create a versioned backup of the current document before editing."""
        try:
            logger.info(f"💾 Creating versioned backup for {file_path}")
            
            # Generate backup path with datetime folder structure
            from datetime import datetime
            import os
            
            # Create datetime-based folder name
            folder_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            commit_hash_short = commit_info['hash'][:8]
            
            # Extract filename and extension
            filename = os.path.basename(file_path)
            name, ext = os.path.splitext(filename)
            
            # Create backup path with datetime folder
            backup_folder = f"backup/{folder_timestamp}"
            
            # Handle DOCX files differently - backup as binary DOCX file
            if file_path.endswith('.docx'):
                # Get binary content for DOCX files
                binary_content_b64 = self._get_docs_file_binary_content(docs_client, docs_provider, file_path)
                if not binary_content_b64:
                    logger.warning(f"⚠️ Could not retrieve binary content for {file_path}")
                    return {"success": False, "error": f"Could not retrieve binary content for {file_path}"}
                
                # Convert base64 string to binary bytes
                import base64
                binary_content_bytes = base64.b64decode(binary_content_b64)
                
                backup_filename = f"{name}_{commit_hash_short}{ext}"
                backup_path = f"{backup_folder}/{backup_filename}"
                
                # Create backup folder structure first (only if not already created)
                if not folder_already_created:
                    folder_created = self._create_backup_folder(docs_client, docs_provider, backup_folder, commit_info)
                    if not folder_created:
                        logger.warning(f"⚠️ Could not create backup folder: {backup_folder}")
                        # Continue anyway, GitLab might create the folder automatically
                else:
                    logger.info(f"📁 Using existing backup folder: {backup_folder}")
                
                commit_message = f"Backup: {commit_info['message']} - {file_path}"
                if docs_client.create_or_update_file_binary(backup_path, binary_content_bytes, commit_message):
                    logger.info(f"✅ Versioned backup created: {backup_path}")
                    
                    # Create a separate metadata file for DOCX
                    metadata_filename = f"{name}_{commit_hash_short}_metadata.md"
                    metadata_path = f"{backup_folder}/{metadata_filename}"
                    metadata_content = f"""# Backup Metadata for {file_path}

## File Information
- **Original File**: {file_path}
- **Backup File**: {backup_filename}
- **Created**: {datetime.now().isoformat()}
- **Commit**: {commit_info['hash']}
- **Message**: {commit_info['message']}
- **Author**: {commit_info['author']}
- **Backup Folder**: {backup_folder}

## Purpose
This is a versioned backup of the original DOCX file. The actual DOCX file is stored as: `{backup_filename}`

## File Type
- **Original**: DOCX (Microsoft Word Document)
- **Backup**: DOCX (Binary format preserved)
"""
                    
                    # Create metadata file
                    metadata_commit_message = f"Backup metadata: {commit_info['message']} - {file_path}"
                    docs_client.create_or_update_file(metadata_path, metadata_content, metadata_commit_message)
                    logger.info(f"✅ Created metadata file: {metadata_filename}")
                    
                    return {"success": True, "backup_path": backup_path}
                else:
                    logger.error(f"❌ Failed to create versioned backup: {backup_path}")
                    return {"success": False, "error": f"Failed to create versioned backup for {file_path}"}
            else:
                # For other files (like .md), keep original extension
                backup_filename = f"{name}_{commit_hash_short}{ext}"
                backup_path = f"{backup_folder}/{backup_filename}"
                
                # Prepare backup content with metadata
                backup_content = f"""# Backup of {file_path}
# Created: {datetime.now().isoformat()}
# Commit: {commit_info['hash']}
# Message: {commit_info['message']}
# Author: {commit_info['author']}
# Original file: {file_path}
# Backup folder: {backup_folder}

---

{content}
"""
            
            # Create backup file in the docs repository
            backup_commit_data = {
                'commit_hash': f"backup_{commit_info['hash']}",
                'commit_message': f"Backup: {commit_info['message']} - {file_path}",
                'author': commit_info['author'],
                'timestamp': commit_info['timestamp'],
                'repository_url': repo_config.get('github_repo_url', repo_config.get('gitlab_repo_url', ''))
            }
            
            # Create backup folder structure first (only if not already created)
            if not folder_already_created:
                folder_created = self._create_backup_folder(docs_client, docs_provider, backup_folder, commit_info)
                if not folder_created:
                    logger.warning(f"⚠️ Could not create backup folder: {backup_folder}")
                    # Continue anyway, GitLab might create the folder automatically
            else:
                logger.info(f"📁 Using existing backup folder: {backup_folder}")
            
            # Use docs client to create the backup file
            commit_message = f"Backup: {commit_info['message']} - {file_path}"
            if docs_client.create_or_update_file(backup_path, backup_content, commit_message):
                logger.info(f"✅ Backup created successfully: {backup_path}")
                return {
                    "success": True,
                    "backup_path": backup_path,
                    "backup_folder": backup_folder,
                    "backup_filename": backup_filename
                }
            else:
                logger.error(f"❌ Failed to create backup file: {backup_path}")
                return {
                    "success": False,
                    "error": "Failed to create backup file in docs repository"
                }
                
        except Exception as e:
            logger.error(f"❌ Backup creation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_backup_folder(self, docs_client, docs_provider, backup_folder, commit_info):
        """Create backup folder structure in the documentation repository."""
        try:
            logger.info(f"📁 Creating backup folder: {backup_folder}")
            
            # Create a README.md file in the backup folder to establish the folder structure
            folder_readme_path = f"{backup_folder}/README.md"
            folder_readme_content = f"""# Backup Session - {backup_folder.split('/')[-1]}

This folder contains backups created during the documentation update process.

## Session Information
- **Created**: {datetime.now().isoformat()}
- **Commit**: {commit_info['hash']}
- **Message**: {commit_info['message']}
- **Author**: {commit_info['author']}
- **Timestamp**: {commit_info['timestamp']}

## Files in this backup session:
*This README will be updated as files are backed up*

## Purpose
These backups preserve the original state of documentation files before they are updated by the automated documentation system.
"""
            
            commit_message = f"Create backup folder: {backup_folder.split('/')[-1]}"
            
            # Create the folder by creating the README file
            if docs_client.create_or_update_file(folder_readme_path, folder_readme_content, commit_message):
                logger.info(f"✅ Backup folder created: {backup_folder}")
                return True
            else:
                logger.warning(f"⚠️ Could not create backup folder: {backup_folder}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Backup folder creation failed: {str(e)}")
            return False
    
    def _get_all_documentation_files(self, docs_client, docs_provider):
        """Get all documentation files from the documentation repository."""
        try:
            import requests
            
            if docs_provider == 'gitlab':
                # Use GitLab API to get repository tree
                url = f"{docs_client.base_url}/api/v4/projects/{docs_client.project_id}/repository/tree"
                
                params = {
                    'ref': docs_client.docs_branch,
                    'recursive': True,
                    'per_page': 100
                }
                
                response = requests.get(url, headers=docs_client.headers, params=params)
                
                if response.status_code == 200:
                    tree_data = response.json()
                    
                    # Filter for documentation files (markdown and docx files in docs folder)
                    doc_files = []
                    for item in tree_data:
                        if (item['type'] == 'blob' and 
                            item['path'].startswith('docs/') and 
                            (item['path'].endswith('.md') or item['path'].endswith('.docx'))):
                            doc_files.append(item['path'])
                    
                    logger.info(f"📚 Found {len(doc_files)} documentation files")
                    return doc_files
                else:
                    logger.error(f"❌ Failed to get repository tree: {response.status_code}")
                    return []
                    
            elif docs_provider == 'github':
                # Use GitHub API to get repository contents
                url = f"https://api.github.com/repos/{docs_client.owner}/{docs_client.repo}/contents/docs"
                
                response = requests.get(url, headers=docs_client.headers)
                
                if response.status_code == 200:
                    contents = response.json()
                    
                    # Filter for documentation files (markdown and docx files)
                    doc_files = []
                    for item in contents:
                        if (item['type'] == 'file' and 
                            (item['name'].endswith('.md') or item['name'].endswith('.docx'))):
                            doc_files.append(f"docs/{item['name']}")
                    
                    logger.info(f"📚 Found {len(doc_files)} documentation files")
                    return doc_files
                else:
                    logger.error(f"❌ Failed to get repository contents: {response.status_code}")
                    return []
            else:
                logger.error(f"❌ Unknown docs provider: {docs_provider}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Error getting documentation files: {str(e)}")
            return []
    
    def _update_backup_folder_readme(self, docs_client, docs_provider, backup_folder, commit_info, backed_up_files, failed_files):
        """Update the backup folder README with backup results."""
        try:
            logger.info(f"📝 Updating backup folder README...")
            
            from datetime import datetime
            
            # Create updated README content
            readme_content = f"""# Backup Session - {backup_folder.split('/')[-1]}

This folder contains a **FULL BACKUP** of all documentation files created during the documentation update process.

## Session Information
- **Created**: {datetime.now().isoformat()}
- **Commit**: {commit_info['hash']}
- **Message**: {commit_info['message']}
- **Author**: {commit_info['author']}
- **Timestamp**: {commit_info['timestamp']}

## Backup Results
- **Total Files Backed Up**: {len(backed_up_files)}
- **Failed Backups**: {len(failed_files)}

## Files Successfully Backed Up:
{chr(10).join([f"- {file}" for file in backed_up_files]) if backed_up_files else "- None"}

## Failed Backups:
{chr(10).join([f"- {file}" for file in failed_files]) if failed_files else "- None"}

## Purpose
This is a **COMPLETE SNAPSHOT** of all documentation files before any updates were made by the automated documentation system. This allows for full rollback capability if needed.

## File Naming Convention
- Original: `docs/user_guide.md`
- Backup: `user_guide_{commit_info['hash'][:8]}.md`
"""
            
            readme_path = f"{backup_folder}/README.md"
            commit_message = f"Update backup folder README: {backup_folder.split('/')[-1]}"
            
            # Update the README file
            if docs_client.create_or_update_file(readme_path, readme_content, commit_message):
                logger.info(f"✅ Backup folder README updated: {readme_path}")
                return True
            else:
                logger.warning(f"⚠️ Could not update backup folder README: {readme_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Backup folder README update failed: {str(e)}")
            return False
    
    def _get_docs_file_content(self, docs_client, docs_provider, file_path):
        """Get current content of a file from the documentation repository."""
        try:
            import requests
            import base64
            
            if docs_provider == 'gitlab':
                # Use GitLab API to get file content
                url = f"{docs_client.base_url}/api/v4/projects/{docs_client.project_id}/repository/files/{file_path.replace('/', '%2F')}"
                
                response = requests.get(url, headers=docs_client.headers, params={'ref': docs_client.docs_branch})
                
                if response.status_code == 200:
                    file_data = response.json()
                    # Decode base64 content
                    content_bytes = base64.b64decode(file_data['content'])
                    
                    # Handle DOCX files
                    if file_path.endswith('.docx'):
                        from utils.docx_handler import DOCXHandler
                        content = DOCXHandler.extract_text_from_docx(content_bytes)
                    else:
                        content = content_bytes.decode('utf-8')
                    
                    return content
                else:
                    logger.warning(f"⚠️ Could not retrieve file content for {file_path}: {response.status_code}")
                    return None
                    
            elif docs_provider == 'github':
                # Use GitHub API to get file content
                url = f"https://api.github.com/repos/{docs_client.owner}/{docs_client.repo}/contents/{file_path}"
                
                response = requests.get(url, headers=docs_client.headers)
                
                if response.status_code == 200:
                    file_data = response.json()
                    # Decode base64 content
                    content_bytes = base64.b64decode(file_data['content'])
                    
                    # Handle DOCX files
                    if file_path.endswith('.docx'):
                        from utils.docx_handler import DOCXHandler
                        content = DOCXHandler.extract_text_from_docx(content_bytes)
                    else:
                        content = content_bytes.decode('utf-8')
                    
                    return content
                else:
                    logger.warning(f"⚠️ Could not retrieve file content for {file_path}: {response.status_code}")
                    return None
            else:
                logger.error(f"❌ Unknown docs provider: {docs_provider}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error retrieving file content for {file_path}: {str(e)}")
            return None
    
    def _get_docs_file_binary_content(self, docs_client, docs_provider, file_path):
        """Get binary content of a file from the documentation repository."""
        try:
            import requests
            import base64
            
            if docs_provider == 'gitlab':
                # Use GitLab API to get file content
                url = f"{docs_client.base_url}/api/v4/projects/{docs_client.project_id}/repository/files/{file_path.replace('/', '%2F')}"
                
                response = requests.get(url, headers=docs_client.headers, params={'ref': docs_client.docs_branch})
                
                if response.status_code == 200:
                    file_data = response.json()
                    # Return base64 encoded content for binary files
                    return file_data['content']
                else:
                    logger.warning(f"⚠️ Could not retrieve file content for {file_path}: {response.status_code}")
                    return None
                    
            elif docs_provider == 'github':
                # Use GitHub API to get file content
                url = f"https://api.github.com/repos/{docs_client.owner}/{docs_client.repo}/contents/{file_path}"
                
                response = requests.get(url, headers=docs_client.headers)
                
                if response.status_code == 200:
                    file_data = response.json()
                    # Return base64 encoded content for binary files
                    return file_data['content']
                else:
                    logger.warning(f"⚠️ Could not retrieve file content for {file_path}: {response.status_code}")
                    return None
            else:
                logger.error(f"❌ Unknown docs provider: {docs_provider}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error retrieving file content for {file_path}: {str(e)}")
            return None
    
    def extract_commit_info(self, payload, headers, webhook_type='github'):
        """Extract commit information from webhook payload."""
        try:
            print("🔍 EXTRACTING COMMIT INFO FROM PAYLOAD")
            print(f"📋 Payload keys: {list(payload.keys())}")
            print(f"📋 Webhook type: {webhook_type}")
            
            if webhook_type == 'github':
                return self._extract_github_commit_info(payload)
            elif webhook_type == 'gitlab':
                return self._extract_gitlab_commit_info(payload)
            else:
                print(f"❌ Unknown webhook type: {webhook_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to extract commit info: {e}")
            print(f"❌ Error extracting commit info: {e}")
            return None
    
    def _extract_github_commit_info(self, payload):
        """Extract commit information from GitHub webhook payload."""
        try:
            # Handle GitHub webhook format - head_commit (most common)
            if 'head_commit' in payload and payload['head_commit']:
                commit = payload['head_commit']
                print("✅ Found head_commit in payload")
                return {
                    "hash": commit.get('id', 'unknown'),
                    "message": commit.get('message', ''),
                    "author": commit.get('author', {}).get('name', 'unknown'),
                    "timestamp": commit.get('timestamp', ''),
                    "url": commit.get('url', ''),
                    "modified": commit.get('modified', []),
                    "added": commit.get('added', []),
                    "removed": commit.get('removed', []),
                    "branch": payload.get('ref', '').replace('refs/heads/', '')
                }
            
            # Handle GitHub webhook format - commits array
            elif 'commits' in payload and payload['commits']:
                commit = payload['commits'][0]  # Get first commit
                print("✅ Found commits array in payload")
                print(f"🔍 Commit details: {json.dumps(commit, indent=2)}")
                
                # Check if there are any file changes in the commit
                modified_files = commit.get('modified', [])
                added_files = commit.get('added', [])
                removed_files = commit.get('removed', [])
                
                print(f"📁 Modified files: {modified_files}")
                print(f"➕ Added files: {added_files}")
                print(f"➖ Removed files: {removed_files}")
                
                return {
                    "hash": commit.get('id', 'unknown'),
                    "message": commit.get('message', ''),
                    "author": commit.get('author', {}).get('name', 'unknown'),
                    "timestamp": commit.get('timestamp', ''),
                    "url": commit.get('url', ''),
                    "modified": modified_files,
                    "added": added_files,
                    "removed": removed_files,
                    "branch": payload.get('ref', '').replace('refs/heads/', '')
                }
            
            # Handle GitHub Actions webhook format
            elif 'workflow_run' in payload:
                workflow_run = payload['workflow_run']
                print("✅ Found workflow_run in payload")
                return {
                    "hash": workflow_run.get('head_sha', 'unknown'),
                    "message": f"GitHub Actions workflow: {workflow_run.get('name', 'Unknown workflow')}",
                    "author": workflow_run.get('head_commit', {}).get('author', {}).get('name', 'GitHub Actions'),
                    "timestamp": workflow_run.get('created_at', ''),
                    "url": workflow_run.get('html_url', ''),
                    "branch": workflow_run.get('head_branch', ''),
                    "modified": [],
                    "added": [],
                    "removed": []
                }
            
            # Handle direct commit data
            elif 'commit_message' in payload:
                print("✅ Found direct commit data in payload")
                return {
                    "hash": payload.get('commit_hash', 'unknown'),
                    "message": payload.get('commit_message', ''),
                    "author": payload.get('author', 'unknown'),
                    "timestamp": datetime.now().isoformat(),
                    "modified": payload.get('changed_files', [])
                }
            
            print("❌ No recognizable GitHub commit format found in payload")
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract GitHub commit info: {e}")
            print(f"❌ Error extracting GitHub commit info: {e}")
            return None
    
    def _extract_gitlab_commit_info(self, payload):
        """Extract commit information from GitLab webhook payload."""
        try:
            # Handle GitLab webhook format - commits array
            if 'commits' in payload and payload['commits']:
                commit = payload['commits'][0]  # Get first commit
                print("✅ Found commits array in GitLab payload")
                print(f"🔍 GitLab commit details: {json.dumps(commit, indent=2)}")
                
                # Check if there are any file changes in the commit
                modified_files = commit.get('modified', [])
                added_files = commit.get('added', [])
                removed_files = commit.get('removed', [])
                
                print(f"📁 Modified files: {modified_files}")
                print(f"➕ Added files: {added_files}")
                print(f"➖ Removed files: {removed_files}")
                
                return {
                    "hash": commit.get('id', 'unknown'),
                    "message": commit.get('message', ''),
                    "author": commit.get('author', {}).get('name', 'unknown'),
                    "timestamp": commit.get('timestamp', ''),
                    "url": commit.get('url', ''),
                    "modified": modified_files,
                    "added": added_files,
                    "removed": removed_files,
                    "branch": payload.get('ref', '').replace('refs/heads/', '')
                }
            
            # Handle GitLab webhook format - single commit
            elif 'object_kind' in payload and payload['object_kind'] == 'push':
                print("✅ Found GitLab push event")
                commits = payload.get('commits', [])
                if commits:
                    commit = commits[0]
                    return {
                        "hash": commit.get('id', 'unknown'),
                        "message": commit.get('message', ''),
                        "author": commit.get('author', {}).get('name', 'unknown'),
                        "timestamp": commit.get('timestamp', ''),
                        "url": commit.get('url', ''),
                        "modified": commit.get('modified', []),
                        "added": commit.get('added', []),
                        "removed": commit.get('removed', []),
                        "branch": payload.get('ref', '').replace('refs/heads/', '')
                    }
            
            print("❌ No recognizable GitLab commit format found in payload")
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract GitLab commit info: {e}")
            print(f"❌ Error extracting GitLab commit info: {e}")
            return None
    
    def _process_docx_with_new_workflow(self, original_content: bytes, current_content: str, commit_context: dict, llm_service) -> bytes:
        """
        Process DOCX files using the new conditional workflow dispatcher.
        
        Args:
            original_content: Original DOCX file as bytes
            current_content: Current document content as text
            commit_context: Commit information and context
            llm_service: LLM service instance
            
        Returns:
            Updated DOCX file as bytes
        """
        try:
            logger.info("🔄 Starting new DOCX workflow dispatcher")
            
            # Import the new workflow modules
            from utils.docx_line_editor import DocxLineEditor
            from utils.docx_paragraph_inserter import DocxParagraphInserter
            from utils.docx_table_editor import DocxTableEditor
            
            # Determine what type of edit is needed
            edit_type = llm_service.determine_docx_edit_type(commit_context, current_content)
            logger.info(f"📋 Determined edit type: {edit_type}")
            
            # Route to appropriate workflow based on edit type
            if edit_type == "edit_line":
                logger.info("📝 Using line editor workflow")
                line_editor = DocxLineEditor()
                return line_editor.edit_document_lines(original_content, commit_context, llm_service)
                
            elif edit_type == "add_paragraph":
                logger.info("📝 Using paragraph inserter workflow")
                paragraph_inserter = DocxParagraphInserter()
                return paragraph_inserter.insert_paragraph(original_content, commit_context, llm_service)
                
            elif edit_type == "edit_table":
                logger.info("📝 Using table editor workflow")
                table_editor = DocxTableEditor()
                return table_editor.edit_tables(original_content, commit_context, llm_service)
                
            elif edit_type == "no_change":
                logger.info("📝 No changes needed, returning original content")
                return original_content
                
            else:
                logger.warning(f"⚠️ Unknown edit type '{edit_type}', defaulting to line editor")
                line_editor = DocxLineEditor()
                return line_editor.edit_document_lines(original_content, commit_context, llm_service)
                
        except Exception as e:
            logger.error(f"❌ New DOCX workflow failed: {str(e)}")
            # Return original content as fallback
            return original_content
    
    def _process_docx_with_new_workflow_fixed(self, original_content: bytes, commit_context: dict, llm_service) -> bytes:
        """
        Process DOCX files using the new fixed conditional workflow dispatcher.
        
        Args:
            original_content: Original DOCX file as bytes
            commit_context: Commit information and context
            llm_service: LLM service instance
            
        Returns:
            Updated DOCX file as bytes
        """
        try:
            logger.info("🔄 Starting new DOCX workflow dispatcher (fixed version)")
            
            # Determine the type of edit needed
            edit_type = llm_service.determine_docx_edit_type(commit_context, "")
            
            logger.info(f"📋 Determined edit type: {edit_type}")
            
            # Route to appropriate workflow using fixed modules
            if edit_type == "edit_line":
                logger.info("📝 Using line editor workflow (fixed)")
                from utils.docx_line_editor_fixed import DocxLineEditor
                editor = DocxLineEditor()
                return editor.edit_document_lines(original_content, commit_context, llm_service)
                
            elif edit_type == "add_paragraph":
                logger.info("📝 Using paragraph inserter workflow (fixed)")
                from utils.docx_paragraph_inserter_fixed import DocxParagraphInserter
                inserter = DocxParagraphInserter()
                return inserter.insert_new_paragraphs(original_content, commit_context, llm_service)
                
            elif edit_type == "edit_table":
                logger.info("📝 Using table editor workflow (fixed)")
                from utils.docx_table_editor_fixed import DocxTableEditor
                editor = DocxTableEditor()
                return editor.edit_document_tables(original_content, commit_context, llm_service)
                
            else:
                logger.info("📝 No changes needed")
                return original_content
                
        except Exception as e:
            logger.error(f"❌ New DOCX workflow failed: {str(e)}")
            return original_content
    
    def start(self):
        """Start the existing repository workflow server."""
        print("[STARTUP] Starting Existing Repository Workflow Server...")
        print("=" * 60)
        
        # Clear settings cache to ensure fresh configuration loading
        from utils.config import clear_settings_cache
        clear_settings_cache()
        
        # Create gRPC server
        self.server = self.create_grpc_server()
        
        # Create webhook Flask app
        self.app = self.create_webhook_app()
        
        # Start gRPC server in a separate thread
        grpc_thread = threading.Thread(target=self.server.start)
        grpc_thread.daemon = True
        grpc_thread.start()
        
        print("[WEBHOOK] Starting webhook server on port 8000...")
        print("[ENDPOINTS] Available endpoints:")
        print("   - Health: http://localhost:8000/health")
        print("   - Configure: http://localhost:8000/configure")
        print("   - Webhook: http://localhost:8000/webhook/{config_id}")
        print("   - Status: http://localhost:8000/status/{config_id}")
        print("   - List Configs: http://localhost:8000/configurations")
        print("   - Generate Embeddings: http://localhost:8000/generate-embeddings/{config_id}")
        print()
        print("[WORKFLOW] Steps:")
        print("   1. POST /configure -> Configure your existing repos")
        print("   2. Get webhook URL for GitHub configuration")
        print("   3. Just commit to your existing GitHub repo -> Automatic processing!")
        print()
        print("[SERVER] Waiting for requests...")
        
        # Start Flask app (webhook server)
        self.app.run(
            host='0.0.0.0',
            port=8000,
            debug=False,
            threaded=True
        )

if __name__ == "__main__":
    server = ExistingRepoWorkflow()
    server.start()

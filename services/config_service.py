"""Configuration service for gRPC services."""

import os
from dotenv import load_dotenv
import structlog

logger = structlog.get_logger("ConfigService")

class ConfigService:
    """Service to manage configuration for gRPC services."""
    
    def __init__(self):
        """Initialize configuration service."""
        # Load environment variables
        load_dotenv()
        
        # GitHub configuration
        self.github_token = os.getenv('GITHUB_TOKEN', '')
        self.github_base_url = os.getenv('GITHUB_BASE_URL', 'https://api.github.com')
        
        # GitLab configuration
        self.gitlab_token = os.getenv('GITLAB_TOKEN', '')
        self.gitlab_base_url = os.getenv('GITLAB_BASE_URL', 'https://gitlab.com')
        self.gitlab_project_id = os.getenv('GITLAB_PROJECT_ID', '')
        self.gitlab_docs_branch = os.getenv('GITLAB_DOCS_BRANCH', 'main')
        
        # ArangoDB configuration
        self.arango_host = os.getenv('ARANGO_HOST', 'localhost')
        self.arango_port = int(os.getenv('ARANGO_PORT', '8529'))
        self.arango_username = os.getenv('ARANGO_USERNAME', 'root')
        self.arango_password = os.getenv('ARANGO_PASSWORD', '')
        self.arango_database = os.getenv('ARANGO_DATABASE', 'rework_docs')
        
        # Webhook configuration
        self.webhook_secret = os.getenv('WEBHOOK_SECRET', '')
        
        # Server configuration
        self.server_host = os.getenv('SERVER_HOST', '0.0.0.0')
        self.server_port = int(os.getenv('SERVER_PORT', '50061'))
        self.max_workers = int(os.getenv('MAX_WORKERS', '10'))
        
        self.logger = logger
        self._validate_config()
    
    def _validate_config(self):
        """Validate that required configuration is present."""
        missing_configs = []
        
        if not self.github_token:
            missing_configs.append('GITHUB_TOKEN')
        
        if not self.gitlab_token:
            missing_configs.append('GITLAB_TOKEN')
        
        if not self.gitlab_project_id:
            missing_configs.append('GITLAB_PROJECT_ID')
        
        if missing_configs:
            self.logger.warning(
                "Missing configuration",
                missing_configs=missing_configs,
                message="Some gRPC services may not function properly"
            )
        else:
            self.logger.info("Configuration validated successfully")
    
    def get_github_config(self):
        """Get GitHub configuration."""
        return {
            'token': self.github_token,
            'base_url': self.github_base_url
        }
    
    def get_gitlab_config(self):
        """Get GitLab configuration."""
        return {
            'token': self.gitlab_token,
            'base_url': self.gitlab_base_url,
            'project_id': self.gitlab_project_id,
            'docs_branch': self.gitlab_docs_branch
        }
    
    def get_arango_config(self):
        """Get ArangoDB configuration."""
        return {
            'host': self.arango_host,
            'port': self.arango_port,
            'username': self.arango_username,
            'password': self.arango_password,
            'database': self.arango_database
        }

# Global config instance
config = ConfigService()

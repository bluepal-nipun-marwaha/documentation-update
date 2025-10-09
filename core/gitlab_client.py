"""
GitLab client for managing documentation updates.
"""
import os
import requests
import json
from typing import Dict, List, Optional, Any
import structlog
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = structlog.get_logger(__name__)

class GitLabClient:
    """Client for GitLab API operations."""
    
    def __init__(self):
        self.base_url = os.getenv('GITLAB_BASE_URL', 'https://gitlab.com')
        self.token = os.getenv('GITLAB_TOKEN', '')
        self.project_id = os.getenv('GITLAB_PROJECT_ID', '')
        self.docs_branch = os.getenv('GITLAB_DOCS_BRANCH', 'main')
        
        self.headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
        
    def create_versioned_folder(self, version: str) -> bool:
        """Create a versioned folder structure in GitLab."""
        try:
            folder_path = f"docs/v{version}"
            
            # Create folder by creating a README.md file in it
            readme_content = f"""# Documentation v{version}

This folder contains documentation for version {version} of the project.

## Structure
- `user_guide.md` - User guide and tutorials
- `api_reference.md` - API documentation  
- `architecture.md` - System architecture
- `bugfix_notes.md` - Bug fix documentation
- `troubleshooting.md` - Troubleshooting guide
- `maintenance.md` - Maintenance procedures
- `performance.md` - Performance optimization guide
- `testing.md` - Testing procedures
- `coding_standards.md` - Coding standards and style guide
- `CHANGELOG.md` - Project changelog
- `README.md` - Documentation overview

Generated on: {datetime.now().isoformat()}
"""
            
            return self.create_or_update_file(
                file_path=f"{folder_path}/README.md",
                content=readme_content,
                commit_message=f"docs: create versioned documentation folder v{version}"
            )
            
        except Exception as e:
            logger.error("Failed to create versioned folder", error=str(e), version=version)
            return False
    
    def create_or_update_file(self, file_path: str, content: str, commit_message: str) -> bool:
        """Create or update a file in GitLab repository."""
        try:
            url = f"{self.base_url}/api/v4/projects/{self.project_id}/repository/files/{file_path.replace('/', '%2F')}"
            
            # First, try to get the existing file to get the blob_id
            try:
                response = requests.get(url, headers=self.headers, params={'ref': self.docs_branch})
                if response.status_code == 200:
                    existing_file = response.json()
                    blob_id = existing_file['blob_id']
                else:
                    blob_id = None
            except:
                blob_id = None
            
            # Prepare the file data
            file_data = {
                'branch': self.docs_branch,
                'content': content,
                'commit_message': commit_message,
                'encoding': 'text'
            }
            
            if blob_id:
                file_data['last_commit_id'] = blob_id
            
            # Create or update the file
            if blob_id:
                # File exists, use PUT to update
                response = requests.put(url, headers=self.headers, json=file_data)
            else:
                # File doesn't exist, use POST to create
                response = requests.post(url, headers=self.headers, json=file_data)
            
            if response.status_code in [200, 201]:
                logger.info("File created/updated successfully", file_path=file_path)
                return True
            else:
                logger.error("Failed to create/update file", 
                           file_path=file_path, 
                           status_code=response.status_code,
                           response=response.text)
                return False
                
        except Exception as e:
            logger.error("Failed to create/update file", error=str(e), file_path=file_path)
            return False
    
    def update_documentation(self, commit_data: Dict[str, Any], files_to_update: List[str]) -> bool:
        """Update documentation files based on commit data."""
        try:
            # Extract version from commit or use timestamp
            version = self._extract_version(commit_data)
            
            # Create versioned folder if it doesn't exist
            if not self.create_versioned_folder(version):
                logger.warning("Failed to create versioned folder", version=version)
            
            success_count = 0
            
            for file_path in files_to_update:
                # Create versioned path
                versioned_path = f"docs/v{version}/{file_path.split('/')[-1]}"
                
                # Generate content based on commit
                content = self._generate_documentation_content(commit_data, file_path)
                
                # Create/update the file
                commit_msg = f"docs: update {file_path.split('/')[-1]} for {commit_data['commit_message']}"
                
                if self.create_or_update_file(versioned_path, content, commit_msg):
                    success_count += 1
                    logger.info("Documentation updated", file_path=versioned_path)
                else:
                    logger.error("Failed to update documentation", file_path=versioned_path)
            
            logger.info("Documentation update completed", 
                       success_count=success_count, 
                       total_files=len(files_to_update))
            
            return success_count > 0
            
        except Exception as e:
            logger.error("Failed to update documentation", error=str(e))
            return False
    
    def update_single_documentation_file(self, file_path: str, content: str, commit_data: Dict[str, Any]) -> bool:
        """Update a single documentation file with LLM-generated content."""
        try:
            # Update the original file path directly (no versioning)
            # file_path should already be in the correct format (e.g., "docs/api_reference.md")
            
            # Create commit message
            commit_msg = f"docs: update {file_path.split('/')[-1]} for {commit_data['commit_message']}"
            
            # Handle DOCX files differently
            if file_path.lower().endswith('.docx'):
                logger.info("📄 Updating DOCX file", file_path=file_path)
                # For DOCX files, content should be bytes
                if isinstance(content, str):
                    # Convert text to DOCX bytes
                    from utils.docx_handler import DOCXHandler
                    content_bytes = DOCXHandler.create_docx_from_text(content, "Updated Document")
                    # Update file with binary content
                    if self.create_or_update_file_binary(file_path, content_bytes, commit_msg):
                        logger.info("DOCX documentation updated", file_path=file_path)
                        return True
                    else:
                        logger.error("Failed to update DOCX documentation", file_path=file_path)
                        return False
                else:
                    # Content is already bytes
                    if self.create_or_update_file_binary(file_path, content, commit_msg):
                        logger.info("DOCX documentation updated", file_path=file_path)
                        return True
                    else:
                        logger.error("Failed to update DOCX documentation", file_path=file_path)
                        return False
            else:
                # Handle text files (markdown, etc.)
                if self.create_or_update_file(file_path, content, commit_msg):
                    logger.info("Documentation updated", file_path=file_path)
                    return True
                else:
                    logger.error("Failed to update documentation", file_path=file_path)
                    return False
                
        except Exception as e:
            logger.error("Single documentation update failed", error=str(e), file_path=file_path)
            return False
    
    def create_or_update_file_binary(self, file_path: str, content: bytes, commit_message: str) -> bool:
        """Create or update a file with binary content (for DOCX files)."""
        try:
            import base64
            
            # Encode binary content to base64 for GitLab API
            encoded_content = base64.b64encode(content).decode('utf-8')
            
            # GitLab API endpoint for file operations
            project_id = self.gitlab_config.get('project_id')
            branch = self.gitlab_config.get('docs_branch', 'main')
            
            url = f"{self.gitlab_config['base_url']}/api/v4/projects/{project_id}/repository/files/{file_path.replace('/', '%2F')}"
            
            headers = {
                'PRIVATE-TOKEN': self.gitlab_config['token'],
                'Content-Type': 'application/json'
            }
            
            # Check if file exists
            check_response = requests.get(url, headers=headers, params={'ref': branch})
            
            if check_response.status_code == 200:
                # File exists, update it
                data = {
                    'branch': branch,
                    'content': encoded_content,
                    'commit_message': commit_message,
                    'encoding': 'base64'
                }
                response = requests.put(url, headers=headers, json=data)
            else:
                # File doesn't exist, create it
                data = {
                    'branch': branch,
                    'content': encoded_content,
                    'commit_message': commit_message,
                    'encoding': 'base64'
                }
                response = requests.post(url, headers=headers, json=data)
            
            if response.status_code in [200, 201]:
                logger.info("Binary file updated successfully", file_path=file_path)
                return True
            else:
                logger.error("Failed to update binary file", 
                           file_path=file_path, 
                           status_code=response.status_code,
                           response=response.text)
                return False
                
        except Exception as e:
            logger.error("Binary file update failed", error=str(e), file_path=file_path)
            return False
    
    def _extract_version(self, commit_data: Dict[str, Any]) -> str:
        """Extract version from commit data."""
        # Try to extract version from commit message
        message = commit_data.get('commit_message', '')
        
        # Look for version patterns like v1.0.0, 1.0.0, etc.
        import re
        version_patterns = [
            r'v?(\d+\.\d+\.\d+)',  # v1.0.0 or 1.0.0
            r'v?(\d+\.\d+)',        # v1.0 or 1.0
            r'v?(\d+)'              # v1 or 1
        ]
        
        for pattern in version_patterns:
            match = re.search(pattern, message)
            if match:
                return match.group(1)
        
        # Fallback to timestamp-based version
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"1.0.{timestamp}"
    
    def _generate_documentation_content(self, commit_data: Dict[str, Any], file_path: str) -> str:
        """Generate documentation content based on commit data."""
        filename = file_path.split('/')[-1]
        commit_message = commit_data.get('commit_message', '')
        author = commit_data.get('author', 'Unknown')
        timestamp = commit_data.get('timestamp', datetime.now().isoformat())
        
        # Base content templates
        templates = {
            'user_guide.md': f"""# User Guide

## Recent Updates

**Commit**: {commit_message}  
**Author**: {author}  
**Date**: {timestamp}  

## Getting Started

This guide will help you get started with the application.

### Installation

1. Clone the repository
2. Install dependencies
3. Run the application

### Basic Usage

[Content will be generated based on the commit changes]

---

*Last updated: {timestamp}*
""",
            
            'api_reference.md': f"""# API Reference

## Recent Changes

**Commit**: {commit_message}  
**Author**: {author}  
**Date**: {timestamp}  

## Endpoints

### Authentication
- `POST /auth/login` - User login
- `POST /auth/logout` - User logout

### Data Management
- `GET /data` - Retrieve data
- `POST /data` - Create new data
- `PUT /data/:id` - Update data
- `DELETE /data/:id` - Delete data

---

*Last updated: {timestamp}*
""",
            
            'architecture.md': f"""# System Architecture

## Recent Updates

**Commit**: {commit_message}  
**Author**: {author}  
**Date**: {timestamp}  

## Overview

The system follows a layered architecture pattern.

### Components

- **Frontend**: User interface layer
- **Backend**: Business logic layer  
- **Database**: Data persistence layer
- **API**: Communication layer

### Data Flow

1. User request → Frontend
2. Frontend → Backend API
3. Backend → Database
4. Response ← Frontend

---

*Last updated: {timestamp}*
""",
            
            'bugfix_notes.md': f"""# Bug Fix Notes

## Recent Fixes

**Commit**: {commit_message}  
**Author**: {author}  
**Date**: {timestamp}  

## Known Issues

### Fixed Issues
- [Fixed] Issue description
- [Fixed] Another issue description

### Open Issues
- [Open] Current issue description

---

*Last updated: {timestamp}*
""",
            
            'troubleshooting.md': f"""# Troubleshooting Guide

## Recent Updates

**Commit**: {commit_message}  
**Author**: {author}  
**Date**: {timestamp}  

## Common Issues

### Issue 1: Description
**Symptoms**: What you see
**Solution**: How to fix it

### Issue 2: Description  
**Symptoms**: What you see
**Solution**: How to fix it

## Getting Help

If you encounter issues not covered here:
1. Check the logs
2. Contact support
3. Create an issue

---

*Last updated: {timestamp}*
""",
            
            'maintenance.md': f"""# Maintenance Guide

## Recent Updates

**Commit**: {commit_message}  
**Author**: {author}  
**Date**: {timestamp}  

## Regular Maintenance Tasks

### Daily
- Check system logs
- Monitor performance metrics

### Weekly  
- Update dependencies
- Review security patches

### Monthly
- Backup data
- Performance optimization

---

*Last updated: {timestamp}*
""",
            
            'performance.md': f"""# Performance Guide

## Recent Updates

**Commit**: {commit_message}  
**Author**: {author}  
**Date**: {timestamp}  

## Performance Optimization

### Database
- Index optimization
- Query optimization
- Connection pooling

### Application
- Caching strategies
- Memory management
- CPU optimization

### Monitoring
- Performance metrics
- Alerting thresholds
- Capacity planning

---

*Last updated: {timestamp}*
""",
            
            'testing.md': f"""# Testing Guide

## Recent Updates

**Commit**: {commit_message}  
**Author**: {author}  
**Date**: {timestamp}  

## Testing Strategy

### Unit Tests
- Test individual components
- Mock external dependencies
- Achieve high coverage

### Integration Tests
- Test component interactions
- Test API endpoints
- Test database operations

### End-to-End Tests
- Test complete user workflows
- Test system integration
- Test performance under load

---

*Last updated: {timestamp}*
""",
            
            'coding_standards.md': f"""# Coding Standards

## Recent Updates

**Commit**: {commit_message}  
**Author**: {author}  
**Date**: {timestamp}  

## Code Style

### Naming Conventions
- Use descriptive names
- Follow language conventions
- Avoid abbreviations

### Documentation
- Comment complex logic
- Document public APIs
- Keep README updated

### Code Organization
- Single responsibility principle
- DRY (Don't Repeat Yourself)
- Clean architecture

---

*Last updated: {timestamp}*
""",
            
            'CHANGELOG.md': f"""# Changelog

## Recent Changes

**Commit**: {commit_message}  
**Author**: {author}  
**Date**: {timestamp}  

## [Unreleased]

### Added
- New features from recent commits

### Changed
- Improvements and modifications

### Fixed
- Bug fixes and corrections

### Removed
- Deprecated features

---

*Last updated: {timestamp}*
""",
            
            'README.md': f"""# Documentation

## Recent Updates

**Commit**: {commit_message}  
**Author**: {author}  
**Date**: {timestamp}  

## Overview

This documentation covers all aspects of the project.

## Sections

- [User Guide](user_guide.md) - Getting started and usage
- [API Reference](api_reference.md) - API documentation
- [Architecture](architecture.md) - System design
- [Troubleshooting](troubleshooting.md) - Common issues
- [Maintenance](maintenance.md) - System maintenance

---

*Last updated: {timestamp}*
"""
        }
        
        return templates.get(filename, f"""# {filename.replace('.md', '').replace('_', ' ').title()}

## Recent Updates

**Commit**: {commit_message}  
**Author**: {author}  
**Date**: {timestamp}  

## Content

This document has been automatically updated based on recent code changes.

---

*Last updated: {timestamp}*
""")

# Global instance
gitlab_client = GitLabClient()

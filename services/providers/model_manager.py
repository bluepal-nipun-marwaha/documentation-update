"""
Model Manager for downloading and caching models from GitHub releases.
"""

import os
import json
import zipfile
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List
import structlog

logger = structlog.get_logger(__name__)

class ModelManager:
    """Manages model downloads and caching from GitHub releases."""
    
    def __init__(self, github_repo: str = "", cache_dir: str = "~/.cache/rework-models"):
        """Initialize the model manager."""
        self.github_repo = github_repo
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        
        # Model versions - pinned to specific versions
        self.model_versions = {
            "qwen2.5-7b": "v1.2.3",
            "gemma-7b": "v2.1.0", 
            "mistral-7b": "v1.0.5",
            "nomic-embed-text": "v1.0.0",
            "e5-large-v2": "v1.0.0",
            "all-MiniLM-L6-v2": "v1.0.0"
        }
    
    def get_model_path(self, model_name: str, version: Optional[str] = None) -> Path:
        """Get the local path for a model."""
        if version is None:
            version = self.model_versions.get(model_name, "latest")
        
        # Sanitize model name for Windows compatibility (replace colons with hyphens)
        safe_model_name = model_name.replace(":", "-")
        
        return self.cache_dir / safe_model_name / version
    
    def is_model_cached(self, model_name: str, version: Optional[str] = None) -> bool:
        """Check if a model is already cached locally."""
        model_path = self.get_model_path(model_name, version)
        return model_path.exists() and (model_path / "model.bin").exists()
    
    def download_model(self, model_name: str, version: Optional[str] = None) -> Path:
        """Download a model from GitHub releases if not cached."""
        if version is None:
            version = self.model_versions.get(model_name, "latest")
        
        model_path = self.get_model_path(model_name, version)
        
        # Check if already cached
        if self.is_model_cached(model_name, version):
            self.logger.info(f"Model {model_name} v{version} already cached")
            return model_path
        
        # Download from GitHub
        self.logger.info(f"Downloading model {model_name} v{version} from GitHub...")
        
        # Create model directory
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Download zip file
        zip_filename = f"{model_name}-{version}.zip"
        zip_url = f"https://github.com/{self.github_repo}/releases/download/{version}/{zip_filename}"
        
        try:
            response = requests.get(zip_url, stream=True)
            response.raise_for_status()
            
            zip_path = model_path / zip_filename
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(model_path)
            
            # Remove zip file
            zip_path.unlink()
            
            self.logger.info(f"Successfully downloaded and extracted {model_name} v{version}")
            return model_path
            
        except Exception as e:
            self.logger.error(f"Failed to download model {model_name} v{version}: {str(e)}")
            raise
    
    def get_model_manifest(self) -> Dict[str, Any]:
        """Get the model manifest from GitHub."""
        try:
            manifest_url = f"https://raw.githubusercontent.com/{self.github_repo}/main/model-manifest.json"
            response = requests.get(manifest_url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.warning(f"Could not fetch model manifest: {str(e)}")
            return {}
    
    def list_available_models(self) -> List[str]:
        """List all available models."""
        return list(self.model_versions.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        version = self.model_versions.get(model_name)
        if not version:
            return {}
        
        return {
            "name": model_name,
            "version": version,
            "cached": self.is_model_cached(model_name, version),
            "path": str(self.get_model_path(model_name, version))
        }

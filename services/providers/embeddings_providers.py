"""
Embeddings Provider implementations for various services.
"""

import requests
import json
import os
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional
from .base_provider import BaseEmbeddingsProvider
from .model_manager import ModelManager

class OpenAIEmbeddingsProvider(BaseEmbeddingsProvider):
    """OpenAI Embeddings provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Force load from .env file with override=True to override any existing env vars
        load_dotenv(override=True)
        self.api_key = os.environ.get('OPENAI_API_KEY', config.get('OPENAI_API_KEY', ''))
        self.model = config.get('openai_embeddings_model', 'text-embedding-3-large')
        self.base_url = config.get('openai_base_url', 'https://api.openai.com/v1')
        
        # Debug logging
        print(f"🔍 DEBUG: OpenAIEmbeddingsProvider initialized")
        print(f"🔍 DEBUG: API key from os.environ: {os.environ.get('OPENAI_API_KEY', 'NOT_FOUND')[:20]}...")
        print(f"🔍 DEBUG: API key from config: {config.get('OPENAI_API_KEY', 'NOT_FOUND')[:20]}...")
        print(f"🔍 DEBUG: Final API key: {self.api_key[:20]}...")
    
    def is_available(self) -> bool:
        """Check if OpenAI embeddings are available and configured."""
        return bool(self.api_key)
    
    def generate_embeddings(self, text: str, **kwargs) -> List[float]:
        """Generate embeddings using OpenAI API."""
        if not self.is_available():
            raise Exception("OpenAI embeddings provider not available - missing API key")
        
        payload = {
            "model": self.model,
            "input": text
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Debug logging
        print(f"🔍 DEBUG: OpenAI Embeddings Request:")
        print(f"🔍 DEBUG: URL: {self.base_url}/embeddings")
        print(f"🔍 DEBUG: Model: {self.model}")
        print(f"🔍 DEBUG: Input length: {len(text)}")
        print(f"🔍 DEBUG: Input preview: {text[:100]}...")
        print(f"🔍 DEBUG: Payload: {payload}")
        
        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result['data'][0]['embedding']
            
        except Exception as e:
            # Add detailed error logging
            if hasattr(e, 'response') and e.response is not None:
                print(f"🔍 DEBUG: Error response status: {e.response.status_code}")
                print(f"🔍 DEBUG: Error response text: {e.response.text}")
            self.logger.error(f"OpenAI embeddings API error: {str(e)}")
            raise Exception(f"OpenAI embeddings provider failed: {str(e)}")
    
    def get_embedding_dimensions(self) -> int:
        """Get the number of dimensions for OpenAI embeddings."""
        if self.model == "text-embedding-3-large":
            return 3072
        elif self.model == "text-embedding-3-small":
            return 1536
        else:
            return 1536  # Default for text-embedding-ada-002
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI embeddings model information."""
        return {
            "provider": "openai",
            "model": self.model,
            "dimensions": self.get_embedding_dimensions(),
            "available": self.is_available()
        }

class CohereEmbeddingsProvider(BaseEmbeddingsProvider):
    """Cohere Embeddings provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('COHERE_API_KEY', '')
        self.model = config.get('cohere_embeddings_model', 'embed-english-v3.0')
    
    def is_available(self) -> bool:
        """Check if Cohere embeddings are available and configured."""
        return bool(self.api_key)
    
    def generate_embeddings(self, text: str, **kwargs) -> List[float]:
        """Generate embeddings using Cohere API."""
        if not self.is_available():
            raise Exception("Cohere embeddings provider not available - missing API key")
        
        payload = {
            "model": self.model,
            "texts": [text],
            "input_type": "search_document"
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                "https://api.cohere.ai/v1/embed",
                json=payload,
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result['embeddings'][0]
            
        except Exception as e:
            self.logger.error(f"Cohere embeddings API error: {str(e)}")
            raise Exception(f"Cohere embeddings provider failed: {str(e)}")
    
    def get_embedding_dimensions(self) -> int:
        """Get the number of dimensions for Cohere embeddings."""
        if self.model == "embed-english-v3.0":
            return 1024
        else:
            return 1024  # Default
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Cohere embeddings model information."""
        return {
            "provider": "cohere",
            "model": self.model,
            "dimensions": self.get_embedding_dimensions(),
            "available": self.is_available()
        }

class GeminiEmbeddingsProvider(BaseEmbeddingsProvider):
    """Google Gemini Embeddings provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('GEMINI_API_KEY', '')
        self.model = config.get('gemini_embeddings_model', 'text-embedding-004')
    
    def is_available(self) -> bool:
        """Check if Gemini embeddings are available and configured."""
        return bool(self.api_key)
    
    def generate_embeddings(self, text: str, **kwargs) -> List[float]:
        """Generate embeddings using Google AI Studio API."""
        if not self.is_available():
            raise Exception("Gemini embeddings provider not available - missing API key")
        
        payload = {
            "model": self.model,
            "content": {
                "parts": [{"text": text}]
            }
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent",
                json=payload,
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result['embedding']['values']
            
        except Exception as e:
            self.logger.error(f"Gemini embeddings API error: {str(e)}")
            raise Exception(f"Gemini embeddings provider failed: {str(e)}")
    
    def get_embedding_dimensions(self) -> int:
        """Get the number of dimensions for Gemini embeddings."""
        return 768  # text-embedding-004 dimensions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Gemini embeddings model information."""
        return {
            "provider": "gemini",
            "model": self.model,
            "dimensions": self.get_embedding_dimensions(),
            "available": self.is_available()
        }

class NomicEmbeddingsProvider(BaseEmbeddingsProvider):
    """Nomic Embeddings provider (via Ollama)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get('ollama_base_url', 'http://localhost:11434')
        self.model = config.get('nomic_embeddings_model', 'nomic-embed-text')
        self.api_key = None  # Nomic via Ollama doesn't need an API key
        self.model_manager = ModelManager(
            github_repo=config.get('github_model_repo', ''),
            cache_dir=config.get('local_model_cache', '~/.cache/rework-models')
        )
    
    def is_available(self) -> bool:
        """Check if Nomic embeddings are available via Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_embeddings(self, text: str, **kwargs) -> List[float]:
        """Generate embeddings using Ollama with Nomic model."""
        if not self.is_available():
            raise Exception("Nomic embeddings provider not available - Ollama not running")
        
        # Ollama manages its own models, no need to download from GitHub
        payload = {
            "model": self.model,
            "prompt": text
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result['embedding']
            
        except Exception as e:
            self.logger.error(f"Nomic embeddings API error: {str(e)}")
            raise Exception(f"Nomic embeddings provider failed: {str(e)}")
    
    def get_embedding_dimensions(self) -> int:
        """Get the number of dimensions for Nomic embeddings."""
        return 768  # nomic-embed-text dimensions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Nomic embeddings model information."""
        return {
            "provider": "nomic",
            "model": self.model,
            "dimensions": self.get_embedding_dimensions(),
            "available": self.is_available()
        }

class E5EmbeddingsProvider(BaseEmbeddingsProvider):
    """E5 Embeddings provider (via Ollama)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get('ollama_base_url', 'http://localhost:11434')
        self.model = config.get('e5_embeddings_model', 'e5-large-v2')
        self.api_key = None  # E5 via Ollama doesn't need an API key
        self.model_manager = ModelManager(
            github_repo=config.get('github_model_repo', ''),
            cache_dir=config.get('local_model_cache', '~/.cache/rework-models')
        )
    
    def is_available(self) -> bool:
        """Check if E5 embeddings are available via Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_embeddings(self, text: str, **kwargs) -> List[float]:
        """Generate embeddings using Ollama with E5 model."""
        if not self.is_available():
            raise Exception("E5 embeddings provider not available - Ollama not running")
        
        # Ollama manages its own models, no need to download from GitHub
        payload = {
            "model": self.model,
            "prompt": text
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result['embedding']
            
        except Exception as e:
            self.logger.error(f"E5 embeddings API error: {str(e)}")
            raise Exception(f"E5 embeddings provider failed: {str(e)}")
    
    def get_embedding_dimensions(self) -> int:
        """Get the number of dimensions for E5 embeddings."""
        return 1024  # e5-large-v2 dimensions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get E5 embeddings model information."""
        return {
            "provider": "e5",
            "model": self.model,
            "dimensions": self.get_embedding_dimensions(),
            "available": self.is_available()
        }

class SentenceTransformersProvider(BaseEmbeddingsProvider):
    """Sentence Transformers provider (via Ollama)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get('ollama_base_url', 'http://localhost:11434')
        self.model = config.get('sentence_transformers_model', 'all-MiniLM-L6-v2')
        self.api_key = None  # Sentence Transformers via Ollama doesn't need an API key
        self.model_manager = ModelManager(
            github_repo=config.get('github_model_repo', ''),
            cache_dir=config.get('local_model_cache', '~/.cache/rework-models')
        )
    
    def is_available(self) -> bool:
        """Check if Sentence Transformers are available via Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_embeddings(self, text: str, **kwargs) -> List[float]:
        """Generate embeddings using Ollama with Sentence Transformers model."""
        if not self.is_available():
            raise Exception("Sentence Transformers provider not available - Ollama not running")
        
        # Ollama manages its own models, no need to download from GitHub
        payload = {
            "model": self.model,
            "prompt": text
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result['embedding']
            
        except Exception as e:
            self.logger.error(f"Sentence Transformers API error: {str(e)}")
            raise Exception(f"Sentence Transformers provider failed: {str(e)}")
    
    def get_embedding_dimensions(self) -> int:
        """Get the number of dimensions for Sentence Transformers embeddings."""
        return 384  # all-MiniLM-L6-v2 dimensions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Sentence Transformers model information."""
        return {
            "provider": "sentence_transformers",
            "model": self.model,
            "dimensions": self.get_embedding_dimensions(),
            "available": self.is_available()
        }

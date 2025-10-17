"""
LLM Provider implementations for various services.
"""

import requests
import json
import os
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional
from .base_provider import BaseLLMProvider
from .model_manager import ModelManager

class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Force load from .env file with override=True to override any existing env vars
        load_dotenv(override=True)
        self.api_key = os.environ.get('OPENAI_API_KEY', config.get('OPENAI_API_KEY', ''))
        self.model = config.get('openai_model', 'gpt-4')
        self.base_url = config.get('openai_base_url', 'https://api.openai.com/v1')
    
    def is_available(self) -> bool:
        """Check if OpenAI is available and configured."""
        return bool(self.api_key)
    
    def generate_response(self, prompt: str, system_prompt: str = "", 
                         temperature: float = 0.3, max_tokens: int = 1000, 
                         **kwargs) -> str:
        """Generate response using OpenAI API."""
        if not self.is_available():
            raise Exception("OpenAI provider not available - missing API key")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=1200
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            raise Exception(f"OpenAI provider failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information."""
        return {
            "provider": "openai",
            "model": self.model,
            "available": self.is_available()
        }

class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('ANTHROPIC_API_KEY', '')
        self.model = config.get('anthropic_model', 'claude-3-sonnet-20240229')
    
    def is_available(self) -> bool:
        """Check if Anthropic is available and configured."""
        return bool(self.api_key)
    
    def generate_response(self, prompt: str, system_prompt: str = "", 
                         temperature: float = 0.3, max_tokens: int = 1000, 
                         **kwargs) -> str:
        """Generate response using Anthropic API."""
        if not self.is_available():
            raise Exception("Anthropic provider not available - missing API key")
        
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers,
                timeout=1200
            )
            response.raise_for_status()
            
            result = response.json()
            return result['content'][0]['text']
            
        except Exception as e:
            self.logger.error(f"Anthropic API error: {str(e)}")
            raise Exception(f"Anthropic provider failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Anthropic model information."""
        return {
            "provider": "anthropic",
            "model": self.model,
            "available": self.is_available()
        }

class MistralProvider(BaseLLMProvider):
    """Mistral LLM provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('MISTRAL_API_KEY', '')
        self.model = config.get('mistral_model', 'mistral-large-latest')
    
    def is_available(self) -> bool:
        """Check if Mistral is available and configured."""
        return bool(self.api_key)
    
    def generate_response(self, prompt: str, system_prompt: str = "", 
                         temperature: float = 0.3, max_tokens: int = 1000, 
                         **kwargs) -> str:
        """Generate response using Mistral API."""
        if not self.is_available():
            raise Exception("Mistral provider not available - missing API key")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=1200
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            self.logger.error(f"Mistral API error: {str(e)}")
            raise Exception(f"Mistral provider failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Mistral model information."""
        return {
            "provider": "mistral",
            "model": self.model,
            "available": self.is_available()
        }

class GemmaProvider(BaseLLMProvider):
    """Gemma LLM provider (via Google AI Studio)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('GEMMA_API_KEY', '')
        self.model = config.get('gemma_model', 'gemma-7b-it')
    
    def is_available(self) -> bool:
        """Check if Gemma is available and configured."""
        return bool(self.api_key)
    
    def generate_response(self, prompt: str, system_prompt: str = "", 
                         temperature: float = 0.3, max_tokens: int = 1000, 
                         **kwargs) -> str:
        """Generate response using Google AI Studio API."""
        if not self.is_available():
            raise Exception("Gemma provider not available - missing API key")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemma-7b-it:generateContent",
                json=payload,
                headers=headers,
                timeout=1200
            )
            response.raise_for_status()
            
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
            
        except Exception as e:
            self.logger.error(f"Gemma API error: {str(e)}")
            raise Exception(f"Gemma provider failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Gemma model information."""
        return {
            "provider": "gemma",
            "model": self.model,
            "available": self.is_available()
        }

class CohereProvider(BaseLLMProvider):
    """Cohere Command R+ LLM provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('COHERE_API_KEY', '')
        self.model = config.get('cohere_model', 'command-r-plus')
    
    def is_available(self) -> bool:
        """Check if Cohere is available and configured."""
        return bool(self.api_key)
    
    def generate_response(self, prompt: str, system_prompt: str = "", 
                         temperature: float = 0.3, max_tokens: int = 1000, 
                         **kwargs) -> str:
        """Generate response using Cohere API."""
        if not self.is_available():
            raise Exception("Cohere provider not available - missing API key")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                "https://api.cohere.ai/v1/chat",
                json=payload,
                headers=headers,
                timeout=1200
            )
            response.raise_for_status()
            
            result = response.json()
            return result['text']
            
        except Exception as e:
            self.logger.error(f"Cohere API error: {str(e)}")
            raise Exception(f"Cohere provider failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Cohere model information."""
        return {
            "provider": "cohere",
            "model": self.model,
            "available": self.is_available()
        }

class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get('ollama_base_url', 'http://localhost:11434')
        self.model = config.get('ollama_model', 'qwen2.5:7b')
        self.use_gpu = config.get('use_gpu', True)
        self.model_manager = ModelManager(
            github_repo=config.get('github_model_repo', ''),
            cache_dir=config.get('local_model_cache', '~/.cache/rework-models')
        )
    
    def is_available(self) -> bool:
        """Check if Ollama is available and running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_response(self, prompt: str, system_prompt: str = "", 
                         temperature: float = 0.3, max_tokens: int = 1000, 
                         **kwargs) -> str:
        """Generate response using Ollama API."""
        if not self.is_available():
            raise Exception("Ollama provider not available - Ollama not running")
        
        # Ensure model is available
        try:
            self.model_manager.download_model(self.model)
        except Exception as e:
            self.logger.warning(f"Could not download model {self.model}: {str(e)}")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        options = {
            "temperature": temperature,
            "num_predict": max_tokens
        }
        
        if self.use_gpu:
            options["gpu_layers"] = -1
            options["num_gpu"] = 1
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=1200
            )
            response.raise_for_status()
            
            result = response.json()
            return result['message']['content']
            
        except Exception as e:
            self.logger.error(f"Ollama API error: {str(e)}")
            raise Exception(f"Ollama provider failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Ollama model information."""
        return {
            "provider": "ollama",
            "model": self.model,
            "available": self.is_available(),
            "gpu_enabled": self.use_gpu
        }

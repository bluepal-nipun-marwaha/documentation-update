"""
Base provider classes for LLM and Embeddings providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import structlog

logger = structlog.get_logger(__name__)

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the provider with configuration."""
        self.config = config
        self.logger = logger
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and configured."""
        pass
    
    @abstractmethod
    def generate_response(self, prompt: str, system_prompt: str = "", 
                         temperature: float = 0.3, max_tokens: int = 1000, 
                         **kwargs) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        pass

class BaseEmbeddingsProvider(ABC):
    """Abstract base class for Embeddings providers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the provider with configuration."""
        self.config = config
        self.logger = logger
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and configured."""
        pass
    
    @abstractmethod
    def generate_embeddings(self, text: str, **kwargs) -> List[float]:
        """Generate embeddings for the given text."""
        pass
    
    @abstractmethod
    def get_embedding_dimensions(self) -> int:
        """Get the number of dimensions for embeddings."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        pass

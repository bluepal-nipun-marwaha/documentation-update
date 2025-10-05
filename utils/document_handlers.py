"""
Document handlers for section-level editing with formatting preservation.
Supports DOCX, Excel, CSV, and Markdown files.
"""

import io
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class UpdateSection:
    """Represents a section to be updated in a document."""
    section_id: str
    section_type: str  # paragraph, table, cell, row, heading, etc.
    action: str  # replace, update_cells, insert, delete
    new_content: Optional[str] = None
    new_values: Optional[List[str]] = None
    target_location: Optional[Dict[str, Any]] = None  # row, col, table_index, etc.
    reason: Optional[str] = None

class BaseDocumentHandler(ABC):
    """Abstract base class for document handlers."""
    
    def __init__(self):
        self.logger = logger
    
    @abstractmethod
    def extract_content_with_structure(self, file_bytes: bytes) -> Dict[str, Any]:
        """
        Extract document content with structural information.
        
        Args:
            file_bytes: Raw file content as bytes
            
        Returns:
            Dictionary containing document structure with sections, IDs, and metadata
        """
        pass
    
    @abstractmethod
    def apply_section_updates(self, file_bytes: bytes, updates: List[UpdateSection]) -> bytes:
        """
        Apply targeted updates to specific sections while preserving formatting.
        
        Args:
            file_bytes: Original file content as bytes
            updates: List of UpdateSection objects describing what to update
            
        Returns:
            Updated file content as bytes
        """
        pass
    
    @abstractmethod
    def get_file_extensions(self) -> List[str]:
        """Get list of file extensions this handler supports."""
        pass
    
    def can_handle(self, file_path: str) -> bool:
        """Check if this handler can process the given file."""
        file_ext = file_path.lower().split('.')[-1] if '.' in file_path else ''
        return file_ext in self.get_file_extensions()

class DocumentHandlerFactory:
    """Factory for creating appropriate document handlers."""
    
    _handlers: List[BaseDocumentHandler] = []
    
    @classmethod
    def register_handler(cls, handler: BaseDocumentHandler):
        """Register a document handler."""
        cls._handlers.append(handler)
    
    @classmethod
    def get_handler(cls, file_path: str) -> Optional[BaseDocumentHandler]:
        """Get the appropriate handler for a file."""
        for handler in cls._handlers:
            if handler.can_handle(file_path):
                return handler
        return None
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get all supported file extensions."""
        extensions = []
        for handler in cls._handlers:
            extensions.extend(handler.get_file_extensions())
        return list(set(extensions))

def get_document_handler(file_path: str) -> Optional[BaseDocumentHandler]:
    """Convenience function to get document handler for a file."""
    return DocumentHandlerFactory.get_handler(file_path)

def is_supported_file_type(file_path: str) -> bool:
    """Check if file type is supported for section-level editing."""
    return DocumentHandlerFactory.get_handler(file_path) is not None


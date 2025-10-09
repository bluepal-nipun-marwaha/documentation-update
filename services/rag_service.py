"""
RAG Service for retrieving relevant embeddings and context.
Now supports multiple embeddings providers.
"""
import json
from typing import Dict, List, Any, Optional, Tuple
import structlog
from datetime import datetime
from utils.docx_handler import DOCXHandler
from utils.config import get_settings
from services.providers import (
    OpenAIEmbeddingsProvider, CohereEmbeddingsProvider, 
    GeminiEmbeddingsProvider, NomicEmbeddingsProvider,
    E5EmbeddingsProvider, SentenceTransformersProvider
)

logger = structlog.get_logger(__name__)

class RAGService:
    """Service for Retrieval-Augmented Generation using embeddings with multiple providers."""
    
    def __init__(self, arango_client, config: Optional[Dict[str, Any]] = None):
        """Initialize RAG service with ArangoDB client and embeddings provider."""
        self.arango_client = arango_client
        
        if config is None:
            settings = get_settings()
            config = settings.ai.dict()
        
        self.config = config
        self.embeddings_provider = self._create_embeddings_provider(config)
        self._initialize_embeddings_provider()
        
        logger.info("[SUCCESS] RAG service initialized successfully")
    
    def _create_embeddings_provider(self, config: Dict[str, Any]):
        """Create the appropriate embeddings provider based on configuration."""
        provider_name = config.get('embeddings_provider', 'nomic').lower()
        
        if provider_name == 'openai':
            return OpenAIEmbeddingsProvider(config)
        elif provider_name == 'cohere':
            return CohereEmbeddingsProvider(config)
        elif provider_name == 'gemini':
            return GeminiEmbeddingsProvider(config)
        elif provider_name == 'nomic':
            return NomicEmbeddingsProvider(config)
        elif provider_name == 'e5':
            return E5EmbeddingsProvider(config)
        elif provider_name == 'sentence_transformers':
            return SentenceTransformersProvider(config)
        else:
            logger.warning(f"Unknown embeddings provider {provider_name}, falling back to Nomic")
            return NomicEmbeddingsProvider(config)
    
    def _initialize_embeddings_provider(self):
        """Initialize the selected embeddings provider."""
        try:
            if self.embeddings_provider.is_available():
                model_info = self.embeddings_provider.get_model_info()
                logger.info(f"[SUCCESS] Embeddings provider initialized: {model_info}")
            else:
                raise Exception(f"Embeddings provider {self.config.get('embeddings_provider')} is not available")
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize embeddings provider: {str(e)}")
            raise
    
    def retrieve_relevant_context(self, commit_context: Dict[str, Any], 
                                 top_k: int = 3) -> str:
        """
        Retrieve relevant documentation context using embeddings.
        
        Args:
            commit_context: Commit information and analysis
            top_k: Number of top relevant documents to retrieve
            
        Returns:
            Concatenated context from relevant documents
        """
        try:
            logger.info("🔍 Retrieving relevant context with RAG...")
            
            # Generate query from commit context
            query_text = self._create_query_from_commit(commit_context)
            
            # Get query embedding
            query_embedding = self._get_query_embedding(query_text)
            if not query_embedding:
                logger.warning("[WARNING] Could not generate query embedding, using fallback")
                return self._fallback_context_retrieval(commit_context)
            
            # Search for similar documents
            similar_docs = self._search_similar_documents(query_embedding, top_k)
            
            # Combine context
            context = self._combine_context(similar_docs, commit_context)
            
            logger.info(f"[SUCCESS] Retrieved context from {len(similar_docs)} relevant documents")
            return context
            
        except Exception as e:
            logger.error(f"[ERROR] RAG context retrieval failed: {str(e)}")
            return self._fallback_context_retrieval(commit_context)
    
    def _create_query_from_commit(self, commit_context: Dict[str, Any]) -> str:
        """Create search query from commit context."""
        
        commit_message = commit_context.get('message', '')
        modified_files = commit_context.get('modified', [])
        analysis = commit_context.get('analysis', {})
        key_changes = analysis.get('key_changes', [])
        
        # Build comprehensive query
        query_parts = []
        
        # Add commit message
        if commit_message:
            query_parts.append(commit_message)
        
        # Add modified files context
        if modified_files:
            query_parts.append(f"Files changed: {', '.join(modified_files)}")
        
        # Add key changes from analysis
        if key_changes:
            query_parts.append(f"Key changes: {', '.join(key_changes)}")
        
        # Add analysis reasoning
        reasoning = analysis.get('reasoning', '')
        if reasoning:
            query_parts.append(f"Impact: {reasoning}")
        
        query = " ".join(query_parts)
        logger.info(f"🔍 Created query: {query[:100]}...")
        return query
    
    def _get_query_embedding(self, query_text: str) -> Optional[List[float]]:
        """Generate embedding for the query text using the configured embeddings provider."""
        try:
            embedding = self.embeddings_provider.generate_embeddings(query_text)
            logger.info(f"[SUCCESS] Generated query embedding: {len(embedding)} dimensions")
            return embedding
        except Exception as e:
            logger.error(f"[ERROR] Query embedding generation failed: {str(e)}")
            return None
    
    def _search_similar_documents(self, query_embedding: List[float], 
                                top_k: int) -> List[Dict[str, Any]]:
        """Search for documents with similar embeddings."""
        try:
            # Convert embedding to string for AQL query
            embedding_str = json.dumps(query_embedding)
            
            # AQL query to find similar documents using cosine similarity
            similarity_query = f"""
            FOR doc IN documents
            LET doc_embedding = (
                FOR emb IN documents_to_embeddings
                FILTER emb._from == doc._id
                FOR embedding IN embeddings
                FILTER embedding._id == emb._to
                RETURN embedding.embeddings
            )
            FILTER LENGTH(doc_embedding) > 0
            LET similarity = (
                LET a = @query_embedding
                LET b = doc_embedding[0]
                LET dot_product = (
                    FOR i IN 0..LENGTH(a)-1
                    RETURN a[i] * b[i]
                )
                LET sum_dot = SUM(dot_product)
                LET norm_a = SQRT(SUM(FOR x IN a RETURN x * x))
                LET norm_b = SQRT(SUM(FOR x IN b RETURN x * x))
                RETURN sum_dot / (norm_a * norm_b)
            )
            SORT similarity DESC
            LIMIT {top_k}
            RETURN {{
                document: doc,
                similarity: similarity,
                path: doc.path,
                content: doc.content
            }}
            """
            
            cursor = self.arango_client.aql.execute(similarity_query, bind_vars={
                'query_embedding': query_embedding
            })
            
            results = list(cursor)
            logger.info(f"🔍 Found {len(results)} similar documents")
            
            return results
            
        except Exception as e:
            logger.error(f"[ERROR] Similarity search failed: {str(e)}")
            return []
    
    def _combine_context(self, similar_docs: List[Dict[str, Any]], 
                        commit_context: Dict[str, Any]) -> str:
        """Combine context from similar documents."""
        try:
            context_parts = []
            
            # Add commit context
            context_parts.append("COMMIT CONTEXT:")
            context_parts.append(f"- Message: {commit_context.get('message', 'N/A')}")
            context_parts.append(f"- Author: {commit_context.get('author', 'N/A')}")
            context_parts.append(f"- Modified files: {', '.join(commit_context.get('modified', []))}")
            
            # Add relevant documentation context
            if similar_docs:
                context_parts.append("\nRELEVANT DOCUMENTATION:")
                for i, doc_result in enumerate(similar_docs, 1):
                    doc = doc_result['document']
                    similarity = doc_result['similarity']
                    content = doc.get('content', '')
                    
                    # Handle similarity as either float or list
                    similarity_str = f"{similarity:.3f}" if isinstance(similarity, (int, float)) else str(similarity)
                    context_parts.append(f"\n{i}. {doc.get('path', 'Unknown')} (similarity: {similarity_str})")
                    context_parts.append(f"Content preview: {content[:300]}...")
            
            context = "\n".join(context_parts)
            logger.info(f"[SUCCESS] Combined context: {len(context)} characters")
            return context
            
        except Exception as e:
            logger.error(f"[ERROR] Context combination failed: {str(e)}")
            return f"Commit: {commit_context.get('message', 'N/A')}"
    
    def _fallback_context_retrieval(self, commit_context: Dict[str, Any]) -> str:
        """Fallback context retrieval when RAG fails."""
        logger.info("🔄 Using fallback context retrieval")
        
        # Simple fallback - just return commit information
        context_parts = [
            "COMMIT CONTEXT (Fallback):",
            f"- Message: {commit_context.get('message', 'N/A')}",
            f"- Author: {commit_context.get('author', 'N/A')}",
            f"- Modified files: {', '.join(commit_context.get('modified', []))}"
        ]
        
        return "\n".join(context_parts)
    
    def get_documentation_files(self, repo_key: str) -> List[str]:
        """Get list of available documentation files for a repository."""
        try:
            query = """
            FOR doc IN documents
            FILTER doc.path LIKE "docs/%"
            RETURN doc.path
            """
            
            cursor = self.arango_client.aql.execute(query)
            files = list(cursor)
            
            logger.info(f"📚 Found {len(files)} documentation files")
            return files
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to get documentation files: {str(e)}")
            return []
    
    def get_document_content(self, file_path: str) -> Optional[str]:
        """Get content of a specific documentation file with enhanced DOCX support."""
        try:
            query = """
            FOR doc IN documents
            FILTER doc.path == @file_path
            RETURN doc.content
            """
            
            cursor = self.arango_client.aql.execute(query, bind_vars={'file_path': file_path})
            results = list(cursor)
            
            if results:
                content = results[0]
                
                # Enhanced DOCX handling with Markdown conversion
                if DOCXHandler.is_docx_file(file_path):
                    logger.info(f"📄 Processing DOCX file with enhanced Markdown conversion: {file_path}")
                    # If content is stored as bytes, convert to Markdown for better LLM processing
                    if isinstance(content, bytes):
                        try:
                            # Use enhanced DOCX to Markdown conversion
                            markdown_content, _ = DOCXHandler.docx_to_markdown_with_metadata(content)
                            logger.info(f"[SUCCESS] Converted DOCX to Markdown: {len(markdown_content)} characters")
                            content = markdown_content
                        except Exception as e:
                            logger.warning(f"[WARNING] Enhanced conversion failed, using fallback: {str(e)}")
                            content = DOCXHandler.extract_text_from_docx(content)
                    elif isinstance(content, str) and content.startswith('b\''):
                        # Handle string representation of bytes
                        import ast
                        try:
                            content_bytes = ast.literal_eval(content)
                            content = DOCXHandler.extract_text_from_docx(content_bytes)
                        except:
                            logger.warning(f"[WARNING] Could not parse DOCX content for {file_path}")
                            return content
                
                return content
            else:
                logger.warning(f"[WARNING] Document not found: {file_path}")
                return None
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to get document content: {str(e)}")
            return None

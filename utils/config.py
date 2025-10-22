"""Configuration management for the rework document update agent."""

from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """Database configuration settings."""
    host: str = Field(default="localhost", description="ArangoDB host")
    port: int = Field(default=8529, description="ArangoDB port")
    username: str = Field(default="root", description="ArangoDB username")
    password: str = Field(default="password", description="ArangoDB password")
    database_name: str = Field(default="mcp_documents", description="Database name")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


class ServerConfig(BaseSettings):
    """Server configuration settings."""
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=50061, description="Server port")
    max_workers: int = Field(default=10, description="Maximum number of worker threads")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


class LoggingConfig(BaseSettings):
    """Logging configuration settings."""
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(default="json", description="Log message format")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


class AIConfig(BaseSettings):
    """AI/LLM configuration settings."""
    # LLM Provider Selection
    llm_provider: str = Field(default="ollama", description="LLM provider: ollama, openai, anthropic, mistral, gemma, cohere", alias="LLM_PROVIDER")
    llm_model: str = Field(default="qwen2.5:7b", description="LLM model name", alias="LLM_MODEL")
    
    # Embeddings Provider Selection
    embeddings_provider: str = Field(default="nomic", description="Embeddings provider: openai, cohere, gemini, nomic, e5, sentence_transformers", alias="EMBEDDINGS_PROVIDER")
    embeddings_model: str = Field(default="nomic-embed-text", description="Embeddings model name", alias="EMBEDDINGS_MODEL")
    
    # Model Management
    use_github_models: bool = Field(default=True, description="Use GitHub for model downloads")
    github_model_repo: str = Field(default="", description="GitHub repository for model releases")
    local_model_cache: str = Field(default="~/.cache/rework-models", description="Local model cache directory")
    
    # Ollama Configuration (for local models)
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama server URL", alias="OLLAMA_BASE_URL")
    use_gpu: bool = Field(default=True, description="Enable GPU acceleration for LLM inference", alias="USE_GPU")
    gpu_layers: int = Field(default=-1, description="Number of GPU layers to use (-1 for all)", alias="GPU_LAYERS")
    
    # OpenAI Configuration
    openai_api_key: str = Field(default="", description="OpenAI API key", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", description="OpenAI LLM model", alias="OPENAI_MODEL")
    openai_embeddings_model: str = Field(default="text-embedding-3-large", description="OpenAI embeddings model", alias="OPENAI_EMBEDDINGS_MODEL")
    openai_base_url: str = Field(default="https://api.openai.com/v1", description="OpenAI API base URL", alias="OPENAI_BASE_URL")
    
    # Anthropic Configuration
    anthropic_api_key: str = Field(default="", description="Anthropic API key", alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-sonnet-20240229", description="Anthropic LLM model", alias="ANTHROPIC_MODEL")
    
    # Mistral Configuration
    mistral_api_key: str = Field(default="", description="Mistral API key", alias="MISTRAL_API_KEY")
    mistral_model: str = Field(default="mistral-large-latest", description="Mistral LLM model", alias="MISTRAL_MODEL")
    
    # Gemma Configuration
    gemma_api_key: str = Field(default="", description="Gemma API key", alias="GEMMA_API_KEY")
    gemma_model: str = Field(default="gemma-7b-it", description="Gemma LLM model", alias="GEMMA_MODEL")
    
    # Cohere Configuration
    cohere_api_key: str = Field(default="", description="Cohere API key", alias="COHERE_API_KEY")
    cohere_model: str = Field(default="command-r-plus", description="Cohere LLM model", alias="COHERE_MODEL")
    cohere_embeddings_model: str = Field(default="embed-english-v3.0", description="Cohere embeddings model", alias="COHERE_EMBEDDINGS_MODEL")
    
    # Gemini Configuration
    gemini_api_key: str = Field(default="", description="Gemini API key", alias="GEMINI_API_KEY")
    gemini_embeddings_model: str = Field(default="text-embedding-004", description="Gemini embeddings model", alias="GEMINI_EMBEDDINGS_MODEL")
    
    # Local Model Configuration
    nomic_embeddings_model: str = Field(default="nomic-embed-text", description="Nomic embeddings model")
    e5_embeddings_model: str = Field(default="e5-large-v2", description="E5 embeddings model")
    sentence_transformers_model: str = Field(default="all-MiniLM-L6-v2", description="Sentence Transformers model")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


class DocumentConfig(BaseSettings):
    """Document processing configuration settings."""
    storage_path: str = Field(default="documents", description="Document storage directory")
    backup_path: str = Field(default="rework/backups", description="Backup directory")
    audit_log_path: str = Field(default="rework/audit/audit.log", description="Audit log path")
    
    # DOCX Workflow Configuration
    new_docx_workflow_enabled: bool = Field(default=True, description="Enable new conditional DOCX workflow", alias="NEW_DOCX_WORKFLOW")
    docx_table_formatting: str = Field(default="github_docs", description="Table formatting style: github_docs, basic", alias="DOCX_TABLE_FORMATTING")
    
    # Table Update Configuration
    table_confidence_threshold: float = Field(default=0.5, description="Minimum confidence threshold for table updates (0.0-1.0)", alias="TABLE_CONFIDENCE_THRESHOLD")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


class VCSConfig(BaseSettings):
    """Version control system configuration."""
    github_token: str = Field(default="", description="GitHub access token", alias="GITHUB_TOKEN")
    gitlab_token: str = Field(default="", description="GitLab access token", alias="GITLAB_TOKEN")
    webhook_secret: str = Field(default="", description="Webhook secret", alias="WEBHOOK_SECRET")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


class Settings(BaseSettings):
    """Main application settings."""
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    document: DocumentConfig = Field(default_factory=DocumentConfig)
    vcs: VCSConfig = Field(default_factory=VCSConfig)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"
        extra = "ignore"  # Allow extra fields from environment variables


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

def clear_settings_cache():
    """Clear the global settings cache to force reload."""
    global _settings
    _settings = None

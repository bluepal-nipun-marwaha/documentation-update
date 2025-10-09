#!/usr/bin/env python3
"""
Model Setup Script for Rework Document Update System
Interactive script to download and configure LLM and Embedding models.
"""

import os
import sys
import requests
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import structlog

# Try to import psutil for system detection
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not installed. Installing for system detection...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        import psutil
        PSUTIL_AVAILABLE = True
        print("✅ psutil installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Failed to install psutil. System detection will be limited.")

# Setup logging
logging = structlog.get_logger(__name__)

class ModelSetup:
    """Interactive model setup and configuration."""
    
    def __init__(self):
        self.ollama_base_url = "http://localhost:11434"
        self.models_dir = Path.home() / ".cache" / "rework-models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # LLM Model options
        self.llm_models = {
            "1": {
                "name": "qwen2.5:7b",
                "description": "Qwen2.5:7b (Recommended) - Excellent code understanding, good documentation generation",
                "ram_usage": "7GB",
                "best_for": "Code analysis, technical writing, commit impact assessment",
                "speed": "Fast inference, good balance of quality/speed"
            },
            "2": {
                "name": "llama3.1:8b", 
                "description": "Llama 3.1:8b (High Quality) - Superior reasoning, excellent for complex analysis",
                "ram_usage": "8GB",
                "best_for": "Complex commit analysis, architectural documentation",
                "speed": "Slightly slower but higher quality"
            },
            "3": {
                "name": "codellama:7b",
                "description": "CodeLlama:7b (Code Specialized) - Specialized for code understanding",
                "ram_usage": "7GB", 
                "best_for": "Code-focused documentation, API reference updates",
                "speed": "Fast, purpose-built for code"
            },
            "4": {
                "name": "mistral:7b",
                "description": "Mistral:7b (Fast) - Fast, efficient, good general performance",
                "ram_usage": "7GB",
                "best_for": "Quick documentation updates, general purpose",
                "speed": "Very fast inference"
            },
            "5": {
                "name": "phi3:3.8b",
                "description": "Phi-3:3.8b (Lightweight) - Very fast, low resource usage",
                "ram_usage": "4GB",
                "best_for": "Quick updates, resource-constrained environments", 
                "speed": "Very fast, minimal resources"
            },
            "6": {
                "name": "gemma:7b",
                "description": "Gemma:7b (Google) - Google's model, good multilingual support",
                "ram_usage": "7GB",
                "best_for": "International projects, diverse documentation",
                "speed": "Good general performance"
            },
            "7": {
                "name": "none",
                "description": "None - Use remote LLM providers only (OpenAI, Anthropic, etc.)",
                "ram_usage": "0GB",
                "best_for": "When you prefer cloud-based LLM services",
                "speed": "Depends on remote provider"
            }
        }
        
        # Remote LLM Model options
        self.remote_llm_models = {
            "1": {
                "provider": "openai",
                "name": "gpt-4o",
                "description": "GPT-4o (OpenAI) - Latest and most capable OpenAI model",
                "best_for": "Complex analysis, high-quality documentation generation",
                "cost": "Higher cost, premium quality"
            },
            "2": {
                "provider": "openai", 
                "name": "gpt-4o-mini",
                "description": "GPT-4o Mini (OpenAI) - Fast and cost-effective",
                "best_for": "Quick updates, balanced quality and cost",
                "cost": "Lower cost, good quality"
            },
            "3": {
                "provider": "anthropic",
                "name": "claude-3-5-sonnet-20241022",
                "description": "Claude 3.5 Sonnet (Anthropic) - Superior reasoning and analysis",
                "best_for": "Complex commit analysis, architectural documentation",
                "cost": "Premium quality, excellent reasoning"
            },
            "4": {
                "provider": "anthropic",
                "name": "claude-3-haiku-20240307", 
                "description": "Claude 3 Haiku (Anthropic) - Fast and efficient",
                "best_for": "Quick documentation updates, general purpose",
                "cost": "Lower cost, fast processing"
            },
            "5": {
                "provider": "mistral",
                "name": "mistral-large-latest",
                "description": "Mistral Large (Mistral) - High-quality European model",
                "best_for": "Multilingual projects, balanced performance",
                "cost": "Competitive pricing, good quality"
            },
            "6": {
                "provider": "mistral",
                "name": "mistral-small-latest",
                "description": "Mistral Small (Mistral) - Fast and cost-effective",
                "best_for": "Quick updates, resource-efficient",
                "cost": "Lower cost, fast processing"
            },
            "7": {
                "provider": "cohere",
                "name": "command-r-plus",
                "description": "Command R+ (Cohere) - Excellent for technical content",
                "best_for": "Code documentation, technical writing",
                "cost": "Good value, technical focus"
            },
            "8": {
                "provider": "cohere",
                "name": "command-r",
                "description": "Command R (Cohere) - Fast and reliable",
                "best_for": "General documentation, reliable performance",
                "cost": "Lower cost, reliable quality"
            }
        }
        
        # Remote Embedding Model options
        self.remote_embedding_models = {
            "1": {
                "provider": "openai",
                "name": "text-embedding-3-large",
                "description": "OpenAI Embedding 3 Large - Highest quality embeddings",
                "dimensions": "3072",
                "best_for": "Complex documentation, best semantic understanding",
                "cost": "Higher cost, premium quality"
            },
            "2": {
                "provider": "openai",
                "name": "text-embedding-3-small", 
                "description": "OpenAI Embedding 3 Small - Fast and cost-effective",
                "dimensions": "1536",
                "best_for": "General use, balanced quality and cost",
                "cost": "Lower cost, good quality"
            },
            "3": {
                "provider": "cohere",
                "name": "embed-english-v3.0",
                "description": "Cohere English v3.0 - Excellent for English technical content",
                "dimensions": "1024",
                "best_for": "English documentation, technical content",
                "cost": "Good value, English-optimized"
            },
            "4": {
                "provider": "cohere",
                "name": "embed-multilingual-v3.0",
                "description": "Cohere Multilingual v3.0 - Great for international projects",
                "dimensions": "1024", 
                "best_for": "Multilingual documentation, global projects",
                "cost": "Good value, multilingual support"
            },
            "5": {
                "provider": "gemini",
                "name": "text-embedding-004",
                "description": "Gemini Embedding 004 - Google's high-quality embeddings",
                "dimensions": "768",
                "best_for": "General purpose, Google ecosystem integration",
                "cost": "Competitive pricing, reliable quality"
            },
            "6": {
                "provider": "gemini",
                "name": "text-multilingual-embedding-002",
                "description": "Gemini Multilingual Embedding - Best for diverse languages",
                "dimensions": "768",
                "best_for": "Multilingual projects, diverse content",
                "cost": "Good value, multilingual support"
            }
        }
        
        # Embedding Model options
        self.embedding_models = {
            "1": {
                "name": "nomic-embed-text",
                "description": "nomic-embed-text (Recommended) - Excellent semantic understanding, good for technical docs",
                "dimensions": "768",
                "size": "1GB",
                "best_for": "Code documentation, technical content retrieval"
            },
            "2": {
                "name": "e5-large-v2",
                "description": "e5-large-v2 (High Quality) - Superior embedding quality, better semantic matching",
                "dimensions": "1024", 
                "size": "1.5GB",
                "best_for": "Complex documentation relationships, better RAG"
            },
            "3": {
                "name": "all-MiniLM-L6-v2",
                "description": "all-MiniLM-L6-v2 (Lightweight) - Fast, lightweight, good general performance",
                "dimensions": "384",
                "size": "500MB", 
                "best_for": "Quick retrieval, resource-constrained environments"
            },
            "4": {
                "name": "none",
                "description": "None - Use remote embedding providers only (OpenAI, Cohere, etc.)",
                "dimensions": "0",
                "size": "0MB",
                "best_for": "When you prefer cloud-based embedding services"
            }
        }
    
    def get_model_type_choice(self) -> str:
        """Get user's choice between local and remote models."""
        print("\n" + "="*60)
        print("🚀 MODEL TYPE SELECTION")
        print("="*60)
        print("Choose your preferred model type:")
        print()
        print("1. Local Models (Ollama)")
        print("   ✅ Privacy - No data sent to external services")
        print("   ✅ No ongoing costs - One-time setup")
        print("   ✅ Full control - Run on your hardware")
        print("   ⚠️ Requires: Local GPU/RAM, Ollama installation")
        print()
        print("2. Remote Models (Cloud APIs)")
        print("   ✅ No local resources required")
        print("   ✅ Latest models - Always up-to-date")
        print("   ✅ High performance - Enterprise-grade infrastructure")
        print("   ⚠️ Requires: API keys, internet connection, ongoing costs")
        print()
        print("="*60)
        
        while True:
            choice = input("Select model type (1-2): ").strip()
            if choice == "1":
                return "local"
            elif choice == "2":
                return "remote"
            else:
                print("❌ Invalid choice. Please select 1 or 2.")
    
    def display_remote_llm_menu(self):
        """Display remote LLM model selection menu."""
        print("\n🤖 REMOTE LLM MODEL SELECTION")
        print("="*60)
        
        for key, model in self.remote_llm_models.items():
            print(f"{key}. {model['description']}")
            print(f"   Provider: {model['provider'].title()}")
            print(f"   Best For: {model['best_for']}")
            print(f"   Cost: {model['cost']}")
            print()
        
        print("💡 Recommendations:")
        print("   • For best quality: Option 1 (GPT-4o) or Option 3 (Claude 3.5 Sonnet)")
        print("   • For cost efficiency: Option 2 (GPT-4o Mini) or Option 4 (Claude 3 Haiku)")
        print("   • For technical content: Option 7 (Command R+)")
        print("="*60)
    
    def display_remote_embedding_menu(self):
        """Display remote embedding model selection menu."""
        print("\n🔍 REMOTE EMBEDDING MODEL SELECTION")
        print("="*60)
        
        for key, model in self.remote_embedding_models.items():
            print(f"{key}. {model['description']}")
            print(f"   Provider: {model['provider'].title()}")
            print(f"   Dimensions: {model['dimensions']}")
            print(f"   Best For: {model['best_for']}")
            print(f"   Cost: {model['cost']}")
            print()
        
        print("💡 Recommendations:")
        print("   • For best quality: Option 1 (OpenAI Embedding 3 Large)")
        print("   • For cost efficiency: Option 2 (OpenAI Embedding 3 Small)")
        print("   • For English technical docs: Option 3 (Cohere English v3.0)")
        print("   • For multilingual projects: Option 4 (Cohere Multilingual) or Option 6 (Gemini Multilingual)")
        print("="*60)
    
    def check_ollama_running(self) -> bool:
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama is running")
                return True
            else:
                print(f"❌ Ollama responded with status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("❌ Ollama is not running. Please start Ollama first:")
            print("   • Install Ollama: https://ollama.ai/download")
            print("   • Start Ollama: ollama serve")
            return False
        except Exception as e:
            print(f"❌ Error checking Ollama: {str(e)}")
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for recommendations."""
        if not PSUTIL_AVAILABLE:
            print("⚠️ psutil not available, cannot detect system specs")
            return {
                "ram_gb": "unknown",
                "gpu_available": False,
                "recommendation": "balanced"
            }
        
        try:
            # Get RAM info
            ram_gb = psutil.virtual_memory().total / (1024**3)
            
            # Check for GPU using multiple methods
            gpu_available = False
            gpu_info = "None detected"
            
            # Method 1: Try PyTorch
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_available = True
                    gpu_count = torch.cuda.device_count()
                    gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                    gpu_info = f"{gpu_name} ({gpu_count} device{'s' if gpu_count > 1 else ''})"
            except ImportError:
                pass
            
            # Method 2: Try nvidia-smi command
            if not gpu_available:
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0 and result.stdout.strip():
                        gpu_available = True
                        gpu_info = result.stdout.strip().split('\n')[0]
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass
            
            # Method 3: Check for AMD GPU (Windows)
            if not gpu_available:
                try:
                    result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0 and 'AMD' in result.stdout:
                        gpu_available = True
                        gpu_info = "AMD GPU detected"
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass
            
            return {
                "ram_gb": round(ram_gb, 1),
                "gpu_available": gpu_available,
                "gpu_info": gpu_info,
                "recommendation": self._get_recommendation(ram_gb, gpu_available)
            }
        except Exception as e:
            print(f"⚠️ Error detecting system specs: {str(e)}")
            return {
                "ram_gb": "unknown",
                "gpu_available": False,
                "gpu_info": "Detection failed",
                "recommendation": "balanced"
            }
    
    def _get_recommendation(self, ram_gb: float, gpu_available: bool) -> str:
        """Get recommendation based on system specs."""
        if ram_gb < 8:
            return "lightweight"
        elif ram_gb < 16:
            # If GPU is available, we can recommend higher quality models
            return "high_quality" if gpu_available else "balanced"
        else:
            return "high_quality"
    
    def display_system_info(self, system_info: Dict[str, Any]):
        """Display system information and recommendations."""
        print("\n" + "="*60)
        print("🖥️  SYSTEM INFORMATION")
        print("="*60)
        print(f"RAM: {system_info['ram_gb']}GB")
        print(f"GPU Available: {'Yes' if system_info['gpu_available'] else 'No'}")
        if 'gpu_info' in system_info:
            print(f"GPU Info: {system_info['gpu_info']}")
        
        recommendation = system_info['recommendation']
        if recommendation == "lightweight":
            print("💡 Recommendation: Lightweight models (Phi-3, MiniLM)")
            print("   Reason: Limited RAM detected, prioritizing speed and efficiency")
        elif recommendation == "balanced":
            print("💡 Recommendation: Balanced models (Qwen2.5, nomic-embed-text)")
            print("   Reason: Good RAM available, balanced quality and performance")
        else:
            print("💡 Recommendation: High-quality models (Llama 3.1, e5-large-v2)")
            print("   Reason: High RAM available, prioritizing quality and accuracy")
        print("="*60)
    
    def display_llm_menu(self, system_info: Dict[str, Any]):
        """Display LLM model selection menu."""
        print("\n🤖 LLM MODEL SELECTION")
        print("="*60)
        
        for key, model in self.llm_models.items():
            if key == "7":  # None option
                print(f"{key}. {model['description']}")
            else:
                print(f"{key}. {model['description']}")
                print(f"   RAM Usage: {model['ram_usage']}")
                print(f"   Best For: {model['best_for']}")
                print(f"   Speed: {model['speed']}")
                print()
        
        # Add recommendation based on system
        recommendation = system_info['recommendation']
        if recommendation == "lightweight":
            print("💡 Recommended for your system: Option 5 (Phi-3:3.8b)")
        elif recommendation == "balanced":
            print("💡 Recommended for your system: Option 1 (Qwen2.5:7b)")
        else:
            print("💡 Recommended for your system: Option 2 (Llama 3.1:8b)")
        
        print("="*60)
    
    def display_embedding_menu(self, system_info: Dict[str, Any]):
        """Display embedding model selection menu."""
        print("\n🔍 EMBEDDING MODEL SELECTION")
        print("="*60)
        
        for key, model in self.embedding_models.items():
            if key == "4":  # None option
                print(f"{key}. {model['description']}")
            else:
                print(f"{key}. {model['description']}")
                print(f"   Dimensions: {model['dimensions']}")
                print(f"   Size: {model['size']}")
                print(f"   Best For: {model['best_for']}")
                print()
        
        # Add recommendation based on system
        recommendation = system_info['recommendation']
        if recommendation == "lightweight":
            print("💡 Recommended for your system: Option 3 (all-MiniLM-L6-v2)")
        elif recommendation == "balanced":
            print("💡 Recommended for your system: Option 1 (nomic-embed-text)")
        else:
            print("💡 Recommended for your system: Option 2 (e5-large-v2)")
        
        print("="*60)
    
    def get_user_choice(self, prompt: str, valid_choices: List[str]) -> str:
        """Get user choice with validation."""
        while True:
            choice = input(f"\n{prompt}: ").strip()
            if choice in valid_choices:
                return choice
            else:
                print(f"❌ Invalid choice. Please select from: {', '.join(valid_choices)}")
    
    def download_model(self, model_name: str) -> bool:
        """Download a model using Ollama."""
        if model_name == "none":
            return True
            
        print(f"\n🔄 Downloading model: {model_name}")
        print("   This may take several minutes depending on model size...")
        
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/pull",
                json={"name": model_name},
                timeout=600  # 10 minutes timeout
            )
            
            if response.status_code == 200:
                print(f"✅ Successfully downloaded model: {model_name}")
                return True
            else:
                print(f"❌ Failed to download model {model_name}: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error downloading model {model_name}: {str(e)}")
            return False
    
    def test_model(self, model_name: str, model_type: str) -> bool:
        """Test a model with a simple prompt."""
        if model_name == "none":
            return True
            
        print(f"🧪 Testing {model_type} model: {model_name}")
        
        try:
            if model_type == "llm":
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": "Hello! Please respond with 'Model is working correctly.'"}
                    ],
                    "stream": False
                }
                endpoint = "/api/chat"
            else:  # embedding
                payload = {
                    "model": model_name,
                    "prompt": "test embedding"
                }
                endpoint = "/api/embeddings"
            
            response = requests.post(
                f"{self.ollama_base_url}{endpoint}",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if model_type == "llm":
                    response_text = result['message']['content']
                    print(f"✅ LLM test successful: {response_text[:100]}...")
                else:
                    embedding = result['embedding']
                    print(f"✅ Embedding test successful: {len(embedding)} dimensions")
                return True
            else:
                print(f"❌ {model_type.upper()} test failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error testing {model_type} model: {str(e)}")
            return False
    
    def generate_env_config(self, llm_model: str, embedding_model: str, model_type: str = "local") -> str:
        """Generate .env configuration based on selected models."""
        if model_type == "local":
            return self._generate_local_config(llm_model, embedding_model)
        else:
            return self._generate_remote_config(llm_model, embedding_model)
    
    def _generate_local_config(self, llm_model: str, embedding_model: str) -> str:
        """Generate complete configuration for local models."""
        config_lines = [
            "# Generated by setup_models.py - Complete Local Configuration",
            "# ==============================================================================",
            "# AI/LLM CONFIGURATION",
            "# ==============================================================================",
            "",
            "# LLM Provider Configuration",
            f"LLM_PROVIDER={'ollama' if llm_model != 'none' else 'openai'}",
            f"LLM_MODEL={llm_model if llm_model != 'none' else 'gpt-4'}",
            "",
            "# Embeddings Provider Configuration", 
            f"EMBEDDINGS_PROVIDER={'nomic' if embedding_model == 'nomic-embed-text' else 'e5' if embedding_model == 'e5-large-v2' else 'sentence_transformers' if embedding_model == 'all-MiniLM-L6-v2' else 'openai' if embedding_model == 'none' else 'nomic'}",
            f"EMBEDDINGS_MODEL={embedding_model if embedding_model != 'none' else 'text-embedding-3-large'}",
            "",
            "# Ollama Configuration (for local models)",
            "OLLAMA_BASE_URL=http://localhost:11434",
            "USE_GPU=true",
            "GPU_LAYERS=-1",
            "",
            "# ==============================================================================",
            "# DATABASE CONFIGURATION (ArangoDB)",
            "# ==============================================================================",
            "",
            "# ArangoDB connection settings - defaults work with docker-compose",
            "ARANGO_HOST=localhost",
            "ARANGO_PORT=8529",
            "ARANGO_USERNAME=root",
            "ARANGO_PASSWORD=openSesame",
            "ARANGO_DATABASE=mcp_documents",
            "",
            "# ==============================================================================",
            "# SERVER CONFIGURATION",
            "# ==============================================================================",
            "",
            "# gRPC server settings",
            "SERVER_HOST=0.0.0.0",
            "SERVER_PORT=50061",
            "SERVER_MAX_WORKERS=10",
            "",
            "# Webhook HTTP server settings",
            "WEBHOOK_HOST=0.0.0.0",
            "WEBHOOK_PORT=8000",
            "",
            "# ==============================================================================",
            "# LOGGING CONFIGURATION",
            "# ==============================================================================",
            "",
            "# Logging settings",
            "LOG_LEVEL=INFO",
            "LOG_FORMAT=json",
            "",
            "# ==============================================================================",
            "# NGROK CONFIGURATION (Required for Webhooks)",
            "# ==============================================================================",
            "",
            "# Ngrok tunnel URL - GET FROM: ngrok http 8000",
            "# This creates a public URL that forwards to your local webhook server",
            "NGROK_BASE_URL=https://your-tunnel-id.ngrok.io",
            "",
            "# ==============================================================================",
            "# VCS CONFIGURATION (Optional - for uploading updated docs)",
            "# ==============================================================================",
            "",
            "# GitHub access token for repository operations - GET FROM: https://github.com/settings/tokens",
            "# Required if you want the system to automatically upload updated documentation",
            "GITHUB_TOKEN=",
            "",
            "# GitLab access token for repository operations - GET FROM: https://gitlab.com/-/profile/personal_access_tokens",
            "# Required if you want the system to automatically upload updated documentation",
            "GITLAB_TOKEN=",
            "GITLAB_BASE_URL=https://gitlab.com",
            "GITLAB_PROJECT_ID=",
            "GITLAB_DOCS_BRANCH=main",
            "",
            "# Webhook secret for validating incoming webhooks",
            "# Set this same secret in your GitHub/GitLab webhook settings for security",
            "WEBHOOK_SECRET=git_token_secret",
            "",
            "# ==============================================================================",
            "# DOCUMENT PROCESSING CONFIGURATION",
            "# ==============================================================================",
            "",
            "# File paths for document storage and processing",
            "DOCUMENT_STORAGE_PATH=documents",
            "DOCUMENT_BACKUP_PATH=rework/backups",
            "DOCUMENT_AUDIT_LOG_PATH=rework/audit/audit.log",
            "",
            "# ==============================================================================",
            "# DEVELOPMENT/TESTING CONFIGURATION",
            "# ==============================================================================",
            "",
            "# Set to 'development' for local testing, 'production' for deployment",
            "ENVIRONMENT=development",
            "",
            "# Enable debug mode (set to 'true' for verbose logging)",
            "DEBUG=false",
            "",
            "# ==============================================================================",
            "# API KEYS (Add your keys below if using remote providers)",
            "# ==============================================================================",
            "",
            "# OpenAI Configuration",
            "OPENAI_API_KEY=",
            "OPENAI_MODEL=gpt-4o",
            "OPENAI_EMBEDDINGS_MODEL=text-embedding-3-large",
            "OPENAI_BASE_URL=https://api.openai.com/v1",
            "",
            "# Anthropic Configuration",
            "ANTHROPIC_API_KEY=",
            "ANTHROPIC_MODEL=claude-3-5-sonnet-20241022",
            "",
            "# Mistral Configuration",
            "MISTRAL_API_KEY=",
            "MISTRAL_MODEL=mistral-large-latest",
            "",
            "# Cohere Configuration",
            "COHERE_API_KEY=",
            "COHERE_MODEL=command-r-plus",
            "COHERE_EMBEDDINGS_MODEL=embed-english-v3.0",
            "",
            "# Gemini Configuration",
            "GEMINI_API_KEY=",
            "GEMINI_EMBEDDINGS_MODEL=text-embedding-004",
            "",
            "# ==============================================================================",
            "# QUICK SETUP INSTRUCTIONS",
            "# ==============================================================================",
            "# 1. Fill in your API keys above if using remote providers",
            "# 2. Set your GitLab/GitHub tokens if using VCS integration",
            "# 3. Run: poetry install && poetry run python scripts/generate_protobuf.py",
            "# 4. Start server: poetry run rework-server",
            "# 5. Configure repositories using the /configure endpoint",
            "# 6. Test with a commit to your GitHub/GitLab repository"
        ]
        return "\n".join(config_lines)
    
    def _generate_remote_config(self, llm_model: str, embedding_model: str) -> str:
        """Generate complete configuration for remote models."""
        llm_provider = self.remote_llm_models[llm_model]["provider"]
        embedding_provider = self.remote_embedding_models[embedding_model]["provider"]
        
        config_lines = [
            "# Generated by setup_models.py - Complete Remote Configuration",
            "# ==============================================================================",
            "# AI/LLM CONFIGURATION",
            "# ==============================================================================",
            "",
            "# LLM Provider Configuration",
            f"LLM_PROVIDER={llm_provider}",
            f"LLM_MODEL={self.remote_llm_models[llm_model]['name']}",
            "",
            "# Embeddings Provider Configuration",
            f"EMBEDDINGS_PROVIDER={embedding_provider}",
            f"EMBEDDINGS_MODEL={self.remote_embedding_models[embedding_model]['name']}",
            "",
            "# ==============================================================================",
            "# DATABASE CONFIGURATION (ArangoDB)",
            "# ==============================================================================",
            "",
            "# ArangoDB connection settings - defaults work with docker-compose",
            "ARANGO_HOST=localhost",
            "ARANGO_PORT=8529",
            "ARANGO_USERNAME=root",
            "ARANGO_PASSWORD=openSesame",
            "ARANGO_DATABASE=mcp_documents",
            "",
            "# ==============================================================================",
            "# SERVER CONFIGURATION",
            "# ==============================================================================",
            "",
            "# gRPC server settings",
            "SERVER_HOST=0.0.0.0",
            "SERVER_PORT=50061",
            "SERVER_MAX_WORKERS=10",
            "",
            "# Webhook HTTP server settings",
            "WEBHOOK_HOST=0.0.0.0",
            "WEBHOOK_PORT=8000",
            "",
            "# ==============================================================================",
            "# LOGGING CONFIGURATION",
            "# ==============================================================================",
            "",
            "# Logging settings",
            "LOG_LEVEL=INFO",
            "LOG_FORMAT=json",
            "",
            "# ==============================================================================",
            "# NGROK CONFIGURATION (Required for Webhooks)",
            "# ==============================================================================",
            "",
            "# Ngrok tunnel URL - GET FROM: ngrok http 8000",
            "# This creates a public URL that forwards to your local webhook server",
            "NGROK_BASE_URL=https://your-tunnel-id.ngrok.io",
            "",
            "# ==============================================================================",
            "# VCS CONFIGURATION (Optional - for uploading updated docs)",
            "# ==============================================================================",
            "",
            "# GitHub access token for repository operations - GET FROM: https://github.com/settings/tokens",
            "# Required if you want the system to automatically upload updated documentation",
            "GITHUB_TOKEN=",
            "",
            "# GitLab access token for repository operations - GET FROM: https://gitlab.com/-/profile/personal_access_tokens",
            "# Required if you want the system to automatically upload updated documentation",
            "GITLAB_TOKEN=",
            "GITLAB_BASE_URL=https://gitlab.com",
            "GITLAB_PROJECT_ID=",
            "GITLAB_DOCS_BRANCH=main",
            "",
            "# Webhook secret for validating incoming webhooks",
            "# Set this same secret in your GitHub/GitLab webhook settings for security",
            "WEBHOOK_SECRET=git_token_secret",
            "",
            "# ==============================================================================",
            "# DOCUMENT PROCESSING CONFIGURATION",
            "# ==============================================================================",
            "",
            "# File paths for document storage and processing",
            "DOCUMENT_STORAGE_PATH=documents",
            "DOCUMENT_BACKUP_PATH=rework/backups",
            "DOCUMENT_AUDIT_LOG_PATH=rework/audit/audit.log",
            "",
            "# ==============================================================================",
            "# DEVELOPMENT/TESTING CONFIGURATION",
            "# ==============================================================================",
            "",
            "# Set to 'development' for local testing, 'production' for deployment",
            "ENVIRONMENT=development",
            "",
            "# Enable debug mode (set to 'true' for verbose logging)",
            "DEBUG=false",
            "",
            "# ==============================================================================",
            "# API KEYS (REQUIRED - Add your actual keys below)",
            "# ==============================================================================",
            "",
            "# OpenAI Configuration",
            "OPENAI_API_KEY=",
            "OPENAI_MODEL=gpt-4o",
            "OPENAI_EMBEDDINGS_MODEL=text-embedding-3-large",
            "OPENAI_BASE_URL=https://api.openai.com/v1",
            "",
            "# Anthropic Configuration", 
            "ANTHROPIC_API_KEY=",
            "ANTHROPIC_MODEL=claude-3-5-sonnet-20241022",
            "",
            "# Mistral Configuration",
            "MISTRAL_API_KEY=",
            "MISTRAL_MODEL=mistral-large-latest",
            "",
            "# Cohere Configuration",
            "COHERE_API_KEY=",
            "COHERE_MODEL=command-r-plus",
            "COHERE_EMBEDDINGS_MODEL=embed-english-v3.0",
            "",
            "# Gemini Configuration",
            "GEMINI_API_KEY=",
            "GEMINI_EMBEDDINGS_MODEL=text-embedding-004",
            "",
            "# Ollama Configuration (for fallback)",
            "OLLAMA_BASE_URL=http://localhost:11434",
            "USE_GPU=true",
            "GPU_LAYERS=-1",
            "",
            "# ==============================================================================",
            "# QUICK SETUP INSTRUCTIONS",
            "# ==============================================================================",
            "# 1. Fill in your API keys above (REQUIRED for remote models)",
            "# 2. Set your GitLab/GitHub tokens if using VCS integration",
            "# 3. Run: poetry install && poetry run python scripts/generate_protobuf.py",
            "# 4. Start server: poetry run rework-server",
            "# 5. Configure repositories using the /configure endpoint",
            "# 6. Test with a commit to your GitHub/GitLab repository",
            "",
            "# ==============================================================================",
            "# API KEY SETUP LINKS",
            "# ==============================================================================",
            "# OpenAI: https://platform.openai.com/api-keys",
            "# Anthropic: https://console.anthropic.com/",
            "# Mistral: https://console.mistral.ai/",
            "# Cohere: https://dashboard.cohere.ai/",
            "# Gemini: https://aistudio.google.com/app/apikey"
        ]
        return "\n".join(config_lines)
    
    def save_config(self, config_content: str, llm_model: str, embedding_model: str):
        """Save configuration to .env file."""
        env_file = Path(".env")
        backup_file = Path(".env.backup")
        
        if env_file.exists():
            # Remove existing backup if it exists (Windows-safe)
            if backup_file.exists():
                try:
                    backup_file.unlink()
                except OSError:
                    # If unlink fails, try to overwrite the backup
                    try:
                        backup_file.write_text(env_file.read_text())
                    except OSError:
                        print("⚠️ Could not remove old backup, will overwrite it")
            
            # Create backup of current .env
            try:
                env_file.rename(backup_file)
                print(f"📁 Backed up existing .env to .env.backup")
            except OSError as e:
                print(f"⚠️ Could not rename .env to backup: {e}")
                print("   Will overwrite existing .env file")
        
        # Write new configuration
        with open(env_file, 'w') as f:
            f.write(config_content)
        
        print(f"💾 Configuration saved to .env")
        print(f"   LLM Model: {llm_model}")
        print(f"   Embedding Model: {embedding_model}")
    
    def run_setup(self):
        """Run the complete model setup process."""
        print("🚀 Model Setup for Rework Document Update System")
        print("="*60)
        
        # Get model type choice first
        model_type = self.get_model_type_choice()
        
        if model_type == "local":
            return self._run_local_setup()
        else:
            return self._run_remote_setup()
    
    def _run_local_setup(self):
        """Run setup for local models."""
        print("\n🏠 LOCAL MODEL SETUP")
        print("="*40)
        
        # Check system info
        system_info = self.get_system_info()
        self.display_system_info(system_info)
        
        # Check if Ollama is running
        ollama_running = self.check_ollama_running()
        
        # Display LLM menu and get selection
        self.display_llm_menu(system_info)
        llm_choice = self.get_user_choice(
            "Select LLM model (1-7)", 
            list(self.llm_models.keys())
        )
        llm_model = self.llm_models[llm_choice]["name"]
        
        # Display embedding menu and get selection
        self.display_embedding_menu(system_info)
        embedding_choice = self.get_user_choice(
            "Select embedding model (1-4)",
            list(self.embedding_models.keys())
        )
        embedding_model = self.embedding_models[embedding_choice]["name"]
        
        # Confirm selection
        print(f"\n📋 SELECTION SUMMARY")
        print("="*40)
        print(f"Model Type: Local (Ollama)")
        print(f"LLM Model: {llm_model}")
        print(f"Embedding Model: {embedding_model}")
        print("="*40)
        
        confirm = input("\nProceed with this configuration? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ Setup cancelled")
            return False
        
        # Download models if Ollama is running and models are not "none"
        if ollama_running:
            success = True
            
            if llm_model != "none":
                if not self.download_model(llm_model):
                    success = False
                else:
                    if not self.test_model(llm_model, "llm"):
                        print(f"⚠️ LLM model {llm_model} downloaded but test failed")
            
            if embedding_model != "none":
                if not self.download_model(embedding_model):
                    success = False
                else:
                    if not self.test_model(embedding_model, "embedding"):
                        print(f"⚠️ Embedding model {embedding_model} downloaded but test failed")
            
            if not success:
                print("❌ Some models failed to download or test")
                return False
        else:
            print("⚠️ Ollama not running - skipping model download")
            print("   You can download models later when Ollama is running")
        
        # Generate and save configuration
        config_content = self.generate_env_config(llm_model, embedding_model, "local")
        self.save_config(config_content, llm_model, embedding_model)
        
        # Final instructions
        print(f"\n🎉 Local model setup completed successfully!")
        print("\n📋 NEXT STEPS:")
        print("1. Fill in any API keys in .env if using remote providers")
        print("2. Set GitLab/GitHub tokens in .env if using VCS integration")
        print("3. Set up ngrok tunnel: ngrok http 8000")
        print("4. Update NGROK_BASE_URL in .env with your ngrok URL")
        print("5. Run: poetry install && poetry run python scripts/generate_protobuf.py")
        print("6. Start server: poetry run rework-server")
        print("7. Configure repositories using the /configure endpoint")
        print("8. Test with a commit to your GitHub/GitLab repository")
        
        if llm_model != "none" or embedding_model != "none":
            print("\n💡 LOCAL MODELS:")
            print("   Make sure Ollama is running: ollama serve")
            print("   Check available models: ollama list")
        
        print("\n🌐 NGROK SETUP:")
        print("   Install ngrok: https://ngrok.com/download")
        print("   Start tunnel: ngrok http 8000")
        print("   Copy the https URL and update NGROK_BASE_URL in .env")
        print("   This enables webhooks from GitHub/GitLab to reach your server")
        
        print("\n📄 COMPLETE .env FILE GENERATED:")
        print("   All necessary configuration sections included")
        print("   Database, server, logging, VCS, and ngrok settings configured")
        print("   Ready for immediate use!")
        
        return True
    
    def _run_remote_setup(self):
        """Run setup for remote models."""
        print("\n☁️ REMOTE MODEL SETUP")
        print("="*40)
        
        # Display remote LLM menu and get selection
        self.display_remote_llm_menu()
        llm_choice = self.get_user_choice(
            "Select LLM model (1-8)",
            list(self.remote_llm_models.keys())
        )
        llm_model_info = self.remote_llm_models[llm_choice]
        
        # Display remote embedding menu and get selection
        self.display_remote_embedding_menu()
        embedding_choice = self.get_user_choice(
            "Select embedding model (1-6)",
            list(self.remote_embedding_models.keys())
        )
        embedding_model_info = self.remote_embedding_models[embedding_choice]
        
        # Confirm selection
        print(f"\n📋 SELECTION SUMMARY")
        print("="*40)
        print(f"Model Type: Remote (Cloud APIs)")
        print(f"LLM Provider: {llm_model_info['provider'].title()}")
        print(f"LLM Model: {llm_model_info['name']}")
        print(f"Embedding Provider: {embedding_model_info['provider'].title()}")
        print(f"Embedding Model: {embedding_model_info['name']}")
        print("="*40)
        
        confirm = input("\nProceed with this configuration? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ Setup cancelled")
            return False
        
        # Generate and save configuration
        config_content = self.generate_env_config(llm_choice, embedding_choice, "remote")
        self.save_config(config_content, llm_model_info['name'], embedding_model_info['name'])
        
        # Final instructions
        print(f"\n🎉 Remote model setup completed successfully!")
        print("\n📋 NEXT STEPS:")
        print("1. Add your API keys to the .env file:")
        print(f"   • {llm_model_info['provider'].upper()}_API_KEY=your-api-key-here")
        print(f"   • {embedding_model_info['provider'].upper()}_API_KEY=your-api-key-here")
        print("2. Set GitLab/GitHub tokens in .env if using VCS integration")
        print("3. Set up ngrok tunnel: ngrok http 8000")
        print("4. Update NGROK_BASE_URL in .env with your ngrok URL")
        print("5. Run: poetry install && poetry run python scripts/generate_protobuf.py")
        print("6. Start server: poetry run rework-server")
        print("7. Configure repositories using the /configure endpoint")
        print("8. Test with a commit to your GitHub/GitLab repository")
        
        print("\n💡 API KEY SETUP:")
        print(f"   • OpenAI: https://platform.openai.com/api-keys")
        print(f"   • Anthropic: https://console.anthropic.com/")
        print(f"   • Mistral: https://console.mistral.ai/")
        print(f"   • Cohere: https://dashboard.cohere.ai/")
        print(f"   • Gemini: https://aistudio.google.com/app/apikey")
        
        print("\n🌐 NGROK SETUP:")
        print("   Install ngrok: https://ngrok.com/download")
        print("   Start tunnel: ngrok http 8000")
        print("   Copy the https URL and update NGROK_BASE_URL in .env")
        print("   This enables webhooks from GitHub/GitLab to reach your server")
        
        print("\n📄 COMPLETE .env FILE GENERATED:")
        print("   All necessary configuration sections included")
        print("   Database, server, logging, VCS, and ngrok settings configured")
        print("   API key placeholders ready for your keys")
        print("   Ready for immediate use after adding API keys!")
        
        return True

def main():
    """Main setup function."""
    setup = ModelSetup()
    
    try:
        success = setup.run_setup()
        if success:
            print("\n✅ Model setup completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Model setup failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

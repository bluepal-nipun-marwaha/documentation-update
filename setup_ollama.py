#!/usr/bin/env python3
"""
Ollama Setup Script for Local LLM Integration
This script helps set up Ollama with the recommended models for the documentation system.
"""

import requests
import json
import time
import sys

class OllamaSetup:
    """Setup and configure Ollama for local LLM processing."""
    
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
    
    def check_ollama_running(self):
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
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
    
    def get_available_models(self):
        """Get list of available models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except Exception as e:
            print(f"❌ Error getting models: {str(e)}")
            return []
    
    def pull_model(self, model_name):
        """Pull a model from Ollama."""
        print(f"🔄 Pulling model: {model_name}")
        print("   This may take several minutes depending on model size...")
        
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=600  # 10 minutes timeout
            )
            
            if response.status_code == 200:
                print(f"✅ Successfully pulled model: {model_name}")
                return True
            else:
                print(f"❌ Failed to pull model {model_name}: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error pulling model {model_name}: {str(e)}")
            return False
    
    def test_model(self, model_name):
        """Test a model with a simple prompt."""
        print(f"🧪 Testing model: {model_name}")
        
        try:
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": "Hello! Please respond with 'Model is working correctly.'"}
                ],
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result['message']['content']
                print(f"✅ Model test successful: {response_text[:100]}...")
                return True
            else:
                print(f"❌ Model test failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error testing model: {str(e)}")
            return False
    
    def setup_recommended_models(self):
        """Setup recommended models for the documentation system."""
        
        # Recommended models for different use cases
        recommended_models = {
            "qwen2.5:7b": "Main LLM for code analysis and documentation (7GB RAM) - RECOMMENDED",
            "llama3.1:8b": "Alternative high-performance model (8GB RAM)", 
            "codellama:7b": "Code-specialized model (7GB RAM)",
            "mistral:7b": "Fast and efficient model (7GB RAM)",
            "nomic-embed-text": "Embeddings model for RAG (1GB RAM)"
        }
        
        print("🎯 Recommended models for documentation system:")
        print("=" * 60)
        
        for model, description in recommended_models.items():
            print(f"• {model}")
            print(f"  {description}")
        print("=" * 60)
        
        available_models = self.get_available_models()
        
        # Check which models are already available
        missing_models = []
        for model in recommended_models.keys():
            if model not in available_models:
                missing_models.append(model)
            else:
                print(f"✅ {model} is already available")
        
        if missing_models:
            print(f"\n📥 Missing models: {', '.join(missing_models)}")
            
            # Ask user which models to install
            print("\nWhich models would you like to install?")
            print("1. All recommended models (recommended)")
            print("2. Just the main LLM (qwen2.5:7b)")
            print("3. Just the embeddings model (nomic-embed-text)")
            print("4. Custom selection")
            print("5. Skip model installation")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            models_to_install = []
            
            if choice == "1":
                models_to_install = missing_models
            elif choice == "2":
                models_to_install = ["qwen2.5:7b"] if "qwen2.5:7b" in missing_models else []
            elif choice == "3":
                models_to_install = ["nomic-embed-text"] if "nomic-embed-text" in missing_models else []
            elif choice == "4":
                print("\nAvailable models to install:")
                for i, model in enumerate(missing_models, 1):
                    print(f"{i}. {model} - {recommended_models[model]}")
                
                selections = input("Enter model numbers (comma-separated): ").strip()
                try:
                    indices = [int(x.strip()) - 1 for x in selections.split(',')]
                    models_to_install = [missing_models[i] for i in indices if 0 <= i < len(missing_models)]
                except (ValueError, IndexError):
                    print("❌ Invalid selection")
                    return False
            elif choice == "5":
                print("⏭️ Skipping model installation")
                return True
            else:
                print("❌ Invalid choice")
                return False
            
            # Install selected models
            if models_to_install:
                print(f"\n🚀 Installing {len(models_to_install)} models...")
                for model in models_to_install:
                    if not self.pull_model(model):
                        print(f"❌ Failed to install {model}")
                        return False
                    
                    # Test the model
                    if not self.test_model(model):
                        print(f"⚠️ Model {model} installed but test failed")
            
            print("\n✅ Model installation completed!")
        
        return True
    
    def run_setup(self):
        """Run the complete Ollama setup."""
        print("🚀 Ollama Setup for Documentation System")
        print("=" * 50)
        
        # Check if Ollama is running
        if not self.check_ollama_running():
            return False
        
        # Setup recommended models
        if not self.setup_recommended_models():
            return False
        
        # Final verification
        print("\n🔍 Final verification...")
        available_models = self.get_available_models()
        
        essential_models = ["qwen2.5:7b", "nomic-embed-text"]
        missing_essential = [model for model in essential_models if model not in available_models]
        
        if missing_essential:
            print(f"⚠️ Missing essential models: {', '.join(missing_essential)}")
            print("   The system may not work properly without these models.")
        else:
            print("✅ All essential models are available!")
        
        print("\n🎉 Ollama setup completed!")
        print("\nNext steps:")
        print("1. Start your documentation system")
        print("2. Configure repositories using the /configure endpoint")
        print("3. Test with a commit to your GitHub repository")
        
        return True

def main():
    """Main setup function."""
    setup = OllamaSetup()
    
    try:
        success = setup.run_setup()
        if success:
            print("\n✅ Setup completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Setup failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

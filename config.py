# config.py - System configuration file

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class OllamaConfig:
    """Ollama configuration class"""
    host: str = "gpu12" #Based on the actual situation
    port: int = 48441 #Based on the actual situation
    base_url: str = None
    timeout: int = 300  # 5-minute timeout
    
    def __post_init__(self):
        if self.base_url is None:
            self.base_url = f"http://{self.host}:{self.port}"

@dataclass
class ModelConfig:
    """Model configuration class"""
    # Small model test environment
    small_model: str = "qwen3:30b-a3b-instruct-2507-fp16"  # Small model used by Agent
    main_model: str = "qwen3:30b-a3b-instruct-2507-fp16"  # Main writing model
    test_model: str = "qwen3:30b-a3b-instruct-2507-q8_0"   # Test model

    # Special configurations for different tasks (only override Ollama defaults when needed)
    task_configs: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.task_configs is None:
            # Only adjust key parameters, others use Ollama defaults
            self.task_configs = {
                "literature_generation": {
                    "temperature": 0.3  # Academic content needs higher accuracy
                },
                "agent_review": {
                    "temperature": 0.5  # Review needs balance between accuracy and creativity
                },
                "chemical_analysis": {
                    "temperature": 0.2  # Chemical analysis needs highest accuracy
                },
                "diagram_generation": {
                    "temperature": 0.1,  # Concept diagrams need structured thinking
                    "max_tokens": 4096   # Chart code might be longer
                }
            }

@dataclass
class SystemConfig:
    """System configuration class"""
    # Environment configuration
    environment: str = "test"  # test | production
    debug: bool = True
    
    # File paths
    input_dir: str = "./files_mmd"
    output_dir: str = "./outputs"
    embeddings_dir: str = "./embeddings"
    logs_dir: str = "./logs"
    
    # Processing parameters
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_concurrent_requests: int = 5
    
    # Agent configuration
    max_agent_rounds: int = 10
    agent_timeout: int = 180
    
    # Chemical formula processing
    preserve_formulas: bool = True
    formula_formats: list = None
    
    def __post_init__(self):
        if self.formula_formats is None:
            self.formula_formats = ["latex", "mathml", "unicode"]
        
        # Adjust configuration based on environment
        if self.environment == "production":
            self.debug = False
            self.max_concurrent_requests = 10

# Global configuration instances
OLLAMA_CONFIG = OllamaConfig()
MODEL_CONFIG = ModelConfig()
SYSTEM_CONFIG = SystemConfig()

# Environment variable override
def load_config_from_env():
    """Load configuration from environment variables"""
    
    # Ollama configuration
    if os.getenv("OLLAMA_HOST"):
        OLLAMA_CONFIG.host = os.getenv("OLLAMA_HOST")
    if os.getenv("OLLAMA_PORT"):
        OLLAMA_CONFIG.port = int(os.getenv("OLLAMA_PORT"))
    
    # Model configuration
    if os.getenv("MAIN_MODEL"):
        MODEL_CONFIG.main_model = os.getenv("MAIN_MODEL")
    if os.getenv("SMALL_MODEL"):
        MODEL_CONFIG.small_model = os.getenv("SMALL_MODEL")
    
    # System configuration
    if os.getenv("ENVIRONMENT"):
        SYSTEM_CONFIG.environment = os.getenv("ENVIRONMENT")
    if os.getenv("DEBUG"):
        SYSTEM_CONFIG.debug = os.getenv("DEBUG").lower() == "true"
    
    # Recalculate dependent configurations
    OLLAMA_CONFIG.__post_init__()

# Configuration validation
def validate_config():
    """Validate configuration validity"""
    import requests
    try:
        response = requests.get(f"{OLLAMA_CONFIG.base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            print(f"✅ Ollama connection successful: {OLLAMA_CONFIG.base_url}")
            return True
        else:
            print(f"❌ Ollama connection failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ollama connection error: {e}")
        return False

# Initialize configuration
load_config_from_env()

if __name__ == "__main__":
    print("🔧 System Configuration Information:")
    print(f"Ollama Server: {OLLAMA_CONFIG.base_url}")
    print(f"Main Model: {MODEL_CONFIG.main_model}")
    print(f"Small Model: {MODEL_CONFIG.small_model}")
    print(f"Environment: {SYSTEM_CONFIG.environment}")
    print(f"Debug Mode: {SYSTEM_CONFIG.debug}")
    
    # Validate connection
    validate_config()
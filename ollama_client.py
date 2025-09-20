# ollama_client.py - Ollama Client Wrapper

import requests
import json
import time
import logging
from typing import Dict, Any, List, Optional, Generator
from config import OLLAMA_CONFIG, MODEL_CONFIG, SYSTEM_CONFIG

logger = logging.getLogger(__name__)

def disable_thinking():
    """Disable thinking mode globally for Ollama."""
    url = f"{OLLAMA_CONFIG.host}/api/set"
    payload = {"nothink": True}
    try:
        r = requests.post(url, json=payload, timeout=5)
        if r.status_code == 200:
            print("[INFO] Thinking mode disabled.")
        else:
            print(f"[WARN] Failed to disable thinking mode: {r.status_code}, {r.text}")
    except Exception as e:
        print(f"[ERROR] Exception while disabling thinking mode: {e}")

class OllamaClient:
    """Ollama API Client"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.base_url = OLLAMA_CONFIG.base_url
        self.timeout = OLLAMA_CONFIG.timeout
        self.session = requests.Session()
        
        # Apply custom configuration
        if config:
            for key, value in config.items():
                if hasattr(OLLAMA_CONFIG, key):
                    setattr(OLLAMA_CONFIG, key, value)
    
    def generate(self, 
                model: str,
                prompt: str,
                system: str = None,
                temperature: float = None,
                max_tokens: int = None,
                top_p: float = None,
                stream: bool = False) -> Dict[str, Any]:
        """
        Generate text
        
        Args:
            model: Model name
            prompt: User input
            system: System prompt
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens
            top_p: top_p parameter
            stream: Whether to stream output
            
        Returns:
            Generated response
        """
        
        # Build request data
        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {}
        }
        
        # Add system message
        if system:
            data["system"] = system
        
        # Add parameters (only set when explicitly specified, otherwise use Ollama defaults)
        if temperature is not None:
            data["options"]["temperature"] = temperature
            
        if max_tokens is not None:
            data["options"]["num_predict"] = max_tokens
            
        if top_p is not None:
            data["options"]["top_p"] = top_p
        
        try:
            if stream:
                return self._stream_generate(data)
            else:
                return self._single_generate(data)
                
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise
    
    def _single_generate(self, data: Dict) -> Dict[str, Any]:
        """Single generation"""
        url = f"{self.base_url}/api/generate"
        
        response = self.session.post(
            url, 
            json=data, 
            timeout=self.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        return {
            "response": result.get("response", ""),
            "done": result.get("done", False),
            "total_duration": result.get("total_duration", 0),
            "load_duration": result.get("load_duration", 0),
            "prompt_eval_count": result.get("prompt_eval_count", 0),
            "eval_count": result.get("eval_count", 0),
        }
    
    def _stream_generate(self, data: Dict) -> Generator[Dict[str, Any], None, None]:
        """Stream generation"""
        url = f"{self.base_url}/api/generate"
        
        with self.session.post(
            url, 
            json=data, 
            timeout=self.timeout,
            stream=True
        ) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        yield chunk
                    except json.JSONDecodeError:
                        continue
    
    def chat(self,
             model: str,
             messages: List[Dict[str, str]],
             temperature: float = None,
             max_tokens: int = None,
             stream: bool = False) -> Dict[str, Any]:
        """
        Chat interface
        
        Args:
            model: Model name
            messages: Chat history [{"role": "user/assistant", "content": "..."}]
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens
            stream: Whether to stream output
            
        Returns:
            Chat response
        """
        
        data = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {}
        }
        
        # Add parameters (only set when explicitly specified, otherwise use Ollama defaults)
        if temperature is not None:
            data["options"]["temperature"] = temperature
            
        if max_tokens is not None:
            data["options"]["num_predict"] = max_tokens
        
        try:
            url = f"{self.base_url}/api/chat"
            
            if stream:
                return self._stream_chat(url, data)
            else:
                response = self.session.post(url, json=data, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()
                
                return {
                    "message": result.get("message", {}),
                    "done": result.get("done", False),
                    "total_duration": result.get("total_duration", 0),
                    "load_duration": result.get("load_duration", 0),
                    "prompt_eval_count": result.get("prompt_eval_count", 0),
                    "eval_count": result.get("eval_count", 0),
                }
                
        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
            raise
    
    def _stream_chat(self, url: str, data: Dict) -> Generator[Dict[str, Any], None, None]:
        """Stream chat"""
        with self.session.post(
            url, 
            json=data, 
            timeout=self.timeout,
            stream=True
        ) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        yield chunk
                    except json.JSONDecodeError:
                        continue
    
    def list_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        try:
            url = f"{self.base_url}/api/tags"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            return result.get("models", [])
            
        except Exception as e:
            logger.error(f"Failed to get model list: {e}")
            return []
    
    def pull_model(self, model: str) -> bool:
        """Pull model"""
        try:
            url = f"{self.base_url}/api/pull"
            data = {"name": model}
            
            response = self.session.post(url, json=data, timeout=600)  # 10 minutes timeout
            response.raise_for_status()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False
    
    def health_check(self) -> bool:
        """Health check"""
        try:
            models = self.list_models()
            return len(models) >= 0
        except:
            return False

# Global client instance
ollama_client = OllamaClient()

# Utility functions
def generate_text(prompt: str, 
                 model: str = None, 
                 system: str = None,
                 task_type: str = None,
                 **kwargs) -> str:
    """
    Convenient text generation function
    
    Args:
        prompt: Input prompt
        model: Model name, defaults to main model in config
        system: System prompt
        task_type: Task type, used to get special configuration
        **kwargs: Other parameters
        
    Returns:
        Generated text
    """
    
    if model is None:
        model = MODEL_CONFIG.main_model
    
    # Apply task-specific configuration
    if task_type and task_type in MODEL_CONFIG.task_configs:
        task_config = MODEL_CONFIG.task_configs[task_type]
        for key, value in task_config.items():
            if key not in kwargs:
                kwargs[key] = value
    
    result = ollama_client.generate(
        model=model,
        prompt=prompt,
        system=system,
        **kwargs
    )
    
    return result["response"]

def chat_with_model(messages: List[Dict[str, str]], 
                   model: str = None,
                   **kwargs) -> str:
    """
    Convenient chat function
    
    Args:
        messages: Chat history
        model: Model name
        **kwargs: Other parameters
        
    Returns:
        Model response
    """
    
    if model is None:
        model = MODEL_CONFIG.small_model  # Default to small model for chat
    
    result = ollama_client.chat(
        model=model,
        messages=messages,
        **kwargs
    )
    
    return result["message"]["content"]

if __name__ == "__main__":
    # Test connection
    client = OllamaClient()
    
    print("🔍 Checking Ollama connection...")
    if client.health_check():
        print("Ollama connection normal")
        
        # List available models
        models = client.list_models()
        print(f"Available models: {[m['name'] for m in models]}")
        
        # Test generation
        if models:
            test_model = models[0]['name']
            print(f"Test model: {test_model}")
            
            response = generate_text(
                prompt="Hello, how are you?",
                model=test_model,
                max_tokens=50
            )
            print(f"Test response: {response}")
        
    else:
        print("Ollama connection failed")
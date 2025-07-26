"""
Ollama Model Wrapper for smolagents compatibility
Provides OpenAI-compatible interface to Ollama API
"""

import requests
import json
import time
from typing import List, Dict, Any, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaModel:
    """
    Ollama model wrapper that provides compatibility with smolagents
    Mimics the InferenceClientModel interface while using Ollama backend
    """
    
    def __init__(
        self,
        model_name: str = "qwen2.5:7b",
        base_url: str = "http://localhost:6399",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        timeout: int = 120,
        **kwargs
    ):
        """
        Initialize Ollama model wrapper
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API (default: http://localhost:6399)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        
        # Token counting for compatibility
        self.last_input_token_count = 0
        self.last_output_token_count = 0
        
        # Verify Ollama is running and model is available
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify Ollama is running and model is available"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"Ollama not responding at {self.base_url}")
            
            # Check if model is available
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]
            
            if not any(self.model_name in name for name in model_names):
                logger.warning(f"Model {self.model_name} not found. Available models: {model_names}")
                logger.info(f"Attempting to pull model {self.model_name}...")
                self._pull_model()
            
            logger.info(f"✅ Ollama connection verified. Model: {self.model_name}")
            
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama: {e}. Make sure Ollama is running with 'ollama serve'")
    
    def _pull_model(self):
        """Pull the model if it's not available"""
        try:
            pull_response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name},
                timeout=300  # 5 minutes for model download
            )
            if pull_response.status_code == 200:
                logger.info(f"✅ Model {self.model_name} pulled successfully")
            else:
                logger.error(f"Failed to pull model {self.model_name}")
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to a single prompt string"""
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                formatted += f"System: {content}\n\n"
            elif role == "user":
                formatted += f"User: {content}\n\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n\n"
        
        formatted += "Assistant: "
        return formatted
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 characters ≈ 1 token)"""
        return len(text) // 4
    
    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Main inference method compatible with smolagents
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            stop: List of stop sequences (optional)
            **kwargs: Additional parameters
            
        Returns:
            Generated response string
        """
        try:
            # Format messages for Ollama
            prompt = self._format_messages(messages)
            
            # Update parameters from kwargs
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            top_p = kwargs.get("top_p", self.top_p)
            
            # Prepare Ollama API request
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": top_p,
                    "stop": stop or []
                }
            }
            
            # Count input tokens
            self.last_input_token_count = self._estimate_tokens(prompt)
            
            # Make request to Ollama
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
            
            result = response.json()
            generated_text = result.get("response", "")
            
            # Count output tokens
            self.last_output_token_count = self._estimate_tokens(generated_text)
            
            # Log performance
            duration = time.time() - start_time
            logger.debug(f"Generated {self.last_output_token_count} tokens in {duration:.2f}s")
            
            return generated_text.strip()
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout after {self.timeout} seconds")
            return "Error: Request timeout. The model may be processing a complex request."
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}")
            return f"Error: Network error - {str(e)}"
        
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return f"Error: Generation failed - {str(e)}"
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Alternative interface method for compatibility"""
        return self.__call__(messages, **kwargs)
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        OpenAI-style chat completion interface
        
        Returns:
            OpenAI-compatible response dictionary
        """
        response_text = self.__call__(messages, **kwargs)
        
        return {
            "id": f"chatcmpl-{hash(response_text) % 10000000}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": self.last_input_token_count,
                "completion_tokens": self.last_output_token_count,
                "total_tokens": self.last_input_token_count + self.last_output_token_count
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check Ollama service health"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return {
                    "status": "healthy",
                    "service": "ollama",
                    "models_available": len(models),
                    "current_model": self.model_name,
                    "base_url": self.base_url
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}"
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    # Properties for smolagents compatibility
    @property
    def model_id(self) -> str:
        """Model identifier for compatibility"""
        return self.model_name
    
    @property
    def tokenizer(self):
        """Dummy tokenizer property for compatibility"""
        return None


class OllamaInferenceClient(OllamaModel):
    """
    Alias class for drop-in replacement of InferenceClientModel
    Provides the same interface as smolagents InferenceClientModel
    """
    
    def __init__(
        self,
        model_id: str = "qwen2.5:7b",
        base_url: str = "http://localhost:6399",
        api_key: str = "not-needed",  # Ignored but kept for compatibility
        max_tokens: int = 512,
        temperature: float = 0.7,
        custom_role_conversions: Optional[Dict] = None,  # Ignored but kept for compatibility
        **kwargs
    ):
        """
        Initialize with InferenceClientModel-compatible parameters
        
        Args:
            model_id: Model identifier (Ollama model name)
            base_url: Ollama API base URL
            api_key: API key (ignored for Ollama but kept for compatibility)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            custom_role_conversions: Custom role conversions (ignored)
        """
        # Map InferenceClientModel parameters to OllamaModel
        super().__init__(
            model_name=model_id,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        # Store compatibility parameters
        self.api_key = api_key
        self.custom_role_conversions = custom_role_conversions
        
        logger.info(f"🔄 Using Ollama model '{model_id}' instead of InferenceClientModel")


# Convenience function for easy import
def create_ollama_model(
    model_name: str = "qwen2.5:7b",
    **kwargs
) -> OllamaInferenceClient:
    """Create an Ollama model instance with sensible defaults"""
    return OllamaInferenceClient(model_id=model_name, **kwargs)


if __name__ == "__main__":
    # Test the Ollama model
    print("Testing Ollama model connection...")
    
    try:
        model = OllamaModel()
        
        # Health check
        health = model.health_check()
        print(f"Health check: {health}")
        
        # Test generation
        test_messages = [
            {"role": "user", "content": "Hello! Can you write a simple Python function to add two numbers?"}
        ]
        
        response = model(test_messages)
        print(f"Response: {response}")
        
        print("✅ Ollama model test successful!")
        
    except Exception as e:
        print(f"❌ Ollama model test failed: {e}")
        print("Make sure Ollama is running with: ollama serve")
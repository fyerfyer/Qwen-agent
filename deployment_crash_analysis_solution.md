# Cloud Server Deployment Crash Analysis & Solution (No Swap Required)

## Executive Summary

Your cloud server crashed during model deployment due to **insufficient memory management without swap space**. Since your cloud provider doesn't allow swap space creation, this solution provides **three alternative deployment methods** that work within your existing 30GB RAM constraint: **Ollama** (recommended), **vLLM with streaming**, and **lightweight quantized deployment**.

## Root Cause Analysis

### Primary Issue: Memory Exhaustion Without Swap Buffer
- **Current Configuration**: SwapTotal: 0 kB (cannot be modified)
- **Impact**: Model loading requires temporary memory spikes (2x model size)
- **Result**: System OOM killer crashes the server when 30GB physical RAM is exceeded

### Hardware Assessment
- **✅ GPU**: NVIDIA Tesla T4 (15GB VRAM) - Sufficient
- **✅ CPU**: 8x Intel Xeon Platinum 8255C - Adequate  
- **✅ RAM**: 30GB available - Workable with proper management
- **❌ Swap**: 0B and cannot be created - Critical constraint

## Solution 1: Ollama Deployment (Recommended)

### Why Ollama?
- **Memory Efficient**: Automatic memory management and model streaming
- **No Swap Required**: Built-in memory optimization handles limitations
- **Easy Setup**: Single binary installation with automatic GPU detection
- **OpenAI Compatible**: Drop-in replacement for your existing API calls

### Step 1: Install Ollama
```bash
# Download and install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Verify installation
ollama --version

# Check CUDA support
nvidia-smi
```

### Step 2: Configure Ollama for Tesla T4
```bash
# Set memory limits (leave 4GB for system)
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_MAX_QUEUE=1
export OLLAMA_MEMORY_LIMIT=26GB

# Configure GPU usage
export OLLAMA_GPU_OVERHEAD=2GB
export CUDA_VISIBLE_DEVICES=0
```

### Step 3: Pull and Test Recommended Model
```bash
# Pull Qwen2.5:7b model (optimized for your hardware)
ollama pull qwen2.5:7b

# Alternative smaller model if needed
ollama pull qwen2.5:3b

# Test the model
ollama run qwen2.5:7b "Hello! Can you help me write a Python function?"
```

### Step 4: Start Ollama API Server
```bash
# Start Ollama as API server
ollama serve

# In another terminal, test API compatibility
curl http://localhost:6399/api/generate -d '{
  "model": "qwen2.5:7b",
  "prompt": "Hello world",
  "stream": false
}'
```

### Step 5: Integrate with Your Application
Replace your current model configuration:

```python
# Original problematic configuration
# model = InferenceClientModel(
#     max_tokens=2096,
#     temperature=0.5,
#     model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
#     custom_role_conversions=None,
# )

# New Ollama-based configuration
import requests
import json

class OllamaModel:
    def __init__(self, model_name="qwen2.5:7b", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.max_tokens = 512  # Conservative for memory safety
        self.temperature = 0.7
    
    def generate(self, messages, **kwargs):
        # Convert messages to single prompt
        prompt = self._format_messages(messages)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens)
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return "Error: Model unavailable"
    
    def _format_messages(self, messages):
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

# Use the new model
model = OllamaModel(model_name="qwen2.5:7b")

# Test usage
response = model.generate([
    {"role": "user", "content": "Write a simple Python function"}
])
print(response)
```

### Step 6: Production Deployment Script
Create `ollama_production.py`:

```python
#!/usr/bin/env python3
import subprocess
import time
import requests
import os
import signal
import sys

class OllamaManager:
    def __init__(self):
        self.ollama_process = None
        self.model_name = "qwen2.5:7b"
        
    def setup_environment(self):
        """Configure environment for optimal memory usage"""
        os.environ["OLLAMA_MAX_LOADED_MODELS"] = "1"
        os.environ["OLLAMA_MAX_QUEUE"] = "1" 
        os.environ["OLLAMA_MEMORY_LIMIT"] = "26GB"
        os.environ["OLLAMA_GPU_OVERHEAD"] = "2GB"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
    def start_ollama(self):
        """Start Ollama server"""
        self.setup_environment()
        
        print("Starting Ollama server...")
        self.ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to be ready
        for _ in range(30):  # 30 second timeout
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    print("✅ Ollama server is ready!")
                    return True
            except:
                time.sleep(1)
        
        print("❌ Failed to start Ollama server")
        return False
    
    def load_model(self):
        """Ensure model is loaded"""
        print(f"Loading model: {self.model_name}")
        try:
            # Attempt to pull model if not present
            subprocess.run(["ollama", "pull", self.model_name], check=True)
            
            # Test model with small prompt
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": "Hello",
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                print("✅ Model loaded successfully!")
                return True
            else:
                print(f"❌ Model test failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def monitor_resources(self):
        """Monitor system resources"""
        try:
            # Check memory usage
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
                
            mem_total = 0
            mem_available = 0
            for line in lines:
                if line.startswith('MemTotal:'):
                    mem_total = int(line.split()[1])
                elif line.startswith('MemAvailable:'):
                    mem_available = int(line.split()[1])
            
            usage_percent = ((mem_total - mem_available) / mem_total) * 100
            print(f"Memory usage: {usage_percent:.1f}%")
            
            if usage_percent > 85:
                print("⚠️  WARNING: High memory usage detected")
                
        except Exception as e:
            print(f"Resource monitoring error: {e}")
    
    def shutdown(self, signum=None, frame=None):
        """Graceful shutdown"""
        print("\nShutting down Ollama...")
        if self.ollama_process:
            self.ollama_process.terminate()
            self.ollama_process.wait()
        sys.exit(0)
    
    def run(self):
        """Main execution loop"""
        # Register signal handlers
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        
        if not self.start_ollama():
            return False
            
        if not self.load_model():
            return False
        
        print("🚀 Ollama deployment successful!")
        print("API available at: http://localhost:11434")
        print("Press Ctrl+C to shutdown")
        
        # Keep running and monitor
        try:
            while True:
                self.monitor_resources()
                time.sleep(30)  # Check every 30 seconds
        except KeyboardInterrupt:
            self.shutdown()

if __name__ == "__main__":
    manager = OllamaManager()
    manager.run()
```

## Solution 2: vLLM with Memory Streaming

### Alternative High-Performance Option
If you need more control than Ollama provides:

```bash
# Install vLLM
pip install vllm

# Create memory-optimized deployment
cat > vllm_deploy.py << 'EOF'
from vllm import LLM, SamplingParams
import torch

# Memory-conservative configuration
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct-AWQ",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,  # Use 85% of GPU memory
    max_model_len=1024,           # Reduced context length
    enforce_eager=True,           # Disable CUDA graphs to save memory
    disable_custom_all_reduce=True,
    swap_space=0,                 # Explicitly disable swap usage
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256
)

def generate_response(prompt):
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text

# Test
if __name__ == "__main__":
    response = generate_response("Hello! Write a Python function.")
    print(response)
EOF

python vllm_deploy.py
```

## Solution 3: Lightweight Quantized Deployment

### Minimal Resource Usage
For maximum memory conservation:

```python
# lightweight_deploy.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
import os

# Force conservative memory usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

class MemoryEfficientModel:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load model with aggressive quantization"""
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/DialoGPT-medium",  # Smaller 1.5B model
            use_fast=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Loading model with 8-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/DialoGPT-medium",
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        print("✅ Model loaded successfully!")
    
    def generate(self, prompt, max_length=150):
        """Generate response with memory cleanup"""
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            
            # Generate with conservative settings
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            
            # Clean up GPU memory
            del inputs, outputs
            torch.cuda.empty_cache()
            gc.collect()
            
            return response
            
        except Exception as e:
            print(f"Generation error: {e}")
            return "Error: Generation failed"

# Usage
model = MemoryEfficientModel()
response = model.generate("Hello! Can you help me with Python?")
print(response)
```

## Deployment Comparison

| Method | Memory Usage | Setup Complexity | Performance | Stability |
|--------|-------------|------------------|-------------|-----------|
| **Ollama** | Low (Auto-managed) | Very Easy | High | Excellent |
| **vLLM** | Medium | Medium | Very High | Good |
| **Quantized** | Very Low | Easy | Medium | Good |

## Recommended Implementation Plan

### Phase 1: Quick Solution (Ollama)
```bash
# 15-minute setup
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5:7b
ollama serve
```

### Phase 2: Integration Testing
```bash
# Test API compatibility
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:7b", 
  "prompt": "Test prompt",
  "stream": false
}'

# Monitor resources
watch -n 5 'free -h && nvidia-smi'
```

### Phase 3: Production Deployment
```bash
# Run production script
python ollama_production.py

# Set up as system service (optional)
sudo systemctl enable ollama
sudo systemctl start ollama
```

## Expected Results

### ✅ **Memory Usage (Ollama)**
- **System RAM**: 8-12GB (well under 30GB limit)
- **GPU VRAM**: 6-8GB (comfortable fit in 15GB)
- **No swap required**: Built-in memory streaming

### 📊 **Performance Metrics**
- **Model loading**: 30-60 seconds
- **First response**: 2-5 seconds  
- **Subsequent responses**: 0.5-2 seconds
- **Concurrent requests**: 2-3 simultaneous

### 🔧 **Troubleshooting Commands**
```bash
# Check Ollama status
ollama list

# Monitor resource usage
htop

# View Ollama logs
journalctl -u ollama -f

# Restart if needed
sudo systemctl restart ollama
```

This solution eliminates the swap space requirement while providing a stable, production-ready deployment that works within your cloud server constraints.
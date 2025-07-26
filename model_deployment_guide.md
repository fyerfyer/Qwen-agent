# Model Deployment Guide: Self-Hosting LLM on NVIDIA Tesla T4

## 1. Model Recommendation

### Recommended Model
**Hugging Face ID:** `microsoft/DialoGPT-medium` or `Qwen/Qwen2.5-7B-Instruct-AWQ`

### Justification
Given your NVIDIA Tesla T4 GPU with 15360MiB VRAM, the `Qwen/Qwen2.5-7B-Instruct-AWQ` model is the optimal choice for the following reasons:

1. **Memory Efficiency**: AWQ (Activation-aware Weight Quantization) reduces memory footprint by ~75% while maintaining performance
2. **Agentic Task Performance**: Qwen2.5 series excels at code generation, reasoning, and tool usage - essential for agent applications
3. **VRAM Compatibility**: 7B AWQ model requires approximately 4-6GB VRAM, leaving comfortable headroom on your 15GB T4
4. **Instruction Following**: Specifically fine-tuned for instruction-following tasks, making it ideal for agent workflows
5. **Active Development**: Regular updates and strong community support from Alibaba Cloud

**Alternative Option**: If AWQ model has compatibility issues, `microsoft/DialoGPT-medium` (1.5B parameters) provides excellent conversational capabilities with minimal VRAM usage (~2GB).

## 2. Environment Setup

### Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n llm-agent python=3.10
conda activate llm-agent

# Or using venv
python3.10 -m venv llm-agent
source llm-agent/bin/activate
```

### Install Required Packages
```bash
# Core ML libraries
pip install torch==2.5.1+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Hugging Face ecosystem
pip install transformers==4.36.0
pip install accelerate==0.25.0
pip install auto-gptq==0.7.1
pip install autoawq==0.1.8

# API server
pip install fastapi==0.104.1
pip install uvicorn[standard]==0.24.0
pip install pydantic==2.5.0

# Additional utilities
pip install requests==2.31.0
pip install numpy==1.24.3
pip install psutil==5.9.6
```

### Verify CUDA Installation
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

## 3. Deployment Script

Create the following file as `deploy_model.py`:

```python
import os
import gc
import torch
import psutil
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import uvicorn
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"
MAX_LENGTH = 2048
TEMPERATURE = 0.7
TOP_P = 0.9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global model and tokenizer
model = None
tokenizer = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = TEMPERATURE
    top_p: Optional[float] = TOP_P
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

def load_model():
    """Load the AWQ quantized model and tokenizer"""
    global model, tokenizer
    
    try:
        logger.info(f"Loading model: {MODEL_NAME}")
        logger.info(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load AWQ quantized model
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=True
        )
        
        # Memory optimization
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        logger.info(f"Model loaded successfully on {DEVICE}")
        
        # Print memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"VRAM used: {memory_used:.2f}GB, reserved: {memory_reserved:.2f}GB")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

def generate_response(messages: List[ChatMessage], max_tokens: int, temperature: float, top_p: float) -> str:
    """Generate response using the loaded model"""
    try:
        # Format messages for Qwen chat template
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({"role": msg.role, "content": msg.content})
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH - max_tokens,
            padding=True
        ).to(DEVICE)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Decode response
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return response
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting model server...")
    success = load_model()
    if not success:
        raise RuntimeError("Failed to load model")
    yield
    # Shutdown
    logger.info("Shutting down model server...")
    global model, tokenizer
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Create FastAPI app
app = FastAPI(
    title="Local LLM API Server",
    description="OpenAI-compatible API for local LLM inference",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {"message": "Local LLM API Server is running"}

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": 1677610602,
                "owned_by": "local"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
    try:
        if not model or not tokenizer:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Generate response
        response_content = generate_response(
            request.messages,
            request.max_tokens or 512,
            request.temperature or TEMPERATURE,
            request.top_p or TOP_P
        )
        
        # Count tokens (approximate)
        prompt_tokens = len(tokenizer.encode(" ".join([msg.content for msg in request.messages])))
        completion_tokens = len(tokenizer.encode(response_content))
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{hash(response_content) % 10000000}",
            created=1677610602,
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content
                    },
                    "finish_reason": "stop"
                }
            ],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        )
        
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    memory_info = psutil.virtual_memory()
    gpu_info = {}
    
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_available": True,
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_used_gb": torch.cuda.memory_allocated(0) / 1024**3,
            "vram_reserved_gb": torch.cuda.memory_reserved(0) / 1024**3,
            "vram_total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
        }
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "ram_usage_percent": memory_info.percent,
        **gpu_info
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
```

## 4. Running the Service

### Start the FastAPI Server
```bash
# Make sure your virtual environment is activated
conda activate llm-agent  # or source llm-agent/bin/activate

# Start the server
python deploy_model.py

# Alternative using uvicorn directly
uvicorn deploy_model:app --host 0.0.0.0 --port 8000 --reload
```

### Monitor Server Logs
The server will display startup logs including:
- Model loading progress
- VRAM usage information
- Server startup confirmation
- Health check endpoint availability

## 5. Testing the Endpoint

### Test Server Health
```bash
curl -X GET http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "Qwen/Qwen2.5-7B-Instruct-AWQ",
  "device": "cuda",
  "ram_usage_percent": 45.2,
  "gpu_available": true,
  "gpu_name": "Tesla T4",
  "vram_used_gb": 4.8,
  "vram_reserved_gb": 5.2,
  "vram_total_gb": 15.0
}
```

### Test Chat Completion
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "messages": [
      {"role": "user", "content": "Hello! Can you help me write a Python function to calculate fibonacci numbers?"}
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

### Test Model List
```bash
curl -X GET http://localhost:8000/v1/models
```

## 6. Integration with `app.py`

### Modify the InferenceClientModel Configuration

Replace the original model configuration in your `app.py`:

**Original:**
```python
model = InferenceClientModel(
    max_tokens=2096,
    temperature=0.5,
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
    custom_role_conversions=None,
)
```

**Updated:**
```python
from smolagents import InferenceClientModel

model = InferenceClientModel(
    max_tokens=512,  # Reduced for local model
    temperature=0.7,
    base_url="http://localhost:8000/v1",  # Point to local server
    api_key="not-needed",  # Required but not used
    model_id="Qwen/Qwen2.5-7B-Instruct-AWQ",
    custom_role_conversions=None,
)
```

### Additional Configuration Options

For production deployment, consider these modifications:

```python
# For external access (if needed)
model = InferenceClientModel(
    max_tokens=512,
    temperature=0.7,
    base_url="http://YOUR_SERVER_IP:8000/v1",
    api_key="not-needed",
    model_id="Qwen/Qwen2.5-7B-Instruct-AWQ",
    custom_role_conversions=None,
)

# For development with error handling
import os
from smolagents import InferenceClientModel

LOCAL_MODEL_URL = os.getenv("LOCAL_MODEL_URL", "http://localhost:8000/v1")

try:
    model = InferenceClientModel(
        max_tokens=512,
        temperature=0.7,
        base_url=LOCAL_MODEL_URL,
        api_key="not-needed",
        model_id="Qwen/Qwen2.5-7B-Instruct-AWQ",
        custom_role_conversions=None,
    )
    print(f"✅ Successfully connected to local model at {LOCAL_MODEL_URL}")
except Exception as e:
    print(f"❌ Failed to connect to local model: {e}")
    raise
```

### Performance Optimization Tips

1. **Batch Processing**: If handling multiple requests, consider implementing request batching in the deployment script
2. **Memory Management**: Monitor VRAM usage and implement automatic garbage collection if needed
3. **Caching**: Add response caching for frequently asked questions
4. **Load Balancing**: For high traffic, consider running multiple model instances behind a load balancer

### Troubleshooting Common Issues

1. **CUDA Out of Memory**: Reduce `max_tokens` or switch to CPU inference temporarily
2. **Model Loading Errors**: Verify internet connection for initial model download
3. **API Connection Issues**: Check firewall settings and ensure port 8000 is accessible
4. **Performance Issues**: Monitor CPU/GPU utilization and adjust batch sizes accordingly

This deployment guide provides a complete solution for self-hosting an LLM on your Tesla T4 GPU while maintaining compatibility with the smolagents framework.
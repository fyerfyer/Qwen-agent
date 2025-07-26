#!/bin/bash

# DataSphere Navigator Ollama Startup Script
# This script helps set up and start Ollama for the first time

echo "🚀 DataSphere Navigator - Ollama Setup & Startup"
echo "=================================================="

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "📦 Ollama not found. Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
    
    if [ $? -eq 0 ]; then
        echo "✅ Ollama installed successfully"
    else
        echo "❌ Failed to install Ollama"
        exit 1
    fi
else
    echo "✅ Ollama is already installed"
fi

# Check if Ollama service is running
if pgrep -x "ollama" > /dev/null; then
    echo "✅ Ollama service is already running"
else
    echo "🔧 Starting Ollama service..."
    ollama serve &
    OLLAMA_PID=$!
    echo "📝 Ollama PID: $OLLAMA_PID"
    
    # Wait for Ollama to start
    echo "⏳ Waiting for Ollama to start..."
    sleep 5
    
    # Check if Ollama is responding
    for i in {1..10}; do
        if curl -s http://localhost:11434/api/tags > /dev/null; then
            echo "✅ Ollama is responding on port 11434"
            break
        else
            echo "⏳ Waiting for Ollama to respond... ($i/10)"
            sleep 2
        fi
        
        if [ $i -eq 10 ]; then
            echo "❌ Ollama failed to respond after 20 seconds"
            exit 1
        fi
    done
fi

# Check if the required model is available
echo "🔍 Checking for qwen2.5:7b model..."
if ollama list | grep -q "qwen2.5:7b"; then
    echo "✅ Model qwen2.5:7b is available"
else
    echo "📥 Pulling qwen2.5:7b model (this may take a few minutes)..."
    ollama pull qwen2.5:7b
    
    if [ $? -eq 0 ]; then
        echo "✅ Model qwen2.5:7b downloaded successfully"
    else
        echo "❌ Failed to download model qwen2.5:7b"
        exit 1
    fi
fi

# Test the model
echo "🧪 Testing model with a simple prompt..."
TEST_RESPONSE=$(ollama run qwen2.5:7b "Hello! Please respond with 'Model test successful' to confirm you're working." --timeout 30)

if [[ "$TEST_RESPONSE" == *"successful"* ]]; then
    echo "✅ Model test successful"
else
    echo "⚠️ Model test response: $TEST_RESPONSE"
fi

# Show available models
echo ""
echo "📋 Available Ollama models:"
ollama list

echo ""
echo "🎉 Ollama setup complete!"
echo ""
echo "🔗 Ollama API URL: http://localhost:11434"
echo "🤖 Active model: qwen2.5:7b"
echo ""
echo "🚀 Next steps:"
echo "   1. Install Python dependencies: pip install -r requirements.txt"
echo "   2. Test integration: python test_ollama_integration.py"
echo "   3. Start the application: python app.py"
echo ""
echo "💡 Useful Ollama commands:"
echo "   - List models: ollama list"
echo "   - Chat with model: ollama run qwen2.5:7b"
echo "   - Stop Ollama: pkill ollama"
echo "   - Check status: curl http://localhost:11434/api/tags"
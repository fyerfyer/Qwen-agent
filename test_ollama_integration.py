#!/usr/bin/env python3
"""
Test script for Ollama integration with DataSphere Navigator
Verifies that the Ollama model wrapper works correctly
"""

import sys
import os
from ollama_model import OllamaInferenceClient, OllamaModel

def test_ollama_connection():
    """Test basic Ollama connection"""
    print("🔧 Testing Ollama connection...")
    
    try:
        model = OllamaModel()
        health = model.health_check()
        
        if health["status"] == "healthy":
            print(f"✅ Ollama is running and healthy")
            print(f"   - Service: {health['service']}")
            print(f"   - Models available: {health['models_available']}")
            print(f"   - Current model: {health['current_model']}")
            print(f"   - Base URL: {health['base_url']}")
            return True
        else:
            print(f"❌ Ollama health check failed: {health.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to connect to Ollama: {e}")
        print("   Make sure Ollama is running with: ollama serve")
        return False

def test_model_generation():
    """Test model text generation"""
    print("\n🤖 Testing model generation...")
    
    try:
        model = OllamaModel(model_name="qwen2.5:7b")
        
        # Test simple generation
        messages = [
            {"role": "user", "content": "Hello! Please respond with exactly 'Test successful' if you can read this."}
        ]
        
        response = model(messages)
        print(f"✅ Model response: {response}")
        
        # Check token counting
        print(f"   - Input tokens: {model.last_input_token_count}")
        print(f"   - Output tokens: {model.last_output_token_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model generation failed: {e}")
        return False

def test_inference_client_compatibility():
    """Test InferenceClientModel compatibility"""
    print("\n🔄 Testing InferenceClientModel compatibility...")
    
    try:
        # This should work as a drop-in replacement
        model = OllamaInferenceClient(
            model_id="qwen2.5:7b",
            base_url="http://localhost:11434",
            api_key="not-needed",
            max_tokens=100,
            temperature=0.7,
            custom_role_conversions=None
        )
        
        # Test with smolagents-style message format
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2? Please answer briefly."}
        ]
        
        response = model.generate(messages)
        print(f"✅ Compatibility test response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        return False

def test_chat_completion_interface():
    """Test OpenAI-style chat completion interface"""
    print("\n💬 Testing chat completion interface...")
    
    try:
        model = OllamaModel()
        
        messages = [
            {"role": "user", "content": "Say 'Interface test successful' to confirm this works."}
        ]
        
        completion = model.chat_completion(messages)
        
        print(f"✅ Chat completion response:")
        print(f"   - ID: {completion['id']}")
        print(f"   - Model: {completion['model']}")
        print(f"   - Content: {completion['choices'][0]['message']['content']}")
        print(f"   - Tokens: {completion['usage']['total_tokens']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Chat completion test failed: {e}")
        return False

def test_app_import():
    """Test that the updated app.py can be imported"""
    print("\n📦 Testing app.py import...")
    
    try:
        # Add current directory to path to import app
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Try to import the app module
        import app
        
        print(f"✅ App imported successfully")
        print(f"   - Model type: {type(app.model).__name__}")
        print(f"   - Model ID: {app.model.model_id}")
        print(f"   - Agent name: {app.agent.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ App import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 DataSphere Navigator Ollama Integration Test")
    print("=" * 50)
    
    tests = [
        ("Ollama Connection", test_ollama_connection),
        ("Model Generation", test_model_generation),
        ("InferenceClient Compatibility", test_inference_client_compatibility),
        ("Chat Completion Interface", test_chat_completion_interface),
        ("App Import", test_app_import)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ollama integration is working correctly.")
        print("\n🚀 You can now start your application with:")
        print("   python app.py")
    else:
        print(f"⚠️  {total - passed} test(s) failed. Please check the issues above.")
        print("\n🔧 Common solutions:")
        print("   1. Make sure Ollama is running: ollama serve")
        print("   2. Pull the model: ollama pull qwen2.5:7b")
        print("   3. Check Ollama status: ollama list")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
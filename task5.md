Now I get the following error:

```
📦 Testing app.py import...
INFO:ollama_model:✅ Ollama connection verified. Model: qwen2.5:7b
INFO:ollama_model:🔄 Using Ollama model 'qwen2.5:7b' instead of InferenceClientModel
❌ App import failed: 'str' object has no attribute 'keys'

==================================================
🏁 Test Results: 4/5 tests passed
⚠️  1 test(s) failed. Please check the issues above.
```

Why is this happening? Please carefully analyze the relevant code and help me identify where the problem is and how to fix it. Note: you only need to check if there are any Python syntax errors in the current code and fix the error and confirm that the `final_answer` prompt template is not missing yourself. You do **not** need to actually run the code—I will run it myself on the cloud server. Please provide a detailed explanation and solution.
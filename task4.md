I am currently working on an Agent project. When I run a simple test script called `test_ollama_integration.py` on my cloud server, I get the following error:

```
📦 Testing app.py import...
❌ App import failed: f-string expression part cannot include a backslash (data_tools.py, line 807)

==================================================
🏁 Test Results: 4/5 tests passed
⚠️  1 test(s) failed. Please check the issues above.

🔧 Common solutions:
   1. Make sure Ollama is running: ollama serve
   2. Pull the model: ollama pull qwen2.5:7b
   3. Check Ollama status: ollama list
```

Why is this happening? Please carefully analyze the relevant code and help me identify where the problem is and how to fix it. Note: you only need to check if there are any Python syntax errors in the current code and fix the error yourself. You do **not** need to actually run the code—I will run it myself on the cloud server. Please provide a detailed explanation and solution.
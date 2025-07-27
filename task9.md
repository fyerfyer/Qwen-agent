When I ran the agent in the cloud server, I got the following error:

```
ERROR:ollama_model:Ollama generation error: 'ChatMessage' object has no attribute 'get'
Error in generating model output:
'str' object has no attribute 'content'
[Step 1: Duration 0.00 seconds]
Traceback (most recent call last):
  File "/root/miniforge3/lib/python3.11/site-packages/smolagents/agents.py", line 1648, in _step_stream
    output_text = chat_message.content
                  ^^^^^^^^^^^^^^^^^^^^
AttributeError: 'str' object has no attribute 'content'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/root/miniforge3/lib/python3.11/site-packages/gradio/queueing.py", line 716, in process_events
    response = await route_utils.call_process_api(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniforge3/lib/python3.11/site-packages/gradio/route_utils.py", line 350, in call_process_api
    output = await app.get_blocks().process_api(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniforge3/lib/python3.11/site-packages/gradio/blocks.py", line 2235, in process_api
    result = await self.call_function(
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniforge3/lib/python3.11/site-packages/gradio/blocks.py", line 1758, in call_function
    prediction = await utils.async_iteration(iterator)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniforge3/lib/python3.11/site-packages/gradio/utils.py", line 762, in async_iteration
    return await anext(iterator)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniforge3/lib/python3.11/site-packages/gradio/utils.py", line 753, in __anext__
    return await anyio.to_thread.run_sync(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniforge3/lib/python3.11/site-packages/anyio/to_thread.py", line 56, in run_sync
    return await get_async_backend().run_sync_in_worker_thread(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniforge3/lib/python3.11/site-packages/anyio/_backends/_asyncio.py", line 2470, in run_sync_in_worker_thread
    return await future
           ^^^^^^^^^^^^
  File "/root/miniforge3/lib/python3.11/site-packages/anyio/_backends/_asyncio.py", line 967, in run
    result = context.run(func, *args)
             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniforge3/lib/python3.11/site-packages/gradio/utils.py", line 736, in run_sync_iterator_async
    return next(iterator)
           ^^^^^^^^^^^^^^
  File "/root/miniforge3/lib/python3.11/site-packages/gradio/utils.py", line 900, in gen_wrapper
    response = next(iterator)
               ^^^^^^^^^^^^^^
  File "/workspace/Qwen-agent/Gradio_UI.py", line 197, in interact_with_agent
    for msg in stream_to_gradio(self.agent, task=prompt, reset_agent_memory=False):
  File "/workspace/Qwen-agent/Gradio_UI.py", line 142, in stream_to_gradio
    for step_log in agent.run(task, stream=True, reset=reset_agent_memory, additional_args=additional_args):
  File "/root/miniforge3/lib/python3.11/site-packages/smolagents/agents.py", line 561, in _run_stream
    raise e
  File "/root/miniforge3/lib/python3.11/site-packages/smolagents/agents.py", line 543, in _run_stream
    for output in self._step_stream(action_step):
  File "/root/miniforge3/lib/python3.11/site-packages/smolagents/agents.py", line 1664, in _step_stream
    raise AgentGenerationError(f"Error in generating model output:\n{e}", self.logger) from e
smolagents.utils.AgentGenerationError: Error in generating model output:
'str' object has no attribute 'content'
```

Why does the error occur? Please analyze related code carefully and help me fix the error. 

ote: you only need to check if there are any Python syntax errors in the current code and fix the error yourself. You do **not** need to actually run the code—I will run it myself on the cloud server. 
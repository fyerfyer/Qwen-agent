**Subject: Resolve `AttributeError` in Agent and Proactively Audit API Usage**

**1. Context**

I am encountering a runtime error in my application that uses the `smolagents` library with a Gradio interface. When the agent attempts to generate a response, the application crashes.

**2. Error Details**

The program fails with the following traceback, indicating that a `ChatMessage` object is missing the expected `token_usage` attribute:

```
Error in generating model output:
'ChatMessage' object has no attribute 'token_usage'
[Step 1: Duration 37.38 seconds]
Traceback (most recent call last):
  File "/root/miniforge3/lib/python3.11/site-packages/smolagents/agents.py", line 1661, in _step_stream
    memory_step.token_usage = chat_message.token_usage
                              ^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'ChatMessage' object has no attribute 'token_usage'

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
'ChatMessage' object has no attribute 'token_usage'
```

**3. Request**

Please perform the following two tasks:

1.  **Fix the Error:** Analyze the relevant code implementation in my workspace and resolve the `AttributeError`. The issue likely stems from an incorrect assumption about the structure of the `ChatMessage` object returned by the underlying model API.
2.  **Proactive Code Review:** After fixing the immediate issue, please research the relevant API documentation (e.g., for `smolagents` or the LLM provider it uses). Review the code to identify and fix any other similar issues where the code might be misaligned with the API's expected data structures.

**Note:** You only need to perform a static analysis of the code and provide the necessary fixes. You do **not** need to execute the code, as I will run it on my cloud server for testing.

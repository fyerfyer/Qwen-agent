**Role:** You are an expert AI Agent architect and a senior MLOps engineer.

**Context:**
I am working on a project from the Hugging Face Agent course. The goal is to build a unique and powerful AI agent using the `smolagents` library. Below is the full text of the tutorial I am following. Read it carefully to understand the framework, the provided code (`app.py`), the concept of tools, and the overall objective.

```markdown
<Let’s Create Our First Agent Using smolagents>
In the last section, we learned how we can create Agents from scratch using Python code, and we saw just how tedious that process can be. Fortunately, many Agent libraries simplify this work by handling much of the heavy lifting for you.

In this tutorial, you’ll create your very first Agent capable of performing actions such as image generation, web search, time zone checking and much more!

You will also publish your agent on a Hugging Face Space so you can share it with friends and colleagues.

Let’s get started!

<What is smolagents?>
To make this Agent, we’re going to use smolagents, a library that provides a framework for developing your agents with ease.

This lightweight library is designed for simplicity, but it abstracts away much of the complexity of building an Agent, allowing you to focus on designing your agent’s behavior.

We’re going to get deeper into smolagents in the next Unit. Meanwhile, you can also check this blog post or the library’s repo in GitHub.

In short, smolagents is a library that focuses on codeAgent, a kind of agent that performs “Actions” through code blocks, and then “Observes” results by executing the code.

Here is an example of what we’ll build!

We provided our agent with an Image generation tool and asked it to generate an image of a cat.

The agent inside smolagents is going to have the same behaviors as the custom one we built previously: it’s going to think, act and observe in cycle until it reaches a final answer.

<Let’s build our Agent!>
To start, duplicate this Space: https://huggingface.co/spaces/agents-course/First_agent_template

After duplicating the Space, you’ll need to add your Hugging Face API token so your agent can access the model API.

Throughout this lesson, the only file you will need to modify is the (currently incomplete) “app.py”.

Let’s break down the code together:

The file begins with some simple but necessary library imports
```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, FinalAnswerTool, InferenceClientModel, load_tool, tool
import datetime
import requests
import pytz
import yaml
```
As outlined earlier, we will directly use the `CodeAgent` class from `smolagents`.

<The Tools>
Now let’s get into the tools!

```python
def my_custom_tool(arg1:str, arg2:int)-> str: # it's important to specify the return type
    # Keep this format for the tool description / args description but feel free to modify the tool
    """A tool that does nothing yet 
    Args:
        arg1: the first argument
        arg2: the second argument
    """
    return "What magic will you build ?"


def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"
```
To define your tool it is important to:
1.  Provide input and output types for your function.
2.  A well formatted docstring with argument descriptions.

<The Agent>
It uses `Qwen/Qwen2.5-Coder-32B-Instruct` as the LLM engine.

```python
final_answer = FinalAnswerTool()
model = InferenceClientModel(
    max_tokens=2096,
    temperature=0.5,
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
    custom_role_conversions=None,
)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
# We're creating our CodeAgent
agent = CodeAgent(
    model=model,
    tools=[final_answer], # add your tools here (don't remove final_answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)

GradioUI(agent).launch()
```
You need to focus on adding new tools to the `tools` list of your Agent.

<The System Prompt>
The agent’s system prompt is stored in a separate `prompts.yaml` file. This file contains predefined instructions that guide the agent’s behavior.

<The complete “app.py” template:>
```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, load_tool, tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool

from Gradio_UI import GradioUI

# Below is an example of a tool that does nothing. Amaze us with your creativity!

def my_custom_tool(arg1:str, arg2:int)-> str: # it's important to specify the return type
    # Keep this format for the tool description / args description but feel free to modify the tool
    """A tool that does nothing yet 
    Args:
        arg1: the first argument
        arg2: the second argument
    """
    return "What magic will you build ?"


def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


final_answer = FinalAnswerTool()
model = InferenceClientModel(
    max_tokens=2096,
    temperature=0.5,
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
    custom_role_conversions=None,
)


# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

# Load system prompt from prompt.yaml file
with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[final_answer], # add your tools here (don't remove final_answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates # Pass system prompt to CodeAgent
)


GradioUI(agent).launch()
```

<Your Goal>
Your Goal is to get familiar with the Space and the Agent. Currently, the agent in the template does not use any tools, so try to provide it with some of the pre-made ones or even make some new tools yourself!
```

---

**Task & Deliverables**

You will perform two main tasks and provide two separate, complete Markdown documents as the final output.

**Part 1: AI Agent Design Document**

**Task:** Based on the context provided, design a concept for a unique, creative, and fully-featured AI agent. Your design should go beyond the simple examples in the tutorial. Propose a coherent set of tools that work together to give the agent a distinct personality and purpose.

**Deliverable:** Produce a detailed design document in a single Markdown file. The document must include the following sections:

1.  **Agent Concept:**
    *   **Name:** A creative name for your agent.
    *   **Purpose & Personality:** A brief description of what the agent does and its character (e.g., "A sarcastic news analyst that summarizes daily events and generates editorial cartoons," or "A helpful travel planner that finds flights, suggests activities, and checks weather forecasts.").

2.  **Core Architecture:**
    *   A high-level overview of how the agent will function.
    *   Explain how the chosen tools will interact to fulfill the agent's purpose.

3.  **Tool Manifest:**
    *   List each tool you will create or use. For each tool, provide:
        *   **Tool Name:** The Python function name (e.g., `get_weather_forecast`).
        *   **Description:** A clear, one-sentence summary of what the tool does.
        *   **Python Function Signature:** The function definition with type hints (e.g., `def get_weather_forecast(city: str, date: str) -> str:`).
        *   **Docstring:** A complete docstring explaining the tool and its arguments, formatted for `smolagents`.

4.  **System Prompt (`prompts.yaml`):**
    *   Write the complete content for the `system_prompt` section of the `prompts.yaml` file. This prompt should instruct the LLM on its role, personality, how to use the tools, and the format for its final answer.

5.  **Example Usage:**
    *   Provide 2-3 sample user prompts and describe the expected output or chain of actions the agent would take, demonstrating how the agent works in practice.

*(Note: After you provide this design document, I will review it and then ask you to write the complete Python code for `app.py` based on your design. For now, only provide the design document.)*

---

**Part 2: Model Deployment Guide**

**Task:** I plan to self-host the LLM for this agent on my own cloud server. Create a step-by-step guide to deploy a suitable model on my hardware.

**Context for this task:**
My server specifications are as follows:
```
- OS: Ubuntu 20.04.6 LTS
- GPU: 1x NVIDIA Tesla T4 (15360MiB VRAM)
- CPU: 8x Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz
- CUDA: 12.4
- Python: 3.10.11
- PyTorch: 2.5.1+cu124
```

The tutorial mentions using `Qwen/Qwen2.5-Coder-32B-Instruct`. **This model is too large for my NVIDIA T4 GPU.**

Your task is to:
1.  Acknowledge this hardware limitation.
2.  Recommend a smaller, high-quality, open-source alternative model (e.g., in the 7B-8B parameter range) that is excellent for agentic tasks and can be deployed on my hardware with sufficient VRAM headroom. Justify your choice.
3.  Provide a detailed, step-by-step deployment guide for your **recommended model**.

**Deliverable:** Produce a complete deployment guide in a single Markdown file. The guide must include:

1.  **Model Recommendation:**
    *   The recommended model Hugging Face ID (e.g., `Qwen/Qwen1.5-7B-Chat-AWQ`).
    *   A brief justification for why this model is a good fit for the hardware and the agentic task.

2.  **Environment Setup:**
    *   Instructions for setting up a virtual environment (e.g., `conda` or `venv`).
    *   A list of all necessary pip packages (e.g., `transformers`, `accelerate`, `bitsandbytes`, `fastapi`, `uvicorn`, etc.).

3.  **Deployment Script:**
    *   A complete Python script (`deploy_model.py`) that:
        *   Loads the recommended model and tokenizer from Hugging Face.
        *   Applies quantization (e.g., 4-bit with `bitsandbytes` or uses a pre-quantized version) to ensure it fits in memory.
        *   Sets up a simple web server using **FastAPI**.
        *   Exposes a `/v1/chat/completions` endpoint that mimics the OpenAI API format. This is crucial for compatibility with `smolagents.InferenceClientModel`.

4.  **Running the Service:**
    *   The exact command-line instruction to start the FastAPI server (e.g., `uvicorn deploy_model:app --host 0.0.0.0 --port 8000`).

5.  **Testing the Endpoint:**
    *   An example `curl` command to send a request to the local server, verifying that the model is running and responding correctly.

6.  **Integration with `app.py`:**
    *   Show exactly how to modify the `InferenceClientModel` instantiation in the original `app.py` to point to the new local model server.
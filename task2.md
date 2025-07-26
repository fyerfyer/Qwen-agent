**Role:** You are an expert AI Agent developer specializing in the `smolagents` library and Hugging Face ecosystem.

**Context:**
I am building a custom AI Agent based on a Hugging Face course template. The project has two key documents:
1.  `agent_design_document.md`: A detailed plan outlining the agent's concept, architecture, tools, and system prompt. This is the primary specification for you to follow.
2.  `model_deployment_guide.md`: Instructions for deploying the required LLM. I have already completed these steps and the model is running on my cloud server, ready for the agent to use.

Your task is to write the Python code to bring the agent described in the design document to life.

**Task:**
Implement the complete AI Agent project based *exactly* on the specifications in the `agent_design_document.md`. You will need to write the Python code for the custom tools and configure the `CodeAgent` with the correct parameters, tools, and system prompt.

**Instructions:**
1.  **Read Carefully:** Thoroughly analyze the attached `agent_design_document.md` and `model_deployment_guide.md` to understand the complete project requirements.
2.  **Implement the Agent:** Write the full Python code for `app.py`. This includes defining all custom tools and instantiating the `CodeAgent` with the tools, model, and system prompt specified in the design document.
3.  **File Management:** The current project directory contains template files. You have the authority to delete any example files (like `my_custom_tool`) that are not part of the final design.

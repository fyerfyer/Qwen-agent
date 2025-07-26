from smolagents import CodeAgent, load_tool
import yaml
from tools.final_answer import FinalAnswerTool
from tools.data_tools import (
    load_data_from_url,
    analyze_data_profile, 
    create_visualization,
    calculate_statistics,
    get_market_data,
    export_analysis_report
)
from ollama_model import OllamaInferenceClient

from Gradio_UI import GradioUI


final_answer = FinalAnswerTool()

# Configure Ollama model connection
model = OllamaInferenceClient(
    model_id="qwen2.5:7b",  # Ollama model name
    base_url="http://localhost:6399",  # Ollama default port
    api_key="not-needed",  # Compatibility parameter
    max_tokens=512,
    temperature=0.7,
    custom_role_conversions=None,
)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[
        final_answer,
        load_data_from_url,
        analyze_data_profile,
        create_visualization,
        calculate_statistics,
        get_market_data,
        export_analysis_report
    ],
    max_steps=10,  # Increased for more complex data analysis workflows
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name="DataSphere Navigator",
    description="Expert data analyst and visualization specialist",
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()
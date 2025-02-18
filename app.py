from smolagents import CodeAgent,DuckDuckGoSearchTool, HfApiModel,load_tool,tool
import datetime
import requests
import pytz
import yaml
from rank_bm25 import BM25Okapi
from tools.final_answer import FinalAnswerTool
from tools.visit_webpage import VisitWebpageTool
from tools.web_search import DuckDuckGoSearchTool

from Gradio_UI import GradioUI

def rank_urls_by_relevance(search_results, query):
    """Ranks URLs based on BM25 similarity between query and article titles/snippets."""
    docs = []
    for res in search_results:
        docs.append(res[0] + " " + res[2])
    
    tokenized_docs = [doc.lower().split() for doc in docs]
    
    bm25 = BM25Okapi(tokenized_docs)
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)

    # Sort results by BM25 score (higher is better)
    ranked_results = sorted(zip(search_results, scores), key=lambda x: x[1], reverse=True)
    return ranked_results

# The tool below uses the DuckDuckGoSearchTool to fetch data from the web that is relevant to the query.
# The fetched results are reranked using BM25 and data from the url with the highest score is fetched.
@tool
def get_answer_from_web(query:str)-> str: #it's import to specify the return type
    """A tool that fetches results from the web. This function first calls DuckDuckGoSearchTool and the url received is passed to VisitWebpageTool to receive the final results.
    Args:
        query: A string that represents the query to search the web for.
    """
    try:
        web_search_tool = DuckDuckGoSearchTool()
        results = web_search_tool.forward(query)
        
        visit_webpage_tool = VisitWebpageTool()

        ranked_results = rank_urls_by_relevance(results, query)

        page_content = visit_webpage_tool.forward(ranked_results[0][0][1])

        return page_content
        
    except Exception as e:
        return f"Error fetching results"

@tool
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

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = HfApiModel(
max_tokens=2096,
temperature=0.5,
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',# it is possible that this model may be overloaded
custom_role_conversions=None,
)


# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[final_answer, image_generation_tool, get_current_time_in_timezone, get_answer_from_web], ## add your tools here (don't remove final answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()
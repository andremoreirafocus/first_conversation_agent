import math
import os

import numexpr
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import tool


@tool("internet_search")
def internet_search(query: str) -> str:
    """Search Google via SerpAPI for up-to-date information."""
    serp_api_key = os.environ["SERPAPI_API_KEY"]
    params = {"engine": "google", "gl": "us", "hl": "en"}
    search = SerpAPIWrapper(params=params, serpapi_api_key=serp_api_key)
    return search.run(query)


@tool("calculator")
def calculator(expression: str) -> str:
    """Evaluate a single-line mathematical expression with numexpr."""
    local_dict = {"pi": math.pi, "e": math.e}
    output = numexpr.evaluate(
        expression.strip(),
        global_dict={},
        local_dict=local_dict,
    )
    return str(output)

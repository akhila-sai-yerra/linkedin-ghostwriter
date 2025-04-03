import os 

from mcp.server.fastmcp import FastMCP
from exa_py import Exa
from typing import Annotated
from datetime import date, timedelta, datetime


mcp = FastMCP("linkedin_tools_stdio")
EXA_API_KEY = os.environ["EXA_API_KEY"]

exa = Exa(api_key=EXA_API_KEY)


start_published_date = datetime.combine(date.today() - timedelta(days=30), datetime.min.time()).isoformat() + ".000Z"

@mcp.tool()
def search_and_content(
    query: str,
    start_published_date: str,
    end_published_date: str
) -> str:
    """Search for webpages based on the query ... """
    
    return exa.search_and_contents(
        query,
        use_autoprompt=False,
        num_results=10,
        start_published_date=start_published_date,
        end_published_date=end_published_date,
        text={"max_characters": 400},
        category="news",
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")

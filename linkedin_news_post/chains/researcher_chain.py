from langgraph.prebuilt import create_react_agent

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from datetime import date

today = date.today()

class search_and_content(BaseModel):

    query: str = Field(description="Query for the search")
    start_published_date: str = Field(description="Start range of publishign range")
    end_published_date: str = Field(description="End range of publishign range")

llm = ChatOpenAI(model="gpt-4o")

llm_with_tools = llm.bind_tools([search_and_content])

system = f"""You are an expert researcher tasked with finding the latest news in quantitative finance in the 1 to 3 months noting that today is {today} in the United States, tailored for advanced undergraduates seeking to apply for quant roles. 

Select a topic and always call the tool "search_and_content" to find relevant content and output the result.

If the supervisor provides you with information, always try a different query to generate results that are as distinct as possible from the previous query.
"""

# Use placeholder instead of messages since we are working with create_react_agent
systemPrompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("placeholder", "{messages}"),
    ]
)

researcher_chain = systemPrompt | llm_with_tools





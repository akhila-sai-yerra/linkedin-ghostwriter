import asyncio

import os
from langgraph.graph import StateGraph, START
from linkedin_news_post import State
from langgraph.prebuilt import ToolNode
from linkedin_news_post.nodes import publisher_node, supervisor_node, researcher_node, writer_node, quality_node

from contextlib import asynccontextmanager
from langchain_mcp_adapters.client import MultiServerMCPClient

from linkedin_news_post.mongo_store import MongoDBBaseStore
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()


CONNECTION_STRING = os.environ["MONGODB_URI"]
COMPOSIO_MCP_URL = os.environ["COMPOSIO_MCP_URL"]


openai_embeddings = OpenAIEmbeddings()

def embed_text(text: str) -> list[float]:
    return openai_embeddings.embed_query(text)

index_config = {
    "embed": embed_text,     
    "fields": ["content.article", "summary"],
    "index_name": "store_index",
}

mongo_store = MongoDBBaseStore(
    mongo_url=CONNECTION_STRING,   
    db_name="checkpointing_db",                
    collection_name="store",    
    index_config=index_config,
    ttl_support=True
)

@asynccontextmanager
async def make_graph():

    async with MultiServerMCPClient(
        connections={
            "linkedin_tools_stdio": {
                "transport": "stdio",
                "command": "python",
                "args": ["linkedin_news_post/mcp_server.py"],
            },
            "linkedin": {
                "transport": "sse",
                "url": COMPOSIO_MCP_URL,
            },
        }
    ) as mcp_client:

        tools = mcp_client.get_tools()
        
        workflow = StateGraph(State)

  

        workflow.add_node("supervisor_node", supervisor_node)
        workflow.add_node("tool_node", ToolNode(tools))
        workflow.add_node("publisher_node", publisher_node)
        workflow.add_node("researcher_node", researcher_node)
        workflow.add_node("quality_node", quality_node)
        workflow.add_node("writer_node", writer_node)

        workflow.add_edge(START, "supervisor_node")
        workflow.add_edge("tool_node", "supervisor_node")

        graph = workflow.compile(store=mongo_store)
        yield graph

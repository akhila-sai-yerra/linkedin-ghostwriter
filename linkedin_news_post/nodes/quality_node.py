
from linkedin_news_post import State
from langgraph.types import Command
from langchain_core.messages import HumanMessage

from langgraph.store.base import BaseStore

from linkedin_news_post.chains import quality_chain
from typing import Literal

def quality_node(state: State, store: BaseStore) -> Command[Literal["supervisor_node"]]:

    
    # Semantic serach using proposed article
    past_articles = store.search(("articles",), query=state["messages"][-2].content, limit=3)


    result = quality_chain.invoke({
        "messages": state["messages"],
        "past_articles": past_articles
    })



    return Command(
        goto="supervisor_node",
        update={"messages": [HumanMessage(content=result.content, name="quality_node")]}
    )
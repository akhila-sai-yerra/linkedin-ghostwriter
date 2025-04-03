import os

from linkedin_news_post import State
from langgraph.constants import END

from langgraph.types import Command
from typing import Literal
from langgraph.store.base import BaseStore

from pydantic import BaseModel, Field
from langmem import create_memory_store_manager
from linkedin_news_post.chains import publisher_chain

class Article(BaseModel):
    article: str = Field(description="Very concise description of what the published article is about")


def publisher_node(state: State, store: BaseStore) -> Command[Literal["tool_node"]]:

    # Memory Managment
    manager = create_memory_store_manager(
        "gpt-4o",
        namespace=("articles",),
        schemas=[Article],
        instructions="Extract the information from the most recent article written by the writer_node message, which will be the newly published article about to be released. Add 1 new entry for the article to the collection, including details such as dates and statistics for future reference, while avoiding content redundancy.",
        store=store,
        enable_inserts=True
    )

    manager.invoke({"messages": state["messages"]})

    result = publisher_chain.invoke(state)

    return Command(
        goto="tool_node",
        update={
            "messages": [result]
        }
    )



    
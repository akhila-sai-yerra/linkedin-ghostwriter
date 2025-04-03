
from linkedin_news_post import State
from linkedin_news_post.chains import researcher_chain
from langchain_core.messages import HumanMessage

from langgraph.types import Command
from typing import Literal

def researcher_node(state: State) -> Command[Literal["tool_node"]]:

    new_messages = state["messages"] + [HumanMessage(content=f"Tell me news about quantitative finance picking a topic of your choice")]
    result = researcher_chain.invoke({
        "messages": new_messages
    })

    return Command(
        goto="tool_node",
        update={
            "messages": [result]
        }
    )





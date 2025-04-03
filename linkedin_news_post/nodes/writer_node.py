
from linkedin_news_post import State
from linkedin_news_post.chains import writer_chain
from langchain_core.messages import HumanMessage

from langgraph.types import Command
from typing import Literal

def writer_node(state: State) -> Command[Literal["supervisor_node"]]:

    result = writer_chain.invoke(state)

    return Command(
        goto="supervisor_node",
        update={
            "messages": [HumanMessage(content=result.content, name="writer_node")]
        }
    )



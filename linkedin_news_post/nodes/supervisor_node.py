

from linkedin_news_post import State
from linkedin_news_post.chains import supervisor_chain
from langchain_core.messages import HumanMessage
from langgraph.constants import END

from langgraph.types import Command
from typing import Literal

def supervisor_node(state: State) -> Command[Literal["publisher_node", "researcher_node", "writer_node", "quality_node", "__end__"]]:

    result = supervisor_chain.invoke(state)

    print(state["messages"])

    if result.next_node == "researcher_node":

        return Command(
            goto="researcher_node",
            update={"messages": [HumanMessage(content="Passing to researcher...", name="supervisor_node")]}
        )

    elif result.next_node == "writer_node":

        return Command(
            goto="writer_node",
            update={"messages": [HumanMessage(content="Passing to writer...", name="supervisor_node")]}
        )
    
    elif result.next_node == "quality_node":

        return Command(
            goto="quality_node",
            update={"messages": [HumanMessage(content="Passing to quality checker...", name="supervisor_node")]}
        )

    elif result.next_node == "publisher_node":

        return Command(
            goto="publisher_node",
            update={"messages": [HumanMessage(content="Passing to publisher...", name="supervisor")]}
        )
    
    elif result.next_node == "end_node":

        return Command(
            goto={END},
            update={"messages": [HumanMessage(content="Finishing the Process...", name="supervisor")]}
        )


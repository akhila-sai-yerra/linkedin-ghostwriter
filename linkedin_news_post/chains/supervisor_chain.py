
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field
from typing import Literal

llm = ChatOpenAI(model="gpt-4o")

class Handout(BaseModel):
    next_node: Literal["researcher_node", "writer_node", "publisher_node", "quality_node", "end_node"] = Field(description="Next node in the workflow")

structured_llm = llm.with_structured_output(Handout)


system = """You are a supervisor tasked with managing a conversation between the following workers: writer_node, quality_node, researcher_node, and publisher_node. You should refer to each worker at least once. Given the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and status. When the post has been succefully published respond with "end_node". 

# Note:
 - Listen to the recommedations of the quality_node
 - Be Ready to suggest different queries to the researcher
"""

systemPrompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("user", "{messages}"),
    ]
)

supervisor_chain = systemPrompt | structured_llm


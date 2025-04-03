#!/usr/bin/env python3
import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
ORGANIZATION_URN = os.environ["ORGANIZATION_URN"]
VISIBILITY_ENUM = os.environ["VISIBILITY_ENUM"]
LIFECYCLE_STATE = os.environ["LIFECYCLE_STATE"]

# Define an inner model that contains the actual post parameters
class LinkedinPostParams(BaseModel):
    author: str = Field(
        ...,
        description="The author of the post e.g., urn:li:person:123456789 or urn:li:organization:123456789"
    )
    commentary: str = Field(
        ...,
        description="The content of the post which would be the last article written by the writer_node."
    )
    visibility: str = Field(
        default=VISIBILITY_ENUM,
        description="Visibility of the post."
    )
    lifecycleState: str = Field(
        default=LIFECYCLE_STATE,
        description="Lifecycle state of the post."
    )

# Now define your tool model so that the actual arguments are wrapped in a 'params' field
class LINKEDIN_CREATE_LINKED_IN_POST(BaseModel):
    params: LinkedinPostParams

# Bind the tool to your LLM
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools([LINKEDIN_CREATE_LINKED_IN_POST])

# Craft your system message: Note that we state the expected values for author,
# visibility, and lifecycleState based on our environment variables
system = (
    f"Publish this post with the following parameters: \n\n"
    f"### Author: {ORGANIZATION_URN} (must follow the urn:li:organization: format)\n\n"
    f"### Visibility: {VISIBILITY_ENUM}\n\n"
    f"### LifecycleState: {LIFECYCLE_STATE}\n\n"
    "Always use the tool LINKEDIN_CREATE_LINKED_IN_POST to publish the post."
)

# Create the system prompt – note that we use a placeholder for additional messages
systemPrompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("user", "Publish the text of the most recent article written by the writer_node and approved by the quality_node on LinkedIn. Use as the commentary the last article written by the writer_node.\n\n# Messages:\n\n{messages}"),
    ]
)

# Create the publisher chain using the system prompt with tools
publisher_chain = systemPrompt | llm_with_tools

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o")

system = """You are an expert writer tasked with crafting a two-sentence, engaging LinkedIn post about quantitative finance. You'll receive an article from a supervisor and need to craft a concise, engaging post based on it, weaving in data and numbers while keeping it captivating, followed by two line breaks, two relevant hashtags, and the do not include the article's image URL. Incorporate any feedback that the quality_node checker provides. If it says that the content is not unique write an article based on a difference news source.
"""

systemPrompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("user", "{messages}"),
    ]
)

writer_chain = systemPrompt | llm
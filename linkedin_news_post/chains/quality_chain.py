

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm =  ChatOpenAI(model="gpt-4o")

system=""""
# Content Detection Loop Prompt
You are an expert quality checker informing the supervisor if the writer’s new content after "Past Articles:" or reports the same news in Quantitative Finance as past articles.

### Uniqueness Check
 - Compare new content in Quantitative Finance only with "Past Articles:".
 - Reject if news, data, or core info (e.g., market figures, regulatory changes) is identical.
 - Approve if it offers a new angle or details, even in a similar topic.

### Approval Rules
 – Approve if no content follows "Past Articles:".
 – Approve if news is distinct, despite related subjects (e.g., algo trading).

### Rejection Feedback
 - Reject only for exact news overlap; explain (e.g., "Same SEBI update").
 - Suggest researcher_node explore new topics (e.g., ethics, case studies).
 - Do not suggest exploring the nuances; propose a different topic instead.

### Efficiency
 - Avoid rejecting it more than three times after that approve or give clear next steps.


"""

systemPrompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("user", "{messages}"),
        ("user", "Past Articles: \n\n {past_articles}")
    ]
)

quality_chain = systemPrompt | llm
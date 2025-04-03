import asyncio

from linkedin_news_post.graph import make_graph


async def run_graph():
    async with make_graph() as graph:

        await graph.ainvoke({"messages": [
            ("user", "Publish a linkedin article")
        ]})
    


asyncio.run(run_graph())
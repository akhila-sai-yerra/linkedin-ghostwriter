# LinkedIn Ghostwriter AI Agent

An open-source LinkedIn Ghostwriter AI Agent powered by MCP and LangChain. This agent was used to grow a LinkedIn account to 900 followers in just 7 weeks (CoachQuant) by leveraging modern tools for agent orchestration, web search, authentication, vector search, and episodic memory management.

> “Opensourcing a LinkedIn Ghostwriter AI Agent fueled by MCP & LangChain💥  
> We leveraged this agent to grow a LinkedIn account to 900 followers in 7 weeks (CoachQuant) using LangChain, Exa, and MongoDB.  
> This is how we did it 👇  
> • LangGraph & LangChain for agent orchestration  
> • Exa for web search  
> • Composio's MCP for agent auth  
> • MongoDB + Embeddings for vector search  
> • Langmem for Episodic memory management  
>  
> **Inner workings:**  
> 🧪 We used a supervisor model with four agents: a researcher, a writer, a quality control agent, and a publisher. These are connected to an MCP Client (tool_node) that provides access to the appropriate tools.  
> **Workflow:**  
> ✍ The supervisor uses the researcher_node powered by Exa's Web API to look up news articles and the writer_node to draft the article.  
> 🛂 The quality_node ensures that the news article is unique compared to previously published articles by using vector search and our custom MongoDB implementation of LangGraph's robust BaseStore.  
> 🛫 The publisher_node leverages Composio's tools to post on LinkedIn.  
> 🧠 Finally, the supervisor stores the episodic memory of the agent in the MongoDB database using Langmem's Store Manager.”

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [License](#license)

## Overview

This project constructs a state graph-based workflow for a LinkedIn Ghostwriter Agent using a combination of:

- **LangGraph & LangChain**: To orchestrate the agent workflows.
- **MCP (Multi-Client Provider)**: For authentication and tool management.
- **MongoDB + OpenAI Embeddings**: For vector-based search and checkpointing the state/episodic memory.
- **Agents**: The supervisor delegates tasks across different nodes—researcher, writer, quality control, and publisher.

The agent follows these basic steps:

1. **Research**: Search for news articles using an external web API.
2. **Writing**: Draft a news article.
3. **Quality Check**: Ensure that the article is unique by comparing it with previously published content using vector search.
4. **Publish**: Post the final content to LinkedIn.
5. **Memory Storage**: Store details of the episode for future reference.

## Architecture

### Core Components

- **StateGraph & State**:  
  The graph is constructed using LangGraph’s `StateGraph` that defines nodes (agents) and edges (workflow connections). The `State` defines the type or structure of data that flows through the graph.

- **Nodes**:  
  - `supervisor_node`: Oversees and delegates work.  
  - `researcher_node`: Searches for news articles (powered by Exa’s Web API).  
  - `writer_node`: Drafts articles.  
  - `quality_node`: Validates the article’s uniqueness using vector search.  
  - `publisher_node`: Publishes the article on LinkedIn.  
  - `tool_node`: Wraps external tools provided by the MCP client.

- **MCP Client & Tools**:  
  The agent uses a `MultiServerMCPClient` to obtain external tools via different transport channels:  
  - A local process using stdio (to run a Python MCP server).  
  - A server-sent events (SSE) connection to Composio’s MCP URL.

- **MongoDB Store**:  
  The `MongoDBBaseStore` provides persistence and checkpointing capabilities using a custom index configuration built on OpenAI embeddings—facilitating vector search to check article uniqueness.

- **Embedding Configuration**:  
  Uses the OpenAI embedding model to convert text into vector representations. The embedding function is integrated into the MongoDB indexing configuration.

### Graph Workflow

1. The workflow begins at the `START` node, which immediately transitions to the `supervisor_node`.  
2. The supervisor then leverages the `tool_node` (with external MCP tools) and connects with the other agent nodes.  
3. After processing by the individual agent nodes (researcher, writer, quality, publisher), results and episodic memory data are stored in MongoDB via the compiled graph.

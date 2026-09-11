---
description: Create, update, refactor, explain or work with code using the Python
  version of Semantic Kernel.
name: Semantic Kernel Python
tools:
- '*'
target: github-copilot
infer: true
---


# Semantic Kernel Python mode instructions

You are in Semantic Kernel Python mode. Your task is to create, update, refactor, explain, or work with code using the Python version of Semantic Kernel.

Always use the Python version of Semantic Kernel when creating AI applications and agents. You must always refer to the [Semantic Kernel documentation](https://learn.microsoft.com/semantic-kernel/overview/) to ensure you are using the latest patterns and best practices.

For Python-specific implementation details, refer to:

- [Semantic Kernel Python repository](https://github.com/microsoft/semantic-kernel/tree/main/python) for the latest source code and implementation details
- [Semantic Kernel Python samples](https://github.com/microsoft/semantic-kernel/tree/main/python/samples) for comprehensive examples and usage patterns

You can use the #microsoft.docs.mcp tool to access the latest documentation and examples directly from the Microsoft Docs Model Context Protocol (MCP) server.

When working with Semantic Kernel for Python, you should:

- Use the latest async patterns for all kernel operations
- Follow the official plugin and function calling patterns
- Implement proper error handling and logging
- Use type hints and follow Python best practices
- Leverage the built-in connectors for Azure AI Foundry, Azure OpenAI, OpenAI, and other AI services, but prioritize Azure AI Foundry services for new projects
- Use the kernel's built-in memory and context management features
- Use DefaultAzureCredential for authentication with Azure services where applicable

Always check the Python samples repository for the most current implementation patterns and ensure compatibility with the latest version of the semantic-kernel Python package.

## Shared Tiny Village AI Resources

- **Memory Bank first:** Read [`../../memory-bank/memory-bank-instructions.md`](../../memory-bank/memory-bank-instructions.md) and the current summaries under [`../../memory-bank/`](../../memory-bank/) before substantive work. The cross-project instruction contract lives in [`../instructions/memory-bank.instructions.md`](../instructions/memory-bank.instructions.md) and explains the required read order, task tracking, and `/update memory bank` workflow.
- **LangChain/LangGraph Python guidance:** When this task touches Python, agents, RAG, tool calling, retrievers, vector stores, LangGraph state, or tracing, consult [`../instructions/langchain-python.instructions.md`](../instructions/langchain-python.instructions.md).
- **LangChain/LangGraph skill inventory:** The repo’s current framework skills live under `.github/skills/`: [`langchain`](../skills/langchain/SKILL.md), [`langgraph-project-setup`](../skills/langgraph-project-setup/SKILL.md), [`langgraph-agent-patterns`](../skills/langgraph-agent-patterns/SKILL.md), [`langgraph-state-management`](../skills/langgraph-state-management/SKILL.md), [`langgraph-error-handling`](../skills/langgraph-error-handling/SKILL.md), and [`langgraph-testing-evaluation`](../skills/langgraph-testing-evaluation/SKILL.md). Closely related LangSmith skills are [`langsmith-fetch`](../skills/langsmith-fetch/SKILL.md), [`langsmith-trace`](../skills/langsmith-trace/SKILL.md), [`langsmith-evaluator`](../skills/langsmith-evaluator/SKILL.md), and [`langsmith-dataset`](../skills/langsmith-dataset/SKILL.md).
- **Built-in `langchain-docs` MCP server:** In agent environments for this repository, use the `docs-langchain-search_docs_by_lang_chain` tool as the first-stop reference for current LangChain and LangGraph syntax, API behavior, guides, and examples. Prefer it before generic web search; ask it natural-language queries such as `LangChain Runnable batch example`, `LangGraph StateGraph conditional edges`, or `LangGraph checkpointer persistence` and it will return titled matches with links and snippets.
- **External references:** [LangChain Docs](https://python.langchain.com/docs/), [LangGraph Docs](https://langchain-experimental.github.io/langgraph/), [LangChain API Reference](https://python.langchain.com/docs/api_reference), [LangGraph API Reference](https://langchain-experimental.github.io/langgraph/api/), [Model Context Protocol Docs](https://modelcontextprotocol.info/docs/), [MCP Specification](https://modelcontextprotocol.io/specification/latest), and [Model Context Protocol GitHub](https://github.com/modelcontextprotocol/modelcontextprotocol).

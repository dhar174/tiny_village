- 🧠 Read `/memory-bank/memory-bank-instructions.md` first.
- 🗂 Load all `/memory-bank/*.md` files before each task.
- 📂 Also load files from the active feature folder (e.g. `/memory-bank/authentication/`).
- 🚦 Follow the Kiro-Lite workflow: PRD → Design → Tasks → Code.
- 🔒 Follow rules in `copilot-rules.md`.
- 📝 On "/update memory bank", refresh activeContext.md & progress.md.

When writing tests, do NOT over-mock or fake classes if avoidable. Write tests so that they will fail if the function does not work as expected, do NOT design tests so that they will pass regardless! Good tests fail when there is an error, NEVER manipulate the test design to make it pass while the tested function does not function as expected!

BE CAREFUL AND CONSERVATIVE about creating fake or mock classes as this may not correctly test the functions.

Also, be cautious in test design, only design tests to accurately test functions, do NOT design tests meant to pass even if the function isn't doing exactly what it should do! In other words, don't design tests to pass, design tests that will only pass if the tested code works as intended and fail otherwise.

## Shared repo AI resources

- **Memory Bank:** The Memory Bank is the repo’s persistent context layer for contributors and AI agents. Read [`../memory-bank/memory-bank-instructions.md`](../memory-bank/memory-bank-instructions.md) plus the current summaries in [`../memory-bank/`](../memory-bank/) before substantive work. Also consult [`instructions/memory-bank.instructions.md`](instructions/memory-bank.instructions.md) for the required read order, task tracking expectations, and `/update memory bank` behavior.
- **LangChain/LangGraph Python instructions:** For Python work involving LangChain, LangGraph, agents, RAG, tool calling, retrievers, vector stores, or tracing, load [`instructions/langchain-python.instructions.md`](instructions/langchain-python.instructions.md).
- **LangChain/LangGraph skills:** This clone’s framework skills currently live under [`skills/`](skills/) inside `.github`: [`langchain`](skills/langchain/SKILL.md), [`langgraph-project-setup`](skills/langgraph-project-setup/SKILL.md), [`langgraph-agent-patterns`](skills/langgraph-agent-patterns/SKILL.md), [`langgraph-state-management`](skills/langgraph-state-management/SKILL.md), [`langgraph-error-handling`](skills/langgraph-error-handling/SKILL.md), and [`langgraph-testing-evaluation`](skills/langgraph-testing-evaluation/SKILL.md). Related LangSmith skills are [`langsmith-fetch`](skills/langsmith-fetch/SKILL.md), [`langsmith-trace`](skills/langsmith-trace/SKILL.md), [`langsmith-evaluator`](skills/langsmith-evaluator/SKILL.md), and [`langsmith-dataset`](skills/langsmith-dataset/SKILL.md).
- **Built-in `langchain-docs` MCP server:** In agent environments, use `docs-langchain-search_docs_by_lang_chain` as the first-stop reference for current LangChain/LangGraph documentation, API usage, guides, and examples. Prefer it before generic web search; use natural-language queries that name the task or API you need, and expect titled results with links and snippets.
- **External references:** [LangChain Docs](https://python.langchain.com/docs/), [LangGraph Docs](https://langchain-experimental.github.io/langgraph/), [LangChain API Reference](https://python.langchain.com/docs/api_reference), [LangGraph API Reference](https://langchain-experimental.github.io/langgraph/api/), [Model Context Protocol Docs](https://modelcontextprotocol.info/docs/), [MCP Specification](https://modelcontextprotocol.io/specification/latest), and [Model Context Protocol GitHub](https://github.com/modelcontextprotocol/modelcontextprotocol).

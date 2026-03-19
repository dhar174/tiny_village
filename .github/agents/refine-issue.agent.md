---
description: 'Refine the requirement or issue with Acceptance Criteria, Technical Considerations, Edge Cases, and NFRs'
name: 'Refine Requirement or Issue'
tools: [ 'list_issues','githubRepo', 'search', 'add_issue_comment','create_issue','create_issue_comment','update_issue','delete_issue','get_issue', 'search_issues']
---

# Refine Requirement or Issue Chat Mode

When activated, this mode allows GitHub Copilot to analyze an existing issue and enrich it with structured details including:

- Detailed description with context and background
- Acceptance criteria in a testable format
- Technical considerations and dependencies
- Potential edge cases and risks
- Expected NFR (Non-Functional Requirements)

## Steps to Run
1. Read the issue description and understand the context.
2. Modify the issue description to include more details.
3. Add acceptance criteria in a testable format.
4. Include technical considerations and dependencies.
5. Add potential edge cases and risks.
6. Provide suggestions for effort estimation.
7. Review the refined requirement and make any necessary adjustments.

## Usage

To activate Requirement Refinement mode:

1. Refer an existing issue in your prompt as `refine <issue_URL>`
2. Use the mode: `refine-issue`

## Output

Copilot will modify the issue description and add structured details to it.

## Shared Tiny Village AI Resources

- **Memory Bank first:** Read [`../../memory-bank/memory-bank-instructions.md`](../../memory-bank/memory-bank-instructions.md) and the current summaries under [`../../memory-bank/`](../../memory-bank/) before substantive work. The cross-project instruction contract lives in [`../instructions/memory-bank.instructions.md`](../instructions/memory-bank.instructions.md) and explains the required read order, task tracking, and `/update memory bank` workflow.
- **LangChain/LangGraph Python guidance:** When this task touches Python, agents, RAG, tool calling, retrievers, vector stores, LangGraph state, or tracing, consult [`../instructions/langchain-python.instructions.md`](../instructions/langchain-python.instructions.md).
- **LangChain/LangGraph skill inventory:** The repo’s current framework skills live under `.github/skills/`: [`langchain`](../skills/langchain/SKILL.md), [`langgraph-project-setup`](../skills/langgraph-project-setup/SKILL.md), [`langgraph-agent-patterns`](../skills/langgraph-agent-patterns/SKILL.md), [`langgraph-state-management`](../skills/langgraph-state-management/SKILL.md), [`langgraph-error-handling`](../skills/langgraph-error-handling/SKILL.md), and [`langgraph-testing-evaluation`](../skills/langgraph-testing-evaluation/SKILL.md). Closely related LangSmith skills are [`langsmith-fetch`](../skills/langsmith-fetch/SKILL.md), [`langsmith-trace`](../skills/langsmith-trace/SKILL.md), [`langsmith-evaluator`](../skills/langsmith-evaluator/SKILL.md), and [`langsmith-dataset`](../skills/langsmith-dataset/SKILL.md).
- **Built-in `langchain-docs` MCP server:** In agent environments for this repository, use the `docs-langchain-search_docs_by_lang_chain` tool as the first-stop reference for current LangChain and LangGraph syntax, API behavior, guides, and examples. Prefer it before generic web search; ask it natural-language queries such as `LangChain Runnable batch example`, `LangGraph StateGraph conditional edges`, or `LangGraph checkpointer persistence` and it will return titled matches with links and snippets.
- **External references:** [LangChain Docs](https://python.langchain.com/docs/), [LangGraph Docs](https://langchain-experimental.github.io/langgraph/), [LangChain API Reference](https://python.langchain.com/docs/api_reference), [LangGraph API Reference](https://langchain-experimental.github.io/langgraph/api/), [Model Context Protocol Docs](https://modelcontextprotocol.info/docs/), [MCP Specification](https://modelcontextprotocol.io/specification/latest), and [Model Context Protocol GitHub](https://github.com/modelcontextprotocol/modelcontextprotocol).

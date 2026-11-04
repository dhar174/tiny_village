import unittest
from pathlib import Path
import sys
import os

# Add the scripts directory to path to import the generator functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.github/skills/langgraph-agent-patterns/scripts')))

from generate_supervisor_graph import generate_python_supervisor, generate_typescript_supervisor

class TestGenerateSupervisorGraph(unittest.TestCase):
    def setUp(self):
        self.graph_name = "test-graph"
        self.subagents = ["researcher", "writer"]

    def test_generate_python_supervisor_contains_llm_logic(self):
        output = generate_python_supervisor(self.graph_name, self.subagents)

        # Check for LLM initialization
        self.assertIn('from langchain_openai import ChatOpenAI', output)
        self.assertIn('model = ChatOpenAI(model="gpt-4")', output)

        # Check for chain invocation
        self.assertIn('chain = prompt | model', output)
        self.assertIn('response = chain.invoke({"messages": messages})', output)

        # Check for validation logic
        self.assertIn('next_agent = response.content.strip()', output)
        self.assertIn('valid_options = ["researcher", "writer", "FINISH"]', output)
        self.assertIn('if next_agent not in valid_options:', output)
        self.assertIn('next_agent = "FINISH"', output)

    def test_generate_typescript_supervisor_contains_llm_logic(self):
        output = generate_typescript_supervisor(self.graph_name, self.subagents)

        # Check for LLM initialization
        self.assertIn('import { ChatOpenAI } from "@langchain/openai";', output)
        self.assertIn('const model = new ChatOpenAI({ model: "gpt-4" });', output)

        # Check for chain invocation
        self.assertIn('const chain = prompt.pipe(model);', output)
        self.assertIn('const response = await chain.invoke({ messages });', output)

        # Check for validation logic
        self.assertIn('let nextAgent = response.content.trim();', output)
        self.assertIn('const validOptions = ["researcher", "writer", "FINISH"];', output)
        self.assertIn('if (!validOptions.includes(nextAgent))', output)
        self.assertIn('nextAgent = "FINISH";', output)

if __name__ == '__main__':
    unittest.main()

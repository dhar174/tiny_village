#!/usr/bin/env python3
"""
Isolated test suite for LLM integration features.
Tests only the new classes without importing the full module.
"""

import unittest
from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation with the LLM."""
    timestamp: str
    character_name: str
    prompt: str
    response: str
    action_taken: Optional[str] = None
    outcome: Optional[str] = None


@dataclass
class FewShotExample:
    """Represents a few-shot learning example for prompt enhancement."""
    situation_context: str
    character_state: Dict[str, Any]
    decision_made: str
    outcome: str
    success_rating: float  # 0.0 to 1.0


class ConversationHistory:
    """Manages multi-turn conversation history for LLM interactions."""
    
    def __init__(self, max_history_length: int = 10):
        self.max_history_length = max_history_length
        self.turns = []
        
    def add_turn(self, character_name: str, prompt: str, response: str, 
                 action_taken: str = None, outcome: str = None):
        """Add a new conversation turn to the history."""
        turn = ConversationTurn(
            timestamp=datetime.now().isoformat(),
            character_name=character_name,
            prompt=prompt,
            response=response,
            action_taken=action_taken,
            outcome=outcome
        )
        self.turns.append(turn)
        
        # Keep only the most recent turns
        if len(self.turns) > self.max_history_length:
            self.turns = self.turns[-self.max_history_length:]
            
    def get_recent_context(self, character_name: str, num_turns: int = 3):
        """Get recent conversation turns for a specific character."""
        character_turns = [turn for turn in self.turns if turn.character_name == character_name]
        return character_turns[-num_turns:] if character_turns else []
        
    def format_context_for_prompt(self, character_name: str, num_turns: int = 3) -> str:
        """Format recent conversation history for inclusion in prompts."""
        recent_turns = self.get_recent_context(character_name, num_turns)
        if not recent_turns:
            return ""
            
        context_lines = ["Previous conversation context:"]
        for i, turn in enumerate(recent_turns, 1):
            context_lines.append(f"Turn {i}:")
            context_lines.append(f"  Decision: {turn.action_taken or 'Unknown'}")
            if turn.outcome:
                context_lines.append(f"  Outcome: {turn.outcome}")
                
        return "\n".join(context_lines) + "\n"


class FewShotExampleManager:
    """Manages few-shot learning examples for prompt enhancement."""
    
    def __init__(self):
        self.examples = []
        self._populate_default_examples()
        
    def _populate_default_examples(self):
        """Add some default few-shot examples."""
        default_examples = [
            FewShotExample(
                situation_context="Character was very hungry (8/10) with low money",
                character_state={"hunger": 8, "money": 2, "energy": 6},
                decision_made="buy_food",
                outcome="Successfully bought bread, hunger reduced to 4/10",
                success_rating=0.9
            ),
            FewShotExample(
                situation_context="Character had low energy (2/10) but good health",
                character_state={"energy": 2, "health": 8, "hunger": 4},
                decision_made="sleep",
                outcome="Energy restored to 9/10, felt much better",
                success_rating=0.95
            ),
        ]
        self.examples.extend(default_examples)
        
    def add_example(self, example: FewShotExample):
        """Add a new few-shot example."""
        self.examples.append(example)
        
    def get_relevant_examples(self, character_state: Dict[str, Any], max_examples: int = 2):
        """Get relevant examples based on character state similarity."""
        if not self.examples:
            return []
            
        # Simple relevance scoring based on state similarity
        scored_examples = []
        for example in self.examples:
            score = self._calculate_relevance_score(character_state, example.character_state)
            scored_examples.append((score, example))
            
        # Sort by relevance and success rating
        scored_examples.sort(key=lambda x: (x[0], x[1].success_rating), reverse=True)
        
        return [example for _, example in scored_examples[:max_examples]]
        
    def _calculate_relevance_score(self, current_state: Dict[str, Any], example_state: Dict[str, Any]) -> float:
        """Calculate how relevant an example is to the current state."""
        if not current_state or not example_state:
            return 0.0
            
        common_keys = set(current_state.keys()) & set(example_state.keys())
        if not common_keys:
            return 0.0
            
        total_similarity = 0.0
        for key in common_keys:
            current_val = float(current_state.get(key, 0))
            example_val = float(example_state.get(key, 0))
            
            # Calculate similarity (closer values = higher similarity)
            max_val = max(abs(current_val), abs(example_val), 1.0)
            difference = abs(current_val - example_val)
            similarity = 1.0 - (difference / max_val)
            total_similarity += similarity
            
        return total_similarity / len(common_keys)
        
    def format_examples_for_prompt(self, examples) -> str:
        """Format few-shot examples for inclusion in prompts."""
        if not examples:
            return ""
            
        lines = ["Examples of past successful decisions:"]
        for i, example in enumerate(examples, 1):
            lines.append(f"Example {i}:")
            lines.append(f"  Situation: {example.situation_context}")
            lines.append(f"  Decision: {example.decision_made}")
            lines.append(f"  Result: {example.outcome}")
            lines.append("")
            
        return "\n".join(lines)


class OutputSchema:
    """Defines structured output schemas for LLM responses."""
    
    @staticmethod
    def get_decision_schema() -> str:
        """Get JSON schema for decision-making responses."""
        return '''Expected response format (JSON):
{
    "reasoning": "Brief explanation of why this decision was made",
    "action": "chosen_action_name",
    "confidence": 0.8,
    "expected_outcome": "What you expect to happen",
    "priority_needs": ["need1", "need2"]
}'''
    
    @staticmethod  
    def get_routine_schema() -> str:
        """Get schema for daily routine responses."""
        return '''Expected response format:
**Decision:** [chosen_action]
**Reasoning:** [why this action was chosen]
**Expected Impact:** [how this will help with current needs]
**Priority:** [high/medium/low]'''
        
    @staticmethod
    def get_crisis_schema() -> str:
        """Get schema for crisis response."""
        return '''Expected response format:
**Immediate Action:** [urgent action to take]
**Reasoning:** [why this is the best response]
**Follow-up:** [what to do next]
**Resources Needed:** [any help or items required]'''


class TestConversationHistory(unittest.TestCase):
    """Test conversation history management."""
    
    def setUp(self):
        self.history = ConversationHistory(max_history_length=5)
        
    def test_add_turn(self):
        """Test adding conversation turns."""
        self.history.add_turn("Alice", "What should I do?", "I choose to eat", "eat_food", "hunger reduced")
        
        self.assertEqual(len(self.history.turns), 1)
        turn = self.history.turns[0]
        self.assertEqual(turn.character_name, "Alice")
        self.assertEqual(turn.action_taken, "eat_food")
        self.assertEqual(turn.outcome, "hunger reduced")
        
    def test_max_history_length(self):
        """Test that history respects maximum length."""
        for i in range(10):
            self.history.add_turn(f"Char{i}", f"Prompt{i}", f"Response{i}")
            
        # Should only keep the last 5 turns
        self.assertEqual(len(self.history.turns), 5)
        self.assertEqual(self.history.turns[0].character_name, "Char5")
        self.assertEqual(self.history.turns[-1].character_name, "Char9")
        
    def test_get_recent_context(self):
        """Test retrieving recent context for specific character."""
        # Add turns for different characters
        self.history.add_turn("Alice", "Prompt1", "Response1", "action1")
        self.history.add_turn("Bob", "Prompt2", "Response2", "action2")
        self.history.add_turn("Alice", "Prompt3", "Response3", "action3")
        
        alice_context = self.history.get_recent_context("Alice", 2)
        self.assertEqual(len(alice_context), 2)
        self.assertEqual(alice_context[0].action_taken, "action1")
        self.assertEqual(alice_context[1].action_taken, "action3")
        
    def test_format_context_for_prompt(self):
        """Test formatting context for prompt inclusion."""
        self.history.add_turn("Alice", "What to do?", "I choose sleep", "sleep", "energy restored")
        
        formatted = self.history.format_context_for_prompt("Alice")
        self.assertIn("Previous conversation context:", formatted)
        self.assertIn("Decision: sleep", formatted)
        self.assertIn("Outcome: energy restored", formatted)


class TestFewShotExampleManager(unittest.TestCase):
    """Test few-shot learning example management."""
    
    def setUp(self):
        self.manager = FewShotExampleManager()
        
    def test_add_example(self):
        """Test adding new examples."""
        initial_count = len(self.manager.examples)
        
        example = FewShotExample(
            situation_context="Test situation",
            character_state={"hunger": 8, "energy": 3},
            decision_made="eat_food",
            outcome="Felt better",
            success_rating=0.9
        )
        self.manager.add_example(example)
        
        self.assertEqual(len(self.manager.examples), initial_count + 1)
        
    def test_get_relevant_examples(self):
        """Test getting relevant examples based on character state."""
        # Add a specific example
        example = FewShotExample(
            situation_context="Very hungry character",
            character_state={"hunger": 8, "energy": 5, "money": 10},
            decision_made="buy_food",
            outcome="Hunger satisfied",
            success_rating=0.95
        )
        self.manager.add_example(example)
        
        # Test with similar state
        current_state = {"hunger": 7, "energy": 6, "money": 8}
        relevant = self.manager.get_relevant_examples(current_state, max_examples=1)
        
        self.assertGreater(len(relevant), 0)
        
    def test_calculate_relevance_score(self):
        """Test relevance scoring algorithm."""
        current_state = {"hunger": 8, "energy": 3}
        example_state = {"hunger": 9, "energy": 2}
        
        score = self.manager._calculate_relevance_score(current_state, example_state)
        
        # Should be high similarity (both states are quite similar)
        self.assertGreater(score, 0.7)
        
    def test_format_examples_for_prompt(self):
        """Test formatting examples for prompt inclusion."""
        example = FewShotExample(
            situation_context="Test situation",
            character_state={"hunger": 8},
            decision_made="eat_food",
            outcome="Hunger reduced",
            success_rating=0.9
        )
        
        formatted = self.manager.format_examples_for_prompt([example])
        self.assertIn("Examples of past successful decisions:", formatted)
        self.assertIn("Test situation", formatted)
        self.assertIn("eat_food", formatted)
        self.assertIn("Hunger reduced", formatted)


class TestOutputSchema(unittest.TestCase):
    """Test structured output schema generation."""
    
    def test_decision_schema(self):
        """Test decision schema format."""
        schema = OutputSchema.get_decision_schema()
        
        self.assertIn("JSON", schema)
        self.assertIn("reasoning", schema)
        self.assertIn("action", schema)
        self.assertIn("confidence", schema)
        
    def test_routine_schema(self):
        """Test routine schema format."""
        schema = OutputSchema.get_routine_schema()
        
        self.assertIn("Decision:", schema)
        self.assertIn("Reasoning:", schema)
        self.assertIn("Expected Impact:", schema)
        
    def test_crisis_schema(self):
        """Test crisis schema format."""
        schema = OutputSchema.get_crisis_schema()
        
        self.assertIn("Immediate Action:", schema)
        self.assertIn("Follow-up:", schema)
        self.assertIn("Resources Needed:", schema)


if __name__ == "__main__":
    unittest.main()
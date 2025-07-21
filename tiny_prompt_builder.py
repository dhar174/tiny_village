"""Prompt generation utilities for Tiny Village characters.

This module forms the core of the Tiny Village *prompt system*.  It contains
helper classes for evaluating a character's current needs and available actions
and, most importantly, the :class:`PromptBuilder`.  ``PromptBuilder`` combines
character state with world context to create textual prompts which are then
passed to the language model via :mod:`tiny_brain_io`.  Responses from the LLM
are interpreted using :mod:`tiny_output_interpreter`.  No networking occurs in
this module; it purely formats information for those other components.
"""

import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

import tiny_characters as tc


class ContextManager:
    """Systematically gathers and organizes relevant context for prompt generation.
    
    This class ensures consistent prompt structure and comprehensive information
    across all prompt types by centralizing context assembly logic.
    """
    
    def __init__(self, character, memory_manager=None):
        """Initialize context manager for a character.
        
        Args:
            character: The character to gather context for
            memory_manager: Optional memory manager for retrieving memories
        """
        self.character = character
        self.memory_manager = memory_manager
        
    def gather_character_context(self):
        """Gather comprehensive character context information.
        
        Returns:
            Dictionary containing organized character context
        """
        context = {
            'basic_info': {
                'name': self.character.name,
                'job': getattr(self.character, 'job', 'Unknown'),
                'personality_traits': getattr(self.character, 'personality_traits', {}),
            },
            'current_state': {
                'health_status': getattr(self.character, 'health_status', 5),
                'hunger_level': getattr(self.character, 'hunger_level', 5),
                'mental_health': getattr(self.character, 'mental_health', 5),
                'social_wellbeing': getattr(self.character, 'social_wellbeing', 5),
                'energy': getattr(self.character, 'energy', 5),
                'wealth_money': getattr(self.character, 'wealth_money', 0),
            },
            'motivations': {
                'long_term_goal': getattr(self.character, 'long_term_goal', None),
                'recent_event': getattr(self.character, 'recent_event', 'default'),
            },
            'inventory': {
                'food_items': getattr(self.character, 'inventory', {})
            }
        }
        
        # Add motives if available
        if hasattr(self.character, 'motives') and self.character.motives:
            try:
                context['motives'] = self._extract_character_motives()
            except Exception:
                context['motives'] = {}
                
        return context
        
    def gather_environmental_context(self, time: str, weather: str):
        """Gather environmental context for prompts.
        
        Args:
            time: Current time description
            weather: Current weather description
            
        Returns:
            Dictionary containing environmental context
        """
        return {
            'time': time,
            'weather': weather,
            'time_formatted': f"it's {time}",
            'weather_formatted': descriptors.get_weather_description(weather)
        }
        
    def gather_memory_context(self, query: str, max_memories: int = 3):
        """Gather relevant memories for the given context query.
        
        Args:
            query: Query string to find relevant memories
            max_memories: Maximum number of memories to return
            
        Returns:
            List of relevant memory objects
        """
        if not self.memory_manager:
            return []
            
        try:
            # Use the memory manager to search for relevant memories
            memory_results = self.memory_manager.search_memories(query)
            
            # Extract top memories based on relevance score
            if isinstance(memory_results, dict):
                sorted_memories = sorted(
                    memory_results.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                return [memory for memory, score in sorted_memories[:max_memories]]
            elif isinstance(memory_results, list):
                return memory_results[:max_memories]
            else:
                return []
        except Exception:
            return []
            
    def gather_goal_context(self):
        """Gather character goal and priority context.
        
        Returns:
            Dictionary containing goal information and priorities
        """
        context = {
            'active_goals': [],
            'goal_priorities': {},
            'needs_priorities': {}
        }
        
        # Get active goals if character has goal evaluation capability
        try:
            if hasattr(self.character, 'evaluate_goals'):
                goal_queue = self.character.evaluate_goals()
                context['active_goals'] = goal_queue[:3]  # Top 3 goals
        except Exception:
            pass
            
        # Calculate needs priorities
        try:
            needs_calculator = NeedsPriorities()
            context['needs_priorities'] = needs_calculator.calculate_needs_priorities(self.character)
        except Exception:
            pass
            
        return context
        
    def assemble_complete_context(self, time: str, weather: str, 
                                  memory_query=None):
        """Assemble all context types into a comprehensive context object.
        
        Args:
            time: Current time description
            weather: Current weather description  
            memory_query: Optional query for relevant memories
            
        Returns:
            Complete context dictionary for prompt generation
        """
        context = {
            'character': self.gather_character_context(),
            'environment': self.gather_environmental_context(time, weather),
            'goals': self.gather_goal_context(),
            'memories': [],
            'timestamp': datetime.now().isoformat(),
        }
        
        # Add memory context if query provided
        if memory_query:
            context['memories'] = self.gather_memory_context(memory_query)
            
        return context
        
    def _extract_character_motives(self):
        """Extract character motives as a dictionary.
        
        Returns:
            Dictionary mapping motive names to scores
        """
        try:
            motives_dict = self.character.motives.to_dict()
            return {name: motive.score for name, motive in motives_dict.items()}
        except Exception:
            return {}


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
        self.turns: List[ConversationTurn] = []
        
    def add_turn(self, character_name: str, prompt: str, response: str, 
                 action_taken: Optional[str] = None, outcome: Optional[str] = None):
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
            
    def get_recent_context(self, character_name: str, num_turns: int = 3) -> List[ConversationTurn]:
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
        self.examples: List[FewShotExample] = []
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
            FewShotExample(
                situation_context="Character needed social interaction, low social wellbeing",
                character_state={"social_wellbeing": 3, "energy": 7, "mental_health": 5},
                decision_made="social_visit",
                outcome="Visited friend, social wellbeing increased to 7/10",
                success_rating=0.85
            )
        ]
        self.examples.extend(default_examples)
        
    def add_example(self, example: FewShotExample):
        """Add a new few-shot example."""
        self.examples.append(example)
        
    def get_relevant_examples(self, character_state: Dict[str, Any], max_examples: int = 2) -> List[FewShotExample]:
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
        
    def format_examples_for_prompt(self, examples: List[FewShotExample]) -> str:
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


class NeedsPriorities:
    """Utility object for scoring how pressing each of a character's needs is."""

    def __init__(self) -> None:
        """Initialise the list of tracked needs and their default priority values."""

        self.needs = [
            "health",
            "hunger",
            "wealth",
            "mental_health",
            "social_wellbeing",
            "happiness",
            "shelter",
            "stability",
            "luxury",
            "hope",
            "success",
            "control",
            "job_performance",
            "beauty",
            "community",
            "material_goods",
            "friendship_grid",
        ]
        # Value represents a character's current need level
        self.needs_priorities: Dict[str, float] = {
            "health": 0,
            "hunger": 0,
            "wealth": 0,
            "mental_health": 0,
            "social_wellbeing": 0,
            "happiness": 0,
            "shelter": 0,
            "stability": 0,
            "luxury": 0,
            "hope": 0,
            "success": 0,
            "control": 0,
            "job_performance": 0,
            "beauty": 0,
            "community": 0,
            "material_goods": 0,
            "friendship_grid": 0,
        }

    def get_needs_priorities(self) -> Dict[str, float]:
        """Return the current mapping of needs to priority scores."""

        return self.needs_priorities

    def get_needs_priorities_list(self) -> dict_keys:
        """Return a view of names of tracked needs."""

        return self.needs_priorities.keys()

    def get_needs_priorities_values(self) -> dict_values[float]:
        """Return a view of priority scores in their current order."""

        return self.needs_priorities.values()

    def get_needs_priorities_sorted(self) -> List[tuple]:
        """Return need/priority pairs sorted from lowest to highest priority."""

        return sorted(self.needs_priorities.items(), key=lambda x: x[1])

    def get_needs_priorities_sorted_list(self) -> List[str]:
        """Return need names sorted from lowest to highest priority."""

        return [name for name, _ in self.get_needs_priorities_sorted()]

    def get_needs_priorities_sorted_values(self) -> List[float]:
        """Return priority scores sorted from lowest to highest."""

        return [score for _, score in self.get_needs_priorities_sorted()]

    def get_needs_priorities_sorted_reverse(self) -> List[tuple]:
        """Return need/priority pairs sorted from highest to lowest."""

        return sorted(self.needs_priorities.items(), key=lambda x: x[1], reverse=True)

    def get_needs_priorities_sorted_list_reverse(self) -> List[str]:
        """Return need names sorted from highest to lowest priority."""

        return [name for name in sorted(self.needs_priorities, key=self.needs_priorities.get, reverse=True)]

    def get_needs_priorities_sorted_values_reverse(self) -> List[float]:
        """Return priority scores sorted from highest to lowest."""

        return [score for _, score in self.get_needs_priorities_sorted_reverse()]

    def get_needs_priorities_sorted_by_value(self) -> List[tuple]:
        """Alias for :meth:`get_needs_priorities_sorted` for backward compatibility."""

        return self.get_needs_priorities_sorted()

    def set_needs_priorities(self, needs_priorities: Dict[str, float]) -> None:
        """Replace the internal priority mapping with ``needs_priorities``."""

        self.needs_priorities = needs_priorities

    def calculate_health_priority(self, character: tc.Character) -> float:
        """Return a priority score (0-100) representing how urgently the
        character needs medical attention."""

        health_status = character.get_health_status()
        health_priority = (
            100 - (health_status * 10)
        ) + character.get_motives().get_health_motive().get_score()
        return health_priority

    def calculate_hunger_priority(self, character: tc.Character) -> float:
        """Return a priority score for the character's hunger level."""

        hunger_level = character.get_hunger_level()
        hunger_priority = (
            hunger_level * 10 + character.get_motives().get_hunger_motive().get_score()
        )
        return hunger_priority

    def calculate_wealth_priority(self, character: tc.Character) -> float:
        """Return a priority score based on the character's financial state."""

        wealth = character.get_wealth()
        wealth_priority = character.get_motives().get_wealth_motive().get_score()
        return wealth_priority

    def calculate_mental_health_priority(self, character: tc.Character) -> float:
        """Return a priority score for improving mental wellness."""

        mental_health = character.get_mental_health()
        mental_health_priority = character.get_motives().get_mental_health_motive().get_score()
        return mental_health_priority

    def calculate_social_wellbeing_priority(self, character: tc.Character) -> float:
        """Return a priority score reflecting the need for social interaction."""

        social_wellbeing = character.get_social_wellbeing()
        social_wellbeing_priority = (
            character.get_motives().get_social_wellbeing_motive().get_score()
        )
        return social_wellbeing_priority

    def calculate_happiness_priority(self, character: tc.Character) -> float:
        """Return a priority score indicating how much the character seeks happiness."""

        happiness = character.get_happiness()
        happiness_priority = character.get_motives().get_happiness_motive().get_score()
        return happiness_priority

    def calculate_shelter_priority(self, character: tc.Character) -> float:
        """Return a priority score describing the need for shelter/housing."""

        shelter = character.get_shelter()
        shelter_priority = character.get_motives().get_shelter_motive().get_score()
        return shelter_priority

    def calculate_stability_priority(self, character: tc.Character) -> float:
        """Return a priority score representing the desire for routine and stability."""

        stability = character.get_stability()
        stability_priority = character.get_motives().get_stability_motive().get_score()
        return stability_priority

    def calculate_luxury_priority(self, character: tc.Character) -> float:
        """Return a priority score representing the desire for luxury items or comfort."""

        luxury = character.get_luxury()
        luxury_priority = character.get_motives().get_luxury_motive().get_score()
        return luxury_priority

    def calculate_hope_priority(self, character: tc.Character) -> float:
        """Return a priority score representing the character's need for optimism."""

        hope = character.get_hope()
        hope_priority = character.get_motives().get_hope_motive().get_score()
        return hope_priority

    def calculate_success_priority(self, character: tc.Character) -> float:
        """Return a priority score for career or personal success."""

        success = character.get_success()
        success_priority = character.get_motives().get_success_motive().get_score()
        return success_priority

    def calculate_control_priority(self, character: tc.Character) -> float:
        """Return a priority score for the character's sense of personal control."""

        control = character.get_control()
        control_priority = character.get_motives().get_control_motive().get_score()
        return control_priority

    def calculate_job_performance_priority(self, character: tc.Character) -> float:
        """Return a priority score for improving job performance."""

        job_performance = character.get_job_performance()
        job_performance_priority = character.get_motives().get_job_performance_motive().get_score()
        return job_performance_priority

    def calculate_beauty_priority(self, character: tc.Character) -> float:
        """Return a priority score describing desire to improve appearance."""

        beauty = character.get_beauty()
        beauty_priority = character.get_motives().get_beauty_motive().get_score() - beauty
        return beauty_priority

    def calculate_community_priority(self, character: tc.Character) -> float:
        """Return a priority score reflecting the need for community involvement."""

        community = character.get_community()
        community_priority = character.get_motives().get_community_motive().get_score()
        return community_priority

    def calculate_material_goods_priority(self, character: tc.Character) -> float:
        """Return a priority score for acquiring material possessions."""

        material_goods = character.get_material_goods()
        material_goods_priority = character.get_motives().get_material_goods_motive().get_score()
        return material_goods_priority


    def calculate_friendship_grid_priority(self, character: tc.Character):
        # Friendship grid priority is based on social connections and relationships
        # Use social_wellbeing_motive as the base since friendship relates to social wellbeing
        # Calculate aggregate friendship score from the character's friendship grid
        friendship_grid = character.get_friendship_grid()
        
        # Calculate average friendship score from the grid
        if friendship_grid and len(friendship_grid) > 0:
            # Filter out empty dictionaries and calculate average friendship score
            valid_friendships = [f for f in friendship_grid if f and 'friendship_score' in f]
            if valid_friendships:
                avg_friendship_score = sum(f['friendship_score'] for f in valid_friendships) / len(valid_friendships)
                # Convert to 0-10 scale for consistency with other priorities
                # Clamp to reasonable bounds (0-100 friendship score range)
                avg_friendship_score = max(0, min(100, avg_friendship_score))
                friendship_state = avg_friendship_score / 10.0
            else:
                friendship_state = 0  # No valid friendships
        else:
            friendship_state = 0  # No friendship data
        
        # Combine with social wellbeing motive (friendship is social)
        social_motive = character.get_motives().get_social_wellbeing_motive().get_score()
        
        # Calculate priority: higher motive with lower current state = higher priority
        # Ensure priority is always non-negative
        friendship_grid_priority = max(0, social_motive + (10 - friendship_state) * 2)
        return friendship_grid_priority

    def calculate_needs_priorities(self, character: tc.Character) -> Dict[str, float]:
        """Calculate priority values for all needs for ``character``."""

        needs_priorities = {
            "health": self.calculate_health_priority(character),
            "hunger": self.calculate_hunger_priority(character),
            "wealth": self.calculate_wealth_priority(character),
            "mental_health": self.calculate_mental_health_priority(character),
            "social_wellbeing": self.calculate_social_wellbeing_priority(character),
            "happiness": self.calculate_happiness_priority(character),
            "shelter": self.calculate_shelter_priority(character),
            "stability": self.calculate_stability_priority(character),
            "luxury": self.calculate_luxury_priority(character),
            "hope": self.calculate_hope_priority(character),
            "success": self.calculate_success_priority(character),
            "control": self.calculate_control_priority(character),
            "job_performance": self.calculate_job_performance_priority(character),
            "beauty": self.calculate_beauty_priority(character),
            "community": self.calculate_community_priority(character),
            "material_goods": self.calculate_material_goods_priority(character),
            "friendship_grid": self.calculate_friendship_grid_priority(character),
        }

        return needs_priorities


class ActionOptions:
    """List and prioritize the actions a character can perform."""

    def __init__(self) -> None:
        """Initialise the set of known action strings."""

        self.actions = [
            "buy_food",
            "eat_food",
            "improve_job_performance",
            "increase_friendship",
            "improve_mental_health",
            "pursue_hobby",
            "volunteer_time",
            "set_goal",
            "leisure_activity",
            "organize_event",
            "research_new_technology",
            "buy_medicine",
            "take_medicine",
            "visit_doctor",
            "collaborate_colleagues",
            "gather_resource",
            "trade_goods",
            "repair_item",
            "get_educated",
            "social_visit",
            "attend_event",
            "go_to_work",
            "clean_up",
            "invest_wealth",
            "buy_property",
            "sell_property",
            "move_to_new_location",
            "commission_service",
            "start_business",
            "craft_item",
            "work_current_job",
        ]

    def prioritize_actions(self, character: tc.Character) -> List[str]:
        """Return a list of plausible actions ordered by likelihood.

        The ordering is determined using a few heuristic checks on ``character``
        state such as hunger level or available money.
        """

        # char_dict = character.to_dict()
        # inv_dict = character.inventory.to_dict()

        # Sample criteria for prioritizing actions
        needs_goals = {
            "buy_food": character.get_hunger_level() > 7
            and character.get_wealth_money() > 1
            and (
                character.get_inventory().count_food_items_total() < 5
                or character.get_inventory().count_food_calories_total()
                < character.get_hunger_level()
            ),
            "eat_food": character.get_hunger_level() > 5
            and character.get_inventory().count_food_items_total() > 0,
            "visit_doctor": character.get_health_status() < 3
            or character.get_mental_health() < 4,
            "take_medicine": character.get_health_status() < 5,
            "improve_shelter": character.get_shelter() < 4,
            "attend_event": character.get_social_wellbeing() < 5
            or character.get_community() < 5,
            "pursue_hobby": character.get_happiness() < 5 or character.get_beauty() < 5,
            "self_care": character.get_mental_health() < 5,
            "social_visit": character.get_friendship_grid() < 5,
            "volunteer_time": character.get_community() < 5,
            "improve_job_performance": character.get_job_performance() < 5,
            "get_educated": character.get_long_term_goal() == "career_advancement",
            "set_goal": character.get_hope() < 5,
            "start_business": character.get_long_term_goal() == "entrepreneurship",
            "trade_goods": character.get_wealth_money() > 5
            or character.get_material_goods() > 5,
            "invest_wealth": character.get_wealth_money() > 8,
            # ... additional mappings ...
        }

        prioritized_actions = []
        prioritized_actions += [action for action, need in needs_goals.items() if need]
        other_actions = [
            action for action in self.actions if action not in prioritized_actions
        ]
        if len(prioritized_actions) < 5:
            prioritized_actions += other_actions[: 5 - len(prioritized_actions)]
        return prioritized_actions


class ParameterizedTemplateEngine:
    """Dynamic template system that can be modified at runtime based on character personality and game state.
    
    This replaces static descriptor matrices with a flexible template system that supports
    placeholders, template definitions, and runtime parameter substitution.
    """
    
    def __init__(self):
        """Initialize the template engine with base templates."""
        self.templates = {}
        self.parameters = {}
        self.character_context = {}
        self._load_base_templates()
        
    def _load_base_templates(self):
        """Load base template definitions with placeholders."""
        self.templates = {
            "character_intro": "You are {character_name}, a {character_adjective} {character_role}",
            "character_activity": "who enjoys {activity_verb} {activity_object}",
            "current_project": "You are currently working on {current_project} {work_location}",
            "future_plans": "and you are excited to see how it turns out. You are also planning to attend a {event_type} in the next few weeks",
            "event_expectations": "and you are hoping to {event_goal} there",
            "health_status": "You're feeling {health_descriptor}",
            "hunger_status": "and {hunger_descriptor}",
            "recent_events": "{recent_event_prefix}",
            "financial_status": "and {financial_descriptor}",
            "motivation": "{motivation_prefix} {goal_description}",
            "weather_context": "it's {time_period}, and {weather_description}",
            "question_framing": "{question_style}"
        }
        
    def set_character_parameters(self, character, personality_modifier: str = None):
        """Set character-specific parameters for template substitution.
        
        Args:
            character: Character object to extract parameters from
            personality_modifier: Optional personality-based template modifier
        """
        job = getattr(character, 'job', 'person')
        
        # Base character parameters
        self.parameters.update({
            'character_name': character.name,
            'character_role': job.lower(),
            'current_project': self._get_project_for_job(job),
            'work_location': self._get_work_location_for_job(job),
            'event_type': self._get_event_for_job(job),
            'event_goal': self._get_event_goal_for_job(job),
        })
        
        # Personality-based modifications
        if personality_modifier:
            self._apply_personality_modifier(personality_modifier)
            
        # Dynamic adjectives based on character state
        self._set_dynamic_descriptors(character)
        
    def _get_project_for_job(self, job: str) -> str:
        """Get appropriate current project based on job."""
        project_map = {
            'Engineer': ['a new software project', 'a new system design', 'debugging a complex issue'],
            'Farmer': ['a new crop rotation', 'preparing the fields', 'planning the harvest'],
            'Waitress': ['improving customer service', 'learning new recipes', 'organizing the restaurant']
        }
        import random
        return random.choice(project_map.get(job, ['a new project']))
        
    def _get_work_location_for_job(self, job: str) -> str:
        """Get work location for job."""
        location_map = {
            'Engineer': 'at your desk',
            'Farmer': 'on the farm', 
            'Waitress': 'at the restaurant'
        }
        return location_map.get(job, 'at work')
        
    def _get_event_for_job(self, job: str) -> str:
        """Get appropriate event type for job."""
        event_map = {
            'Engineer': ['tech conference', 'hackathon', 'developer meetup'],
            'Farmer': ['farmers market', 'agricultural fair', 'farming conference'],
            'Waitress': ['culinary event', 'service training', 'restaurant expo']
        }
        import random
        return random.choice(event_map.get(job, ['professional event']))
        
    def _get_event_goal_for_job(self, job: str) -> str:
        """Get event goal for job."""
        goal_map = {
            'Engineer': ['network with other developers', 'learn new technologies', 'showcase your work'],
            'Farmer': ['sell your produce', 'learn about new techniques', 'meet other farmers'],
            'Waitress': ['improve your skills', 'learn new recipes', 'meet other service professionals']
        }
        import random
        return random.choice(goal_map.get(job, ['learn something new']))
        
    def _apply_personality_modifier(self, modifier: str):
        """Apply personality-based template modifications."""
        personality_modifiers = {
            'analytical': {
                'character_adjective': ['methodical', 'precise', 'logical'],
                'activity_verb': ['analyzing', 'optimizing', 'systematizing'],
                'question_style': 'What is the most logical next step?'
            },
            'creative': {
                'character_adjective': ['innovative', 'imaginative', 'artistic'],
                'activity_verb': ['creating', 'designing', 'innovating'],
                'question_style': 'What creative solution will you pursue?'
            },
            'social': {
                'character_adjective': ['friendly', 'outgoing', 'collaborative'],
                'activity_verb': ['connecting with', 'helping', 'supporting'],
                'question_style': 'How will you engage with others?'
            }
        }
        
        if modifier in personality_modifiers:
            for key, value in personality_modifiers[modifier].items():
                if isinstance(value, list):
                    import random
                    self.parameters[key] = random.choice(value)
                else:
                    self.parameters[key] = value
                    
    def _set_dynamic_descriptors(self, character):
        """Set descriptors based on current character state."""
        # Health descriptor based on health status
        health = getattr(character, 'health_status', 5)
        if health >= 8:
            self.parameters['health_descriptor'] = 'excellent'
        elif health >= 6:
            self.parameters['health_descriptor'] = 'good'
        elif health >= 4:
            self.parameters['health_descriptor'] = 'okay'
        else:
            self.parameters['health_descriptor'] = 'unwell'
            
        # Hunger descriptor
        hunger = getattr(character, 'hunger_level', 5)
        if hunger >= 8:
            self.parameters['hunger_descriptor'] = 'very hungry'
        elif hunger >= 6:
            self.parameters['hunger_descriptor'] = 'somewhat hungry'
        elif hunger >= 3:
            self.parameters['hunger_descriptor'] = 'satisfied'
        else:
            self.parameters['hunger_descriptor'] = 'full'
            
        # Financial descriptor
        wealth = getattr(character, 'wealth_money', 0)
        if wealth >= 100:
            self.parameters['financial_descriptor'] = 'you are financially comfortable'
        elif wealth >= 50:
            self.parameters['financial_descriptor'] = 'your finances are stable'
        elif wealth >= 10:
            self.parameters['financial_descriptor'] = 'you have some money saved'
        else:
            self.parameters['financial_descriptor'] = 'money is tight'
            
    def set_environmental_parameters(self, time: str, weather: str):
        """Set environmental parameters for templates."""
        self.parameters.update({
            'time_period': time,
            'weather_description': weather
        })
        
    def generate_text(self, template_key: str, additional_params: Dict[str, str] = None) -> str:
        """Generate text from a template with parameter substitution.
        
        Args:
            template_key: Key of the template to use
            additional_params: Additional parameters for substitution
            
        Returns:
            Generated text with parameters substituted
        """
        if template_key not in self.templates:
            return f"[Template '{template_key}' not found]"
            
        template = self.templates[template_key]
        params = self.parameters.copy()
        
        if additional_params:
            params.update(additional_params)
            
        try:
            return template.format(**params)
        except KeyError as e:
            return f"[Missing parameter {e} for template '{template_key}']"
            
    def add_custom_template(self, key: str, template: str):
        """Add a custom template definition.
        
        Args:
            key: Template identifier
            template: Template string with {parameter} placeholders
        """
        self.templates[key] = template
        
    def modify_template(self, key: str, new_template: str):
        """Modify an existing template at runtime.
        
        Args:
            key: Template identifier to modify
            new_template: New template string
        """
        if key in self.templates:
            self.templates[key] = new_template
        else:
            self.add_custom_template(key, new_template)


class DescriptorMatrices:
    """Repository of descriptors used to enrich generated text prompts."""
    def __init__(self):

        self.job_adjective = {
            "default": [
                "skilled",
                "hardworking",
                "friendly",
                "friendly, outgoing",
                "average",
            ]
        }

        self.job_pronoun = {
            "default": ["person"],
            "Engineer": [
                "person",
                "engineer",
                "programmer",
                "developer",
                "coder",
                "software engineer",
                "hardware engineer",
                "computer scientist",
                "computer engineer",
                "computer programmer",
                "computer scientist",
                "computer technician",
                "computer repair technician",
                "computer repairman",
                "computer repairwoman",
                "computer repair person",
                "computer repair specialist",
                "computer repair expert",
                "computer repair professional",
                "computer repair master",
                "computer repair guru",
                "computer repair wizard",
                "computer repair genius",
                "computer repair prodigy",
                "computer repair whiz",
                "computer repair wiz",
                "computer nerd",
                "computer geek",
            ],
            "Farmer": [
                "person",
                "farmer",
                "agriculturalist",
                "agricultural scientist",
                "agricultural engineer",
                "agricultural technician",
                "agricultural nerd",
                "agricultural geek",
            ],
        }

        self.job_place = {
            "default": ["at your job"],
            "Engineer": [""],
            "Farmer": ["at your farm"],
        }

        self.job_enjoys_verb = {
            "default": ["working with", "helping"],
            "Engineer": [
                "building",
                "designing",
                "creating",
                "developing",
                "programming",
                "testing",
                "debugging",
                "fixing",
                "improving",
                "optimizing",
                "learning",
                "teaching",
                "mentoring",
                "leading",
                "managing",
                "collaborating",
                "working",
                "writing",
                "reading",
                "researching",
                "analyzing",
                "planning",
                "documenting",
                "communicating",
                "presenting",
                "speaking",
                "talking",
                "discussing",
                "debating",
                "arguing",
                "solving",
                "simplifying",
                "automating",
                "optimizing",
            ],
            "Farmer": [
                "planting",
                "growing",
                "harvesting",
                "watering",
                "feeding",
                "tending",
                "caring",
                "cultivating",
                "nurturing",
                "pruning",
                "weeding",
                "fertilizing",
                "sowing",
                "reaping",
                "mowing",
                "raking",
                "plowing",
                "tilling",
                "hoeing",
                "digging",
                "shoveling",
                "raking",
            ],
        }

        self.job_verb_acts_on_noun = {
            "default": ["your hands", "others"],
            "Engineer": [
                "things",
                "machines",
                "doo-dads",
                "gizmos",
                "widgets",
                "programs",
                "software",
                "hardware",
                "systems",
                "components",
                "parts",
                "circuits",
                "circuits",
                "devices",
                "solutions",
            ],
            "Farmer": [
                "plants",
                "crops",
                "vegetables",
                "fruits",
                "grains",
                "flowers",
                "trees",
                "shrubs",
                "bushes",
                "grass",
                "weeds",
                "soil",
                "land",
                "fields",
                "gardens",
                "orchards",
                "vineyards",
                "pastures",
                "meadows",
                "ranches",
                "farms",
                "livestock",
                "animals",
                "cattle",
                "pigs",
                "chickens",
                "sheep",
                "goats",
                "horses",
                "llamas",
                "alpacas",
                "ostriches",
                "turkeys",
                "geese",
                "ducks",
                "fish",
                "aquatic life",
                "wildlife",
            ],
        }

        self.job_currently_working_on = {
            "Engineer": [
                "a new project",
                "a new software project",
                "a new hardware project",
                "a new product",
                "a new feature",
                "a new design",
                "a new system",
                "a new solution",
                "a new component",
                "a new part",
                "a new circuit",
                "a new device",
                "a new machine",
                "a new tool",
                "a new program",
                "a new algorithm",
                "a new technology",
                "a new language",
                "a new framework",
                "a new library",
                "a new interface",
                "a new API",
                "a new database",
                "a new website",
                "a new app",
                "a new game",
                "a new tool",
                "a new service",
                "a new business",
                "a new company",
                "a new startup",
                "a new project",
                "a new idea",
                "a new concept",
                "a new invention",
                "a new discovery",
                "a new theory",
                "a new hypothesis",
                "a new experiment",
                "a new method",
                "a new technique",
                "a new approach",
                "a new strategy",
                "a new plan",
                "a new goal",
                "a new objective",
                "a new target",
                "a new milestone",
                "a new task",
                "a new assignment",
                "a new mission",
                "a new quest",
                "a new adventure",
                "a new journey",
                "a new adventure",
                "a new experience",
                "a new opportunity",
                "a new challenge",
                "debugging a new bug",
                "fixing a new defect",
                "solving  a new error",
            ],
            "Farmer": [
                "a new crop",
                "a new harvest",
                "a new field",
                "a new garden",
                "a new orchard",
                "a new vineyard",
                "a new pasture",
                "a new meadow",
                "a new ranch",
                "a new farm",
                "a new livestock",
                "a new animal",
                "a new cattle",
                "a new pig",
                "a new chicken",
                "a new sheep",
                "a new goat",
                "a new horse",
                "a new llama",
                "a new alpaca",
                "a new ostrich",
                "a new turkey",
                "a new goose",
                "a new duck",
                "a new fish",
                "a new aquatic life",
                "a new wildlife",
                "a new plant",
                "a new vegetable",
                "a new fruit",
                "a new grain",
                "a new flower",
                "a new tree",
                "a new shrub",
                "a new bush",
                "a new grass",
                "a new weed",
                "a new soil",
                "a new land",
                "a new field",
                "a new garden",
                "a new orchard",
                "a new vineyard",
                "a new pasture",
                "a new meadow",
                "a new ranch",
                "a new farm",
                "a new livestock",
                "a new animal",
                "a new cattle",
                "a new pig",
                "a new chicken",
                "a new sheep",
                "a new goat",
                "a new horse",
                "a new llama",
                "a new alpaca",
                "a new ostrich",
                "a new turkey",
                "a new goose",
                "a new duck",
                "a new fish",
                "a new aquatic life",
                "a new wildlife",
            ],
        }

        self.job_planning_to_attend = {
            "Engineer": [
                "tech conference",
                "tech meetup",
                "developers conference",
                "maker faire",
                "hackathon",
                "startup conference",
                "tech talk",
                "tech event",
                "tech meetup",
                "tech gathering",
                "tech party",
                "tech event",
                "tech festival",
                "tech expo",
                "tech convention",
                "tech summit",
                "tech fair",
                "tech showcase",
                "tech competition",
            ],
            "Farmer": [
                "farmers market",
                "farmers conference",
                "farmers meetup",
                "farmers convention",
                "farmers fair",
                "farmers showcase",
                "farmers competition",
                "farmers festival",
                "farmers expo",
                "farmers gathering",
                "farmers party",
                "farmers event",
                "farmers summit",
                "farmers fair",
                "farmers showcase",
                "farmers competition",
            ],
        }

        self.job_hoping_to_there = {
            "Engineer": [
                "meet some of your colleagues",
                "encounter some new innovations",
            ],
            "Farmer": ["sell some of your produce", "buy a new tool"],
        }

        self.job_hoping_to_learn = {
            "Engineer": [
                "new programming languages",
                "new frameworks",
                "new libraries",
                "new technologies",
                "new tools",
                "new techniques",
                "new methods",
                "new approaches",
                "new strategies",
                "new plans",
                "new goals",
                "new objectives",
                "new targets",
                "new milestones",
                "new tasks",
                "new assignments",
                "new missions",
                "new quests",
                "new adventures",
                "new journeys",
                "new experiences",
                "new opportunities",
                "new challenges",
                "new ideas",
                "new concepts",
                "new inventions",
                "new discoveries",
                "new theories",
                "new hypotheses",
                "new experiments",
                "new algorithms",
                "new designs",
                "new systems",
                "new solutions",
                "new components",
                "new parts",
                "new circuits",
                "new devices",
                "new machines",
                "new programs",
                "new software",
                "new hardware",
                "new products",
                "new features",
                "new designs",
                "new systems",
                "new solutions",
                "new components",
                "new parts",
                "new circuits",
                "new devices",
                "new machines",
                "new programs",
                "new software",
                "new hardware",
                "new products",
                "new features",
            ],
            "Farmer": [
                "new farming techniques",
                "new farming methods",
                "new farming approaches",
                "new farming strategies",
                "new farming plans",
                "new farming goals",
                "new farming objectives",
                "new farming targets",
                "new farming milestones",
                "new farming tasks",
                "new farming assignments",
                "new farming missions",
                "new farming quests",
                "new farming adventures",
                "new farming journeys",
                "new farming experiences",
                "new farming opportunities",
                "new farming challenges",
                "new farming ideas",
                "new farming concepts",
                "new farming inventions",
                "new farming discoveries",
                "new farming theories",
                "new farming hypotheses",
                "new farming experiments",
                "new farming algorithms",
                "new farming designs",
                "new farming systems",
                "new farming solutions",
                "new farming components",
                "new farming parts",
                "new farming circuits",
                "new farming devices",
                "new farming machines",
                "new farming programs",
                "new farming software",
                "new farming hardware",
                "new farming products",
                "new farming features",
            ],
        }

        self.job_hoping_to_meet = {
            "Engineer": [
                "new people",
                "new friends",
                "new colleagues",
                "new mentors",
                "new leaders",
                "new managers",
                "new collaborators",
                "new partners",
                "new investors",
                "new customers",
                "new clients",
                "new users",
                "new developers",
                "new engineers",
                "new designers",
                "new programmers",
                "new testers",
                "new marketers",
                "new salespeople",
                "new businesspeople",
                "new entrepreneurs",
                "new founders",
                "new CEOs",
                "new CTOs",
                "new CIOs",
                "new CMOs",
                "new COOs",
                "new CFOs",
                "new VPs",
                "new directors",
                "new managers",
                "new supervisors",
                "new employees",
                "new interns",
                "new contractors",
                "new consultants",
                "new freelancers",
                "new remote workers",
                "new coworkers",
                "new teammates",
                "new colleagues",
                "new peers",
                "new subordinates",
                "new superiors",
                "new bosses",
                "new leaders",
                "new managers",
                "new mentors",
                "new teachers",
                "new students",
                "new professors",
                "new researchers",
                "new scientists",
                "new engineers",
                "new designers",
                "new programmers",
                "new testers",
                "new marketers",
                "new salespeople",
                "new businesspeople",
                "new entrepreneurs",
                "new founders",
                "new CEOs",
                "new CTOs",
                "new CIOs",
                "new CMOs",
                "new COOs",
                "new CFOs",
                "new VPs",
                "new directors",
                "new managers",
                "new supervisors",
                "new employees",
                "new interns",
                "new contractors",
                "new consultants",
                "new freelancers",
                "new remote workers",
                "new coworkers",
                "new teammates",
                "new colleagues",
                "new peers",
                "new subordinates",
                "new superiors",
                "new bosses",
                "new leaders",
                "new managers",
                "new mentors",
                "new teachers",
                "new students",
                "new professors",
                "new researchers",
                "new scientists",
            ],
            "Farmer": [
                "new people",
                "new friends",
                "new colleagues",
                "new mentors",
                "new leaders",
                "new managers",
                "new collaborators",
                "new partners",
                "new investors",
                "new customers",
                "new clients",
                "new users",
                "new developers",
                "new engineers",
                "new designers",
                "new programmers",
                "new testers",
                "new marketers",
                "new salespeople",
                "new businesspeople",
                "new entrepreneurs",
                "new founders",
                "new CEOs",
                "new CTOs",
                "new CIOs",
                "new CMOs",
                "new COOs",
                "new CFOs",
                "new VPs",
                "new directors",
                "new managers",
                "new supervisors",
                "new employees",
                "new interns",
                "new contractors",
                "new consultants",
                "new freelancers",
                "new remote workers",
                "new coworkers",
                "new teammates",
                "new colleagues",
                "new peers",
                "new subordinates",
                "new superiors",
                "new bosses",
                "new leaders",
                "new managers",
                "new mentors",
                "new teachers",
                "new students",
                "new professors",
                "new researchers",
                "new scientists",
                "new engineers",
                "new designers",
                "new programmers",
                "new testers",
                "new marketers",
                "new salespeople",
                "new businesspeople",
                "new entrepreneurs",
                "new founders",
                "new CEOs",
                "new CTOs",
                "new CIOs",
                "new CMOs",
                "new COOs",
                "new CFOs",
                "new VPs",
                "new directors",
                "new managers",
                "new supervisors",
                "new employees",
                "new interns",
                "new contractors",
                "new consultants",
                "new freelancers",
                "new remote workers",
                "new coworkers",
                "new teammates",
                "new colleagues",
                "new peers",
                "new subordinates",
                "new superiors",
                "new bosses",
                "new leaders",
                "new managers",
                "new mentors",
                "new teachers",
                "new students",
                "new professors",
                "new researchers",
                "new scientists",
            ],
        }

        self.job_hoping_to_find = {
            "Engineer": [
                "new opportunities",
                "new challenges",
                "new ideas",
                "new concepts",
                "new inventions",
                "new discoveries",
                "new theories",
                "new hypotheses",
                "new experiments",
                "new algorithms",
                "new designs",
                "new systems",
                "new solutions",
                "new components",
                "new parts",
                "new circuits",
                "new devices",
                "new machines",
                "new programs",
                "new software",
                "new hardware",
                "new products",
                "new features",
                "new designs",
                "new systems",
                "new solutions",
                "new components",
                "new parts",
                "new circuits",
                "new devices",
                "new machines",
                "new programs",
                "new software",
                "new hardware",
                "new products",
                "new features",
            ],
            "Farmer": [
                "new opportunities",
                "new challenges",
                "new ideas",
                "new concepts",
                "new inventions",
                "new discoveries",
                "new theories",
                "new hypotheses",
                "new experiments",
                "new algorithms",
                "new designs",
                "new systems",
                "new solutions",
                "new components",
                "new parts",
                "new circuits",
                "new devices",
                "new machines",
                "new programs",
                "new software",
                "new hardware",
                "new products",
                "new features",
                "new designs",
                "new systems",
                "new solutions",
                "new components",
                "new parts",
                "new circuits",
                "new devices",
                "new machines",
                "new programs",
                "new software",
                "new hardware",
                "new products",
                "new features",
            ],
        }

        self.feeling_health = {
            "healthy": [
                "in excellent health",
                "healthy",
                "doing well",
                "feeling good",
                "feeling great",
                "feeling amazing",
                "feeling fantastic",
                "feeling excellent",
                "feeling energetic",
                "strong",
                "fit",
                "feeling invincible",
            ],
            "sick": [
                "feeling sick",
                "feeling ill",
                "feeling unwell",
                "feeling bad",
                "feeling terrible",
                "feeling horrible",
                "feeling awful",
                "feeling absolutely dreadful",
                "miserable",
            ],
            "injured": ["injured", "hurt", "wounded", "damaged", "broken", "bruised"],
        }

        self.feeling_hunger = {
            "full": [
                "you are full",
                "you are satisfied",
                "you are not hungry",
                "you are barely peckish",
                "you are not hungry at all",
                "you are not hungry in the slightest",
                "you are not hungry whatsoever",
                "you are not hungry in the least",
            ],
            "moderate": [
                "your hunger is moderate",
                "you are only slightly hungry",
                "you are moderately hungry",
                "you are a bit hungry",
                "you could use a bite to eat",
                "you could do with a snack",
                "you could do with a meal",
                "you could do with a bite",
            ],
            "hungry": [
                "you are hungry",
                "you are starving",
                "you are famished",
                "you are ravenous",
                "you are starving",
            ],
            "starving": [
                "you are starving",
                "you are famished",
                "you are ravenous",
                "you are starving",
            ],
        }

        self.event_recent = {
            "default": ["Recently"],
            "craft fair": ["After your success at the craft fair"],
            "community center": ["After you helped at the community center"],
            "hospital": ["After you were recently in the hospital"],
            "nursing home": ["Since you helped out at the nursing home"],
            "outbreak": ["With the recent outbreak"],
            "rains": ["The recent rains"],
            "learning": ["Recently, you learned"],
        }

        self.financial_situation = {
            "default": ["financially, you are doing okay"],
            "rich": [
                "you are financially well-off",
                "you are rich",
                "you are wealthy",
                "you are well-off",
                "you are well-to-do",
                "you are well-heeled",
                "you are well-fixed",
                "you are well-situated",
                "you are well-provided",
                "you are well-provided for",
                "you are well-endowed",
                "you are well-furnished",
                "you are well-supplied",
                "you are well-stocked",
                "you are well-equipped",
                "you are well-prepared",
                "you are well-organized",
                "you are well-ordered",
                "you are well-regulated",
                "you are well-arranged",
                "you are well-balanced",
                "you are well-adjusted",
                "you are well-kept",
                "you are well-maintained",
                "you are well-preserved",
                "you are well-protected",
                "you are well-secured",
                "you are well-kept",
                "you are well-maintained",
                "you are well-preserved",
                "you are well-protected",
                "you are well-secured",
                "you are well-kept",
                "you are well-maintained",
                "you are well-preserved",
                "you are well-protected",
                "you are well-secured",
                "you are well-kept",
                "you are well-maintained",
                "you are well-preserved",
                "you are well-protected",
                "you are well-secured",
                "you are well-kept",
                "you are well-maintained",
                "you are well-preserved",
                "you are well-protected",
                "you are well-secured",
                "you are well-kept",
                "you are well-maintained",
                "you are well-preserved",
                "you are well-protected",
                "you are well-secured",
            ],
            "stable": [
                "your financial situation is stable",
                "you are financially stable",
                "you are financially secure",
                "you are financially comfortable",
            ],
            "poor": [
                "you are financially poor",
                "you are financially struggling",
                "you are financially unstable",
                "you are financially insecure",
                "you are financially uncomfortable",
                "you are financially squeezed",
                "your finances are tight",
                "you are financially strapped",
                "you are financially stressed",
                "you are financially burdened",
                "you are struggling to make ends meet",
                "you are struggling to get by",
                "you are struggling to get through financially",
                "you are struggling to pay the bills",
                "you are struggling to pay the rent",
                "you are broke",
                "you are in debt",
                "you are in the red",
                "you are in the hole",
                "you are in the negative",
            ],
            "bankrupt": [
                "you are bankrupt",
                "you are insolvent",
                "you are in debt",
                "you are in the red",
                "you are in the hole",
                "you are in the negative",
                "you are destitute",
            ],
        }

        self.motivation = {
            "default": [
                "You're motivated to ",
                "Today, you aim to ",
                "Today offers the chance to",
                "You remind yourself of your goal to",
                "You're closer to your goal of",
            ]
        }

        self.weather_description = {
            "default": [
                "it's an average day outside",
                "it's a typical day outside",
                "it's a normal day outside",
                "it's a regular day outside",
                "it's a standard day outside",
                "it's a typical day outside",
                "it's a usual day outside",
                "it's a common day outside",
                "it's a standard day out there today",
            ],
            "sunny": [
                "it's a sunny day outside",
                "it's a bright day outsid",
                "it's a clear day out",
            ],
            "cloudy": [
                "it's a cloudy day outside",
                "it's a cloudy day out",
                "it's a bit overcast outside",
            ],
            "rainy": [
                "it's a rainy day outside",
                "it's a bit drizzly outside",
                "it's a bit rainy outside",
                "it's a bit wet outside",
                "it's a bit damp outside",
                "it's a bit moist outside",
            ],
            "snowy": [
                "it's a snowy day outside",
                "it's a bit snowy outside",
                "it's a bit icy outside",
                "it's a bit frosty outside",
                "it's a bit slushy outside",
                "it's a bit cold outside",
                "it's a bit chilly outside",
                "it's a bit freezing outside",
                "it's a bit frigid outside",
                "it's a bit wintry outside",
                "it's a bit wintery outside",
                "it's a bit frosty outside",
                "it's a bit icy outside",
                "it's a bit snowy outside",
                "it's a bit slushy outside",
                "it's a bit cold outside",
                "it's a bit chilly outside",
                "it's a bit freezing outside",
                "it's a bit frigid outside",
                "it's a bit wintry outside",
                "it's a bit wintery outside",
            ],
            "windy": [
                "it's a windy day outside",
                "it's a bit windy outside",
                "it's a bit breezy outside",
                "it's a bit gusty outside",
                "it's a bit blustery outside",
                "it's a bit windy out there today",
            ],
            "stormy": [
                "it's a stormy day outside",
                "it's a bit stormy outside",
                "it's a bit stormy out there today",
            ],
            "foggy": [
                "it's a foggy day outside",
                "it's a bit foggy outside",
                "it's a bit misty outside",
                "it's a bit hazy outside",
                "it's a bit smoky outside",
                "it's a bit smoggy outside",
                "it's a bit foggy out there today",
            ],
        }

        self.routine_question_framing = {
            "default": [
                "Considering the weather and your current situation, what do you choose to do next?\n",
                "What do you choose to do next?\n",
                "What do you do next?\n",
                "What will your focus be?\n",
                "What will you do?\n",
                "What will you focus on?\n",
                "What will you work on?\n",
                "What will you do next?\n",
                "What will you do?\n",
                "What is your next move?\n",
                "What is your next step?\n",
                "What is your next action?\n",
                "What is your next priority?\n",
            ]
        }

        self.action_descriptors = {
            "buy_food": [
                "Go to the market",
                "Go to the grocery store",
                "Go to the supermarket",
                "Go to the bodega",
                "Go to the corner store",
                "Go to the convenience store",
                "Go to the deli",
                "Go to the farmers market",
                "Go to the farm",
                "Go to the farm stand",
                "Go to the market",
            ]
        }

        # self.health_status = ["healthy", "sick", "injured", "disabled", "dying"]
        # self.hunger_level = ["full", "moderate", "hungry", "starving"]
        # self.wealth_money = ["rich", "moderate", "poor", "bankrupt"]
        # self.mental_health = ["stable", "unstable", "depressed", "anxious", "suicidal"]
        # self.social_wellbeing = ["connected", "lonely", "isolated", "disconnected"]
        # self.happiness = ["happy", "content", "sad", "depressed", "suicidal"]
        # self.shelter = ["stable", "unstable", "homeless"]
        # self.stability = ["stable", "unstable"]
        # self.luxury = ["luxurious", "comfortable", "uncomfortable", "unlivable"]
        # self.hope = ["hopeful", "hopeless"]
        # self.success = ["successful", "unsuccessful"]
        # self.control = ["in control", "out of control"]
        # self.job_performance = ["good", "bad"]
        # self.beauty = ["beautiful", "ugly"]
        # self.community = ["connected", "disconnected"]
        # self.material_goods = ["plentiful", "scarce"]
        # self.friendship_grid = ["connected", "disconnected"]

    def get_job_adjective(self, job):
        return random.choice(self.job_adjective.get(job, self.job_adjective["default"]))

    def get_job_pronoun(self, job):
        return random.choice(self.job_pronoun.get(job, self.job_pronoun["default"]))

    def get_job_enjoys_verb(self, job):
        return random.choice(
            self.job_enjoys_verb.get(job, self.job_enjoys_verb["default"])
        )

    def get_job_verb_acts_on_noun(self, job):
        return random.choice(
            self.job_verb_acts_on_noun.get(job, self.job_verb_acts_on_noun["default"])
        )

    def get_job_currently_working_on(self, job):
        return random.choice(
            self.job_currently_working_on.get(
                job, self.job_currently_working_on.get("Engineer", ["a new project"])
            )
        )

    def get_job_place(self, job):
        return random.choice(self.job_place.get(job, self.job_place["default"]))

    def get_job_planning_to_attend(self, job):
        return random.choice(
            self.job_planning_to_attend.get(job, self.job_planning_to_attend.get("Engineer", ["tech conference"]))
        )

    def get_job_hoping_to_there(self, job):
        return random.choice(
            self.job_hoping_to_there.get(job, self.job_hoping_to_there.get("Engineer", ["meet some colleagues"]))
        )

    def get_job_hoping_to_learn(self, job):
        return random.choice(
            self.job_hoping_to_learn.get(job, self.job_hoping_to_learn.get("Engineer", ["new technologies"]))
        )

    def get_job_hoping_to_meet(self, job):
        return random.choice(
            self.job_hoping_to_meet.get(job, self.job_hoping_to_meet.get("Engineer", ["new people"]))
        )

    def get_job_hoping_to_find(self, job):
        return random.choice(
            self.job_hoping_to_find.get(job, self.job_hoping_to_find.get("Engineer", ["new opportunities"]))
        )

    def get_feeling_health(self, health_status):
        return random.choice(
            self.feeling_health.get(health_status, self.feeling_health.get("healthy", ["doing well"]))
        )

    def get_feeling_hunger(self, hunger_level):
        return random.choice(
            self.feeling_hunger.get(hunger_level, self.feeling_hunger.get("moderate", ["moderately hungry"]))
        )

    def get_event_recent(self, recent_event):
        return random.choice(
            self.event_recent.get(recent_event, ["Recently"])
        )

    def get_financial_situation(self, wealth_money):
        return random.choice(
            self.financial_situation.get(
                wealth_money, self.financial_situation.get("stable", ["your financial situation is stable"])
            )
        )

    def get_motivation(self, motivation=None):
        """Return a motivational phrase.

        If ``motivation`` is ``None`` or not found in the matrix, a random
        choice from the ``"default"`` list is returned.
        """
        return random.choice(
            self.motivation.get(motivation, self.motivation["default"])
        )

    def get_motivation_zero(self, motivation, job):
        return (
            random.choice(self.motivation.get(motivation, self.motivation["default"]))
            + random.choice(
                self.job_enjoys_verb.get(job, self.job_enjoys_verb["default"])
            )
            + random.choice(
                self.job_verb_acts_on_noun.get(
                    job, self.job_verb_acts_on_noun["default"]
                )
            )
        )

    def get_weather_description(self, weather_description):
        return random.choice(
            self.weather_description.get(
                weather_description, self.weather_description["default"]
            )
        )

    def get_routine_question_framing(self, routine_question_framing=None):
        """Return a question framing string for routine prompts."""
        return random.choice(
            self.routine_question_framing.get(
                routine_question_framing, self.routine_question_framing["default"]
            )
        )

    def get_character_voice_descriptor(self, job: str, trait: str) -> str:
        """Get character voice descriptors based on job and trait type."""
        voice_descriptors = {
            "Engineer": {
                "speech_pattern": ["analytical", "precise", "methodical", "logical"],
                "decision_approach": ["systematic", "data-driven", "problem-solving oriented", "detail-focused"],
                "personality_tone": ["thoughtful", "curious", "innovative", "efficiency-minded"],
                "motivation_style": ["achievement-oriented", "mastery-focused", "improvement-driven"]
            },
            "Farmer": {
                "speech_pattern": ["practical", "down-to-earth", "straightforward", "patient"],
                "decision_approach": ["grounded", "seasonal-minded", "nature-connected", "long-term thinking"],
                "personality_tone": ["nurturing", "steady", "traditional", "community-minded"], 
                "motivation_style": ["growth-oriented", "sustainability-focused", "family-centered"]
            },
            "Waitress": {
                "speech_pattern": ["friendly", "service-oriented", "empathetic", "conversational"],
                "decision_approach": ["people-focused", "relationship-aware", "socially conscious", "helpful"],
                "personality_tone": ["warm", "attentive", "energetic", "customer-focused"],
                "motivation_style": ["service-driven", "connection-seeking", "harmony-oriented"]
            },
            "default": {
                "speech_pattern": ["casual", "balanced", "straightforward"],
                "decision_approach": ["thoughtful", "practical", "considered"],
                "personality_tone": ["friendly", "approachable", "reasonable"],
                "motivation_style": ["goal-oriented", "balanced", "adaptive"]
            }
        }
        
        import random
        job_traits = voice_descriptors.get(job, voice_descriptors["default"])
        trait_options = job_traits.get(trait, job_traits["speech_pattern"])
        return random.choice(trait_options)
        
    def get_character_response_style(self, job: str, personality_traits: dict = None) -> str:
        """Get how the character typically responds based on job and personality."""
        base_style = self.get_character_voice_descriptor(job, "decision_approach")
        
        if personality_traits:
            # Modify based on personality traits
            if personality_traits.get('extraversion', 50) > 70:
                base_style += " and expressive"
            if personality_traits.get('conscientiousness', 50) > 70:
                base_style += " and organized"
            if personality_traits.get('openness', 50) > 70:
                base_style += " and creative"
                
        return base_style
        
    def get_action_descriptors(self, action):
        return random.choice(
            self.action_descriptors.get(action, self.action_descriptors.get("default", [action]))
        )


descriptors = DescriptorMatrices()


class PromptBuilder:
    """Construct complex prompts for the Tiny Village language model.

    A ``PromptBuilder`` instance collects information from a :class:`~tiny_characters.Character`
    and formats it into strings which the LLM can understand.  It does not
    communicate with the model directly; :mod:`tiny_brain_io` is responsible for
    that step.
    """

    def __init__(self, character, memory_manager=None) -> None:
        """Initialize the builder for ``character``."""

        self.character = character
        self.action_options = ActionOptions()
        self.needs_priorities_func = NeedsPriorities()
        
        # New LLM integration features
        self.conversation_history = ConversationHistory()
        self.few_shot_manager = FewShotExampleManager()
        self.character_voice_traits = self._initialize_character_voice()
        
        # Enhanced context management
        self.context_manager = ContextManager(character, memory_manager)
        self.memory_manager = memory_manager
        
        # Prompt versioning
        self.prompt_version = "1.0.0"
        self.prompt_metadata = {
            "created_at": datetime.now().isoformat(),
            "version": self.prompt_version,
            "character_id": getattr(character, 'name', 'unknown'),
            "performance_metrics": {}
        }
        
    def _initialize_character_voice(self) -> Dict[str, str]:
        """Initialize character-specific voice and personality traits."""
        # Default voice patterns that can be customized per character
        voice_traits = {
            "speech_style": "casual",
            "decision_style": "thoughtful", 
            "personality_descriptor": "balanced",
            "motivational_phrase": "focused on personal growth"
        }
        
        # Customize based on character job and personality if available
        if hasattr(self.character, 'job') and self.character.job:
            job_voice_map = {
                "Engineer": {
                    "speech_style": "analytical and precise",
                    "decision_style": "methodical and logical",
                    "personality_descriptor": "problem-solving oriented",
                    "motivational_phrase": "driven by innovation and efficiency"
                },
                "Farmer": {
                    "speech_style": "practical and down-to-earth", 
                    "decision_style": "grounded and patient",
                    "personality_descriptor": "connected to nature",
                    "motivational_phrase": "focused on growth and nurturing"
                },
                "Waitress": {
                    "speech_style": "friendly and service-oriented",
                    "decision_style": "people-focused and empathetic", 
                    "personality_descriptor": "socially aware",
                    "motivational_phrase": "dedicated to helping others"
                }
            }
            voice_traits.update(job_voice_map.get(self.character.job, {}))
            
        # Further customize based on personality traits if available
        if hasattr(self.character, 'personality_traits') and self.character.personality_traits:
            traits = self.character.personality_traits
            if isinstance(traits, dict):
                # High extraversion
                if traits.get('extraversion', 50) > 70:
                    voice_traits["speech_style"] += ", outgoing and expressive"
                # High conscientiousness  
                if traits.get('conscientiousness', 50) > 70:
                    voice_traits["decision_style"] += ", organized and deliberate"
                # High openness
                if traits.get('openness', 50) > 70:
                    voice_traits["personality_descriptor"] += ", creative and curious"
                    
        return voice_traits
        
    def apply_character_voice(self, base_prompt: str) -> str:
        """Apply character voice consistency to prompts."""
        voice_prefix = f"You approach decisions in a {self.character_voice_traits['decision_style']} manner. "
        voice_prefix += f"Your communication is {self.character_voice_traits['speech_style']}. "
        voice_prefix += f"As someone who is {self.character_voice_traits['personality_descriptor']}, "
        voice_prefix += f"you are {self.character_voice_traits['motivational_phrase']}. "
        
        # Insert voice guidance after the system prompt
        if "<|user|>" in base_prompt:
            parts = base_prompt.split("<|user|>", 1)
            return parts[0] + voice_prefix + "<|user|>" + parts[1]
        else:
            return voice_prefix + base_prompt
            
    def record_conversation_turn(self, prompt: str, response: str, action_taken: Optional[str] = None, outcome: Optional[str] = None):
        """Record a conversation turn for future context."""
        self.conversation_history.add_turn(
            character_name=self.character.name,
            prompt=prompt,
            response=response, 
            action_taken=action_taken,
            outcome=outcome
        )
        
    def add_few_shot_example(self, situation: str, character_state: Dict[str, Any], decision: str, outcome: str, success_rating: float = 0.8):
        """Add a new few-shot example from character experience."""
        example = FewShotExample(
            situation_context=situation,
            character_state=character_state,
            decision_made=decision,
            outcome=outcome,
            success_rating=success_rating
        )
        self.few_shot_manager.add_example(example)

    def integrate_relevant_memories(self, context_query: str, max_memories: int = 3):
        """Integrate relevant memories from MemoryManager into prompts.
        
        Args:
            context_query: Query to find relevant memories
            max_memories: Maximum number of memories to retrieve
            
        Returns:
            List of relevant memory objects
        """
        return self.context_manager.gather_memory_context(context_query, max_memories)
        
    def format_memories_for_prompt(self, memories):
        """Format memories for inclusion in prompts.
        
        Args:
            memories: List of memory objects to format
            
        Returns:
            Formatted string for prompt inclusion
        """
        if not memories:
            return ""
            
        lines = ["Relevant memories to consider:"]
        for i, memory in enumerate(memories, 1):
            # Handle different memory object types
            if hasattr(memory, 'description'):
                desc = memory.description
            elif hasattr(memory, 'content'):
                desc = memory.content
            else:
                desc = str(memory)
                
            lines.append(f"{i}. {desc}")
            
        return "\n".join(lines) + "\n"
        
    def add_prompt_metadata(self, prompt_type: str, context_info = None):
        """Add versioning and metadata to prompts.
        
        Args:
            prompt_type: Type of prompt being generated
            context_info: Additional context information
            
        Returns:
            Metadata dictionary for the prompt
        """
        metadata = {
            "prompt_version": self.prompt_version,
            "prompt_type": prompt_type,
            "character_name": self.character.name,
            "timestamp": datetime.now().isoformat(),
        }
        
        if context_info:
            metadata.update(context_info)
            
        return metadata
        
    def collect_performance_feedback(self, prompt_type: str, success_rating: float, 
                                   response_quality: float = None, user_feedback: str = None):
        """Collect performance metrics for prompt versions.
        
        Args:
            prompt_type: Type of prompt that was used
            success_rating: Rating of how successful the prompt was (0-1)
            response_quality: Optional quality rating of the response
            user_feedback: Optional user feedback text
        """
        if prompt_type not in self.prompt_metadata["performance_metrics"]:
            self.prompt_metadata["performance_metrics"][prompt_type] = []
            
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "version": self.prompt_version,
            "success_rating": success_rating,
            "response_quality": response_quality,
            "user_feedback": user_feedback
        }
        
        self.prompt_metadata["performance_metrics"][prompt_type].append(metrics)

    def calculate_needs_priorities(self) -> None:
        """Compute and store the character's current need priorities."""

        self.needs_priorities = self.needs_priorities_func.calculate_needs_priorities(
            self.character
        )

    def prioritize_actions(self) -> List[str]:
        """Query the planning system for top actions and build choice strings."""

        try:
            from tiny_strategy_manager import StrategyManager
            from tiny_utility_functions import calculate_action_utility
        except ImportError:  # pragma: no cover - gracefully handle missing deps
            self.prioritized_actions = []
            self.action_choices = []
            return []

        manager = StrategyManager(use_llm=False)
        actions = manager.get_daily_actions(self.character)

        self.prioritized_actions = actions
        self.action_choices = []
        char_state = self._get_character_state_dict()
        current_goal = (
            self.character.get_current_goal()
            if hasattr(self.character, "get_current_goal")
            else None
        )

        for i, action in enumerate(actions[:5]):
            try:
                util = calculate_action_utility(char_state, action, current_goal)
            except (ValueError, TypeError):  # Replace with specific exceptions
                util = 0.0
                print(f"Error calculating utility for action {action}: {e}")  # Optional logging

            effects_str = ""
            if hasattr(action, "effects") and action.effects:
                parts = [
                    f"{eff.get('attribute', '')}: {eff.get('change_value', 0):+.1f}"
                    for eff in action.effects
                    if eff.get("attribute")
                ]
                if parts:
                    effects_str = f" - Effects: {', '.join(parts)}"

            desc = getattr(action, "description", getattr(action, "name", str(action)))
            choice = f"{i+1}. {desc} (Utility: {util:.1f}){effects_str}"
            self.action_choices.append(choice)

        return self.action_choices

    def generate_completion_message(self, character, action: str) -> str:
        """Return a short message describing successful completion of ``action``."""

        return f"{character.name} has {DescriptorMatrices.get_action_descriptors(action)} {action}."

    def generate_failure_message(self, character, action: str) -> str:
        """Return a short message describing failure to perform ``action``."""

        return f"{character.name} has failed to {DescriptorMatrices.get_action_descriptors(action)} {action}."

    def _get_character_state_dict(self) -> Dict[str, float]:
        """Return a simplified state dictionary for utility calculations."""

        state = {
            "hunger": getattr(self.character, "hunger_level", 5.0) / 10.0,
            "energy": getattr(self.character, "energy", 5.0) / 10.0,
            "health": getattr(self.character, "health_status", 5.0) / 10.0,
            "mental_health": getattr(self.character, "mental_health", 5.0) / 10.0,
            "social_wellbeing": getattr(self.character, "social_wellbeing", 5.0)
            / 10.0,
            "money": float(getattr(self.character, "wealth_money", 0.0)),
        }
        return state

    def calculate_action_utility(self, current_goal: Optional[object] = None) -> Dict[str, float]:
        """Calculate and return utility values for the prioritized actions."""
        from tiny_utility_functions import UtilityEvaluator, calculate_action_utility
        from tiny_output_interpreter import OutputInterpreter

        self.action_utilities = {}
        evaluator = UtilityEvaluator()
        char_state = self._get_character_state_dict()
        interpreter = OutputInterpreter()

        for action_name in self.prioritized_actions:
            try:
                action_cls = interpreter.action_class_map.get(action_name)
                action_obj = action_cls() if action_cls else None
            except Exception as e:
                print(f"Error creating action {action_name}: {e}")
                continue

            if not action_obj:
                print(f"Warning: Unknown action {action_name}")
                continue

            try:
                utility = evaluator.evaluate_action_utility(
                    self.character.name,
                    char_state,
                    action_obj,
                    current_goal,
                )
            except Exception:
                try:
                    utility = calculate_action_utility(char_state, action_obj, current_goal)
                except Exception as e:
                    print(f"Failed to evaluate utility for {action_name}: {e}")
                    continue

            self.action_utilities[action_name] = utility

        return self.action_utilities

    def generate_daily_routine_prompt(
        self, 
        time: str, 
        weather: str, 
        include_conversation_context: bool = True,
        include_few_shot_examples: bool = True,
        include_memories: bool = True,
        output_format: str = "structured"
    ) -> str:
        """Generate a basic daily routine prompt with enhanced context management and memory integration."""
        
        # Use ContextManager to gather comprehensive context
        context = self.context_manager.assemble_complete_context(
            time, weather, 
            memory_query=f"daily routine for {self.character.name}" if include_memories else None
        )
        
        # Add prompt metadata for versioning
        metadata = self.add_prompt_metadata("daily_routine", {
            "time": time,
            "weather": weather,
            "include_memories": include_memories
        })
        
        prompt = "<|system|>"
        prompt += f"<!-- Prompt Version: {metadata['prompt_version']} -->\n"
        
        # Use enhanced character context
        char_info = context['character']
        prompt += (
            f"You are {char_info['basic_info']['name']}, a {char_info['basic_info']['job']} in a small town. "
            f"You are a {descriptors.get_job_adjective(char_info['basic_info']['job'])} "
            f"{descriptors.get_job_pronoun(char_info['basic_info']['job'])} who enjoys "
            f"{descriptors.get_job_enjoys_verb(char_info['basic_info']['job'])} "
            f"{descriptors.get_job_verb_acts_on_noun(char_info['basic_info']['job'])}. "
            f"You are currently working on {descriptors.get_job_currently_working_on(char_info['basic_info']['job'])} "
            f"{descriptors.get_job_place(char_info['basic_info']['job'])}, and you are excited to see how it turns out. "
            f"You are also planning to attend a {descriptors.get_job_planning_to_attend(char_info['basic_info']['job'])} "
            f"in the next few weeks, and you are hoping to {descriptors.get_job_hoping_to_there(char_info['basic_info']['job'])} there."
        )
        
        # Add conversation context if available and requested
        if include_conversation_context:
            context_text = self.conversation_history.format_context_for_prompt(self.character.name)
            if context_text:
                prompt += f"\n{context_text}"

        # Add relevant memories
        if include_memories and context['memories']:
            memory_text = self.format_memories_for_prompt(context['memories'])
            if memory_text:
                prompt += f"\n{memory_text}"

        # Add few-shot examples if requested
        if include_few_shot_examples:
            current_state = self._get_character_state_dict()
            relevant_examples = self.few_shot_manager.get_relevant_examples(current_state)
            if relevant_examples:
                examples_text = self.few_shot_manager.format_examples_for_prompt(relevant_examples)
                prompt += f"\n{examples_text}"
        
        prompt += f"<|user|>"
        prompt += f"{self.character.name}, it's {time}, and {descriptors.get_weather_description(weather)}. You're feeling {descriptors.get_feeling_health(self.character.health_status)}, and {descriptors.get_feeling_hunger(self.character.hunger_level)}. "
        prompt += f"{descriptors.get_event_recent(self.character.recent_event)}, and {descriptors.get_financial_situation(self.character.wealth_money)}. {descriptors.get_motivation()} {getattr(self.character, 'long_term_goal', 'personal growth')}. {descriptors.get_routine_question_framing()}"
        prompt += "Options:\n"
        prompt += "1. Go to the market to Buy_Food.\n"
        prompt += f"2. Work at your job to Improve_{getattr(self.character, 'job_performance', 'job_performance')}.\n"
        prompt += "3. Visit a friend to Increase_Friendship.\n"
        prompt += "4. Engage in a Leisure_Activity to improve Mental_Health.\n"
        prompt += "5. Work on a personal project to Pursue_Hobby.\n"
        actions = self.action_options.prioritize_actions(self.character)
        for i, action in enumerate(actions[:5], 1):
            try:
                descriptor = descriptors.get_action_descriptors(action)
            except (KeyError, AttributeError):
                descriptor = action.replace("_", " ").title()
            action_name = action.replace("_", " ").title().replace(" ", "_")
            prompt += f"{i}. {descriptor} to {action_name}.\n"
        # Add structured output format instructions
        if output_format == "json":
            prompt += f"\n\n{OutputSchema.get_decision_schema()}"
        else:
            prompt += f"\n\n{OutputSchema.get_routine_schema()}"
            
        prompt += "</s>"
        prompt += "<|assistant|>"
        
        # Apply character voice consistency
        prompt = self.apply_character_voice(prompt)
        
        prompt += f"{self.character.name}, I choose "
        return prompt

    def generate_decision_prompt(
        self,
        time: str,
        weather: str,
        action_choices: List[str],
        character_state_dict: Optional[Dict[str, float]] = None,
        memories: Optional[List] = None,
        include_conversation_context: bool = True,
        include_few_shot_examples: bool = True,
        include_memory_integration: bool = True,
        output_format: str = "json",
    ) -> str:
        """Create a decision prompt with enhanced context management and memory integration."""
        
        # Use ContextManager for comprehensive context gathering
        context = self.context_manager.assemble_complete_context(
            time, weather,
            memory_query=f"decision making for {self.character.name}" if include_memory_integration else None
        )
        
        # Add prompt metadata for versioning
        metadata = self.add_prompt_metadata("decision", {
            "time": time,
            "weather": weather,
            "include_memory_integration": include_memory_integration,
            "action_choices_count": len(action_choices)
        })
        
        # Get character's current goals prioritized by importance
        goal_context = context['goals']
        goal_queue = goal_context.get('active_goals', [])
        needs_priorities = goal_context.get('needs_priorities', {})

        # Build enhanced prompt with rich character context
        prompt = f"<|system|>"
        prompt += f"<!-- Prompt Version: {metadata['prompt_version']} -->\n"

        # Basic character identity and role using context
        char_info = context['character']
        prompt += (
            f"You are {char_info['basic_info']['name']}, a {char_info['basic_info']['job']} in a small town. "
        )
        prompt += f"You are a {descriptors.get_job_adjective(char_info['basic_info']['job'])} {descriptors.get_job_pronoun(char_info['basic_info']['job'])} "
        prompt += f"who enjoys {descriptors.get_job_enjoys_verb(char_info['basic_info']['job'])} {descriptors.get_job_verb_acts_on_noun(char_info['basic_info']['job'])}. "

        # Add conversation context if available and requested
        if include_conversation_context:
            context_text = self.conversation_history.format_context_for_prompt(self.character.name)
            if context_text:
                prompt += f"\n{context_text}"

        # Add memory integration - prioritize new integration over legacy memories parameter
        relevant_memories = context['memories'] if include_memory_integration else (memories or [])
        if relevant_memories:
            memory_text = self.format_memories_for_prompt(relevant_memories)
            if memory_text:
                prompt += f"\n{memory_text}"

        # Add few-shot examples if requested
        if include_few_shot_examples:
            current_state = character_state_dict or self._get_character_state_dict()
            relevant_examples = self.few_shot_manager.get_relevant_examples(current_state)
            if relevant_examples:
                examples_text = self.few_shot_manager.format_examples_for_prompt(relevant_examples)
                prompt += f"\n{examples_text}"

        # Current goals and motivations - Enhanced for LLM guidance
        if goal_queue and len(goal_queue) > 0:
            prompt += f"\n\n🎯 **CURRENT ACTIVE GOALS** (in priority order):\n"
            for i, (utility_score, goal) in enumerate(goal_queue[:3]):  # Top 3 goals
                # Enhanced goal description with urgency indicator
                urgency = "🔥 URGENT" if utility_score > self.URGENCY_THRESHOLD_URGENT else "⚡ HIGH" if utility_score > self.URGENCY_THRESHOLD_HIGH else "📌 MODERATE"
                prompt += f"{i+1}. **{goal.name}**: {goal.description}\n"
                prompt += f"   → Priority Score: {utility_score:.1f}/10 ({urgency})\n"
        else:
            prompt += f"\n\n🎯 **CURRENT ACTIVE GOALS** (in priority order):\n"
            prompt += f"   → No active goals currently. 🌱 Consider establishing new objectives to guide your actions.\n"

        # Character's pressing needs and motivations - Enhanced priority display
        top_needs = sorted(needs_priorities.items(), key=lambda x: x[1], reverse=True)[
            :5
        ]
        if top_needs:
            prompt += f"\n🚨 **MOST PRESSING NEEDS** (requiring immediate attention):\n"
            for need_name, priority_score in top_needs:
                need_desc = self._get_need_description(need_name, priority_score)
                # Add visual urgency indicators
                if priority_score > self.NEEDS_PRIORITY_CRITICAL_THRESHOLD:
                    urgency_icon = "🔴 CRITICAL"
                elif priority_score > self.NEEDS_PRIORITY_HIGH_THRESHOLD:
                    urgency_icon = "🟡 HIGH"
                else:
                    urgency_icon = "🟢 MODERATE"
                prompt += f"- {urgency_icon} {need_desc}\n"

        # Character motives and personality context
        if hasattr(self.character, "motives") and self.character.motives:
            prompt += f"\nYour key motivations:\n"
            top_motives = self._get_top_motives(self.character.motives, 4)
            for motive_name, motive_score in top_motives:
                prompt += f"- {motive_name.replace('_', ' ').title()}: {self._get_motive_description(motive_name, motive_score)}\n"

        # Current comprehensive state
        prompt += f"\n<|user|>"
        prompt += f"{self.character.name}, it's {time}, and {descriptors.get_weather_description(weather)}. "

        # Enhanced status description
        prompt += f"Current state: "
        prompt += f"Health {self.character.health_status}/10, "
        prompt += f"Hunger {self.character.hunger_level}/10, "
        prompt += f"Energy {getattr(self.character, 'energy', 5):.1f}/10, "
        prompt += f"Mental Health {self.character.mental_health}/10, "
        prompt += f"Social Wellbeing {self.character.social_wellbeing}/10. "

        # Financial and life context
        prompt += f"{descriptors.get_event_recent(self.character.recent_event)}, and {descriptors.get_financial_situation(self.character.wealth_money)}. "

        # Long-term aspiration context
        if hasattr(self.character, "long_term_goal") and self.character.long_term_goal:
            prompt += f"Your long-term aspiration is: {self.character.long_term_goal}. "

 
        # Include short memory descriptions if provided
        if memories:
            prompt += "\nRecent memories influencing you:\n"
            for mem in memories[:2]:
                desc = getattr(mem, "description", str(mem))
                prompt += f"- {desc}\n"

        # Include any additional character state provided
        if isinstance(character_state_dict, dict):
            prompt += "\nAdditional state:\n"
            for key, value in character_state_dict.items():
                formatted_key = key.replace("_", " ").title()
                prompt += f"- {formatted_key}: {value}\n"
        elif character_state_dict is not None:
            raise TypeError("character_state_dict must be a dictionary.")
 

        prompt += f"\n{descriptors.get_routine_question_framing()}"

        # Enhanced action choices with better formatting
        prompt += f"\nAvailable actions:\n"
        for i, action_choice in enumerate(action_choices):
            prompt += f"{action_choice}\n"

        prompt += f"\nChoose the action that best addresses your ACTIVE GOALS and PRESSING NEEDS listed above. "
        prompt += f"Prioritize actions that: (1) advance your highest-priority active goals, "
        prompt += f"(2) address your most critical needs (🔴/🟡), and (3) support your long-term aspirations. "
        prompt += f"Consider both immediate urgency and strategic value."

        # Add structured output format instructions
        if output_format == "json":
            prompt += f"\n\n{OutputSchema.get_decision_schema()}"
        else:
            prompt += f"\n\n{OutputSchema.get_routine_schema()}"

        prompt += f"\n</s>"
        prompt += f"<|assistant|>"
        
        # Apply character voice consistency
        prompt = self.apply_character_voice(prompt)
        
        prompt += f"{self.character.name}, I choose "
        return prompt

    def _get_need_description(self, need_name: str, priority_score: float) -> str:
        """Generate human-readable description for character needs."""
        need_descriptions = {
            "health": f"Physical health needs attention (priority: {priority_score:.0f}/100)",
            "hunger": f"Nutritional needs are pressing (priority: {priority_score:.0f}/100)",
            "wealth": f"Financial security is important (priority: {priority_score:.0f}/100)",
            "mental_health": f"Mental wellness requires care (priority: {priority_score:.0f}/100)",
            "social_wellbeing": f"Social connections need nurturing (priority: {priority_score:.0f}/100)",
            "happiness": f"Personal happiness and fulfillment (priority: {priority_score:.0f}/100)",
            "shelter": f"Housing and shelter security (priority: {priority_score:.0f}/100)",
            "stability": f"Life stability and routine (priority: {priority_score:.0f}/100)",
            "luxury": f"Comfort and luxury desires (priority: {priority_score:.0f}/100)",
            "hope": f"Optimism and future outlook (priority: {priority_score:.0f}/100)",
            "success": f"Achievement and success drive (priority: {priority_score:.0f}/100)",
            "control": f"Sense of control and agency (priority: {priority_score:.0f}/100)",
            "job_performance": f"Professional excellence (priority: {priority_score:.0f}/100)",
            "beauty": f"Aesthetic and beauty appreciation (priority: {priority_score:.0f}/100)",
            "community": f"Community involvement and belonging (priority: {priority_score:.0f}/100)",
            "material_goods": f"Material possessions and acquisitions (priority: {priority_score:.0f}/100)",
        }
        return need_descriptions.get(
            need_name,
            f"{need_name.replace('_', ' ').title()} (priority: {priority_score:.0f}/100)",
        )

    def _get_top_motives(self, motives: object, count: int = 4) -> List[tuple]:
        """Get the top character motives by score."""
        try:
            motive_dict = motives.to_dict()
            motive_scores = [
                (name, motive.score) for name, motive in motive_dict.items()
            ]
            return sorted(motive_scores, key=lambda x: x[1], reverse=True)[:count]
        except Exception as e:
            print(f"Warning: Could not extract motives: {e}")
            return []

    def _get_motive_description(self, motive_name: str, score: float) -> str:
        """Generate human-readable description for character motives."""
        intensity = (
            "Very High"
            if score >= 8
            else "High" if score >= 6 else "Moderate" if score >= 4 else "Low"
        )
        return f"{intensity} ({score:.1f}/10)"


    def generate_crisis_response_prompt(
        self, 
        crisis_description: str, 
        urgency: str = "high",
        include_conversation_context: bool = True,
        include_few_shot_examples: bool = False  # Usually not needed for crisis
    ):
        """Generate a short crisis response prompt for the LLM.

        Parameters
        ----------
        crisis_description : str
            Description of the crisis situation.
        urgency : str, optional
            Qualitative urgency indicator (e.g. "low", "medium", "high").
        include_conversation_context : bool, optional
            Whether to include recent conversation history.
        include_few_shot_examples : bool, optional
            Whether to include few-shot examples (usually not needed for crisis).
        """

        prompt = "<|system|>"
        prompt += (
            f"You are {self.character.name}, a {descriptors.get_job_adjective(self.character.job)} "
            f"{descriptors.get_job_pronoun(self.character.job)} prepared for emergencies."
        )
        
        # Add conversation context if available and requested
        if include_conversation_context:
            context = self.conversation_history.format_context_for_prompt(self.character.name, num_turns=1)
            if context:
                prompt += f"\n{context}"

        # Add few-shot examples if requested (typically not for crisis)
        if include_few_shot_examples:
            current_state = self._get_character_state_dict()
            relevant_examples = self.few_shot_manager.get_relevant_examples(current_state, max_examples=1)
            if relevant_examples:
                examples_text = self.few_shot_manager.format_examples_for_prompt(relevant_examples)
                prompt += f"\n{examples_text}"

        prompt += "<|user|>"
        prompt += (
            f"A crisis has occurred: {crisis_description}. Urgency: {urgency}. "
            f"{descriptors.get_event_recent(self.character.recent_event)} "
            f"{descriptors.get_financial_situation(self.character.wealth_money)}."
        )

        try:
            from tiny_utility_functions import UtilityEvaluator
            from tiny_output_interpreter import OutputInterpreter

            evaluator = UtilityEvaluator()
            interpreter = OutputInterpreter()
            actions = ActionOptions().prioritize_actions(self.character)
            action_objects = []
            for name in actions[:1]:
                cls = interpreter.action_class_map.get(name)
                if cls:
                    action_objects.append(cls())

            if action_objects:
                char_state = self._get_character_state_dict()
                _, analysis = evaluator.evaluate_plan_utility_advanced(
                    self.character.name, char_state, action_objects
                )
                breakdown = analysis.get("action_breakdown")
                if breakdown:
                    best_action = breakdown[0].get("action")
                    if best_action:
                        prompt += f" Recommended immediate action: {best_action}."
        except ImportError:
            # Utility evaluation is optional and may not work if dependencies are missing
            pass
        except Exception as e:
            # Log unexpected exceptions for debugging purposes
            import logging
            logging.error(f"An unexpected error occurred during utility evaluation: {e}")

        # Add crisis-specific output schema
        prompt += f"\n\n{OutputSchema.get_crisis_schema()}"

        prompt += "<|assistant|>"
        
        # Apply character voice consistency
        prompt = self.apply_character_voice(prompt)
        
        return prompt

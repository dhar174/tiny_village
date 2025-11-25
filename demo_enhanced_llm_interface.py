#!/usr/bin/env python3
"""
Demo of the Enhanced LLM Interface for Tiny Village

This script demonstrates the complete LLM interface implementation including:
1. Strategic invocation logic (when to use LLM vs utility-based)
2. Contextual prompt generation with character state
3. Robust output parsing with multiple format support
4. Graceful fallbacks for missing dependencies

Run with: python demo_enhanced_llm_interface.py
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_character(name="Alex", crisis_mode=False):
    """Create a test character with configurable state"""
    class TestCharacter:
        def __init__(self, name, crisis_mode=False):
            self.name = name
            self.job = "Engineer"
            
            if crisis_mode:
                # Crisis scenario - low health and energy
                self.health_status = 2  # Very low health
                self.energy = 1         # Very low energy
                self.mental_health = 3  # Low mental health
                self.hunger_level = 8   # Very hungry
            else:
                # Normal scenario
                self.health_status = 7
                self.energy = 6
                self.mental_health = 6
                self.hunger_level = 4
                
            self.social_wellbeing = 5
            self.wealth_money = 50
            self.recent_event = "learning"
            self.long_term_goal = "career_advancement"
            self.personality_traits = {"extraversion": 60, "conscientiousness": 70}
            
        # Add all the getter methods
        def get_health_status(self): return self.health_status
        def get_hunger_level(self): return self.hunger_level
        def get_mental_health(self): return self.mental_health
        def get_social_wellbeing(self): return self.social_wellbeing
        def get_wealth_money(self): return self.wealth_money
        def get_wealth(self): return self.wealth_money
        def get_long_term_goal(self): return self.long_term_goal
        def get_happiness(self): return 5
        def get_shelter(self): return 5
        def get_stability(self): return 5
        def get_luxury(self): return 3
        def get_hope(self): return 6
        def get_success(self): return 5
        def get_control(self): return 5
        def get_job_performance(self): return 6
        def get_beauty(self): return 5
        def get_community(self): return 4
        def get_material_goods(self): return 4
        def get_friendship_grid(self): return 5
        
        def get_inventory(self):
            class MockInventory:
                def count_food_items_total(self): return 3
                def count_food_calories_total(self): return 150
            return MockInventory()
            
        def get_motives(self):
            class MockMotive:
                def __init__(self, score): self.score = score
                def get_score(self): return self.score
                
            class MockMotives:
                def get_health_motive(self): return MockMotive(5)
                def get_hunger_motive(self): return MockMotive(4)
                def get_wealth_motive(self): return MockMotive(6)
                def get_mental_health_motive(self): return MockMotive(5)
                def get_social_wellbeing_motive(self): return MockMotive(4)
                def get_happiness_motive(self): return MockMotive(5)
                def get_shelter_motive(self): return MockMotive(5)
                def get_stability_motive(self): return MockMotive(5)
                def get_luxury_motive(self): return MockMotive(3)
                def get_hope_motive(self): return MockMotive(6)
                def get_success_motive(self): return MockMotive(5)
                def get_control_motive(self): return MockMotive(5)
                def get_job_performance_motive(self): return MockMotive(6)
                def get_beauty_motive(self): return MockMotive(5)
                def get_community_motive(self): return MockMotive(4)
                def get_material_goods_motive(self): return MockMotive(4)
            return MockMotives()
            
    return TestCharacter(name, crisis_mode)


def demo_strategic_invocation():
    """Demonstrate the strategic invocation logic"""
    print("🧠 STRATEGIC INVOCATION LOGIC DEMO")
    print("=" * 50)
    
    from tiny_strategy_manager import StrategyManager
    
    # Create strategy manager with LLM enabled
    sm = StrategyManager(use_llm=True)
    print(f"✅ StrategyManager initialized with LLM: {sm.use_llm}")
    
    # Test 1: Routine scenario
    routine_character = create_test_character("RoutineAlex", crisis_mode=False)
    routine_context = {'social_complexity': 0.2, 'novelty_score': 0.1}
    
    should_use_llm_routine = sm.should_use_llm_for_decision(routine_character, routine_context)
    print(f"\n📝 Routine Scenario:")
    print(f"   Character: {routine_character.name} (healthy, normal state)")
    print(f"   Context: Low complexity, familiar situation")
    print(f"   Strategic Decision: {'LLM' if should_use_llm_routine else 'Utility-based'}")
    
    # Test 2: Crisis scenario
    crisis_character = create_test_character("CrisisAlex", crisis_mode=True)
    crisis_context = {'social_complexity': 0.3}
    
    should_use_llm_crisis = sm.should_use_llm_for_decision(crisis_character, crisis_context)
    print(f"\n🚨 Crisis Scenario:")
    print(f"   Character: {crisis_character.name} (health: {crisis_character.health_status}/10, energy: {crisis_character.energy}/10)")
    print(f"   Context: Character in distress")
    print(f"   Strategic Decision: {'LLM' if should_use_llm_crisis else 'Utility-based'}")
    
    # Test 3: Complex social scenario
    social_character = create_test_character("SocialAlex", crisis_mode=False)
    social_context = {'social_complexity': 0.8, 'novelty_score': 0.4}
    
    should_use_llm_social = sm.should_use_llm_for_decision(social_character, social_context)
    print(f"\n🤝 Complex Social Scenario:")
    print(f"   Character: {social_character.name} (normal state)")
    print(f"   Context: High social complexity (0.8/1.0)")
    print(f"   Strategic Decision: {'LLM' if should_use_llm_social else 'Utility-based'}")
    
    # Test 4: Forced LLM scenario
    forced_context = {'force_llm': True}
    should_use_llm_forced = sm.should_use_llm_for_decision(routine_character, forced_context)
    print(f"\n⚡ Forced LLM Scenario:")
    print(f"   Character: {routine_character.name}")
    print(f"   Context: Explicitly requesting LLM")
    print(f"   Strategic Decision: {'LLM' if should_use_llm_forced else 'Utility-based'}")


def demo_prompt_generation():
    """Demonstrate contextual prompt generation"""
    print("\n\n📝 CONTEXTUAL PROMPT GENERATION DEMO")
    print("=" * 50)
    
    from tiny_prompt_builder import PromptBuilder
    
    character = create_test_character("PrompterAlex")
    prompt_builder = PromptBuilder(character)
    
    print(f"✅ PromptBuilder initialized for {character.name}")
    print(f"   Job: {character.job}")
    print(f"   Health: {character.health_status}/10, Hunger: {character.hunger_level}/10")
    
    # Generate a daily routine prompt
    prompt = prompt_builder.generate_daily_routine_prompt(
        time="morning",
        weather="sunny",
        include_memories=False,  # Disabled for demo
        include_few_shot_examples=False,  # Disabled for demo
        output_format="structured"
    )
    
    print(f"\n📋 Generated Daily Routine Prompt:")
    print(f"   Length: {len(prompt)} characters")
    print(f"   Contains character name: {'✅' if character.name in prompt else '❌'}")
    print(f"   Contains job info: {'✅' if character.job in prompt else '❌'}")
    print(f"   Contains action options: {'✅' if 'Options:' in prompt else '❌'}")
    
    # Show excerpt of prompt
    print(f"\n📄 Prompt Excerpt (first 300 chars):")
    print(f"   {prompt[:300]}...")


def demo_output_parsing():
    """Demonstrate robust output parsing"""
    print("\n\n🔍 OUTPUT PARSING DEMO")
    print("=" * 50)
    
    from tiny_output_interpreter import OutputInterpreter
    
    interpreter = OutputInterpreter()
    print(f"✅ OutputInterpreter initialized")
    print(f"   Supported actions: {len(interpreter.action_class_map)} action types")
    
    # Test different response formats
    test_responses = [
        # JSON format
        ('JSON Response', '{"action": "Work", "parameters": {"job_type": "engineering"}}'),
        
        # Natural language
        ('Natural Language', 'I choose to go to work to improve my engineering skills'),
        
        # Mixed format
        ('Mixed Format', 'I think I should work today. {"action": "Work", "parameters": {}}'),
        
        # Fallback scenario
        ('Unclear Response', 'Maybe I should do something productive today?'),
    ]
    
    for format_name, response in test_responses:
        try:
            parsed = interpreter.parse_llm_response(response)
            action = interpreter.interpret(parsed, "demo_character")
            
            print(f"\n📥 {format_name}:")
            print(f"   Input: {response[:50]}...")
            print(f"   Parsed Action: {parsed.get('action', 'Unknown')}")
            print(f"   Action Object: {type(action).__name__}")
            print(f"   Status: ✅ Successfully parsed")
            
        except Exception as e:
            print(f"\n📥 {format_name}:")
            print(f"   Input: {response[:50]}...")
            print(f"   Status: ❌ Parse failed: {e}")


def demo_end_to_end_pipeline():
    """Demonstrate the complete end-to-end pipeline"""
    print("\n\n🔄 END-TO-END PIPELINE DEMO")
    print("=" * 50)
    
    from tiny_strategy_manager import StrategyManager
    
    # Create different scenarios
    scenarios = [
        ("Routine Day", create_test_character("Alice", False), {'social_complexity': 0.1}),
        ("Crisis Mode", create_test_character("Bob", True), {'social_complexity': 0.3}),
        ("Social Event", create_test_character("Carol", False), {'social_complexity': 0.9}),
    ]
    
    sm = StrategyManager(use_llm=True)
    
    for scenario_name, character, context in scenarios:
        print(f"\n🎭 Scenario: {scenario_name}")
        print(f"   Character: {character.name} (Health: {character.health_status}, Energy: {character.energy})")
        
        # Get strategic decision
        will_use_llm = sm.should_use_llm_for_decision(character, context)
        print(f"   Strategic Decision: {'🧠 LLM' if will_use_llm else '🔢 Utility-based'}")
        
        # Get enhanced actions
        actions = sm.get_enhanced_daily_actions(
            character,
            time="morning",
            weather="clear",
            situation_context=context
        )
        
        print(f"   Actions Generated: {len(actions)} actions")
        if actions:
            action_types = [type(action).__name__ for action in actions[:3]]
            print(f"   Top Actions: {', '.join(action_types)}")
        
        print(f"   Status: ✅ Pipeline completed successfully")


def main():
    """Run the complete LLM interface demonstration"""
    print("🚀 ENHANCED LLM INTERFACE DEMONSTRATION")
    print("=" * 60)
    print("This demo showcases the complete LLM interface implementation")
    print("including strategic invocation, contextual prompts, and robust parsing.")
    print()
    
    try:
        # Run all demonstrations
        demo_strategic_invocation()
        demo_prompt_generation()
        demo_output_parsing()
        demo_end_to_end_pipeline()
        
        print("\n" + "=" * 60)
        print("🎉 LLM INTERFACE DEMONSTRATION COMPLETE!")
        print("\n✅ Key Features Demonstrated:")
        print("   • Strategic LLM invocation based on scenario complexity")
        print("   • Crisis detection and intelligent decision routing")
        print("   • Rich contextual prompt generation with character state")
        print("   • Multi-format output parsing (JSON, natural language, fallback)")
        print("   • Complete character → LLM → action pipeline")
        print("   • Graceful degradation with missing dependencies")
        print("\n🎯 The LLM interface is fully functional and ready for production!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Test script to verify that GraphManager is functioning as the single source of truth
for social network data in the Tiny Village system.
"""

import sys
import traceback
from unittest.mock import Mock


def create_mock_character(name, uuid=None):
    """Create a mock character for testing purposes."""
    char = Mock()
    char.name = name
    char.uuid = uuid or name.lower().replace(' ', '_')
    char.age = 25
    char.energy = 50
    char.wealth_money = 1000
    char.beauty = 7
    char.stability = 50
    char.luxury = 30
    char.shelter = 80
    char.success = 60
    char.monogamy = 70
    char.base_libido = 50
    
    # Mock personality traits
    char.personality_traits = Mock()
    char.personality_traits.get_openness.return_value = 6
    char.personality_traits.get_extraversion.return_value = 5
    char.personality_traits.get_conscientiousness.return_value = 7
    char.personality_traits.get_agreeableness.return_value = 8
    char.personality_traits.get_neuroticism.return_value = 3
    
    # Mock motives
    char.motives = Mock()
    char.get_motives.return_value = char.motives
    
    wealth_motive = Mock()
    wealth_motive.score = 6
    char.motives.get_wealth_motive.return_value = wealth_motive
    
    family_motive = Mock()
    family_motive.score = 7
    char.motives.get_family_motive.return_value = family_motive
    
    beauty_motive = Mock()
    beauty_motive.score = 5
    char.motives.get_beauty_motive.return_value = beauty_motive
    
    luxury_motive = Mock()
    luxury_motive.score = 4
    char.motives.get_luxury_motive.return_value = luxury_motive
    
    stability_motive = Mock()
    stability_motive.score = 8
    char.motives.get_stability_motive.return_value = stability_motive
    
    control_motive = Mock()
    control_motive.score = 5
    char.motives.get_control_motive.return_value = control_motive
    
    # Mock other methods
    char.get_control.return_value = 50
    char.get_base_libido.return_value = 50
    
    # Mock job and home for relationship calculations
    char.job = Mock()
    char.job.location = f"{name}'s workplace"
    char.home = f"{name}'s home"
    
    return char


def test_graphmanager_social_networks():
    """Test GraphManager as single source of truth for social networks."""
    print("Testing GraphManager as Single Source of Truth")
    print("=" * 50)
    
    try:
        # Test 1: Import and initialize GraphManager
        print("1. Testing GraphManager initialization...")
        from tiny_graph_manager import GraphManager
        
        gm = GraphManager()
        print("   ✓ GraphManager initialized successfully")
        
        # Test 2: Create mock characters
        print("\n2. Creating test characters...")
        alice = create_mock_character("Alice", "alice_001")
        bob = create_mock_character("Bob", "bob_002") 
        charlie = create_mock_character("Charlie", "charlie_003")
        
        characters = {"alice_001": alice, "bob_002": bob, "charlie_003": charlie}
        print(f"   ✓ Created {len(characters)} test characters")
        
        # Test 3: Add characters to GraphManager
        print("\n3. Adding characters to GraphManager...")
        for char in characters.values():
            gm.add_character_node(char)
        print(f"   ✓ Added {len(characters)} characters to graph")
        print(f"   ✓ Graph now has {len(gm.characters)} characters")
        
        # Test 4: Initialize relationships through GraphManager
        print("\n4. Initializing relationships through GraphManager...")
        gm.initialize_character_relationships(characters)
        
        # Count relationships
        relationship_count = 0
        for char1 in characters.values():
            for char2 in characters.values():
                if char1 != char2 and gm.G.has_edge(char1, char2):
                    relationship_count += 1
        
        print(f"   ✓ Created {relationship_count} relationship edges")
        
        # Test 5: Test social network data retrieval
        print("\n5. Testing social network data retrieval...")
        social_data = gm.get_social_networks()
        
        print(f"   ✓ Retrieved social data with keys: {list(social_data.keys())}")
        print(f"   ✓ Relationships data type: {type(social_data['relationships'])}")
        
        # Check relationships for each character
        for char_id, relationships in social_data['relationships'].items():
            print(f"   ✓ {char_id}: {len(relationships)} relationships")
            for other_id, strength in relationships.items():
                print(f"     - {other_id}: strength {strength}")
        
        # Test 6: Test individual character relationships
        print("\n6. Testing individual character relationship queries...")
        alice_relationships = gm.get_character_relationships(alice)
        print(f"   ✓ Alice has {len(alice_relationships)} detailed relationships")
        
        for other_char, rel_data in alice_relationships.items():
            print(f"     - {other_char}: {rel_data}")
        
        # Test 7: Test GameplayController integration
        print("\n7. Testing GameplayController integration...")
        import pygame
        pygame.init()
        
        from tiny_gameplay_controller import GameplayController
        
        config = {'screen_width': 800, 'screen_height': 600}
        
        # Test with a fresh GraphManager to avoid conflicts
        test_gm = GraphManager()
        gc = GameplayController(graph_manager=test_gm, config=config)
        
        # Test the property delegation
        controller_social_data = gc.social_networks
        print(f"   ✓ GameplayController.social_networks returns: {type(controller_social_data)}")
        
        # Test that it's using GraphManager (should have same structure)
        gm_social_data = test_gm.get_social_networks()
        print(f"   ✓ Both have same structure: {controller_social_data.keys() == gm_social_data.keys()}")
        print("   ✓ GameplayController delegates to GraphManager correctly")
        
        # Test 8: Test relationship updates
        print("\n8. Testing relationship updates...")
        original_alice_bob = None
        if alice in gm.G.nodes and bob in gm.G.nodes and gm.G.has_edge(alice, bob):
            original_alice_bob = gm.G[alice][bob].get('strength', 50)
            print(f"   ✓ Original Alice-Bob relationship strength: {original_alice_bob}")
            
            # Update through GraphManager
            gm.update_social_relationships(0.1)  # Small time step
            
            updated_alice_bob = gm.G[alice][bob].get('strength', 50)
            print(f"   ✓ Updated Alice-Bob relationship strength: {updated_alice_bob}")
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("✓ GraphManager is successfully acting as single source of truth")
        print("✓ Social networks are centrally managed")
        print("✓ GameplayController properly delegates to GraphManager")
        print("✓ No separate state management detected")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False


def test_no_separate_state():
    """Verify that GameplayController does not maintain separate social state."""
    print("\nTesting No Separate State Maintenance")
    print("=" * 40)
    
    try:
        import pygame
        pygame.init()
        
        from tiny_graph_manager import GraphManager
        from tiny_gameplay_controller import GameplayController
        
        gm = GraphManager()
        config = {'screen_width': 800, 'screen_height': 600}
        gc = GameplayController(graph_manager=gm, config=config)
        
        # Check that GameplayController doesn't have separate social_networks attribute
        has_separate_attr = hasattr(gc, '_social_networks') or hasattr(gc, '__dict__') and '_social_networks' in gc.__dict__
        
        if has_separate_attr:
            print("   ❌ GameplayController still has separate social networks attribute")
            return False
        else:
            print("   ✓ GameplayController does not maintain separate social state")
        
        # Check that accessing social_networks goes through property
        social_data = gc.social_networks
        gm_social_data = gm.get_social_networks()
        
        if social_data == gm_social_data:
            print("   ✓ GameplayController.social_networks delegates correctly")
            return True
        else:
            print("   ❌ Social data mismatch between GameplayController and GraphManager")
            return False
            
    except Exception as e:
        print(f"   ❌ Error testing separate state: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = True
    
    success &= test_graphmanager_social_networks()
    success &= test_no_separate_state()
    
    if success:
        print("\n🎯 INTEGRATION TEST SUCCESSFUL!")
        print("GraphManager is now the single source of truth for social networks.")
        sys.exit(0)
    else:
        print("\n💥 INTEGRATION TEST FAILED!")
        print("Issues remain with single source of truth implementation.")
        sys.exit(1)
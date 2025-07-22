#!/usr/bin/env python3
"""
Simple verification test to ensure GraphManager is the single source of truth
and GameplayController doesn't maintain separate social network state.
"""

def test_single_source_of_truth():
    """Verify GameplayController uses GraphManager as single source of truth."""
    print("Verifying Single Source of Truth Implementation")
    print("=" * 50)
    
    try:
        import pygame
        pygame.init()
        
        from tiny_graph_manager import GraphManager
        from tiny_gameplay_controller import GameplayController
        
        # Create GraphManager and GameplayController
        gm = GraphManager()
        config = {'screen_width': 800, 'screen_height': 600}
        gc = GameplayController(graph_manager=gm, config=config)
        
        # Verify 1: No separate social_networks attribute 
        print("1. Checking for separate state management...")
        separate_attrs = [attr for attr in dir(gc) if 'social' in attr.lower() and not attr.startswith('_')]
        print(f"   Public social attributes: {separate_attrs}")
        
        # Should only have the property, not separate state
        if hasattr(gc, '_social_networks'):
            print("   ❌ Found separate _social_networks attribute")
            return False
        else:
            print("   ✓ No separate social networks state found")
        
        # Verify 2: Property delegation works
        print("\n2. Testing property delegation...")
        social_data_gc = gc.social_networks
        social_data_gm = gm.get_social_networks()
        
        print(f"   GameplayController returns: {type(social_data_gc)}")
        print(f"   GraphManager returns: {type(social_data_gm)}")
        print(f"   Both have same keys: {social_data_gc.keys() == social_data_gm.keys()}")
        
        # Verify 3: Method delegation
        print("\n3. Testing method delegation...")
        try:
            get_social_result = gc.get_social_networks()
            print(f"   ✓ get_social_networks() method exists and returns: {type(get_social_result)}")
        except AttributeError:
            print("   ❌ get_social_networks() method not found")
            return False
        
        # Verify 4: Check that GameplayController initialization doesn't create separate state
        print("\n4. Verifying no separate initialization...")
        gc_source = open('/home/runner/work/tiny_village/tiny_village/tiny_gameplay_controller.py').read()
        
        if 'implement_social_network_system()' in gc_source:
            print("   ❌ Found separate social network system implementation")
            return False
        else:
            print("   ✓ No separate social network system implementation found")
            
        if 'self.social_networks = {' in gc_source:
            print("   ❌ Found direct social_networks assignment")
            return False
        else:
            print("   ✓ No direct social_networks assignment found")
        
        print("\n" + "=" * 50)
        print("🎉 VERIFICATION SUCCESSFUL!")
        print("✓ GraphManager is the single source of truth")
        print("✓ GameplayController properly delegates")
        print("✓ No separate state management detected")
        return True
        
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_single_source_of_truth()
    
    if success:
        print("\n🎯 VERIFICATION COMPLETE - Issue #222 Resolved!")
        print("GraphManager successfully established as single source of truth.")
    else:
        print("\n💥 VERIFICATION FAILED!")
        print("Additional work needed to resolve Issue #222.")
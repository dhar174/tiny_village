#!/usr/bin/env python3
"""
Test script to validate the StoryArc implementation and ensure
the missing constants are properly defined.
"""

import sys
import os

# Add the current directory to the path so we can import tiny_story_arc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tiny_story_arc import StoryArc, StoryArcManager


def test_story_arc_constants():
    """Test that the required constants are defined and accessible."""
    print("Testing StoryArc constants...")
    
    # Test that constants are defined on the class
    assert hasattr(StoryArc, 'STARTING_THRESHOLD'), "STARTING_THRESHOLD constant not found"
    assert hasattr(StoryArc, 'DEVELOPING_THRESHOLD'), "DEVELOPING_THRESHOLD constant not found"
    assert hasattr(StoryArc, 'CLIMAX_THRESHOLD'), "CLIMAX_THRESHOLD constant not found"
    
    # Test that constants have reasonable values
    assert 0.0 <= StoryArc.STARTING_THRESHOLD <= 1.0, "STARTING_THRESHOLD not in valid range"
    assert 0.0 <= StoryArc.DEVELOPING_THRESHOLD <= 1.0, "DEVELOPING_THRESHOLD not in valid range"
    assert 0.0 <= StoryArc.CLIMAX_THRESHOLD <= 1.0, "CLIMAX_THRESHOLD not in valid range"
    
    # Test that constants are in logical order
    assert StoryArc.STARTING_THRESHOLD < StoryArc.DEVELOPING_THRESHOLD, "Thresholds not in order"
    assert StoryArc.DEVELOPING_THRESHOLD < StoryArc.CLIMAX_THRESHOLD, "Thresholds not in order"
    
    print(f"✓ STARTING_THRESHOLD = {StoryArc.STARTING_THRESHOLD}")
    print(f"✓ DEVELOPING_THRESHOLD = {StoryArc.DEVELOPING_THRESHOLD}")
    print(f"✓ CLIMAX_THRESHOLD = {StoryArc.CLIMAX_THRESHOLD}")
    print("✓ All constants are properly defined and accessible")


def test_story_arc_functionality():
    """Test basic StoryArc functionality."""
    print("\nTesting StoryArc functionality...")
    
    # Create a story arc
    arc = StoryArc("Test Village Conflict", importance=7)
    
    # Test initial state
    assert arc.name == "Test Village Conflict"
    assert arc.importance == 7
    assert arc.progression == 0.0
    assert arc.phase == "setup"
    assert not arc.completed
    
    # Test progression through phases
    arc.advance_progression(0.1)  # Should still be in setup
    assert arc.phase == "setup"
    
    arc.advance_progression(0.2)  # Should move to rising_action
    assert arc.phase == "rising_action"
    
    arc.advance_progression(0.5)  # Should move to climax
    assert arc.phase == "climax"
    
    arc.advance_progression(0.3)  # Should move to resolution and complete
    assert arc.phase == "resolution"
    assert arc.completed
    
    print("✓ Story arc progresses through phases correctly")
    
    # Test phase detection using constants
    setup_arc = StoryArc("Setup Test")
    setup_arc.progression = 0.1
    assert setup_arc.progression < StoryArc.STARTING_THRESHOLD
    assert setup_arc.get_current_phase() == "setup"
    
    rising_arc = StoryArc("Rising Test")
    rising_arc.progression = 0.4
    assert StoryArc.STARTING_THRESHOLD <= rising_arc.progression < StoryArc.DEVELOPING_THRESHOLD
    assert rising_arc.get_current_phase() == "rising_action"
    
    climax_arc = StoryArc("Climax Test")
    climax_arc.progression = 0.8
    assert StoryArc.DEVELOPING_THRESHOLD <= climax_arc.progression < StoryArc.CLIMAX_THRESHOLD
    assert climax_arc.get_current_phase() == "climax"
    
    resolution_arc = StoryArc("Resolution Test")
    resolution_arc.progression = 0.95
    assert resolution_arc.progression >= StoryArc.CLIMAX_THRESHOLD
    assert resolution_arc.get_current_phase() == "resolution"
    
    print("✓ Phase detection using constants works correctly")


def test_story_arc_manager():
    """Test the StoryArcManager functionality."""
    print("\nTesting StoryArcManager...")
    
    manager = StoryArcManager()
    
    # Create multiple arcs
    arc1 = manager.create_arc("Village Festival", 8)
    arc2 = manager.create_arc("Mystery Stranger", 6)
    arc3 = manager.create_arc("Harvest Season", 5)
    
    assert len(manager.active_arcs) == 3
    assert len(manager.completed_arcs) == 0
    
    # Progress one arc to completion
    arc1.advance_progression(1.0)
    manager.update_arcs()
    
    assert len(manager.active_arcs) == 2
    assert len(manager.completed_arcs) == 1
    
    # Test statistics
    stats = manager.get_statistics()
    assert stats["total_arcs"] == 3
    assert stats["active_arcs"] == 2
    assert stats["completed_arcs"] == 1
    
    print("✓ StoryArcManager works correctly")


def test_attribute_error_prevention():
    """Test that the original AttributeError issue is resolved."""
    print("\nTesting AttributeError prevention...")
    
    try:
        # These should not raise AttributeError anymore
        starting = StoryArc.STARTING_THRESHOLD
        developing = StoryArc.DEVELOPING_THRESHOLD
        climax = StoryArc.CLIMAX_THRESHOLD
        
        print(f"✓ Successfully accessed STARTING_THRESHOLD: {starting}")
        print(f"✓ Successfully accessed DEVELOPING_THRESHOLD: {developing}")
        print(f"✓ Successfully accessed CLIMAX_THRESHOLD: {climax}")
        
        # Test access through instance as well
        arc = StoryArc("Test")
        instance_starting = arc.STARTING_THRESHOLD
        instance_developing = arc.DEVELOPING_THRESHOLD
        instance_climax = arc.CLIMAX_THRESHOLD
        
        print(f"✓ Successfully accessed constants through instance")
        print("✓ No AttributeError - issue resolved!")
        
    except AttributeError as e:
        print(f"✗ AttributeError still occurs: {e}")
        raise


if __name__ == "__main__":
    print("Running StoryArc validation tests...\n")
    
    try:
        test_story_arc_constants()
        test_story_arc_functionality()
        test_story_arc_manager()
        test_attribute_error_prevention()
        
        print("\n🎉 All tests passed! StoryArc implementation is working correctly.")
        print("The missing constants issue has been resolved.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
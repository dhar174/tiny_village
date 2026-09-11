#!/usr/bin/env python3
"""
Test script for comprehensive error handling in file operations and external dependencies.

This script tests the error handling improvements made to:
- MapController image loading
- FAISS operations in memory management  
- Memory persistence file operations
"""

import sys
import os
import tempfile
import shutil
import logging
from unittest.mock import Mock, patch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_map_controller_error_handling():
    """Test MapController image loading error handling."""
    print("\n=== Testing MapController Error Handling ===")
    
    try:
        import pygame
        pygame.init()
        
        from tiny_map_controller import MapController
        
        # Test 1: Non-existent image file
        print("Test 1: Non-existent image file")
        try:
            map_controller = MapController("non_existent_image.png", {"width": 100, "height": 100, "buildings": []})
            print("✓ MapController created with fallback image")
            assert map_controller.map_image is not None, "Map image should not be None"
            print("✓ Fallback image created successfully")
        except Exception as e:
            print(f"✗ Error creating MapController with non-existent image: {e}")
            return False
        
        # Test 2: Empty image path
        print("Test 2: Empty image path")
        try:
            map_controller = MapController("", {"width": 100, "height": 100, "buildings": []})
            print("✓ MapController created with empty image path")
            assert map_controller.map_image is not None, "Map image should not be None"
            print("✓ Fallback image created successfully")
        except Exception as e:
            print(f"✗ Error creating MapController with empty image path: {e}")
            return False
        
        # Test 3: None image path
        print("Test 3: None image path")
        try:
            map_controller = MapController(None, {"width": 100, "height": 100, "buildings": []})
            print("✓ MapController created with None image path")
            assert map_controller.map_image is not None, "Map image should not be None"
            print("✓ Fallback image created successfully")
        except Exception as e:
            print(f"✗ Error creating MapController with None image path: {e}")
            return False
        
        # Test 4: Rendering with fallback image
        print("Test 4: Rendering with fallback image")
        try:
            surface = pygame.Surface((800, 600))
            map_controller.render(surface)
            print("✓ Rendering with fallback image successful")
        except Exception as e:
            print(f"✗ Error rendering with fallback image: {e}")
            return False
        
        pygame.quit()
        print("✓ All MapController error handling tests passed")
        return True
        
    except ImportError as e:
        print(f"✗ Could not import required modules: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error in MapController tests: {e}")
        return False


def test_memory_error_handling():
    """Test memory persistence error handling."""
    print("\n=== Testing Memory Persistence Error Handling ===")
    
    try:
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # Mock the FlatMemoryAccess class since we need to test its methods
            class MockFlatMemoryAccess:
                def __init__(self):
                    self.index_is_normalized = False
                    self.faiss_index = None
                
                def get_specific_memories(self):
                    return []
                    
                def set_all_memory_embeddings_to_normalized(self, normalized):
                    pass
                
                def get_specific_memory_by_description(self, description):
                    return None
            
            # Test the save/load methods we added error handling to
            mock_memory = MockFlatMemoryAccess()
            
            # Import the methods we want to test
            from tiny_memories import FlatMemoryAccess
            
            # Create a real FlatMemoryAccess instance for testing
            flat_access = FlatMemoryAccess()
            
            # Test 1: Save with empty filename
            print("Test 1: Save embeddings with empty filename")
            try:
                result = flat_access.save_all_specific_memories_embeddings_to_file("")
                assert result == False, "Should return False for empty filename"
                print("✓ Correctly handled empty filename")
            except Exception as e:
                print(f"✗ Error handling empty filename: {e}")
                return False
            
            # Test 2: Load from non-existent file
            print("Test 2: Load embeddings from non-existent file")
            try:
                result = flat_access.load_all_specific_memories_embeddings_from_file("non_existent_file")
                assert result == False, "Should return False for non-existent file"
                print("✓ Correctly handled non-existent file")
            except Exception as e:
                print(f"✗ Error handling non-existent file: {e}")
                return False
            
            # Test 3: Save to invalid directory
            print("Test 3: Save to invalid directory")
            invalid_path = os.path.join(temp_dir, "non_existent_dir", "file")
            try:
                result = flat_access.save_all_specific_memories_embeddings_to_file(invalid_path)
                # This should either succeed (if directory is created) or fail gracefully
                print(f"✓ Handled invalid directory path (result: {result})")
            except Exception as e:
                print(f"✗ Error handling invalid directory: {e}")
                return False
        
        print("✓ All memory persistence error handling tests passed")
        return True
        
    except ImportError as e:
        print(f"✗ Could not import required modules: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error in memory tests: {e}")
        return False


def test_faiss_error_handling():
    """Test FAISS operations error handling."""
    print("\n=== Testing FAISS Error Handling ===")
    
    try:
        from tiny_memories import FlatMemoryAccess
        import tempfile
        
        # Create a FlatMemoryAccess instance
        flat_access = FlatMemoryAccess()
        
        # Test 1: Save index when index is None
        print("Test 1: Save FAISS index when index is None")
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                result = flat_access.save_index_to_file(temp_file.name)
                assert result == False, "Should return False when index is None"
                print("✓ Correctly handled None index")
                os.unlink(temp_file.name)
        except Exception as e:
            print(f"✗ Error handling None index: {e}")
            return False
        
        # Test 2: Load index from non-existent file
        print("Test 2: Load FAISS index from non-existent file")
        try:
            result = flat_access.load_index_from_file("non_existent_index.bin")
            assert result == False, "Should return False for non-existent file"
            print("✓ Correctly handled non-existent index file")
        except Exception as e:
            print(f"✗ Error handling non-existent index file: {e}")
            return False
        
        # Test 3: Save index with empty filename
        print("Test 3: Save FAISS index with empty filename")
        try:
            result = flat_access.save_index_to_file("")
            assert result == False, "Should return False for empty filename"
            print("✓ Correctly handled empty filename")
        except Exception as e:
            print(f"✗ Error handling empty filename: {e}")
            return False
        
        print("✓ All FAISS error handling tests passed")
        return True
        
    except ImportError as e:
        print(f"✗ Could not import required modules: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error in FAISS tests: {e}")
        return False


def main():
    """Run all error handling tests."""
    print("Testing Comprehensive Error Handling for File Operations and External Dependencies")
    print("=" * 80)
    
    tests_passed = 0
    total_tests = 3
    
    # Test MapController error handling
    if test_map_controller_error_handling():
        tests_passed += 1
    
    # Test memory persistence error handling
    if test_memory_error_handling():
        tests_passed += 1
    
    # Test FAISS error handling
    if test_faiss_error_handling():
        tests_passed += 1
    
    print("\n" + "=" * 80)
    print(f"Test Results: {tests_passed}/{total_tests} test suites passed")
    
    if tests_passed == total_tests:
        print("✓ All error handling tests passed!")
        return 0
    else:
        print("✗ Some error handling tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
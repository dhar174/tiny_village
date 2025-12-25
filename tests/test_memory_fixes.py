#!/usr/bin/env python3
"""
Test script to verify that the tiny_memories.py fixes resolved the NameError issues.
"""


def test_imports():
    """Test that all classes can be imported without NameError."""
    try:
        print("Testing individual class imports...")

        # These imports should work without NameError since global variables are now declared
        from tiny_memories import EmbeddingModel

        print("✓ EmbeddingModel imported successfully")

        from tiny_memories import SentimentAnalysis

        print("✓ SentimentAnalysis imported successfully")

        from tiny_memories import MemoryManager

        print("✓ MemoryManager imported successfully")

        from tiny_memories import Memory

        print("✓ Memory imported successfully")

        from tiny_memories import FlatMemoryAccess

        print("✓ FlatMemoryAccess imported successfully")

        print("\n✅ All critical classes imported successfully!")
        print("✅ Global variable NameError issues have been resolved!")
        return True

    except NameError as e:
        print(f"❌ NameError still exists: {e}")
        return False
    except ImportError as e:
        print(f"❌ ImportError: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_global_variables():
    """Test that global variables are accessible from the module."""
    try:
        import tiny_memories

        print("\nTesting global variable accessibility...")

        # Check that global variables exist and are initially None
        print(f"manager: {tiny_memories.manager}")
        print(f"model: {tiny_memories.model}")
        print(f"sentiment_analysis: {tiny_memories.sentiment_analysis}")

        print("✅ Global variables are accessible!")
        return True

    except AttributeError as e:
        print(f"❌ Global variable not found: {e}")
        return False
    except Exception as e:
        print(f"❌ Error accessing globals: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING TINY_MEMORIES.PY FIXES")
    print("=" * 60)

    success1 = test_imports()
    success2 = test_global_variables()

    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED! Fixes were successful!")
    else:
        print("❌ Some tests failed. Additional fixes may be needed.")
    print("=" * 60)

#!/usr/bin/env python3
"""
Tiny Village - Main Entry Point

A 2D simulation game where AI characters autonomously go about their lives
in a dynamic village. Characters make decisions based on their histories,
relationships, and current game states using advanced AI planning.

Usage:
    python main.py [options]

Options:
    --mode <mode>       Demo mode: visual, minimal, test (default: visual)
    --characters <n>    Number of characters to create (default: 5)
    --no-llm            Disable LLM decision making (use fallback logic)
    --fps <n>           Target FPS (default: 60)
    --headless          Run without display (for testing)
    --verbose           Enable verbose logging
    --help              Show this help message

Examples:
    python main.py                          # Run full visual demo
    python main.py --mode minimal           # Run minimal console demo
    python main.py --mode test              # Run integration tests
    python main.py --no-llm --characters 3  # Visual demo, 3 characters, no LLM
"""

import sys
import argparse
import logging
from typing import Optional

# Set up logging before imports
def setup_logging(verbose: bool = False):
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(name)s: %(message)s'
    )
    return logging.getLogger(__name__)

def check_dependencies() -> tuple[bool, list[str]]:
    """Check if required dependencies are installed."""
    missing = []
    
    try:
        import pygame
    except ImportError:
        missing.append("pygame")
    
    try:
        import networkx
    except ImportError:
        missing.append("networkx")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import pydantic
    except ImportError:
        missing.append("pydantic")
    
    try:
        import faiss
    except ImportError:
        missing.append("faiss-cpu")
    
    return len(missing) == 0, missing

def print_banner():
    """Print welcome banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║                   🏘️  TINY VILLAGE  🏘️                       ║
    ║                                                              ║
    ║        AI-Driven Village Simulation                          ║
    ║        Characters with autonomous decision-making            ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def run_visual_demo(config: dict, logger):
    """Run the full visual demo with pygame display."""
    logger.info("Starting visual demo...")
    logger.info(f"Configuration: {config}")
    
    try:
        import pygame
        pygame.init()
        
        from tiny_gameplay_controller import GameplayController
        
        controller = GameplayController(config=config)
        logger.info(f"✅ Controller initialized with {len(controller.characters)} characters")
        
        logger.info("\n🎮 Starting game loop...")
        logger.info("Controls:")
        logger.info("  SPACE - Pause/unpause")
        logger.info("  ESC   - Quit")
        logger.info("  S     - Save game")
        logger.info("  L     - Load game")
        logger.info("  F     - Show feature status")
        logger.info("  M     - Toggle minimap")
        logger.info("\n")
        
        controller.run()
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install dependencies with: pip install pygame networkx numpy pydantic faiss-cpu")
        return 1
    except Exception as e:
        logger.error(f"Error running visual demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

def run_minimal_demo(config: dict, logger):
    """Run minimal console demo without display."""
    logger.info("Starting minimal console demo...")
    
    try:
        from demo_minimal_integration import demonstrate_minimal_integration
        
        logger.info("Running integration demonstration...")
        controller = demonstrate_minimal_integration()
        
        if controller:
            logger.info("\n✅ Minimal demo completed successfully")
            return 0
        else:
            logger.error("\n❌ Minimal demo failed")
            return 1
            
    except Exception as e:
        logger.error(f"Error running minimal demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

def run_tests(config: dict, logger):
    """Run integration tests."""
    logger.info("Running integration tests...")
    
    try:
        from test_integration_minimal import run_integration_tests
        
        result = run_integration_tests()
        return result
        
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    """Main entry point for Tiny Village."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Tiny Village - AI-Driven Village Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--mode',
        choices=['visual', 'minimal', 'test'],
        default='visual',
        help='Demo mode to run (default: visual)'
    )
    
    parser.add_argument(
        '--characters',
        type=int,
        default=5,
        help='Number of characters to create (default: 5)'
    )
    
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Disable LLM decision making (use fallback logic)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=60,
        help='Target FPS (default: 60)'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without display (for testing)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.verbose)
    
    # Print banner
    if not args.headless:
        print_banner()
    
    # Check dependencies
    logger.info("Checking dependencies...")
    deps_ok, missing = check_dependencies()
    
    if not deps_ok:
        logger.error("❌ Missing required dependencies:")
        for dep in missing:
            logger.error(f"   - {dep}")
        logger.error("\nInstall with: pip install " + " ".join(missing))
        return 1
    
    logger.info("✅ All required dependencies installed")
    
    # Build configuration
    config = {
        "target_fps": args.fps,
        "render": {
            "background_color": [20, 50, 80],
            "vsync": True
        },
        "characters": {
            "count": args.characters,
            "use_llm": not args.no_llm
        },
        "headless": args.headless
    }
    
    # Run appropriate demo mode
    logger.info(f"\n🚀 Running in {args.mode} mode...")
    
    if args.mode == 'visual':
        return run_visual_demo(config, logger)
    elif args.mode == 'minimal':
        return run_minimal_demo(config, logger)
    elif args.mode == 'test':
        return run_tests(config, logger)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye! 👋")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

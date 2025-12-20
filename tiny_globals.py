# tiny_globals.py
# A minimalistic global variable manager for Python scripts.

import tiny_time_manager
import threading
import sys


class TinyGlobals:
    def __init__(self):
        self._globals = {}

    def set(self, key, value):
        """Set a global variable."""
        self._globals[key] = value

    def get(self, key, default=None):
        """Get a global variable, return default if not found."""
        return self._globals.get(key, default)

    def remove(self, key):
        """Remove a global variable."""
        if key in self._globals:
            del self._globals[key]

    def clear(self):
        """Clear all global variables."""
        self._globals.clear()

    def __contains__(self, key):
        """Check if a global variable exists."""
        return key in self._globals

    def __getitem__(self, key):
        """Get a global variable, raise KeyError if not found."""
        if key in self._globals:
            return self._globals[key]
        else:
            raise KeyError(f"Global variable '{key}' not found.")

    def __setitem__(self, key, value):
        """Set a global variable."""
        self._globals[key] = value

    def __delitem__(self, key):
        """Delete a global variable."""
        if key in self._globals:
            del self._globals[key]
        else:
            raise KeyError(f"Global variable '{key}' not found.")

    def __repr__(self):
        """Return a string representation of the global variables."""
        return f"TinyGlobals({self._globals})"

    def __len__(self):
        """Return the number of global variables."""
        return len(self._globals)

    def keys(self):
        """Return a list of keys of global variables."""
        return list(self._globals.keys())

    def values(self):
        """Return a list of values of global variables."""
        return list(self._globals.values())

    def items(self):
        """Return a list of (key, value) pairs of global variables."""
        return list(self._globals.items())

    def update(self, other):
        """Update global variables with another dictionary."""
        if isinstance(other, dict):
            self._globals.update(other)
        else:
            raise TypeError("Argument must be a dictionary.")

    def __bool__(self):
        """Return True if there are any global variables, else False."""
        return bool(self._globals)

    def __iter__(self):
        """Return an iterator over the global variables."""
        return iter(self._globals)

    def __getattr__(self, name):
        """Get a global variable as an attribute."""
        if name in self._globals:
            return self._globals[name]
        raise AttributeError(f"'TinyGlobals' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        """Set a global variable as an attribute."""
        if name == "_globals":
            super().__setattr__(name, value)
        else:
            self._globals[name] = value

    def __delattr__(self, name):
        """Delete a global variable as an attribute."""
        if name in self._globals:
            del self._globals[name]
        else:
            raise AttributeError(f"'TinyGlobals' object has no attribute '{name}'")


# use todays date as the default calendar
from datetime import datetime

global_calendar = tiny_time_manager.GameCalendar(
    start_year=datetime.now().year,
    start_month=datetime.now().month,
    start_day=datetime.now().day,
    start_hours=datetime.now().hour,
    start_minutes=datetime.now().minute,
    start_seconds=datetime.now().second,
)
global_time_manager = tiny_time_manager.GameTimeManager(global_calendar)


# Create a global instance
tiny_globals_obj = TinyGlobals()

tiny_globals_obj.set("global_calendar", global_calendar)
tiny_globals_obj.set("global_time_manager", global_time_manager)

# Initialize global GraphManager instance
_global_graph_manager = None


# Add a convenience function to access the global instance
def get_globals():
    """Get the global instance of TinyGlobals."""
    return tiny_globals_obj


# Add a convenience function to set a global variable
def set_global(key, value):
    """Set a global variable."""
    tiny_globals_obj.set(key, value)


# Add a convenience function to get a global variable
def get_global(key, default=None):
    """Get a global variable, return default if not found."""
    return tiny_globals_obj.get(key, default)


# Add a convenience function to remove a global variable
def remove_global(key):
    """Remove a global variable."""
    tiny_globals_obj.remove(key)


# Add a convenience function to clear all global variables
def clear_globals():
    """Clear all global variables."""
    tiny_globals_obj.clear()


# Add a convenience function to check if a global variable exists
def has_global(key) -> bool:
    """Check if a global variable exists. Returns True if it exists, else False."""
    return key in tiny_globals_obj or hasattr(tiny_globals_obj, key)


# Add a convenience function to get all global variable keys
def global_keys():
    """Get all global variable keys."""
    return tiny_globals_obj.keys()


# Add a convenience function to get all global variable values
def global_values():
    """Get all global variable values."""
    return tiny_globals_obj.values()


# GraphManager global instance management
# 
# Note: GraphManager itself implements a singleton pattern (see line 717 in tiny_graph_manager.py),
# which ensures only one GraphManager instance can exist in the application. The global instance
# management here in tiny_globals provides:
#   1. Explicit initialization control via initialize_global_graph_manager()
#   2. Thread-safe access with locks for concurrent initialization
#   3. Test reset functionality via reset_global_graph_manager()
#   4. Clear global access point via get_global_graph_manager()
#
# The two patterns work together: GraphManager's singleton ensures a single instance exists,
# while tiny_globals provides controlled, thread-safe access to that singleton.
_graph_manager_lock = threading.Lock()

def initialize_global_graph_manager():
    """Initialize the global GraphManager instance. Should be called on game start.
    
    Thread-safe initialization ensures only one GraphManager instance is created
    even if called simultaneously from multiple threads.
    """
    global _global_graph_manager
    with _graph_manager_lock:
        if _global_graph_manager is None:
            try:
                from tiny_graph_manager import GraphManager
                _global_graph_manager = GraphManager()
                return _global_graph_manager
            except ImportError as e:
                raise ImportError(f"Failed to import GraphManager: {e}") from e
        return _global_graph_manager


def get_global_graph_manager():
    """Get the global GraphManager instance. Initializes it if not already done.
    
    Thread-safe access to the global GraphManager instance using
    a double-checked locking pattern for optimal performance.
    
    Note: The initialization code is duplicated here rather than calling
    initialize_global_graph_manager() because the double-checked locking
    pattern requires the initialization to be inline within the lock block
    for proper thread safety and performance.
    """
    global _global_graph_manager
    # First check without acquiring the lock for fast-path access
    if _global_graph_manager is None:
        with _graph_manager_lock:
            # Check again inside the lock to ensure only one initializer
            if _global_graph_manager is None:
                try:
                    from tiny_graph_manager import GraphManager
                    _global_graph_manager = GraphManager()
                except ImportError as e:
                    raise ImportError(f"Failed to import GraphManager: {e}") from e
    return _global_graph_manager


def set_global_graph_manager(graph_manager):
    """Set the global GraphManager instance. Use with caution.
    
    Args:
        graph_manager: The GraphManager instance to set as global.
    """
    global _global_graph_manager
    with _graph_manager_lock:
        _global_graph_manager = graph_manager


def has_global_graph_manager():
    """Check if the global GraphManager instance is initialized.
    
    Thread-safe check that acquires the lock to ensure consistent reads.
    
    Note: While reading a reference in Python is atomic, we acquire the lock
    for consistency with other functions and to ensure proper memory visibility
    across threads. The performance overhead is negligible for this check operation.
    """
    with _graph_manager_lock:
        return _global_graph_manager is not None


def reset_global_graph_manager():
    """Reset the global GraphManager instance. Primarily for testing purposes.
    
    This clears both the module-level reference to the global GraphManager
    and the GraphManager singleton instance itself. This is important for tests
    that need a truly clean GraphManager state between runs.
    
    Warning: This should only be used in test code to ensure a clean state
    between test runs. Do not use this in production code.
    """
    global _global_graph_manager
    with _graph_manager_lock:
        # Clear the module-level reference
        _global_graph_manager = None
        
        # Also clear the GraphManager singleton instance, if available.
        # This ensures that subsequent calls to GraphManager() return a fresh
        # instance instead of reusing stale state from previous tests.
        try:
            from tiny_graph_manager import GraphManager
            if hasattr(GraphManager, "_instance"):
                GraphManager._instance = None
        except ImportError:
            # If GraphManager cannot be imported, there is no singleton to reset.
            # This preserves the previous behavior of this function.
            pass

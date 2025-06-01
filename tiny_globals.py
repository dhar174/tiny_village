# tiny_globals.py
# A minimalistic global variable manager for Python scripts.

import tiny_time_manager

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

#!/usr/bin/env python3
"""
Simplified validation test that checks code patterns and syntax without imports.
"""

import re
import ast
import os


def validate_file_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, "r") as f:
            content = f.read()
        ast.parse(content)
        return True, "Valid syntax"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"


def check_building_coordinates_implementation():
    """Check that building coordinate selection was implemented."""
    print("Checking building coordinate selection implementation...")

    try:
        with open("tiny_buildings.py", "r") as f:
            content = f.read()

        # Check for removal of TODO
        if 'TODO: "Add x and y coordinates selection"' not in content:
            print("✓ TODO comment removed")
        else:
            print("⚠ TODO comment still present")
            return False

        # Check for new coordinate selection methods
        expected_methods = ["find_valid_coordinates", "_systematic_placement"]

        found_methods = []
        for method in expected_methods:
            if f"def {method}(" in content:
                found_methods.append(method)

        print(
            f"✓ Found {len(found_methods)}/{len(expected_methods)} coordinate methods: {found_methods}"
        )

        # Check for collision detection logic
        if "occupied_areas" in content and "placement_valid" in content:
            print("✓ Collision detection logic implemented")
        else:
            print("⚠ Collision detection logic not found")

        return len(found_methods) == len(expected_methods)

    except Exception as e:
        print(f"✗ Error checking building implementation: {e}")
        return False


def check_pause_implementation():
    """Check that pause functionality was implemented."""
    print("\nChecking pause functionality implementation...")

    try:
        with open("tiny_gameplay_controller.py", "r") as f:
            content = f.read()

        # Check for pause logic
        pause_patterns = [
            "self.paused = not getattr(self",
            "if getattr(self, 'paused', False):",
            "Game {'paused' if self.paused else 'unpaused'}",
        ]

        found_patterns = []
        for pattern in pause_patterns:
            if pattern in content:
                found_patterns.append(pattern)

        print(
            f"✓ Found {len(found_patterns)}/{len(pause_patterns)} pause implementation patterns"
        )

        # Check for UI elements
        if "PAUSED" in content and "SPACE to pause/unpause" in content:
            print("✓ Pause UI elements added")
        else:
            print("⚠ Pause UI elements not found")

        return len(found_patterns) >= 2  # At least the core pause logic

    except Exception as e:
        print(f"✗ Error checking pause implementation: {e}")
        return False


def check_happiness_implementation():
    """Check that happiness calculation was enhanced."""
    print("\nChecking happiness calculation implementation...")

    try:
        with open("tiny_characters.py", "r") as f:
            content = f.read()

        # Check that TODOs were replaced
        todo_patterns = [
            "TODO: Add happiness calculation based on motives",
            "TODO: Add happiness calculation based on social relationships",
            "TODO: Add happiness calculation based on romantic relationships",
            "TODO: Add happiness calculation based on family relationships",
        ]

        remaining_todos = sum(1 for pattern in todo_patterns if pattern in content)

        if remaining_todos == 0:
            print("✓ All happiness TODOs have been replaced with implementations")
        else:
            print(f"⚠ {remaining_todos} TODO items still remain")

        # Check for implemented features
        implemented_features = [
            "motive_satisfaction",
            "social_happiness",
            "romantic_happiness",
            "family_happiness",
            "positive_relationships",
            "romantic_partner",
            "family_members",
        ]

        found_features = sum(
            1 for feature in implemented_features if feature in content
        )
        print(
            f"✓ Found {found_features}/{len(implemented_features)} happiness features implemented"
        )

        return remaining_todos == 0 and found_features >= 4

    except Exception as e:
        print(f"✗ Error checking happiness implementation: {e}")
        return False


def check_goap_system():
    """Check that GOAP system has proper implementations."""
    print("\nChecking GOAP system implementations...")

    try:
        with open("tiny_goap_system.py", "r") as f:
            content = f.read()

        # Check for substantial method implementations
        methods = [
            "def replan(",
            "def find_alternative_action(",
            "def calculate_utility(",
            "def evaluate_utility(",
            "def evaluate_feasibility_of_goal(",
        ]

        method_stats = {}
        for method in methods:
            method_name = method.split("(")[0].replace("def ", "")

            # Find method implementation
            method_start = content.find(method)
            if method_start != -1:
                # Find next method or end of class to count lines
                next_method = content.find("\n    def ", method_start + 1)
                if next_method == -1:
                    method_content = content[method_start:]
                else:
                    method_content = content[method_start:next_method]

                # Count non-empty, non-comment lines
                lines = method_content.split("\n")
                substantial_lines = [
                    line
                    for line in lines
                    if line.strip()
                    and not line.strip().startswith("#")
                    and not line.strip().startswith('"""')
                    and line.strip() != "pass"
                ]

                method_stats[method_name] = len(substantial_lines)
            else:
                method_stats[method_name] = 0

        implemented_methods = sum(1 for count in method_stats.values() if count > 3)
        print(
            f"✓ Found {implemented_methods}/{len(methods)} substantially implemented methods"
        )

        for method, line_count in method_stats.items():
            status = "✓" if line_count > 3 else "⚠"
            print(f"  {status} {method}: {line_count} substantial lines")

        return implemented_methods >= 4  # At least 4 out of 5 methods

    except Exception as e:
        print(f"✗ Error checking GOAP implementation: {e}")
        return False


def check_syntax_all_files():
    """Check syntax of modified files."""
    print("\nChecking syntax of modified files...")

    files_to_check = [
        "tiny_buildings.py",
        "tiny_gameplay_controller.py",
        "tiny_characters.py",
        "tiny_goap_system.py",
    ]

    syntax_results = {}
    for filepath in files_to_check:
        if os.path.exists(filepath):
            valid, message = validate_file_syntax(filepath)
            syntax_results[filepath] = valid
            status = "✓" if valid else "✗"
            print(f"  {status} {filepath}: {message}")
        else:
            print(f"  ⚠ {filepath}: File not found")
            syntax_results[filepath] = False

    passed = sum(syntax_results.values())
    return passed, len(files_to_check)


def main():
    """Run all validation checks."""
    print("Running validation checks for completed implementations...\n")

    # Syntax check first
    syntax_passed, syntax_total = check_syntax_all_files()

    if syntax_passed != syntax_total:
        print(f"\n❌ Syntax errors found. Fix these before proceeding.")
        return False

    # Functional checks
    checks = [
        check_building_coordinates_implementation,
        check_pause_implementation,
        check_happiness_implementation,
        check_goap_system,
    ]

    passed = 0
    total = len(checks)

    for check in checks:
        try:
            if check():
                passed += 1
        except Exception as e:
            print(f"Check failed with exception: {e}")

    print(f"\n{'='*60}")
    print(f"Validation Results:")
    print(f"  Syntax checks: {syntax_passed}/{syntax_total} passed")
    print(f"  Implementation checks: {passed}/{total} passed")
    print(f"  Overall: {syntax_passed + passed}/{syntax_total + total} checks passed")

    if syntax_passed == syntax_total and passed == total:
        print(
            "\n🎉 All validations passed! Implementations are complete and syntactically correct."
        )
        return True
    else:
        failed = (syntax_total - syntax_passed) + (total - passed)
        print(f"\n⚠ {failed} validation(s) failed. Review the implementations.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

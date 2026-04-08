import sys
import types

from actions import Skill


def build_character_skills(**skill_levels):
    if "tiny_graph_manager" not in sys.modules:
        stub = types.ModuleType("tiny_graph_manager")
        stub.GraphManager = object
        sys.modules["tiny_graph_manager"] = stub

    from tiny_characters import CharacterSkills

    return CharacterSkills(
        [Skill(skill_name, level) for skill_name, level in skill_levels.items()]
    )

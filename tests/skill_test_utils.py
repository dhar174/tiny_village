import sys
import types
from unittest.mock import patch

from actions import Skill


def build_character_skills(**skill_levels):
    if "tiny_graph_manager" in sys.modules:
        from tiny_characters import CharacterSkills
    else:
        stub = types.ModuleType("tiny_graph_manager")
        stub.GraphManager = object
        with patch.dict(sys.modules, {"tiny_graph_manager": stub}):
            from tiny_characters import CharacterSkills

    return CharacterSkills(
        [Skill(skill_name, level) for skill_name, level in skill_levels.items()]
    )

from actions import State
from tests.mock_character import MockCharacter, MockPersonalityTraits


class CharacterWithState(MockCharacter):
    """MockCharacter variant that can participate in real Condition checks."""

    def get_state(self):
        return State(self)


def build_character(name, default_energy=10.0, **overrides):
    defaults = {
        "name": name,
        "energy": default_energy,
        "friendship_grid": {},
        "personality_traits": MockPersonalityTraits(
            agreeableness=5,
            extraversion=5,
        ),
    }
    defaults.update(overrides)
    return CharacterWithState(**defaults)

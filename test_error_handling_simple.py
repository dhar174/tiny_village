#!/usr/bin/env python3
"""
Regression tests for real error-handling paths.

These tests exercise the actual fallback helpers instead of simulating their
logic with hand-written booleans.
"""

import os
import tempfile
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import tiny_map_controller
from tiny_memories import FlatMemoryAccess


class FakeSurface:
    def __init__(self, size):
        self.size = size
        self.fill_calls = []

    def fill(self, color):
        self.fill_calls.append(color)


class FakeDraw:
    def __init__(self):
        self.rect_calls = []
        self.circle_calls = []

    def rect(self, surface, color, rect, width=0):
        self.rect_calls.append((surface, color, rect, width))

    def circle(self, surface, color, center, radius):
        self.circle_calls.append((surface, color, center, radius))


@dataclass
class DummySpecificMemory:
    description: str
    embedding_tuple: tuple
    embedding: np.ndarray | None = None
    att_mask: np.ndarray | None = None

    def get_embedding(self):
        return self.embedding_tuple


class DummyFlatMemoryAccess:
    def __init__(self, memories):
        self._memories = list(memories)
        self.index_is_normalized = False
        self.normalization_calls = []

    def set_all_memory_embeddings_to_normalized(self, normalized=None):
        self.normalization_calls.append(normalized)

    def get_specific_memories(self):
        return list(self._memories)

    def get_specific_memory_by_description(self, description):
        for memory in self._memories:
            if memory.description == description:
                return memory
        return None


class TestErrorHandlingLogic(unittest.TestCase):
    def setUp(self):
        self.controller = tiny_map_controller.MapController.__new__(
            tiny_map_controller.MapController
        )

    def test_load_map_image_returns_default_surface_when_path_missing(self):
        sentinel_surface = object()

        with patch.object(
            self.controller, "_create_default_map_image", return_value=sentinel_surface
        ) as create_default:
            self.assertIs(
                self.controller._load_map_image_safely(""), sentinel_surface
            )
            create_default.assert_called_once_with()

        with patch.object(
            self.controller, "_create_default_map_image", return_value=sentinel_surface
        ) as create_default, patch(
            "tiny_map_controller.os.path.exists", return_value=False
        ):
            self.assertIs(
                self.controller._load_map_image_safely("missing-map.png"),
                sentinel_surface,
            )
            create_default.assert_called_once_with()

    def test_load_map_image_uses_pygame_loader_for_existing_file(self):
        loaded_surface = object()

        with patch("tiny_map_controller.os.path.exists", return_value=True), patch.object(
            tiny_map_controller.pygame.image, "load", return_value=loaded_surface
        ) as image_load:
            result = self.controller._load_map_image_safely("valid-map.png")

        self.assertIs(result, loaded_surface)
        image_load.assert_called_once_with("valid-map.png")

    def test_create_default_map_image_draws_expected_fallback_features(self):
        fake_draw = FakeDraw()
        fake_pygame = SimpleNamespace(
            Surface=lambda size: FakeSurface(size),
            draw=fake_draw,
        )

        with patch.object(tiny_map_controller, "pygame", fake_pygame):
            surface = self.controller._create_default_map_image(width=120, height=80)

        self.assertEqual(surface.fill_calls, [(34, 139, 34)])
        self.assertIn(
            (surface, (139, 69, 19), (40, 0, 40, 80), 0),
            fake_draw.rect_calls,
        )
        self.assertIn(
            (surface, (139, 69, 19), (0, 20, 120, 40), 0),
            fake_draw.rect_calls,
        )
        self.assertIn(
            (surface, (101, 67, 33), (0, 0, 120, 80), 5),
            fake_draw.rect_calls,
        )
        self.assertIn(
            (surface, (65, 105, 225), (30, 20), 50),
            fake_draw.circle_calls,
        )

    def test_save_specific_memory_embeddings_uses_real_file_operations(self):
        memories = [
            DummySpecificMemory(
                "market memory",
                (np.array([1.0, 2.0]), np.array([1, 1])),
            ),
            DummySpecificMemory(
                "river memory",
                (np.array([3.0, 4.0]), np.array([1, 0])),
            ),
        ]
        flat_access = DummyFlatMemoryAccess(memories)

        with tempfile.TemporaryDirectory() as temp_dir:
            base_filename = os.path.join(temp_dir, "nested", "memory_store")
            result = FlatMemoryAccess.save_all_specific_memories_embeddings_to_file(
                flat_access, base_filename
            )

            self.assertTrue(result)
            self.assertTrue(
                os.path.exists(f"{base_filename}_embeddings.npy")
            )
            self.assertTrue(
                os.path.exists(f"{base_filename}_att_mask.npy")
            )

    def test_save_specific_memory_embeddings_rejects_empty_filename(self):
        flat_access = DummyFlatMemoryAccess([])

        result = FlatMemoryAccess.save_all_specific_memories_embeddings_to_file(
            flat_access, ""
        )

        self.assertFalse(result)

    def test_load_specific_memory_embeddings_populates_real_memory_objects(self):
        stored_memories = [
            DummySpecificMemory(
                "market memory",
                (np.array([1.0, 2.0]), np.array([1, 1])),
            ),
            DummySpecificMemory(
                "river memory",
                (np.array([3.0, 4.0]), np.array([1, 0])),
            ),
        ]
        saving_access = DummyFlatMemoryAccess(stored_memories)

        with tempfile.TemporaryDirectory() as temp_dir:
            base_filename = os.path.join(temp_dir, "memory_store")
            self.assertTrue(
                FlatMemoryAccess.save_all_specific_memories_embeddings_to_file(
                    saving_access, base_filename
                )
            )

            reloaded_memories = [
                DummySpecificMemory("market memory", (np.array([]), np.array([]))),
                DummySpecificMemory("river memory", (np.array([]), np.array([]))),
            ]
            loading_access = DummyFlatMemoryAccess(reloaded_memories)

            result = FlatMemoryAccess.load_all_specific_memories_embeddings_from_file(
                loading_access, base_filename
            )

        self.assertTrue(result)
        np.testing.assert_array_equal(
            reloaded_memories[0].embedding, np.array([1.0, 2.0])
        )
        np.testing.assert_array_equal(
            reloaded_memories[0].att_mask, np.array([1, 1])
        )
        np.testing.assert_array_equal(
            reloaded_memories[1].embedding, np.array([3.0, 4.0])
        )
        np.testing.assert_array_equal(
            reloaded_memories[1].att_mask, np.array([1, 0])
        )


if __name__ == "__main__":
    unittest.main()

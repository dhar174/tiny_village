"""
Animation System for TinyVillage

This module provides a comprehensive animation system that manages character animations,
state transitions, and visual feedback for character actions.
"""

import logging
import time
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


class AnimationState(Enum):
    """Enumeration of possible animation states."""

    IDLE = "idle"
    WALKING = "walking"
    RUNNING = "running"
    TALKING = "talking"
    WORKING = "working"
    EATING = "eating"
    SLEEPING = "sleeping"
    DANCING = "dancing"
    FIGHTING = "fighting"
    CRAFTING = "crafting"
    READING = "reading"
    THINKING = "thinking"
    CELEBRATING = "celebrating"
    MOURNING = "mourning"
    EXERCISING = "exercising"
    TAKING_ITEM = "taking_item"
    GIVING_ITEM = "giving_item"
    BUILDING = "building"
    HEALING = "healing"


@dataclass
class AnimationFrame:
    """Represents a single frame in an animation sequence."""

    frame_id: int
    duration: float  # Duration in seconds
    sprite_path: Optional[str] = None
    position_offset: tuple = (0, 0)
    scale: float = 1.0
    rotation: float = 0.0
    effects: List[str] = None  # Special effects like particles, sounds

    def __post_init__(self):
        if self.effects is None:
            self.effects = []


@dataclass
class Animation:
    """Represents a complete animation sequence."""

    name: str
    state: AnimationState
    frames: List[AnimationFrame]
    loop: bool = False
    priority: int = 0  # Higher priority animations can interrupt lower ones
    duration: float = 0.0

    def __post_init__(self):
        if not self.duration:
            self.duration = sum(frame.duration for frame in self.frames)


class AnimationController:
    """Controls animation playback for a single character."""

    def __init__(self, character_name: str):
        self.character_name = character_name
        self.current_animation: Optional[Animation] = None
        self.current_frame_index: int = 0
        self.frame_start_time: float = 0.0
        self.animation_start_time: float = 0.0
        self.is_playing: bool = False
        self.animation_queue: List[Animation] = []

    def play_animation(self, animation: Animation, force: bool = False) -> bool:
        """
        Play an animation.

        Args:
            animation: The animation to play
            force: If True, interrupt current animation regardless of priority

        Returns:
            bool: True if animation started playing, False if rejected
        """
        try:
            # Check if we can interrupt current animation
            if self.current_animation and not force:
                if animation.priority <= self.current_animation.priority:
                    # Queue the animation for later
                    self.animation_queue.append(animation)
                    logging.debug(
                        f"Queued animation {animation.name} for {self.character_name}"
                    )
                    return False

            # Start the new animation
            self.current_animation = animation
            self.current_frame_index = 0
            self.animation_start_time = time.time()
            self.frame_start_time = time.time()
            self.is_playing = True

            logging.info(
                f"Started animation {animation.name} for {self.character_name}"
            )
            return True

        except Exception as e:
            logging.error(
                f"Error playing animation {animation.name} for {self.character_name}: {e}"
            )
            return False

    def update(self, delta_time: float) -> Optional[AnimationFrame]:
        """
        Update the animation controller.

        Args:
            delta_time: Time elapsed since last update in seconds

        Returns:
            Current animation frame or None if no animation playing
        """
        if not self.is_playing or not self.current_animation:
            return None

        try:
            current_time = time.time()

            # Check if current frame is finished
            if len(self.current_animation.frames) > 0:
                current_frame = self.current_animation.frames[self.current_frame_index]

                if current_time - self.frame_start_time >= current_frame.duration:
                    # Move to next frame
                    self.current_frame_index += 1
                    self.frame_start_time = current_time

                    # Check if animation is complete
                    if self.current_frame_index >= len(self.current_animation.frames):
                        if self.current_animation.loop:
                            self.current_frame_index = 0
                        else:
                            self.stop_animation()
                            return None

                # Return current frame
                if self.current_frame_index < len(self.current_animation.frames):
                    return self.current_animation.frames[self.current_frame_index]

            return None

        except Exception as e:
            logging.error(f"Error updating animation for {self.character_name}: {e}")
            self.stop_animation()
            return None

    def stop_animation(self):
        """Stop the current animation and play next queued animation if any."""
        if self.current_animation:
            logging.info(
                f"Stopped animation {self.current_animation.name} for {self.character_name}"
            )

        self.current_animation = None
        self.current_frame_index = 0
        self.is_playing = False

        # Play next animation in queue
        if self.animation_queue:
            next_animation = self.animation_queue.pop(0)
            self.play_animation(next_animation)

    def get_current_state(self) -> AnimationState:
        """Get the current animation state."""
        if self.current_animation:
            return self.current_animation.state
        return AnimationState.IDLE


class AnimationSystem:
    """Global animation system managing all character animations."""

    def __init__(self):
        self.controllers: Dict[str, AnimationController] = {}
        self.animation_library: Dict[str, Animation] = {}
        self.default_animations: Dict[AnimationState, Animation] = {}
        self._initialize_default_animations()

    def _initialize_default_animations(self):
        """Initialize default animations for all states."""
        # Create basic animations for each state
        for state in AnimationState:
            animation_name = f"default_{state.value}"

            # Create simple single-frame animation
            frame = AnimationFrame(
                frame_id=0,
                duration=1.0,  # 1 second duration
                sprite_path=f"animations/{state.value}/default.png",
            )

            animation = Animation(
                name=animation_name,
                state=state,
                frames=[frame],
                loop=True if state == AnimationState.IDLE else False,
                priority=1,
            )

            self.animation_library[animation_name] = animation
            self.default_animations[state] = animation

    def get_controller(self, character_name: str) -> AnimationController:
        """Get or create animation controller for a character."""
        if character_name not in self.controllers:
            self.controllers[character_name] = AnimationController(character_name)
        return self.controllers[character_name]

    def play_animation(
        self, character_name: str, animation_name: str, force: bool = False
    ) -> bool:
        """
        Play an animation for a character.

        Args:
            character_name: Name of the character
            animation_name: Name of the animation or animation state
            force: If True, interrupt current animation

        Returns:
            bool: True if animation started playing
        """
        try:
            controller = self.get_controller(character_name)

            # Try to find animation by name first
            animation = None
            if animation_name in self.animation_library:
                animation = self.animation_library[animation_name]
            else:
                # Try to find by state name
                try:
                    state = AnimationState(animation_name.lower())
                    animation = self.default_animations.get(state)
                except ValueError:
                    logging.warning(f"Unknown animation: {animation_name}")
                    return False

            if animation:
                return controller.play_animation(animation, force)
            else:
                logging.warning(f"Animation {animation_name} not found")
                return False

        except Exception as e:
            logging.error(
                f"Error playing animation {animation_name} for {character_name}: {e}"
            )
            return False

    def update_all(self, delta_time: float):
        """Update all animation controllers."""
        for controller in self.controllers.values():
            controller.update(delta_time)

    def stop_animation(self, character_name: str):
        """Stop animation for a character."""
        if character_name in self.controllers:
            self.controllers[character_name].stop_animation()

    def get_character_state(self, character_name: str) -> AnimationState:
        """Get current animation state for a character."""
        if character_name in self.controllers:
            return self.controllers[character_name].get_current_state()
        return AnimationState.IDLE

    def register_animation(self, animation: Animation):
        """Register a new animation in the library."""
        self.animation_library[animation.name] = animation
        logging.info(f"Registered animation: {animation.name}")

    def create_custom_animation(
        self,
        name: str,
        state: AnimationState,
        frame_data: List[Dict[str, Any]],
        loop: bool = False,
        priority: int = 1,
    ) -> Animation:
        """
        Create a custom animation from frame data.

        Args:
            name: Name of the animation
            state: Animation state
            frame_data: List of frame dictionaries
            loop: Whether animation should loop
            priority: Animation priority

        Returns:
            Created Animation object
        """
        frames = []
        for i, frame_info in enumerate(frame_data):
            frame = AnimationFrame(
                frame_id=i,
                duration=frame_info.get("duration", 1.0),
                sprite_path=frame_info.get("sprite_path"),
                position_offset=frame_info.get("position_offset", (0, 0)),
                scale=frame_info.get("scale", 1.0),
                rotation=frame_info.get("rotation", 0.0),
                effects=frame_info.get("effects", []),
            )
            frames.append(frame)

        animation = Animation(
            name=name, state=state, frames=frames, loop=loop, priority=priority
        )

        self.register_animation(animation)
        return animation


# Global animation system instance
_animation_system = None


def get_animation_system() -> AnimationSystem:
    """Get the global animation system instance."""
    global _animation_system
    if _animation_system is None:
        _animation_system = AnimationSystem()
    return _animation_system


def play_character_animation(
    character_name: str, animation_name: str, force: bool = False
) -> bool:
    """
    Convenience function to play an animation for a character.

    Args:
        character_name: Name of the character
        animation_name: Name of the animation or state
        force: Whether to force interrupt current animation

    Returns:
        bool: True if animation started playing
    """
    return get_animation_system().play_animation(character_name, animation_name, force)

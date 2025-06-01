import json
import re
import json
import inspect
from typing import Dict, Any, Optional, Union
from actions import (
    Action,
    TalkAction,
    ExploreAction,
    GreetAction,
    ShareNewsAction,
    OfferComplimentAction,
)


# Custom Exceptions
class InvalidLLMResponseFormatError(Exception):
    """Raised when the LLM response string is not valid JSON."""

    pass


class UnknownActionError(Exception):
    """Raised when the action specified in the LLM response is not recognized."""

    pass


class InvalidActionParametersError(Exception):
    """Raised when the parameters for a recognized action are missing or invalid."""

    pass


class LLMResponseParsingError(Exception):
    """Raised when LLM response cannot be parsed or understood."""

    pass


# Comprehensive Action Classes for Output Interpreter
# These inherit from actions.Action to be compatible with the expected Action type hint.


class EatAction(Action):
    def __init__(self, item_name, initiator_id=None, **kwargs):
        super().__init__(
            name="Eat",
            initiator=initiator_id,
            target=item_name,
            cost=kwargs.pop("cost", 0.1),
            preconditions=kwargs.pop("preconditions", []),
            effects=kwargs.pop(
                "effects",
                [
                    {"attribute": "hunger", "change_value": -2, "target": "initiator"},
                    {"attribute": "energy", "change_value": 1, "target": "initiator"},
                ],
            ),
            **kwargs,
        )
        self.item_name = item_name


class GoToLocationAction(Action):
    def __init__(self, location_name, initiator_id=None, **kwargs):
        super().__init__(
            name="GoToLocation",
            initiator=initiator_id,
            target=location_name,
            cost=kwargs.pop("cost", 0.2),
            preconditions=kwargs.pop("preconditions", []),
            effects=kwargs.pop(
                "effects",
                [
                    {
                        "attribute": "location",
                        "change_value": location_name,
                        "target": "initiator",
                    }
                ],
            ),
            **kwargs,
        )
        self.location_name = location_name


class NoOpAction(Action):
    def __init__(self, initiator_id=None, **kwargs):
        super().__init__(
            name="NoOp",
            initiator=initiator_id,
            cost=kwargs.pop("cost", 0),
            preconditions=kwargs.pop("preconditions", []),
            effects=kwargs.pop("effects", []),
            **kwargs,
        )


class BuyFoodAction(Action):
    def __init__(self, food_type="food", initiator_id=None, **kwargs):
        super().__init__(
            name="BuyFood",
            initiator=initiator_id,
            target=food_type,
            cost=kwargs.pop("cost", 0.3),
            preconditions=kwargs.pop("preconditions", []),
            effects=kwargs.pop(
                "effects",
                [
                    {"attribute": "money", "change_value": -5, "target": "initiator"},
                    {
                        "attribute": "inventory",
                        "change_value": f"add:{food_type}",
                        "target": "initiator",
                    },
                ],
            ),
            **kwargs,
        )
        self.food_type = food_type


class WorkAction(Action):
    def __init__(self, job_type="current_job", initiator_id=None, **kwargs):
        super().__init__(
            name="Work",
            initiator=initiator_id,
            target=job_type,
            cost=kwargs.pop("cost", 0.4),
            preconditions=kwargs.pop("preconditions", []),
            effects=kwargs.pop(
                "effects",
                [
                    {"attribute": "money", "change_value": 20, "target": "initiator"},
                    {"attribute": "energy", "change_value": -2, "target": "initiator"},
                    {
                        "attribute": "job_performance",
                        "change_value": 1,
                        "target": "initiator",
                    },
                ],
            ),
            **kwargs,
        )
        self.job_type = job_type


class SleepAction(Action):
    def __init__(self, duration=8, initiator_id=None, **kwargs):
        super().__init__(
            name="Sleep",
            initiator=initiator_id,
            cost=kwargs.pop("cost", 0.0),
            preconditions=kwargs.pop("preconditions", []),
            effects=kwargs.pop(
                "effects",
                [
                    {"attribute": "energy", "change_value": 5, "target": "initiator"},
                    {"attribute": "health", "change_value": 1, "target": "initiator"},
                ],
            ),
            **kwargs,
        )
        self.duration = duration


class SocialVisitAction(Action):
    def __init__(self, target_person, initiator_id=None, **kwargs):
        super().__init__(
            name="SocialVisit",
            initiator=initiator_id,
            target=target_person,
            cost=kwargs.pop("cost", 0.3),
            preconditions=kwargs.pop("preconditions", []),
            effects=kwargs.pop(
                "effects",
                [
                    {
                        "attribute": "social_wellbeing",
                        "change_value": 2,
                        "target": "initiator",
                    },
                    {
                        "attribute": "happiness",
                        "change_value": 1,
                        "target": "initiator",
                    },
                    {
                        "attribute": "friendship",
                        "change_value": 1,
                        "target": target_person,
                    },
                ],
            ),
            **kwargs,
        )
        self.target_person = target_person


class ImproveJobPerformanceAction(Action):
    def __init__(self, method="study", initiator_id=None, **kwargs):
        super().__init__(
            name="ImproveJobPerformance",
            initiator=initiator_id,
            cost=kwargs.pop("cost", 0.5),
            preconditions=kwargs.pop("preconditions", []),
            effects=kwargs.pop(
                "effects",
                [
                    {
                        "attribute": "job_performance",
                        "change_value": 3,
                        "target": "initiator",
                    },
                    {"attribute": "energy", "change_value": -1, "target": "initiator"},
                ],
            ),
            **kwargs,
        )
        self.method = method


class PursueHobbyAction(Action):
    def __init__(self, hobby_type="reading", initiator_id=None, **kwargs):
        super().__init__(
            name="PursueHobby",
            initiator=initiator_id,
            target=hobby_type,
            cost=kwargs.pop("cost", 0.2),
            preconditions=kwargs.pop("preconditions", []),
            effects=kwargs.pop(
                "effects",
                [
                    {
                        "attribute": "happiness",
                        "change_value": 2,
                        "target": "initiator",
                    },
                    {
                        "attribute": "mental_health",
                        "change_value": 1,
                        "target": "initiator",
                    },
                ],
            ),
            **kwargs,
        )
        self.hobby_type = hobby_type


class VisitDoctorAction(Action):
    def __init__(self, reason="checkup", initiator_id=None, **kwargs):
        super().__init__(
            name="VisitDoctor",
            initiator=initiator_id,
            cost=kwargs.pop("cost", 0.6),
            preconditions=kwargs.pop("preconditions", []),
            effects=kwargs.pop(
                "effects",
                [
                    {"attribute": "health", "change_value": 3, "target": "initiator"},
                    {"attribute": "money", "change_value": -10, "target": "initiator"},
                ],
            ),
            **kwargs,
        )
        self.reason = reason


class OutputInterpreter:
    def __init__(self, graph_manager=None):
        self.graph_manager = graph_manager

        # Comprehensive action class mapping for all 50+ defined actions
        self.action_class_map = {
            # Basic Actions
            "Eat": EatAction,
            "eat": EatAction,  # Add lowercase variant
            "GoTo": GoToLocationAction,
            "GoToLocation": GoToLocationAction,
            "Talk": TalkAction,
            "talk": TalkAction,  # Add lowercase variant
            "NoOp": NoOpAction,
            "Sleep": SleepAction,
            "Work": WorkAction,
            # Social Actions
            "Greet": GreetAction,
            "greet": GreetAction,  # Add lowercase variant
            "ShareNews": ShareNewsAction,
            "OfferCompliment": OfferComplimentAction,
            "SocialVisit": SocialVisitAction,
            "Explore": ExploreAction,
            # Food & Health Actions
            "BuyFood": BuyFoodAction,
            "buy_food": BuyFoodAction,
            "eat_food": EatAction,
            "VisitDoctor": VisitDoctorAction,
            "visit_doctor": VisitDoctorAction,
            "take_medicine": self._create_generic_action("TakeMedicine", cost=0.1),
            "buy_medicine": self._create_generic_action("BuyMedicine", cost=0.2),
            # Work & Career Actions
            "improve_job_performance": ImproveJobPerformanceAction,
            "go_to_work": WorkAction,
            "work_current_job": WorkAction,
            "collaborate_colleagues": self._create_social_action(
                "CollaborateColleagues", cost=0.4
            ),
            "get_educated": self._create_generic_action("GetEducated", cost=0.8),
            "start_business": self._create_generic_action("StartBusiness", cost=1.0),
            # Mental Health & Wellness Actions
            "improve_mental_health": self._create_generic_action(
                "ImproveMentalHealth", cost=0.3
            ),
            "pursue_hobby": PursueHobbyAction,
            "self_care": self._create_generic_action("self_care", cost=0.2),
            "leisure_activity": self._create_generic_action(
                "LeisureActivity", cost=0.2
            ),
            # Social & Community Actions
            "increase_friendship": self._create_social_action(
                "IncreaseFriendship", cost=0.3
            ),
            "social_visit": SocialVisitAction,
            "attend_event": self._create_generic_action("AttendEvent", cost=0.4),
            "organize_event": self._create_generic_action("OrganizeEvent", cost=0.6),
            "volunteer_time": self._create_generic_action("VolunteerTime", cost=0.5),
            # Goal & Planning Actions
            "set_goal": self._create_generic_action("SetGoal", cost=0.1),
            # Material & Economic Actions
            "trade_goods": self._create_generic_action("TradeGoods", cost=0.3),
            "gather_resource": self._create_generic_action("GatherResource", cost=0.4),
            "repair_item": self._create_generic_action("RepairItem", cost=0.3),
            "craft_item": self._create_generic_action("CraftItem", cost=0.5),
            "invest_wealth": self._create_generic_action("InvestWealth", cost=0.2),
            "buy_property": self._create_generic_action("BuyProperty", cost=1.5),
            "sell_property": self._create_generic_action("SellProperty", cost=0.5),
            # Location & Environment Actions
            "move_to_new_location": self._create_movement_action(
                "MoveToNewLocation", cost=0.8
            ),
            "clean_up": self._create_generic_action("CleanUp", cost=0.3),
            "improve_shelter": self._create_generic_action("ImproveShelter", cost=0.7),
            # Service & Utility Actions
            "commission_service": self._create_generic_action(
                "CommissionService", cost=0.4
            ),
            "research_new_technology": self._create_generic_action(
                "ResearchNewTechnology", cost=0.9
            ),
            # Fallback actions
            "unknown": NoOpAction,
            "default": NoOpAction,
        }

        # Action categories for specialized parsing
        self.movement_actions = {"GoTo", "GoToLocation", "move_to_new_location"}

        self.social_actions = {
            "Talk",
            "Greet",
            "ShareNews",
            "OfferCompliment",
            "SocialVisit",
            "increase_friendship",
            "social_visit",
            "collaborate_colleagues",
        }

        self.work_actions = {
            "Work",
            "work_current_job",
            "go_to_work",
            "improve_job_performance",
            "get_educated",
            "start_business",
        }

        self.creative_actions = {
            "pursue_hobby",
            "craft_item",
            "organize_event",
            "research_new_technology",
        }

        # Common parameter patterns for validation
        self.required_parameters = {
            "movement": ["location", "destination", "location_name"],
            "social": ["target", "target_name", "target_person"],
            "work": ["job_type", "task", "method"],
            "creative": ["hobby_type", "item_type", "topic", "subject"],
        }

    def _create_generic_action(self, name: str, cost: float = 0.3):
        """Factory method to create generic action classes"""

        class GenericAction(Action):
            def __init__(self, initiator_id=None, **kwargs):
                super().__init__(
                    name=name,
                    initiator=initiator_id,
                    cost=kwargs.pop("cost", cost),
                    preconditions=kwargs.pop("preconditions", []),
                    effects=kwargs.pop(
                        "effects",
                        [
                            {
                                "attribute": "satisfaction",
                                "change_value": 1,
                                "target": "initiator",
                            }
                        ],
                    ),
                    **kwargs,
                )

        return GenericAction

    def _create_social_action(self, name: str, cost: float = 0.3):
        """Factory method to create social action classes"""

        class SocialAction(Action):
            def __init__(
                self, initiator_id=None, target_person=None, target=None, **kwargs
            ):
                # Handle target parameter variations
                if target_person is None and target is not None:
                    target_person = target
                elif target_person is None and target is None:
                    # Try to get from other target variations in kwargs
                    target_person = kwargs.pop("target_name", None)

                if target_person is None:
                    raise ValueError(
                        f"Missing required target_person parameter for {name}"
                    )

                # Filter out parameters that base Action class doesn't accept
                base_action_params = {
                    "name",
                    "preconditions",
                    "effects",
                    "cost",
                    "target",
                    "initiator",
                    "related_skills",
                    "default_target_is_initiator",
                    "impact_rating_on_target",
                    "impact_rating_on_initiator",
                    "impact_rating_on_other",
                    "action_id",
                    "created_at",
                    "expires_at",
                    "completed_at",
                    "priority",
                    "related_goal",
                }
                filtered_kwargs = {
                    k: v for k, v in kwargs.items() if k in base_action_params
                }

                super().__init__(
                    name=name,
                    initiator=initiator_id,
                    target=target_person,
                    cost=kwargs.pop("cost", cost),
                    preconditions=kwargs.pop("preconditions", []),
                    effects=kwargs.pop(
                        "effects",
                        [
                            {
                                "attribute": "social_wellbeing",
                                "change_value": 1,
                                "target": "initiator",
                            },
                            {
                                "attribute": "relationship",
                                "change_value": 1,
                                "target": target_person,
                            },
                        ],
                    ),
                    **filtered_kwargs,
                )
                self.target_person = target_person

        return SocialAction

    def _create_movement_action(self, name: str, cost: float = 0.4):
        """Factory method to create movement action classes"""

        class MovementAction(Action):
            def __init__(self, destination, initiator_id=None, **kwargs):
                super().__init__(
                    name=name,
                    initiator=initiator_id,
                    target=destination,
                    cost=kwargs.pop("cost", cost),
                    preconditions=kwargs.pop("preconditions", []),
                    effects=kwargs.pop(
                        "effects",
                        [
                            {
                                "attribute": "location",
                                "change_value": destination,
                                "target": "initiator",
                            }
                        ],
                    ),
                    **kwargs,
                )
                self.destination = destination

        return MovementAction

    def parse_llm_response(self, llm_response_str: str) -> dict:
        """
        Enhanced parsing of LLM response with multiple format support and error recovery.
        Supports JSON format, natural language, and mixed formats.
        Priority: JSON extraction > strict JSON > natural language > fallback
        """
        if not llm_response_str:
            # For enhanced interpreter, return NoOp for empty strings instead of raising error
            return {"action": "NoOp", "parameters": {"reason": "empty_response"}}

        # Clean and normalize the response
        cleaned_response = llm_response_str.strip()

        # First priority: Try extracting JSON from mixed format (handles mixed responses)
        try:
            parsed_response = self._extract_json_from_text(cleaned_response)
            return parsed_response
        except (json.JSONDecodeError, InvalidLLMResponseFormatError):
            pass

        # Second priority: Try strict JSON parsing
        try:
            parsed_response = self._parse_json_response(cleaned_response)
            return parsed_response
        except (json.JSONDecodeError, InvalidLLMResponseFormatError):
            pass

        # Third priority: Try natural language parsing
        try:
            parsed_response = self._parse_natural_language_response(cleaned_response)
            return parsed_response
        except LLMResponseParsingError:
            pass

        # Fallback: Create a basic action from keywords
        return self._create_fallback_action_dict(cleaned_response)

    def _parse_json_response(self, response_str: str) -> dict:
        """Parse strict JSON format response"""
        try:
            parsed_response = json.loads(response_str)
        except json.JSONDecodeError as e:
            raise InvalidLLMResponseFormatError(f"Invalid JSON format: {e}")

        if not isinstance(parsed_response, dict):
            raise InvalidLLMResponseFormatError("LLM response must be a JSON object.")
        if "action" not in parsed_response:
            raise InvalidLLMResponseFormatError("LLM response missing 'action' key.")
        if not isinstance(parsed_response["action"], str):
            raise InvalidLLMResponseFormatError("'action' key must be a string.")
        if "parameters" not in parsed_response:
            raise InvalidLLMResponseFormatError(
                "LLM response missing 'parameters' key."
            )
        if not isinstance(parsed_response["parameters"], dict):
            raise InvalidLLMResponseFormatError(
                "'parameters' key must be a dictionary."
            )

        return parsed_response

    def _parse_natural_language_response(self, response_str: str) -> dict:
        """Parse natural language responses and extract action/parameters"""
        response_lower = response_str.lower()

        # Common action patterns - ordered by specificity to avoid conflicts
        action_patterns = [
            (
                r"(?:go to|move to|travel to|visit)\s+([^\s,]+(?:\s+[^\s,]+)*)",
                "GoTo",
                "location_name",
            ),
            (
                r"(?:talk to|speak with|chat with)\s+([^\s,]+(?:\s+[^\s,]+)*)",
                "Talk",
                "target_name",
            ),
            (
                r"(?:greet|say hello to|welcome)\s+([^\s,]+(?:\s+[^\s,]+)*)",
                "Greet",
                "target_name",
            ),
            (
                r"(?:share news with|tell news to)\s+([^\s,]+(?:\s+[^\s,]+)*)",
                "ShareNews",
                "target_name",
            ),
            (
                r"(?:compliment|praise)\s+([^\s,]+(?:\s+[^\s,]+)*)",
                "OfferCompliment",
                "target_name",
            ),
            (r"(?:visit doctor|see doctor|medical checkup)", "VisitDoctor", None),
            (r"(?:buy food|purchase food)", "BuyFood", None),
            (r"(?:eat|consume)\s+([^\s,]+(?:\s+[^\s,]+)*)", "Eat", "item_name"),
            (
                r"(?:improve job|work better|enhance performance)",
                "improve_job_performance",
                None,
            ),
            (
                r"(?:pursue hobby|do hobby|engage in hobby)(?:\s+([^\s,]+(?:\s+[^\s,]+)*))?",
                "pursue_hobby",
                "hobby_type",
            ),
            (r"(?:work|do work|perform job)", "Work", None),
            (r"(?:sleep|rest|take a nap)", "Sleep", None),
        ]

        for pattern, action, param_key in action_patterns:
            match = re.search(pattern, response_lower)
            if match:
                parameters = {}
                if param_key and len(match.groups()) > 0 and match.group(1):
                    parameters[param_key] = match.group(1).strip()
                return {"action": action, "parameters": parameters}

        # If no specific pattern matches, try to extract any recognizable action words
        # But be more careful about conflicts
        for action_name in sorted(self.action_class_map.keys(), key=len, reverse=True):
            if len(action_name) > 2 and action_name.lower() in response_lower:
                # Make sure it's a word boundary match to avoid partial matches
                word_pattern = r"\b" + re.escape(action_name.lower()) + r"\b"
                if re.search(word_pattern, response_lower):
                    return {"action": action_name, "parameters": {}}

        raise LLMResponseParsingError(
            f"Could not parse natural language response: {response_str}"
        )

    def _extract_json_from_text(self, text: str) -> dict:
        """Extract JSON from mixed text format"""
        # Look for JSON-like structures in the text
        json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        matches = re.findall(json_pattern, text)

        for match in matches:
            try:
                parsed = json.loads(match)
                if (
                    isinstance(parsed, dict)
                    and "action" in parsed
                    and isinstance(parsed["action"], str)
                ):
                    # Ensure parameters dict exists
                    if "parameters" not in parsed:
                        parsed["parameters"] = {}
                    elif not isinstance(parsed["parameters"], dict):
                        parsed["parameters"] = {}
                    return parsed
            except json.JSONDecodeError:
                continue

        raise InvalidLLMResponseFormatError("No valid JSON found in response")

    def _create_fallback_action_dict(self, response_str: str) -> dict:
        """Create a fallback action when parsing fails"""
        # Try to find any action keywords
        response_lower = response_str.lower()
        for action_name in self.action_class_map.keys():
            if action_name.lower() in response_lower:
                return {"action": action_name, "parameters": {}}

        # Ultimate fallback
        return {
            "action": "NoOp",
            "parameters": {
                "reason": "parsing_failed",
                "original_response": response_str,
            },
        }

    def parse_movement_action(self, action_name: str, parameters: dict) -> dict:
        """Parse and validate movement action parameters"""
        validated_params = parameters.copy()

        # Look for location parameters with various names
        location_keys = ["location", "destination", "location_name", "target", "place"]
        location = None

        for key in location_keys:
            if key in parameters:
                location = parameters[key]
                break

        if not location:
            # Try to extract from action name if it contains location info
            if "to" in action_name.lower():
                parts = action_name.lower().split("to")
                if len(parts) > 1:
                    location = parts[1].strip()

        if location:
            validated_params["location_name"] = location
            validated_params["destination"] = location
        else:
            raise InvalidActionParametersError(
                f"Missing location parameter for movement action {action_name}"
            )

        return validated_params

    def parse_social_action(self, action_name: str, parameters: dict) -> dict:
        """Parse and validate social action parameters"""
        validated_params = parameters.copy()

        # Look for target person parameters
        target_keys = [
            "target",
            "target_name",
            "target_person",
            "person",
            "character",
            "who",
        ]
        target = None

        for key in target_keys:
            if key in parameters:
                target = parameters[key]
                break

        if target:
            validated_params["target_name"] = target
            validated_params["target_person"] = target
        else:
            # Some social actions might not require a target (like general socializing)
            if action_name in ["Talk", "Greet", "ShareNews", "OfferCompliment"]:
                raise InvalidActionParametersError(
                    f"Missing target person for social action {action_name}"
                )

        # Handle specific social action parameters
        if action_name == "ShareNews":
            news_keys = ["news", "news_item", "information", "message", "content"]
            for key in news_keys:
                if key in parameters:
                    validated_params["news_item"] = parameters[key]
                    break

        elif action_name == "OfferCompliment":
            compliment_keys = [
                "compliment",
                "compliment_topic",
                "topic",
                "about",
                "subject",
            ]
            for key in compliment_keys:
                if key in parameters:
                    validated_params["compliment_topic"] = parameters[key]
                    break

        return validated_params

    def parse_work_action(self, action_name: str, parameters: dict) -> dict:
        """Parse and validate work action parameters"""
        validated_params = parameters.copy()

        # Handle job-related parameters
        job_keys = ["job", "job_type", "work_type", "task", "role"]
        for key in job_keys:
            if key in parameters:
                validated_params["job_type"] = parameters[key]
                break

        # Handle method parameters for improvement actions
        if "improve" in action_name.lower():
            method_keys = ["method", "approach", "way", "how", "strategy", "task"]
            method_value = None
            for key in method_keys:
                if key in parameters:
                    method_value = parameters[key]
                    break

            if method_value:
                validated_params["method"] = method_value
            # Don't set a default here - let the action class handle it

        return validated_params

    def parse_creative_action(self, action_name: str, parameters: dict) -> dict:
        """Parse and validate creative action parameters"""
        validated_params = parameters.copy()

        # Handle hobby parameters
        if "hobby" in action_name.lower():
            hobby_keys = ["hobby", "hobby_type", "activity", "type"]
            for key in hobby_keys:
                if key in parameters:
                    validated_params["hobby_type"] = parameters[key]
                    break
            else:
                validated_params["hobby_type"] = "reading"  # Default hobby

        # Handle crafting parameters
        if "craft" in action_name.lower():
            item_keys = ["item", "item_type", "object", "thing", "product"]
            for key in item_keys:
                if key in parameters:
                    validated_params["item_type"] = parameters[key]
                    break

        # Handle research parameters
        if "research" in action_name.lower():
            topic_keys = ["topic", "subject", "area", "field", "technology"]
            for key in topic_keys:
                if key in parameters:
                    validated_params["topic"] = parameters[key]
                    break

        return validated_params

    def validate_action_parameters(self, action_name: str, parameters: dict) -> dict:
        """Comprehensive parameter validation for all action types"""
        try:
            # Determine action category and apply appropriate parsing
            if action_name in self.movement_actions:
                return self.parse_movement_action(action_name, parameters)
            elif action_name in self.social_actions:
                return self.parse_social_action(action_name, parameters)
            elif action_name in self.work_actions:
                return self.parse_work_action(action_name, parameters)
            elif action_name in self.creative_actions:
                return self.parse_creative_action(action_name, parameters)
            else:
                # Generic validation for other actions
                return self._validate_generic_parameters(action_name, parameters)

        except Exception as e:
            raise InvalidActionParametersError(
                f"Parameter validation failed for {action_name}: {e}"
            )

    def _validate_generic_parameters(self, action_name: str, parameters: dict) -> dict:
        """Generic parameter validation and cleanup"""
        validated_params = parameters.copy()

        # Clean up parameter values
        for key, value in validated_params.items():
            if isinstance(value, str):
                validated_params[key] = value.strip()

        # Add default parameters if missing for specific actions
        if action_name == "Sleep" and "duration" not in validated_params:
            validated_params["duration"] = 8

        if action_name in ["BuyFood", "Eat"] and "food_type" not in validated_params:
            validated_params["food_type"] = "food"
            validated_params["item_name"] = "food"

        return validated_params

    def interpret(self, parsed_response: dict, initiator_id_context=None):
        """
        Interprets the parsed dictionary and instantiates the corresponding Action subclass.
        'initiator_id_context' provides the character ID if not in LLM params.
        """
        action_name_str = parsed_response[
            "action"
        ]  # Already checked for presence in parse_llm_response
        parameters = parsed_response["parameters"]  # Already checked for presence/type

        action_class = self.action_class_map.get(action_name_str)

        if not action_class:
            # For enhanced interpreter, fall back to NoOp for unknown actions
            return NoOpAction(
                initiator_id=parameters.get("initiator_id", initiator_id_context),
                reason=f"unknown_action_{action_name_str}",
            )

        initiator_id = parameters.get("initiator_id", initiator_id_context)

        try:
            # Create a clean parameter dict to avoid duplication
            clean_params = parameters.copy()

            # Food & Health Actions
            if action_class == EatAction:
                item_name = clean_params.pop(
                    "item_name", clean_params.pop("food_type", "food")
                )
                clean_params.pop(
                    "initiator_id", None
                )  # Remove to avoid duplicate parameter
                return EatAction(
                    item_name=item_name, initiator_id=initiator_id, **clean_params
                )

            elif action_class == BuyFoodAction:
                food_type = clean_params.pop(
                    "food_type", clean_params.pop("item_name", "food")
                )
                clean_params.pop(
                    "initiator_id", None
                )  # Remove to avoid duplicate parameter
                return BuyFoodAction(
                    food_type=food_type, initiator_id=initiator_id, **clean_params
                )

            elif action_class == VisitDoctorAction:
                reason = clean_params.pop("reason", "checkup")
                clean_params.pop(
                    "initiator_id", None
                )  # Remove to avoid duplicate parameter
                return VisitDoctorAction(
                    reason=reason, initiator_id=initiator_id, **clean_params
                )

            # Movement Actions
            elif action_class == GoToLocationAction:
                location_name = clean_params.pop(
                    "location_name",
                    clean_params.pop(
                        "destination", clean_params.pop("location", "home")
                    ),
                )
                clean_params.pop(
                    "initiator_id", None
                )  # Remove to avoid duplicate parameter
                return GoToLocationAction(
                    location_name=location_name,
                    initiator_id=initiator_id,
                    **clean_params,
                )

            # Work & Career Actions
            elif action_class == WorkAction:
                job_type = clean_params.pop("job_type", "current_job")
                clean_params.pop(
                    "initiator_id", None
                )  # Remove to avoid duplicate parameter
                return WorkAction(
                    job_type=job_type, initiator_id=initiator_id, **clean_params
                )

            elif action_class == ImproveJobPerformanceAction:
                method = clean_params.pop("method", "study")
                clean_params.pop(
                    "initiator_id", None
                )  # Remove to avoid duplicate parameter
                return ImproveJobPerformanceAction(
                    method=method, initiator_id=initiator_id, **clean_params
                )

            # Social Actions
            if action_class == TalkAction:
                # Extract target with clear logic
                target = None
                if "target_name" in clean_params:
                    target = clean_params.pop("target_name")
                elif "target" in clean_params:
                    target = clean_params.pop("target")

                if not target:
                    raise InvalidActionParametersError(
                        "Missing 'target_name' or 'target' for Talk action."
                    )
                if initiator_id is None:
                    raise InvalidActionParametersError(
                        "Missing 'initiator_id' for Talk action."
                    )

                # Remove ALL parameters that could conflict with base Action class
                # Make a completely clean parameter dict
                clean_params.pop("target_person", None)
                clean_params.pop("initiator_id", None)
                clean_params.pop("topic", None)
                clean_params.pop(
                    "target_name", None
                )  # Remove any remaining target_name
                clean_params.pop("target", None)  # Remove any remaining target

                # Keep only parameters that are safe to pass to base Action class
                safe_params = {
                    k: v
                    for k, v in clean_params.items()
                    if k
                    in [
                        "name",
                        "preconditions",
                        "effects",
                        "cost",
                        "related_skills",
                        "default_target_is_initiator",
                        "impact_rating_on_target",
                        "impact_rating_on_initiator",
                        "impact_rating_on_other",
                        "action_id",
                        "created_at",
                        "expires_at",
                        "completed_at",
                        "priority",
                        "related_goal",
                    ]
                }
                return TalkAction(initiator=initiator_id, target=target, **safe_params)

            elif action_class == GreetAction:
                target = clean_params.pop(
                    "target_name", clean_params.pop("target", None)
                )
                if not target:
                    raise InvalidActionParametersError(
                        "Missing 'target_name' or 'target' for Greet action."
                    )
                clean_params.pop("target_person", None)
                clean_params.pop(
                    "initiator_id", None
                )  # Remove to avoid duplicate parameter
                return GreetAction(
                    initiator=initiator_id, target=target, **clean_params
                )

            elif action_class == ShareNewsAction:
                target = clean_params.pop(
                    "target_name", clean_params.pop("target", None)
                )
                if not target:
                    raise InvalidActionParametersError(
                        "Missing 'target_name' or 'target' for ShareNews action."
                    )
                news_item = clean_params.pop("news_item", clean_params.pop("news", ""))
                clean_params.pop("target_person", None)
                clean_params.pop(
                    "initiator_id", None
                )  # Remove to avoid duplicate parameter
                return ShareNewsAction(
                    initiator=initiator_id,
                    target=target,
                    news_item=news_item,
                    **clean_params,
                )

            elif action_class == OfferComplimentAction:
                target = clean_params.pop(
                    "target_name", clean_params.pop("target", None)
                )
                if not target:
                    raise InvalidActionParametersError(
                        "Missing 'target_name' or 'target' for OfferCompliment action."
                    )
                compliment_topic = clean_params.pop(
                    "compliment_topic", clean_params.pop("topic", "")
                )
                clean_params.pop("target_person", None)
                clean_params.pop(
                    "initiator_id", None
                )  # Remove to avoid duplicate parameter
                return OfferComplimentAction(
                    initiator=initiator_id,
                    target=target,
                    compliment_topic=compliment_topic,
                    **clean_params,
                )

            elif action_class == SocialVisitAction:
                target_person = clean_params.pop(
                    "target_person",
                    clean_params.pop("target_name", clean_params.pop("target", None)),
                )
                if not target_person:
                    raise InvalidActionParametersError(
                        "Missing 'target_person', 'target_name' or 'target' for SocialVisit action."
                    )
                clean_params.pop(
                    "initiator_id", None
                )  # Remove to avoid duplicate parameter
                return SocialVisitAction(
                    target_person=target_person,
                    initiator_id=initiator_id,
                    **clean_params,
                )

            # Mental Health & Wellness Actions
            elif action_class == PursueHobbyAction:
                hobby_type = clean_params.pop(
                    "hobby_type", clean_params.pop("hobby", "reading")
                )
                clean_params.pop(
                    "initiator_id", None
                )  # Remove to avoid duplicate parameter
                return PursueHobbyAction(
                    hobby_type=hobby_type, initiator_id=initiator_id, **clean_params
                )

            # Base Actions
            elif action_class == NoOpAction:
                clean_params.pop(
                    "initiator_id", None
                )  # Remove to avoid duplicate parameter
                return NoOpAction(initiator_id=initiator_id, **clean_params)

            elif action_class == SleepAction:
                duration = clean_params.pop("duration", 8)
                clean_params.pop(
                    "initiator_id", None
                )  # Remove to avoid duplicate parameter
                return SleepAction(
                    duration=duration, initiator_id=initiator_id, **clean_params
                )

            elif action_class == ExploreAction:
                # ExploreAction from actions.py may need special handling based on its constructor
                target = clean_params.pop(
                    "target", clean_params.pop("location", "area")
                )
                clean_params.pop(
                    "initiator_id", None
                )  # Remove to avoid duplicate parameter
                return ExploreAction(
                    initiator=initiator_id, target=target, **clean_params
                )

            # Generic Action Factory - handles dynamically created action classes
            elif (
                hasattr(action_class, "__call__")
                and hasattr(action_class, "__name__")
                and (
                    "GenericAction" in action_class.__name__
                    or "SocialAction" in action_class.__name__
                    or "MovementAction" in action_class.__name__
                )
            ):
                # This is a factory-created class
                clean_params.setdefault("initiator_id", initiator_id)
                # Filter out parameters that base Action class doesn't accept
                base_action_params = {
                    "name",
                    "preconditions",
                    "effects",
                    "cost",
                    "target",
                    "initiator",
                    "related_skills",
                    "default_target_is_initiator",
                    "impact_rating_on_target",
                    "impact_rating_on_initiator",
                    "impact_rating_on_other",
                    "action_id",
                    "created_at",
                    "expires_at",
                    "completed_at",
                    "priority",
                    "related_goal",
                    "initiator_id",  # Add initiator_id for factory-created actions
                }
                filtered_params = {
                    k: v for k, v in clean_params.items() if k in base_action_params
                }
                return action_class(**filtered_params)

            else:
                # Ultimate fallback for any remaining mapped actions
                # Try different constructor patterns based on action type
                if initiator_id is not None:
                    clean_params.setdefault("initiator_id", initiator_id)
                    clean_params.setdefault("initiator", initiator_id)

                # Try to handle target parameter variations
                if "target_name" in clean_params and "target" not in clean_params:
                    clean_params["target"] = clean_params["target_name"]
                elif "target_person" in clean_params and "target" not in clean_params:
                    clean_params["target"] = clean_params["target_person"]

                try:
                    return action_class(**clean_params)
                except TypeError as init_error:
                    # Try with just the essential parameters
                    try:
                        if hasattr(action_class, "__init__"):
                            import inspect

                            sig = inspect.signature(action_class.__init__)
                            filtered_args = {
                                k: v
                                for k, v in clean_params.items()
                                if k in sig.parameters or "kwargs" in str(sig)
                            }
                            return action_class(**filtered_args)
                        else:
                            return action_class()
                    except Exception:
                        raise InvalidActionParametersError(
                            f"Could not instantiate {action_name_str} with available parameters: {init_error}"
                        )

        except TypeError as e:
            # This can happen if an Action subclass's __init__ is called with
            # arguments it doesn't accept (e.g., unexpected kwargs due to **parameters).
            # Or if required arguments (like 'initiator' for TalkAction) are effectively None.
            raise InvalidActionParametersError(
                f"Parameter mismatch or missing required argument for action {action_name_str}: {e}"
            )
        except KeyError as e:  # Should be caught by specific checks, but as a fallback
            raise InvalidActionParametersError(
                f"Missing essential parameter {e} for action {action_name_str}."
            )

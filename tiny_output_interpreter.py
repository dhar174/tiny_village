import json
from actions import Action, TalkAction # Assuming actions.py is in the same directory or PYTHONPATH

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

# Placeholder Action Classes (if not available or suitable in actions.py)
# These inherit from actions.Action to be compatible with the expected Action type hint.

class EatAction(Action):
    def __init__(self, item_name, initiator_id=None, **kwargs):
        # For simplicity, we might treat item_name as the 'target' in the base Action,
        # or just store it. The base Action's 'target' is generic.
        # We also pass along other kwargs to the base Action constructor.
        # Ensure 'name' is passed correctly to base Action.
        super().__init__(name="Eat", initiator=initiator_id, target=item_name, cost=kwargs.pop('cost', 0.1), **kwargs)
        self.item_name = item_name
        # Any other specific initialization for EatAction

class GoToLocationAction(Action):
    def __init__(self, location_name, initiator_id=None, **kwargs):
        super().__init__(name="GoToLocation", initiator=initiator_id, target=location_name, cost=kwargs.pop('cost', 0.2), **kwargs)
        self.location_name = location_name
        # Any other specific initialization for GoToLocationAction

class NoOpAction(Action):
    def __init__(self, initiator_id=None, **kwargs):
        super().__init__(name="NoOp", initiator=initiator_id, cost=kwargs.pop('cost', 0), **kwargs)
        # Any other specific initialization for NoOpAction

class OutputInterpreter:
    def __init__(self, graph_manager=None):
        self.graph_manager = graph_manager
        self.action_class_map = {
            "Eat": EatAction,
            "GoTo": GoToLocationAction,
            "Talk": TalkAction, # From actions.py
            "NoOp": NoOpAction,
        }

    def parse_llm_response(self, llm_response_str: str) -> dict:
        """
        Parses the raw JSON string from the LLM into a Python dictionary.
        Raises InvalidLLMResponseFormatError if parsing fails or format is incorrect.
        """
        if not llm_response_str:
            raise InvalidLLMResponseFormatError("LLM response string cannot be empty.")
        try:
            parsed_response = json.loads(llm_response_str)
        except json.JSONDecodeError as e:
            raise InvalidLLMResponseFormatError(f"Invalid JSON format: {e}")
        
        if not isinstance(parsed_response, dict):
            raise InvalidLLMResponseFormatError("LLM response must be a JSON object.")
        if "action" not in parsed_response:
            raise InvalidLLMResponseFormatError("LLM response missing 'action' key.")
        if not isinstance(parsed_response["action"], str):
            raise InvalidLLMResponseFormatError("'action' key must be a string.")
        if "parameters" not in parsed_response:
            raise InvalidLLMResponseFormatError("LLM response missing 'parameters' key.")
        if not isinstance(parsed_response["parameters"], dict):
            raise InvalidLLMResponseFormatError("'parameters' key must be a dictionary.")
            
        return parsed_response

    def interpret(self, parsed_response: dict, initiator_id_context=None) -> Action:
        """
        Interprets the parsed dictionary and instantiates the corresponding Action subclass.
        'initiator_id_context' provides the character ID if not in LLM params.
        """
        action_name_str = parsed_response["action"] # Already checked for presence in parse_llm_response
        parameters = parsed_response["parameters"] # Already checked for presence/type

        action_class = self.action_class_map.get(action_name_str)

        if not action_class:
            raise UnknownActionError(f"Unknown action: {action_name_str}")

        initiator_id = parameters.get("initiator_id", initiator_id_context)

        try:
            if action_class == EatAction: # Using direct class comparison for clarity
                if "item_name" not in parameters:
                    raise InvalidActionParametersError("Missing 'item_name' for Eat action.")
                # Pass all parameters through, EatAction's __init__ will pick what it needs.
                return EatAction(item_name=parameters["item_name"], initiator_id=initiator_id, **parameters)
            
            elif action_class == GoToLocationAction:
                if "location_name" not in parameters:
                    raise InvalidActionParametersError("Missing 'location_name' for GoTo action.")
                return GoToLocationAction(location_name=parameters["location_name"], initiator_id=initiator_id, **parameters)
            
            elif action_class == TalkAction:
                # TalkAction from actions.py: __init__(self, initiator, target, name="Talk", ..., **kwargs)
                if "target_name" not in parameters:
                    raise InvalidActionParametersError("Missing 'target_name' for Talk action.")
                if initiator_id is None: # TalkAction requires an initiator
                    raise InvalidActionParametersError("Missing 'initiator_id' for Talk action (expected in LLM parameters or context).")
                
                # Pass relevant known params explicitly, rest via **parameters for flexibility (e.g. topic)
                return TalkAction(initiator=initiator_id, 
                                  target=parameters["target_name"], 
                                  **parameters) # This will pass 'topic', 'target_name', 'initiator_id' etc.

            elif action_class == NoOpAction:
                return NoOpAction(initiator_id=initiator_id, **parameters)
            
            else: 
                # Fallback for any actions added to map but not given specific logic.
                # This assumes their constructors can handle 'initiator_id' and **parameters.
                # Or, they might not need initiator_id explicitly if they get it from **parameters.
                constructor_args = parameters.copy()
                if initiator_id is not None: # Add initiator_id if available and not already in params
                    constructor_args.setdefault('initiator_id', initiator_id)
                # This is a generic call; specific actions might need more tailored instantiation.
                # For now, this path should ideally not be hit if all mapped actions have explicit blocks.
                return action_class(**constructor_args)


        except TypeError as e:
            # This can happen if an Action subclass's __init__ is called with
            # arguments it doesn't accept (e.g., unexpected kwargs due to **parameters).
            # Or if required arguments (like 'initiator' for TalkAction) are effectively None.
            raise InvalidActionParametersError(f"Parameter mismatch or missing required argument for action {action_name_str}: {e}")
        except KeyError as e: # Should be caught by specific checks, but as a fallback
            raise InvalidActionParametersError(f"Missing essential parameter {e} for action {action_name_str}.")

```

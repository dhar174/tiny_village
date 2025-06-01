System File: `tiny_event_handler.py`
Status from Controller: Placeholder System (explicitly mentioned in `controller_analysis.txt`), `event_driven_storytelling` is `NOT_STARTED`.
Current Implementation:
The file defines an `Event` class and an `EventHandler` class.
- `Event` class: Has attributes for name, date, type, importance, impact, items, location, recurrence, effects, preconditions, cascading events, and participants. It includes methods for checking recurrence, next occurrence, if it should trigger (based on time and preconditions), item requirements, and participant management. Precondition checking is basic and includes attribute checks, time checks, and placeholders for weather/location. Effect application (`_modify_entity_attribute`) is present but relies on entities having `get_state` or direct attribute access.
- `EventHandler` class: Manages a list of `Event` objects. It can add, remove, update, and get events.
    - `check_events()`: Identifies events that should trigger based on `event.should_trigger()`.
    - `process_events()`: Processes triggered events, applies effects, handles cascading events (basic implementation).
    - `_check_event_requirements()`: Basic check for item availability (placeholder logic for fetching items from graph).
    - `_apply_event_effects()` & `_apply_single_effect()`: Applies defined effects, mainly attribute changes and relationship changes (rudimentary).
    - `_update_event_relationships()`: Adds character-event and location-event edges to the graph.
    - Cascading event logic is present (`_trigger_cascading_events`, `_create_event_from_definition`, `process_cascading_queue`).
    - Daily event checking (`check_daily_events`) includes recurring events and special date events (holidays, market days - with helper creation methods).
    - Event template system (`get_event_templates`, `create_event_from_template`) allows creating events like festivals, harvest, merchant arrival, disasters.
    - Utility methods for scheduling, getting events by type/location/timeframe, statistics, and cleanup are present.

Comparison with Documentation:
- The `documentation_summary.txt` describes `EventHandler` as detecting/queuing game events and that these events trigger `StrategyManager`.
- The current implementation has a fairly detailed `Event` class and logic for event triggering based on time, recurrence, and basic preconditions. It also has systems for effects and cascading events.
- The "Placeholder Systems" section in `controller_analysis.txt` correctly notes that while `EventHandler` is initialized in `GameplayController` and `check_events()` is called in a legacy update, its role in driving the main game loop via `StrategyManager` is not clearly implemented in `tiny_gameplay_controller.py`. The controller's `_process_pending_events` method seems to be the main path for event handling in the loop, and it's marked with many TODOs.
- The system has more than just a "stub" structure; many foundational elements for a reasonably complex event system are there. However, its integration into the main decision-making loop of `GameplayController` is weak, and the "event_driven_storytelling" feature is `NOT_STARTED`, meaning the potential of this event system isn't fully realized in terms of narrative impact or complex emergent situations. The effect application is also quite generic.

---
System File: `tiny_prompt_builder.py`
Status from Controller: Part of conceptual `LLM_Interface`; "advanced_ai_behaviors" is `NOT_STARTED`.
Current Implementation:
- `NeedsPriorities` class:
    - Defines a list of needs.
    - Calculates priority for each need based on character attributes (e.g., `calculate_health_priority` based on `character.get_health_status()` and motives).
    - `calculate_needs_priorities` method compiles these individual priority calculations.
- `ActionOptions` class:
    - Defines a list of possible action strings (e.g., "buy_food", "eat_food").
    - `prioritize_actions` method: A very basic attempt to prioritize actions based on character needs (e.g., "buy_food" if hunger is high and has money). It creates a small list of prioritized actions and fills the rest with other actions.
- `DescriptorMatrices` class:
    - Contains many dictionaries mapping job names (e.g., "Engineer", "Farmer") or conditions (e.g., "healthy", "full") to lists of descriptive phrases.
    - Examples: `job_adjective`, `job_pronoun`, `job_enjoys_verb`, `feeling_health`, `weather_description`.
    - Has `get_` methods to randomly select a phrase from these lists based on a key (e.g., `get_job_adjective(job)`).
- `PromptBuilder` class:
    - Takes a `Character` object.
    - Initializes `ActionOptions` and `NeedsPriorities`.
    - `calculate_needs_priorities()` and `prioritize_actions()` call the respective methods in the helper classes.
    - `generate_daily_routine_prompt(time, weather)`: Constructs a detailed prompt string.
        - It uses the `DescriptorMatrices` to create a narrative intro about the character and their job.
        - It incorporates character status (health, hunger from `DescriptorMatrices`), recent events, financial situation, and long-term goal.
        - It includes a fixed list of 5 action options (hardcoded strings like "Go to the market to Buy_Food.").
        - The prompt is formatted with `<|system|>`, `<|user|>`, and `<|assistant|>` tags, ending with "I choose ".
    - `generate_crisis_response_prompt(crisis)`: A placeholder, returns a minimal prompt structure.
    - `generate_completion_message` and `generate_failure_message`: Placeholders using `DescriptorMatrices`.
    - `calculate_action_utility`: Placeholder, seems to attempt to modify `self.needs_priorities` (which is a dictionary of need priorities) as if it were action utilities, likely incorrect.

Comparison with Documentation:
- `documentation_summary.txt` describes `PromptBuilder` as part of the `LLM_Interface` responsible for constructing text prompts for the LLM, using character/world context.
- The current implementation focuses heavily on generating a "daily routine" prompt.
- It does use character details (name, job, status) and some context (time, weather, recent events).
- The use of `DescriptorMatrices` provides varied textual descriptions, making prompts less repetitive.
- The action options presented to the LLM are currently hardcoded in the daily routine prompt string and do not seem to directly use the `prioritized_actions` list generated by `ActionOptions.prioritize_actions()`.
- The needs calculation is detailed, but its output isn't explicitly shown to be part of the prompt string itself, though it might be used by `ActionOptions`.
- The "advanced_ai_behaviors" feature being `NOT_STARTED` aligns with the current state; while prompts can be built, the strategic selection of when and what to ask the LLM, and how to use its responses for complex behaviors, is not yet developed.
- The crisis response prompt is a clear stub.

---
System File: `tiny_brain_io.py`
Status from Controller: Part of conceptual `LLM_Interface`; "advanced_ai_behaviors" is `NOT_STARTED`.
Current Implementation:
- `TinyBrainIO` class:
    - Initializes a Hugging Face Transformer model (default: "alexredna/TinyLlama-1.1B-Chat-v1.0-reasoning-v2") or a GGUF model using `llama_cpp`.
    - `load_model()`: Handles loading either type of model. It includes parameters like `n_ctx`, `n_threads`, `n_gpu_layers`.
    - `input_to_model(prompts, reset_model=True)`:
        - Takes a prompt string or a list of prompts.
        - Cleans the prompt by removing certain characters.
        - For GGUF models, calls `self.model(text, max_tokens=256, stop="</s>", echo=True)` and extracts the choice.
        - For Transformer models, tokenizes the input, generates output using `self.model.generate()`, and decodes it.
        - It has a basic attempt to clean the generated text by removing the input prompt part or "I choose " prefix.
        - Measures and prints processing time.
        - Returns a list of tuples `(generated_text, processing_time_str)`.

Comparison with Documentation:
- `documentation_summary.txt` describes `TinyBrainIO` as handling communication with the LLM.
- The implementation directly addresses this by loading an LLM and providing a method to send prompts to it and get generated text.
- It supports both full Transformer models and GGUF quantized models.
- The output processing is rudimentary, mainly trying to isolate the LLM's actual response from the input prompt.
- This component seems functional for basic LLM interaction. Its role in "advanced_ai_behaviors" would depend on how its outputs are used by the `OutputInterpreter` and integrated into the broader decision-making systems.

---
System File: `tiny_output_interpreter.py`
Status from Controller: Part of conceptual `LLM_Interface`; "advanced_ai_behaviors" is `NOT_STARTED`.
Current Implementation:
- Defines custom exceptions: `InvalidLLMResponseFormatError`, `UnknownActionError`, `InvalidActionParametersError`.
- Includes placeholder `Action` subclasses (`EatAction`, `GoToLocationAction`, `NoOpAction`) that inherit from `actions.Action`. These are used if the main `actions.py` doesn't provide suitable classes or if a simpler version is needed. It also imports `TalkAction` from `actions.py`.
- `OutputInterpreter` class:
    - `action_class_map`: Maps action name strings (e.g., "Eat", "GoTo") to their respective action classes.
    - `parse_llm_response(llm_response_str)`:
        - Expects the LLM response to be a JSON string.
        - Parses the JSON into a Python dictionary.
        - Validates that the dictionary contains "action" (string) and "parameters" (dictionary) keys.
    - `interpret(parsed_response, initiator_id_context=None)`:
        - Takes the parsed dictionary and an optional `initiator_id`.
        - Looks up the action name in `action_class_map`.
        - Instantiates the corresponding action class using parameters from the parsed response.
        - Has specific logic for `EatAction`, `GoToLocationAction`, and `TalkAction` to ensure required parameters like "item_name", "location_name", or "target_name" are present.
        - Includes a fallback for generically instantiating other actions in the map.
        - Raises custom exceptions for issues like unknown actions or missing/invalid parameters.

Comparison with Documentation:
- `documentation_summary.txt` states `OutputInterpreter` (conceptual) parses LLM responses into game actions.
- The current implementation directly attempts this. It expects a structured JSON output from the LLM.
- It maps string action names to Python `Action` objects.
- Parameter validation is included.
- The main challenge, as noted in `documentation_summary.txt`, is translating "potentially creative/abstract LLM outputs into specific, executable Action instances." This implementation sidesteps the "creative/abstract" part by enforcing a strict JSON format with predefined action names and parameters. If the LLM doesn't adhere to this, parsing or interpretation will fail.
- The use of placeholder action classes within this file suggests that the main `actions.py` might not yet fully support all actions the LLM could specify, or that a simplified interface is preferred for LLM-generated actions.

---
System File: `tiny_items.py`
Status from Controller: "economic_simulation" is `STUB_IMPLEMENTED`. Items are fundamental to economy.
Current Implementation:
- `Stock` class: Represents an ownable stock with name, value, quantity. Basic getters/setters.
- `InvestmentPortfolio` class: A list of `Stock` objects, can calculate total portfolio value.
- `ItemObject` class:
    - Base class for items with name, description, value, weight, quantity, UUID, type, subtype, location, status, possible interactions.
    - `possible_interactions` is a list of `Action` objects.
    - Basic getters/setters.
    - `to_dict()` method.
- `FoodItem(ItemObject)` class:
    - Adds `calories`, `perishable`, `cooked` attributes.
    - Initializes an "Eat Food" `Action` in its `possible_interactions`. The effect of this action includes calorie change and an animation call. Preconditions are taken from a global `preconditions_dict`.
- `Door(ItemObject)` class:
    - Initializes an "Open Door" `Action`.
- `ItemInventory` class:
    - Contains separate lists for different item types (food, clothing, tools, etc.) and an `all_items` list.
    - Methods to add, remove, count items (total, by name, by type).
    - Methods to get total value and weight of inventory.
    - `check_has_item_by_name` and `check_has_item_by_type` methods for checking quantities.
- Global `effect_dict` and `preconditions_dict` for defining parts of actions related to items.

Comparison with Documentation:
- `README.md` lists `tiny_items.py` for managing game items and their interactions.
- `documentation_summary.txt` doesn't go into deep detail on items but implies their existence for actions and character state.
- The implementation provides a solid foundation for items, including typed items (FoodItem) with specific interactions.
- The `ItemInventory` class is reasonably well-developed for managing collections of items.
- The connection to "economic_simulation" (STUB_IMPLEMENTED) is present through item values and quantities. However, a dedicated economy manager that handles production, consumption, trade, and price dynamics based on these items is missing. The current item system provides the "what" but not the "how" of a simulated economy.
- The `Action` objects associated with items (e.g., "Eat Food") are defined within `tiny_items.py` itself, using global dictionaries for effects/preconditions. This is a bit different from the main `ActionSystem` described in `action_system_deep_dive.md`, which has a more centralized `ActionSystem` class for defining and managing actions.

---
System File: `tiny_jobs.py`
Status from Controller: "economic_simulation" is `STUB_IMPLEMENTED`. Jobs are fundamental to economy.
Current Implementation:
- `JobRoles` class:
    - Represents a job template with name, title, description, salary, skills, education requirements, experience requirements, motives, and location.
    - Contains getters and setters for its attributes.
- `JobRules` class:
    - Loads job role definitions from a `job_roles.json` file.
    - Stores a list of `JobRoles` objects in `ValidJobRoles`.
    - `check_job_role_validity` and `check_job_name_validity` methods.
- `Job(JobRoles)` class:
    - Inherits from `JobRoles` and adds an `available` flag and specific `job_title` and `location`. This seems to represent an instance of a job.
- `JobManager` class:
    - Contains an instance of `JobRules`.
    - Provides methods to get details about job roles (skills, education, salary, etc.) by name.
    - `get_all_job_roles` and `get_all_job_role_names`.

Comparison with Documentation:
- `README.md` lists `tiny_jobs.py` for handling job-related data and interactions.
- `documentation_summary.txt` doesn't detail the job system, but jobs are mentioned as part of character attributes and influence decisions (e.g., `StrategyManager` responding to job offers).
- The implementation provides a structure for defining job roles and their attributes, loading them from an external JSON file.
- The `JobManager` allows querying these job definitions.
- This system defines *what* jobs are, their requirements, and salaries.
- For the "economic_simulation" to be more than a stub, this job system would need to interact with characters (assigning them jobs), an economic manager (paying salaries, generating output/value from jobs), and potentially a system for job availability/creation within the simulation. Currently, it's a definitional system.
---
System File: `tiny_weather_system.py` (Conceptual - actually in `GameplayController`)
Status from Controller: `weather_system` is `STUB_IMPLEMENTED`.
Current Implementation:
- Implemented directly in `tiny_gameplay_controller.py` via `implement_weather_system()` and `self.weather_system` dictionary.
- `self.weather_system` stores `current_weather`, `temperature`, `season`, `weather_effects_active`.
- Logic: 10% chance to randomly change `current_weather` from a predefined list (`clear`, `cloudy`, `rainy`, `sunny`). Temperature and season are static after initialization.
- No actual effects on characters or game world are implemented beyond setting these dictionary values. UI in `_render_ui` displays this.
Comparison with Documentation:
- Not specifically detailed in `documentation_summary.txt`, but weather is a common simulation element.
- `controller_analysis.txt` notes the `STUB_IMPLEMENTED` status.
- The implementation is indeed a stub. It changes a string label for weather but has no impact. TODOs in `GameplayController` mention adding seasonal changes, effects on characters, weather-based events, and visual effects.

---
System File: `tiny_social_network_system.py` (Conceptual - actually in `GameplayController`)
Status from Controller: `social_network_system` is `STUB_IMPLEMENTED`.
Current Implementation:
- Implemented directly in `tiny_gameplay_controller.py` via `implement_social_network_system()` and `self.social_networks` dictionary.
- `self.social_networks` stores `relationships` (dict mapping char_id to other_id:strength), `groups`, `social_events`.
- Initialization creates random relationship strengths (30-70) between existing characters.
- `_update_social_relationships(dt)` in `GameplayController` implements a very slow decay/growth of relationships towards a neutral 50.
Comparison with Documentation:
- `documentation_summary.txt` mentions `GraphManager` storing relationships, which is more aligned with the architectural diagrams. The `StrategyManager` and `GOAPPlanner` are also noted to use relationship information.
- The implementation in `GameplayController` is a simple dictionary-based approach, separate from `GraphManager`'s potential role.
- It's a stub because relationship changes are minimal (slow decay/growth) and there's no complex group dynamics or social event processing. TODOs in `GameplayController` mention adding relationship events and social influence on decisions.

---
System File: `tiny_quest_system.py` (Conceptual - actually in `GameplayController`)
Status from Controller: `quest_system` is `STUB_IMPLEMENTED`.
Current Implementation:
- Implemented directly in `tiny_gameplay_controller.py` via `implement_quest_system()` and `self.quest_system` dictionary.
- `self.quest_system` stores `active_quests`, `completed_quests`, `available_quests`, and `quest_templates`.
- Initialization assigns a random quest from templates if a character has no active quests.
- `_update_quest_timers(dt)`: Removes expired quests (hardcoded 5 min), 10% chance to assign a new quest if none active.
- `_update_quest_progress(character, action)`: Basic logic to increment quest progress based on action names and quest type.
- `_complete_quest(character, quest)`: Moves quest to completed, gives random wealth and satisfaction.
Comparison with Documentation:
- Not detailed in `documentation_summary.txt`.
- `controller_analysis.txt` notes the `STUB_IMPLEMENTED` status.
- The implementation is a basic framework. Quest generation is random from a small template list. Progress is simplistic. Rewards are generic. TODOs mention quest generation algorithms, rewards/consequences, multi-step quests, and sharing.

---
System File: `tiny_sound_manager.py` (Conceptual - no file exists)
Status from Controller: `sound_and_music_system` is `NOT_STARTED`.
Current Implementation: N/A. No file found in README or imports.
Comparison with Documentation: N/A.
---

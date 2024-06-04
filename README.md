README.md

# TinyVillage

TinyVillage is a 2D simulation game where AI characters autonomously go about their lives in a dynamic village. Each AI character has a history, likes and dislikes, relationships, careers, bank accounts, and more. The game is driven by an advanced AI system using TinyLlama, a locally-run language model based on Llama 2, which powers the decision-making processes of the characters.

## Features

- **Autonomous AI Characters**: Characters make decisions based on their personal histories, relationships, and current game states.
- **Dynamic Interactions**: Characters interact with each other and the environment, creating a constantly evolving game world.
- **Complex Simulation**: The game includes various systems like buildings, jobs, items, and events, all interconnected to provide a rich simulation experience.
- **Strategic AI Planning**: Utilizes Goal-Oriented Action Planning (GOAP) and utility-based evaluations to simulate realistic character behaviors.

## Directory Structure

- `actions.py`: Defines possible actions characters can take.
- `tiny_brain_io.py`: Manages input/output operations for the AI brain.
- `tiny_buildings.py`: Handles interactions related to game buildings.
- `tiny_characters.py`: Manages character data and attributes.
- `tiny_event_handler.py`: Processes game events and routes them to appropriate components.
- `tiny_gameplay_controller.py`: Coordinates the main game loop and updates.
- `tiny_goap_system.py`: Implements the GOAP system for action planning.
- `tiny_graph_manager.py`: Manages the graph of all game entities and their relationships.
- `tiny_items.py`: Manages game items and their interactions.
- `tiny_jobs.py`: Handles job-related data and interactions.
- `tiny_memories.py`: Stores and retrieves characters' memories.
- `tiny_output_interpreter.py`: Translates AI decisions into game commands.
- `tiny_prompt_builder.py`: Constructs dynamic prompts for the AI.
- `tiny_strategy_manager.py`: Formulates overarching strategies for the AI.
- `tiny_time_manager.py`: Manages game time and synchronizes processes.
- `tiny_utility_functions.py`: Evaluates utility scores for action planning.
- `tiny_util_funcs.py`: Provides various utility functions supporting game logic and AI decisions.

## Getting Started

### Prerequisites

- Python 3.8+
- Required Python packages (listed in `requirements.txt`)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/tinyvillage.git
    ```
2. Navigate to the project directory:
    ```bash
    cd tinyvillage
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Game

To start the game, run the following command:
```bash
python main.py

Contributing
We welcome contributions! Please read our Contributing Guide for details on our code of conduct and the process for submitting pull requests.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
The TinyLlama model, based on Llama 2, for powering our AI characters.
OpenAI for providing the language model architecture.
Contact
For any inquiries or issues, please contact your-email@example.com.

css
Copy code
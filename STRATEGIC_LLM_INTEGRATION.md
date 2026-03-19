# Strategic LLM Integration Guide

## Overview

The Tiny Village project uses a hybrid AI decision-making system that strategically routes decisions between LLM-based and utility-based planning. This document explains the design, implementation, and usage of the strategic LLM integration system.

## Architecture

### Decision Flow

```
Character Decision Request
         ↓
    StrategyManager
         ↓
should_use_llm_for_decision() ← Evaluates criteria
         ↓                 ↓
    LLM Path          Utility Path
         ↓                 ↓
decide_action_with_llm()  get_daily_actions()
         ↓                 ↓
    BrainIO           GOAP Planner
         ↓                 ↓
 OutputInterpreter   Utility Functions
         ↓                 ↓
         └─────→ Actions ←─────┘
```

## When to Use LLM vs Utility-Based Planning

### LLM is Used For:

1. **Crisis Situations** (Threshold: 0.3)
   - When any critical stat (health, mental_health, energy) falls below 30%
   - Requires creative problem-solving and prioritization
   - Example: Character with 25% health needs immediate action

2. **Complex Social Interactions** (Threshold: 0.7)
   - When social_complexity score exceeds 0.7 (70%)
   - Nuanced understanding of relationships required
   - Example: Negotiating with multiple parties with conflicting interests

3. **Novel Situations** (Threshold: 0.6)
   - When novelty_score exceeds 0.6 (60%)
   - Unfamiliar scenarios outside routine patterns
   - Example: First-time diplomatic encounter

4. **Complex Goals** (Threshold: 0.7)
   - When goal.complexity exceeds 0.7 (70%)
   - Multi-step or ambiguous objectives
   - Example: "Establish trade network across regions"

5. **Emergent Behavior** (Probability: 0.2)
   - 20% random chance for variety and unpredictability
   - Prevents deterministic, repetitive behavior
   - Creates more engaging character actions

6. **Explicit Override**
   - When `force_llm` flag is set in situation_context
   - For testing or special scenarios requiring LLM

### Utility-Based Planning is Used For:

- Routine daily activities
- Well-defined optimization problems
- When LLM components are unavailable
- When no special criteria are met
- As fallback when LLM fails

## Implementation

### Key Code Locations

**File:** `tiny_strategy_manager.py`

#### Constants (Lines 57-63)
```python
CRISIS_THRESHOLD = 0.3          # Below 30% triggers crisis mode
SOCIAL_COMPLEXITY_THRESHOLD = 0.7  # Above 70% uses LLM
NOVELTY_THRESHOLD = 0.6         # Above 60% uses LLM
GOAL_COMPLEXITY_THRESHOLD = 0.7 # Above 70% uses LLM
VARIETY_PROBABILITY = 0.2       # 20% chance for variety
```

#### Decision Method (Lines 189-376)
```python
def should_use_llm_for_decision(self, character, situation_context=None) -> bool:
    """Strategic decision logic for when to invoke LLM vs utility-based planning."""
    # Check 0: LLM availability
    # Check 1: Explicit override
    # Check 2: Social complexity
    # Check 3: Crisis detection
    # Check 4: Novelty
    # Check 5: Goal complexity
    # Check 6: Variety/emergent behavior
```

#### Instrumentation (Lines 378-408)
```python
def _log_decision(self, metadata: dict, use_llm: bool):
    """Log LLM decision metadata for analysis and debugging."""
    # Tracks decision history
    # Creates structured logs
    # Enables pattern analysis
```

#### Action Planning (Lines 410-498)
```python
def get_enhanced_daily_actions(self, character, time="morning", 
                              weather="clear", situation_context=None):
    """Enhanced daily action planning with strategic routing."""
    # Routes to LLM or utility-based path
    # Implements fallback mechanisms
    # Comprehensive error handling
```

#### Analytics (Lines 500-566)
```python
def get_decision_analytics(self) -> dict:
    """Get analytics about LLM vs utility-based decision patterns."""
    # Returns decision statistics
    # Breakdown by reason
    # Failure tracking
```

## Usage Examples

### Basic Usage

```python
from tiny_strategy_manager import StrategyManager

# Initialize with LLM enabled
strategy_manager = StrategyManager(use_llm=True)

# Get actions for a character
actions = strategy_manager.get_enhanced_daily_actions(
    character=my_character,
    time="morning",
    weather="clear"
)
```

### With Situation Context

```python
# Complex social scenario
context = {
    'social_complexity': 0.85,
    'novelty_score': 0.75,
    'event_type': 'trade_negotiation'
}

actions = strategy_manager.get_enhanced_daily_actions(
    character=merchant_character,
    situation_context=context
)
# → Will use LLM due to high social complexity
```

### Forced LLM Override

```python
# Force LLM for testing or special cases
context = {
    'force_llm': True,
    'force_llm_reason': 'tutorial_scenario'
}

actions = strategy_manager.get_enhanced_daily_actions(
    character=tutorial_character,
    situation_context=context
)
# → Will always use LLM
```

### Decision Analytics

```python
# Get decision pattern statistics
analytics = strategy_manager.get_decision_analytics()

print(f"Total decisions: {analytics['total_decisions']}")
print(f"LLM usage: {analytics['llm_percentage']:.1f}%")
print(f"Top reason: {analytics['top_reason']}")
print(f"Reasons: {analytics['reasons_breakdown']}")

# Example output:
# Total decisions: 100
# LLM usage: 35.0%
# Top reason: crisis
# Reasons: {'crisis': 15, 'social_complexity': 10, 'variety': 10}
```

## Testing

### Running Tests

```bash
cd /home/runner/work/tiny_village/tiny_village
python tests/test_strategic_llm_integration.py
```

### Test Coverage

The test suite (`tests/test_strategic_llm_integration.py`) includes:

1. **Criteria Tests**
   - LLM unavailability
   - Forced override
   - Crisis detection (health, energy, mental_health)
   - Social complexity threshold
   - Novelty threshold
   - Goal complexity
   - Routine situations

2. **Instrumentation Tests**
   - Decision history tracking
   - Analytics generation

3. **Integration Tests**
   - Enhanced daily actions LLM path
   - Fallback mechanisms

4. **Complex Scenario Test**
   - Full integration demonstration
   - Multiple criteria triggering
   - Decision logging validation

## Monitoring and Debugging

### Logging Levels

The system uses Python's logging module:

```python
import logging
logging.basicConfig(level=logging.INFO)

# INFO: Key decision points
# DEBUG: Detailed decision context
# WARNING: Fallback activations
# ERROR: Component failures
```

### Log Examples

```
INFO - LLM Decision: Merchant - HIGH SOCIAL COMPLEXITY (0.85 > 0.7)
INFO - Strategic Routing: Merchant → LLM-based decision making (time=morning, weather=clear)
INFO - LLM Decision Success: Merchant selected 3 actions: Negotiate, Trade, Rest
DEBUG - Decision Log: character=Merchant, decision=llm, reason=social_complexity
```

### Failure Handling

```
ERROR - LLM Decision Failed: Merchant - ConnectionError: LLM timeout - falling back to utility-based
INFO - Strategic Routing: Merchant → Utility-based planning (LLM fallback)
DEBUG - Utility Decision: Merchant generated 5 actions: Work, Eat, Rest...
```

## Performance Considerations

### LLM Call Optimization

- LLM calls are expensive (time and potentially cost)
- Strategic routing ensures LLM is only used when beneficial
- Expected LLM usage: 20-40% of decisions
- Fallback mechanisms ensure reliability

### Caching (Future Enhancement)

Consider adding:
- Decision caching for similar situations
- LLM response caching for repeated queries
- GOAP plan caching with validity checks

## Tuning Thresholds

### How to Adjust

Edit `tiny_strategy_manager.py` constants:

```python
# More aggressive LLM usage
CRISIS_THRESHOLD = 0.4  # Use LLM earlier in decline
VARIETY_PROBABILITY = 0.3  # 30% variety

# More conservative LLM usage  
CRISIS_THRESHOLD = 0.2  # Only true emergencies
VARIETY_PROBABILITY = 0.1  # 10% variety
```

### Monitoring Impact

```python
analytics = strategy_manager.get_decision_analytics()
print(f"LLM usage: {analytics['llm_percentage']:.1f}%")

# Adjust thresholds based on:
# - Character behavior quality
# - LLM response time
# - System performance
# - Narrative engagement
```

## Integration with GameplayController

The StrategyManager is integrated into `GameplayController`:

```python
# In tiny_gameplay_controller.py
def _execute_character_actions(self, character) -> bool:
    # Check if character should use LLM decision-making
    use_llm_decisions = getattr(character, 'use_llm_decisions', False)
    
    if use_llm_decisions:
        # Use comprehensive LLM-based decision making
        return self.process_character_turn(character)
    else:
        # Use traditional strategy manager approach
        actions = self.strategy_manager.get_daily_actions(character)
        # Execute actions...
```

## Future Enhancements

### Planned Features

1. **Adaptive Thresholds**
   - Learn optimal thresholds from outcomes
   - Adjust based on character personality
   - Time-of-day dependent criteria

2. **Hybrid Decisions**
   - LLM for strategy, utility for tactics
   - LLM validates GOAP plans
   - Combined confidence scores

3. **Cost-Aware Routing**
   - Track LLM API costs
   - Budget-based decision limits
   - Quality vs. cost trade-offs

4. **Context Learning**
   - Learn which situations benefit from LLM
   - Improve novelty detection
   - Pattern recognition for routing

## Troubleshooting

### LLM Always Returns False

Check:
- `use_llm` flag is True
- `brain_io` and `output_interpreter` are initialized
- LLM model is loaded successfully

### Too Many LLM Calls

- Reduce `VARIETY_PROBABILITY`
- Increase thresholds
- Check for unintended crisis states

### Too Few LLM Calls

- Decrease thresholds
- Increase `VARIETY_PROBABILITY`
- Verify situation_context is passed correctly

### Analytics Show Unexpected Patterns

```python
# Debug decision history
for decision in strategy_manager._decision_history[-10:]:
    print(f"Character: {decision['character']}")
    print(f"Decision: {decision['decision_type']}")
    print(f"Reason: {decision['reason']}")
    print(f"Metadata: {decision}")
    print("---")
```

## References

- **Issue:** [#163 - Strategic LLM Calls](https://github.com/dhar174/tiny_village/issues/163)
- **Code:** `tiny_strategy_manager.py` lines 57-566
- **Tests:** `tests/test_strategic_llm_integration.py`
- **Related:** `tiny_brain_io.py`, `tiny_output_interpreter.py`, `tiny_goap_system.py`

## Contributing

When modifying the strategic LLM integration:

1. Update threshold constants with clear rationale
2. Add logging for new criteria
3. Update tests to cover new scenarios
4. Document changes in this guide
5. Update analytics to track new patterns

## License

Part of the Tiny Village project.

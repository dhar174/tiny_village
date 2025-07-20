# Phase 1: Core Decision Loop Implementation

## Overview
Implement the core AI decision-making loop that connects characters to the LLM system for intelligent behavior. This is the foundation for all character AI functionality in Tiny Village.

## Description
Currently, the main character AI decision cycle is disconnected from the LLM system, preventing characters from making intelligent decisions. This phase focuses on completing the output interpretation system and integrating LLM decision-making into the character turn processing.

## Phase Goals
- Complete the Output Interpreter system for parsing LLM responses
- Integrate LLM decision-making into the character turn cycle
- Establish the foundational decision loop: Character Turn → LLM Query → Action Execution

## Detailed Tasks

### 1.1 Complete Output Interpreter (`tiny_output_interpreter.py`)
**Estimated Time**: 2 days  
**Priority**: CRITICAL  
**Dependencies**: None

**Tasks**:
- [ ] Expand action mapping for all 50+ defined actions (currently only ~15% implemented)
- [ ] Implement parameter extraction from LLM responses
- [ ] Add validation and error handling for malformed outputs  
- [ ] Create fallback behaviors for unrecognized responses
- [ ] Add comprehensive logging for debugging

**Key Methods to Implement**:
```python
def parse_movement_action(self, llm_response: str) -> Dict
def parse_social_action(self, llm_response: str) -> Dict  
def parse_work_action(self, llm_response: str) -> Dict
def parse_creative_action(self, llm_response: str) -> Dict
def validate_action_parameters(self, action: Dict) -> bool
def handle_malformed_response(self, response: str) -> Dict
```

### 1.2 Integrate LLM Decision-Making (`tiny_gameplay_controller.py`)
**Estimated Time**: 3 days  
**Priority**: CRITICAL  
**Dependencies**: Task 1.1 Complete

**Tasks**:
- [ ] Connect StrategyManager to BrainIO for LLM queries
- [ ] Implement decision request formatting in PromptBuilder
- [ ] Route LLM responses through OutputInterpreter
- [ ] Handle decision failures and timeouts
- [ ] Add performance monitoring for decision latency

**Integration Points**:
- `process_character_turn()` → `strategy_manager.decide_action()`
- `decide_action()` → `brain_io.query_llm()`  
- LLM response → `output_interpreter.parse_response()`
- Parsed action → `execute_character_action()`

## Acceptance Criteria
- [ ] Characters can successfully query the LLM for decisions
- [ ] LLM responses are parsed into valid game actions >90% of the time
- [ ] Invalid or malformed responses are handled gracefully with fallbacks
- [ ] Decision latency is < 5 seconds per character turn
- [ ] Integration tests pass for complete decision cycle
- [ ] All action types have corresponding parser implementations
- [ ] Error handling prevents crashes from unexpected LLM outputs

## Technical Requirements

### Decision Loop Flow
```
Character Turn → StrategyManager.decide_action()
→ PromptBuilder.build_decision_prompt()  
→ BrainIO.query_llm()
→ OutputInterpreter.parse_response()
→ GameController.execute_action()
→ MemoryManager.store_experience()
```

### Error Handling Strategy
- **LLM Failures**: Fallback to rule-based decisions
- **Action Validation**: Reject invalid actions, request clarification
- **Timeout Handling**: Use cached decisions or default behaviors
- **Parsing Errors**: Log warnings, attempt recovery, use fallbacks

### Performance Targets
- Decision latency: < 5 seconds per character turn
- LLM response parsing success rate: > 90%
- Memory usage: Stable during extended sessions
- Error recovery: Graceful handling without crashes

## Testing Requirements
- [ ] Unit tests for all OutputInterpreter parsing methods
- [ ] Integration tests for complete decision cycle
- [ ] Error handling tests for malformed responses
- [ ] Performance tests for decision latency
- [ ] Stress tests with multiple characters

## Definition of Done
- [ ] All tasks completed and tested
- [ ] Integration tests pass
- [ ] Performance targets met
- [ ] Documentation updated
- [ ] Code reviewed and approved
- [ ] Demo scenario works with basic character decisions

## Impact
This phase unblocks the core AI functionality, enabling characters to make intelligent, context-aware decisions. Without this foundation, characters cannot exhibit the emergent behaviors that make Tiny Village compelling.

## Dependencies
- **Blocks**: All subsequent phases depend on this foundation
- **Requires**: Existing BrainIO and PromptBuilder systems
- **Enables**: GOAP planning, social interactions, storytelling

## Estimated Timeline
**Total**: 5 days (1 week)
- Task 1.1: 2 days  
- Task 1.2: 3 days

## Priority
**CRITICAL** - Blocks all other AI functionality
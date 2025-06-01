# Tiny Village Implementation Plan

## Executive Summary

Based on comprehensive analysis of the design documentation and existing codebase, this report identifies critical unimplemented features blocking a working demo and provides a fine-grained implementation plan. The architecture is well-designed with 10 modular components, but several key integration points and core systems remain unimplemented.

## Current State Analysis

### Implemented Components ✅
- **Basic Infrastructure**: Core classes for Characters, Buildings, Items, Locations
- **LLM Communication**: `tiny_brain_io.py` with Transformers/GGUF support
- **Prompt Generation**: `tiny_prompt_builder.py` with descriptor matrices
- **Memory System Foundation**: `tiny_memories.py` with FAISS indexing
- **Graph Manager**: `tiny_graph_manager.py` with NetworkX MultiDiGraph
- **Time Management**: `tiny_time_manager.py` with tick-based progression
- **Basic Action System**: Action/Condition/State classes in `actions.py`
- **Game Controller Shell**: `tiny_gameplay_controller.py` main loop structure

### Critical Gaps Blocking Demo 🚫

#### 1. **Decision-Making Loop Integration** (CRITICAL)
- **Status**: Main character AI decision cycle is disconnected from LLM system
- **Impact**: Characters cannot make intelligent decisions
- **Location**: `tiny_gameplay_controller.py` lines 1800-2000 (process_character_turn). ALSO, `prompt_builder.py` needs to be fully implemented for decision prompts.
- **Current Coverage**: Only basic action execution without LLM input

#### 2. **Output Interpretation System** (CRITICAL)
- **Status**: `tiny_output_interpreter.py` has basic structure but minimal action mapping
- **Impact**: LLM responses cannot be converted to game actions
- **Current Coverage**: Only ~15% of defined actions have interpretation logic

#### 3. **GOAP Planning Engine** (HIGH PRIORITY)
- **Status**: `tiny_goap_system.py` exists but not integrated with character decision-making
- **Impact**: No intelligent goal-driven behavior
- **Integration Point**: StrategyManager → GOAP → ActionSystem chain broken

#### 4. **Event-Driven Storytelling** (HIGH PRIORITY)
- **Status**: Event handler exists but storytelling system marked "NOT_STARTED"
- **Impact**: No dynamic narrative generation
- **Components**: Event triggers, story arc management, narrative coherence

#### 5. **Social Interaction System** (MEDIUM PRIORITY)
- **Status**: Social network in GraphManager but interaction logic missing
- **Impact**: Characters cannot form relationships or have conversations
- **Dependencies**: OutputInterpreter, MemoryManager integration

## Detailed Implementation Plan

### Phase 1: Core Decision Loop (Week 1) 🎯

#### 1.1 Complete Output Interpreter (`tiny_output_interpreter.py`)
**Estimated Time**: 2 days
**Dependencies**: None

**Tasks**:
- Expand action mapping for all 50+ defined actions
- Implement parameter extraction from LLM responses
- Add validation and error handling for malformed outputs
- Create fallback behaviors for unrecognized responses

**Key Methods to Implement**:
```python
def parse_movement_action(self, llm_response: str) -> Dict
def parse_social_action(self, llm_response: str) -> Dict
def parse_work_action(self, llm_response: str) -> Dict
def parse_creative_action(self, llm_response: str) -> Dict
def validate_action_parameters(self, action: Dict) -> bool
```

#### 1.2 Integrate LLM Decision-Making (`tiny_gameplay_controller.py`)
**Estimated Time**: 3 days
**Dependencies**: 1.1 Complete

**Tasks**:
- Connect StrategyManager to BrainIO for LLM queries
- Implement decision request formatting in PromptBuilder
- Route LLM responses through OutputInterpreter
- Handle decision failures and timeouts

**Integration Points**:
- `process_character_turn()` → `strategy_manager.decide_action()`
- `decide_action()` → `brain_io.query_llm()`
- LLM response → `output_interpreter.parse_response()`
- Parsed action → `execute_character_action()`

### Phase 2: GOAP Planning Integration (Week 2) 🧠

#### 2.1 Complete GOAP System (`tiny_goap_system.py`)
**Estimated Time**: 2 days
**Dependencies**: Phase 1 complete

**Tasks**:
- Implement A* pathfinding for action sequences
- Add goal priority weighting system
- Create action precondition validation
- Implement plan caching and revalidation

**Key Components**:
```python
class GOAPPlanner:
    def plan_actions(self, character: Character, goal: Goal) -> List[Action]
    def validate_plan(self, plan: List[Action], world_state: Dict) -> bool
    def replan_on_failure(self, character: Character, failed_action: Action) -> List[Action]
```

#### 2.2 Integrate GOAP with Strategy Manager
**Estimated Time**: 2 days
**Dependencies**: 2.1 Complete

**Tasks**:
- Connect StrategyManager to GOAPPlanner
- Implement goal selection logic based on character state
- Add plan execution monitoring
- Handle plan interruptions and replanning

### Phase 3: Event-Driven Storytelling (Week 3) 📖

#### 3.1 Implement Story Arc Management
**Estimated Time**: 3 days
**Dependencies**: Phase 2 complete

**Tasks**:
- Create story event detection system
- Implement narrative coherence tracking
- Add story beat generation
- Create character arc progression

**New Components**:
```python
class StoryManager:
    def detect_story_events(self, recent_actions: List[Action]) -> List[StoryEvent]
    def generate_narrative_beat(self, event: StoryEvent) -> str
    def update_character_arcs(self, characters: List[Character]) -> None
    def maintain_narrative_coherence(self) -> None
```

#### 3.2 Enhanced Event Processing
**Estimated Time**: 2 days
**Dependencies**: 3.1 Complete

**Tasks**:
- Extend EventHandler for story events
- Implement event propagation to MemoryManager
- Add event-based memory formation
- Create story-driven goal generation

### Phase 4: Social Systems (Week 4) 👥

#### 4.1 Social Interaction Engine
**Estimated Time**: 3 days
**Dependencies**: Phase 3 complete

**Tasks**:
- Implement conversation system
- Add relationship formation/decay mechanics
- Create social influence on decision-making
- Implement group behavior patterns

**Key Features**:
- Dynamic conversation topics based on shared memories
- Relationship strength affecting action choices
- Social roles influencing behavior patterns
- Group coordination for collective actions

#### 4.2 Advanced Social Behaviors
**Estimated Time**: 2 days
**Dependencies**: 4.1 Complete

**Tasks**:
- Implement conflict resolution mechanisms
- Add reputation and status systems
- Create social learning and adaptation
- Implement cultural transmission of knowledge

### Phase 5: Polish and Integration (Week 5) ✨

#### 5.1 System Integration Testing
**Estimated Time**: 2 days
**Dependencies**: All previous phases

**Tasks**:
- End-to-end integration testing
- Performance optimization
- Memory leak detection and fixing
- Error handling robustness

#### 5.2 Demo Scenario Implementation
**Estimated Time**: 3 days
**Dependencies**: 5.1 Complete

**Tasks**:
- Create compelling demo scenario
- Implement demo-specific content
- Add visualization and logging
- Create demo documentation

## Critical Implementation Details

### Decision Loop Flow
```
Character Turn → StrategyManager.decide_action()
→ PromptBuilder.build_decision_prompt()
→ BrainIO.query_llm()
→ OutputInterpreter.parse_response()
→ GOAPPlanner.validate_action()
→ GameController.execute_action()
→ MemoryManager.store_experience()
→ EventHandler.process_consequences()
```

### Error Handling Strategy
- **LLM Failures**: Fallback to rule-based decisions
- **Action Validation**: Reject invalid actions, request clarification
- **GOAP Planning**: Use cached plans or simplified goals
- **Memory Errors**: Log warnings, continue with partial data

### Performance Considerations
- **LLM Query Batching**: Process multiple characters simultaneously
- **Memory Optimization**: Implement memory cleanup cycles
- **GOAP Caching**: Cache and reuse valid action plans
- **Event Processing**: Use event queues to prevent blocking

## Testing Strategy

### Unit Tests Priority
1. **OutputInterpreter**: All action parsing methods
2. **GOAPPlanner**: Planning algorithm edge cases
3. **StrategyManager**: Decision-making logic
4. **MemoryManager**: Memory storage and retrieval
5. **EventHandler**: Event propagation and processing

### Integration Tests
1. **Complete Decision Cycle**: From character turn to action execution
2. **Memory Formation**: From action execution to memory storage
3. **Social Interactions**: Multi-character conversation scenarios
4. **Story Generation**: Event detection to narrative creation

### Demo Scenarios
1. **Basic Survival**: Characters finding food, shelter, rest
2. **Social Interaction**: Characters meeting, forming relationships
3. **Conflict Resolution**: Characters handling disagreements
4. **Creative Collaboration**: Characters working together on projects

## Risk Mitigation

### Technical Risks
- **LLM Response Quality**: Implement response validation and retry logic
- **Performance Degradation**: Monitor and optimize critical paths
- **Memory Consumption**: Implement memory management strategies
- **Integration Complexity**: Use staged integration approach

### Schedule Risks
- **Scope Creep**: Maintain strict focus on demo requirements
- **Technical Debt**: Allocate time for refactoring in each phase
- **Testing Overhead**: Integrate testing throughout development

## Success Metrics

### Demo Readiness Criteria
- ✅ Characters make intelligent, goal-driven decisions
- ✅ LLM responses are consistently parsed and executed
- ✅ Characters form and maintain relationships
- ✅ Emergent stories arise from character interactions
- ✅ System runs stably for 30+ minute demo sessions
- ✅ Actions feel natural and contextually appropriate

### Performance Targets
- **Decision Latency**: < 5 seconds per character turn
- **Memory Usage**: < 2GB for 10 characters over 1 hour
- **Success Rate**: > 90% of LLM responses successfully parsed
- **Uptime**: > 99% stability during demo sessions

## Conclusion

The Tiny Village architecture is fundamentally sound, but requires focused implementation of 5 critical systems to achieve demo readiness. The proposed 5-week plan prioritizes the most impactful components first, ensuring early wins while building toward full system integration.

**Immediate Next Steps**:
1. Begin Phase 1.1: Complete OutputInterpreter action mapping
2. Set up comprehensive testing framework
3. Create daily integration checkpoints
4. Establish performance monitoring baselines

The modular architecture and existing foundation provide a solid base for rapid development. With focused execution of this plan, a compelling AI village simulation demo is achievable within the 5-week timeline.

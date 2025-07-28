# Fallback System Architecture Design

## Current State Analysis

### Architecture Layers
1. **CLI Layer** (`cli_commands.py`, `cli.py`)
   - Entry point for `llm-orc invoke` commands
   - Handles output format selection (text, JSON, streaming)
   - Calls either `run_streaming_execution()` or `run_standard_execution()`

2. **Visualization Layer** (`cli_modules/utils/visualization.py`)
   - `run_streaming_execution()` - Rich console with live updates and event handling
   - `run_standard_execution()` - Basic execution without rich formatting
   - Event consumers that listen for structured events

3. **Execution Layer** (`core/execution/ensemble_execution.py`)
   - `EnsembleExecutor._execute_agent()` - Core agent execution logic
   - Currently has enhanced fallback handling that emits structured events
   - Should be the single source of truth for fallback decisions

4. **Model Layer** (`core/models/model_factory.py`)
   - `load_model_from_agent_config()` - Loads models from agent configurations
   - `load_model()` - Low-level model loading with authentication
   - Currently has competing fallback logic using `click.echo()`

### The Problem: Conflicting Fallback Systems

**Issue 1: Multiple Fallback Handlers**
- ModelFactory handles basic fallbacks with `click.echo()` (immediate display)
- EnsembleExecutor handles enhanced fallbacks with structured events (queued display)
- ModelFactory fallbacks prevent EnsembleExecutor fallbacks from ever being triggered

**Issue 2: Inconsistent User Experience**
- Some fallback messages show as immediate text output (ModelFactory)
- Other fallback messages should show as structured Rich output (EnsembleExecutor events)
- No coordination between the two systems

**Issue 3: Dual Event Systems Architecture Problem**
- Performance hooks system (`_emit_performance_event()`) - Used by EnsembleExecutor
- Streaming events system (`execute_streaming()` yields) - Used by CLI visualization  
- Events must be forwarded between systems (band-aid solution in StreamingProgressTracker)
- Creates complexity and potential for events to be lost or duplicated

**Issue 4: Output Format Inconsistency**
- Enhanced fallback events work with Rich/streaming output
- But not properly handled for JSON or plain text output formats
- User experience varies based on output format choice

## Ideal System Design

### Core Principles

1. **Single Responsibility**: Each layer has one clear job
   - ModelFactory: Load models OR throw exceptions (no fallback logic)
   - EnsembleExecutor: Handle ALL fallback logic and decisions
   - Visualization: Format and display fallback events

2. **Event-Driven Architecture**: All fallback information flows through structured events
   - Consistent data structure regardless of failure type (OAuth, model loading, runtime)
   - Events can be consumed by different output formatters

3. **Unified Event System**: Single event flow for all execution information  
   - Eliminate dual event systems (performance hooks + streaming events)
   - All events flow through the same streaming mechanism
   - Performance monitoring integrates with execution events seamlessly

4. **Output Format Agnostic**: Same fallback information works for all output types
   - Rich/streaming: Real-time visual feedback
   - JSON: Structured data with fallback events included
   - Text: Simple formatted messages

### Proposed Flow

```
User Request
    ↓
CLI Layer (determines output format)
    ↓
Visualization Layer (chooses appropriate execution method)
    ↓
EnsembleExecutor
    ├─ Load Agent Model
    │   ├─ ModelFactory.load_model() → Success ✓
    │   └─ ModelFactory.load_model() → Exception ✗
    │       └─ EnsembleExecutor handles fallback:
    │           ├─ Emit "agent_fallback_started" event
    │           ├─ Determine fallback model (configurable → system default)
    │           ├─ Load fallback model
    │           └─ Emit "agent_fallback_completed" or "agent_fallback_failed"
    └─ Execute Agent
        ├─ agent.respond_to_message() → Success ✓
        └─ agent.respond_to_message() → Exception ✗ (OAuth, runtime errors)
            └─ EnsembleExecutor handles fallback (same pattern as above)
```

### Event Structure

```yaml
agent_fallback_started:
  agent_name: "validator"
  failure_type: "model_loading" | "oauth_error" | "runtime_error" 
  original_model_profile: "premium-claude"
  original_error: "OAuth token refresh failed with status 400"
  fallback_model_profile: "standard-claude"  # if configurable fallback
  fallback_model_name: "llama3"              # actual model being used

agent_fallback_completed:
  agent_name: "validator"
  fallback_model_name: "llama3"
  response_preview: "First 100 chars of response..."

agent_fallback_failed:
  agent_name: "validator"
  fallback_error: "Even fallback model failed: connection timeout"
```

### Output Format Handling

**Streaming/Rich Mode** (`run_streaming_execution`)
- Real-time event display with Rich formatting
- Progress indicators and visual feedback
- Enhanced error messages with colors and symbols

**JSON Mode** 
- Include fallback events in response structure
- Machine-readable format for integration
- All event data preserved

**Text Mode**
- Simple formatted messages without Rich markup
- Fallback information included in plain text
- Suitable for scripts and basic terminals

## Unified Event System Architecture

### Current Dual System Problems
```
EnsembleExecutor → _emit_performance_event() → Performance Hooks → [Lost for CLI]
                                                                 ↓
StreamingProgressTracker → [Forward fallback events] → Streaming Events → CLI Visualization

EnsembleExecutor → execute_streaming() → Streaming Events → CLI Visualization  
                                      ↓
                   [Direct events: agent_started, agent_completed, execution_completed]
```

### Proposed Unified System
```
EnsembleExecutor → execute_streaming() → Single Event Stream → Output Formatters
                                      ↓                    ↓
                   [All events: fallback, progress, performance, completion]
                                                           ↓
                                              ┌─ Rich/Streaming Display
                                              ├─ JSON Output  
                                              └─ Text Output
```

### Benefits of Unified System
1. **Simplicity**: Single event flow eliminates forwarding complexity
2. **Consistency**: All events available to all output formats automatically  
3. **Performance**: No duplicate event processing or forwarding overhead
4. **Extensibility**: New event types automatically work across all formats
5. **Debugging**: Single event stream easier to trace and debug

### Migration Strategy
1. **Phase 1**: Move all `_emit_performance_event()` calls to direct streaming yields
2. **Phase 2**: Remove performance hooks system entirely
3. **Phase 3**: Enhance streaming event consumers for JSON/text output
4. **Phase 4**: Remove StreamingProgressTracker forwarding logic

## Implementation Strategy

### Phase 1: Clean Separation
1. Remove all fallback logic from ModelFactory
2. ModelFactory only loads models or throws exceptions
3. All fallback handling moves to EnsembleExecutor

### Phase 2: Enhanced Event System
1. Standardize event data structure
2. Add failure_type classification
3. Include preview information for success cases

### Phase 3: Output Format Support
1. Enhance JSON output to include events
2. Add plain text event formatting
3. Ensure consistent experience across all modes

### Phase 4: Testing & Validation
1. Test all failure scenarios (OAuth, model loading, runtime)
2. Verify output consistency across formats
3. Test configurable fallback chains and cycle detection

### Phase 5: Unify Event Systems (Architectural Improvement)
1. **Move performance events to streaming**: Replace all `_emit_performance_event()` calls with direct `yield` statements in `execute_streaming()`
2. **Remove performance hooks system**: Delete `_performance_hooks`, `register_performance_hook()`, `_emit_performance_event()`
3. **Simplify StreamingProgressTracker**: Remove event forwarding logic, focus on progress tracking only
4. **Enhance output formatters**: JSON and text outputs automatically get all events from unified stream
5. **Test unified architecture**: Verify all events work across all output formats seamlessly

## Success Criteria

✅ **Clear User Feedback**: Users always know which model profile failed and which fallback is used
✅ **Consistent Experience**: Same fallback information regardless of output format
✅ **Proper Error Context**: OAuth failures, model loading failures, and runtime failures all handled
✅ **Configurable Fallbacks**: Model profiles can specify custom fallback chains
✅ **Safety Features**: Cycle detection prevents infinite fallback loops

## Next Steps

1. Document detailed implementation plan for each phase
2. Start with Phase 1: Clean separation of concerns
3. Test each phase thoroughly before proceeding
4. Maintain backward compatibility where possible
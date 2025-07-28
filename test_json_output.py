#!/usr/bin/env python3
"""Quick test script for JSON output format validation."""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_orc.core.execution.ensemble_execution import EnsembleExecutor
from llm_orc.core.config.config_manager import ConfigurationManager


async def test_json_output():
    """Test JSON output format with fallback events."""
    config_manager = ConfigurationManager()
    executor = EnsembleExecutor()
    
    # Load the fallback demo ensemble
    ensemble_config = config_manager.load_ensemble("fallback-demo-ensemble")
    
    print("Testing JSON output with fallback events:")
    print("=" * 50)
    
    fallback_events_found = []
    event_count = 0
    
    async for event in executor.execute_streaming(ensemble_config, "Analyze this data"):
        event_count += 1
        print(f"Event {event_count}: {json.dumps(event, indent=2)}")
        
        # Check for fallback events
        if event.get("type", "").startswith("agent_fallback"):
            fallback_events_found.append(event)
            
        # Stop after a reasonable number of events
        if event_count >= 10:
            break
    
    print(f"\nSummary:")
    print(f"Total events processed: {event_count}")
    print(f"Fallback events found: {len(fallback_events_found)}")
    
    for i, event in enumerate(fallback_events_found):
        print(f"  Fallback Event {i+1}: {event['type']} for {event['data']['agent_name']}")


if __name__ == "__main__":
    asyncio.run(test_json_output())
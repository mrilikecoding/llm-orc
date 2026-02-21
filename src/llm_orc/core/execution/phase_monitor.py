"""Phase-level metrics monitoring for ensemble execution."""

import time
from collections.abc import Callable
from typing import Any

from llm_orc.core.execution.agent_resource_monitor import AgentResourceMonitor
from llm_orc.schemas.agent_config import AgentConfig


class PhaseMonitor:
    """Manages phase-level metrics collection during ensemble execution.

    Encapsulates the start/stop monitoring lifecycle for execution phases,
    collecting CPU, memory, and timing metrics via AgentResourceMonitor's monitor.
    """

    def __init__(
        self,
        agent_executor: AgentResourceMonitor,
        emit_event_fn: Callable[[str, dict[str, Any]], None],
    ) -> None:
        self._agent_executor = agent_executor
        self._emit_event = emit_event_fn

    async def start(self, phase_index: int, phase_agents: list[AgentConfig]) -> None:
        """Start monitoring for a specific phase."""
        phase_name = f"phase_{phase_index}"
        agent_names = [agent.name for agent in phase_agents]

        phase_metrics = await self._agent_executor.monitor.collect_phase_metrics(
            phase_index=phase_index,
            phase_name=phase_name,
            agent_count=len(phase_agents),
        )

        phase_metrics.update(
            {
                "agent_names": agent_names,
                "start_time": time.time(),
            }
        )

        self._agent_executor._phase_metrics.append(phase_metrics)

        await self._agent_executor.monitor.start_execution_monitoring()

        self._emit_event(
            "phase_monitoring_started",
            {
                "phase_index": phase_index,
                "agent_count": len(phase_agents),
                "agent_names": agent_names,
            },
        )

    async def stop(
        self,
        phase_index: int,
        phase_agents: list[AgentConfig],
        duration: float,
    ) -> None:
        """Stop monitoring for a specific phase and collect final metrics."""
        phase_metrics = None
        for metrics in self._agent_executor._phase_metrics:
            if metrics.get("phase_index") == phase_index:
                phase_metrics = metrics
                break

        if phase_metrics:
            try:
                phase_execution_metrics = await (
                    self._agent_executor.monitor.stop_execution_monitoring()
                )

                phase_metrics.update(
                    {
                        "duration_seconds": duration,
                        "end_time": time.time(),
                        "agents_completed": len(phase_agents),
                        "peak_cpu": phase_execution_metrics.get("peak_cpu", 0.0),
                        "avg_cpu": phase_execution_metrics.get("avg_cpu", 0.0),
                        "peak_memory": phase_execution_metrics.get("peak_memory", 0.0),
                        "avg_memory": phase_execution_metrics.get("avg_memory", 0.0),
                        "sample_count": phase_execution_metrics.get("sample_count", 0),
                    }
                )
            except Exception:
                try:
                    current_metrics = await (
                        self._agent_executor.monitor.get_current_metrics()
                    )
                    phase_metrics.update(
                        {
                            "duration_seconds": duration,
                            "end_time": time.time(),
                            "agents_completed": len(phase_agents),
                            "final_cpu_percent": current_metrics.get(
                                "cpu_percent", 0.0
                            ),
                            "final_memory_percent": current_metrics.get(
                                "memory_percent", 0.0
                            ),
                        }
                    )
                except Exception:
                    phase_metrics.update(
                        {
                            "duration_seconds": duration,
                            "end_time": time.time(),
                            "agents_completed": len(phase_agents),
                        }
                    )

        self._emit_event(
            "phase_monitoring_stopped",
            {
                "phase_index": phase_index,
                "duration_seconds": duration,
                "agent_count": len(phase_agents),
            },
        )

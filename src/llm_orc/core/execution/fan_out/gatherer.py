"""Fan-out result gathering for parallel agent execution (issue #73)."""

from typing import Any

from llm_orc.core.execution.fan_out.expander import FanOutExpander


class FanOutGatherer:
    """Gathers results from fan-out instances into ordered arrays."""

    def __init__(self, expander: FanOutExpander) -> None:
        """Initialize gatherer with expander for instance name parsing.

        Args:
            expander: FanOutExpander instance for parsing instance names
        """
        self._expander = expander
        # Map: original_agent_name -> {index: (result, success, error)}
        self._results: dict[str, dict[int, tuple[Any, bool, str | None]]] = {}

    def record_instance_result(
        self,
        instance_name: str,
        result: Any,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Record result from a single fan-out instance.

        Args:
            instance_name: Instance name like 'extractor[0]'
            result: Result from the instance execution
            success: Whether the execution succeeded
            error: Error message if failed
        """
        original_name = self._expander.get_original_agent_name(instance_name)
        index = self._expander.get_instance_index(instance_name)

        if index is None:
            # Not a fan-out instance, treat as single result at index 0
            index = 0

        if original_name not in self._results:
            self._results[original_name] = {}

        self._results[original_name][index] = (result, success, error)

    def has_instance_result(self, instance_name: str) -> bool:
        """Check if a result has been recorded for an instance.

        Args:
            instance_name: Instance name like 'extractor[0]'

        Returns:
            True if result recorded, False otherwise
        """
        original_name = self._expander.get_original_agent_name(instance_name)
        index = self._expander.get_instance_index(instance_name)

        if index is None:
            index = 0

        return original_name in self._results and index in self._results[original_name]

    def gather_results(self, original_agent_name: str) -> dict[str, Any]:
        """Gather all instance results for an original agent.

        Args:
            original_agent_name: Original agent name (e.g., 'extractor')

        Returns:
            Dict with:
            - response: Ordered list of results (None for failed instances)
            - status: 'success' | 'partial' | 'failed'
            - fan_out: True (marker for fan-out result)
            - instances: Per-instance status info
        """
        if original_agent_name not in self._results:
            return {
                "response": [],
                "status": "success",
                "fan_out": True,
                "instances": [],
            }

        instance_data = self._results[original_agent_name]

        # Sort by index
        sorted_indices = sorted(instance_data.keys())

        response: list[Any] = []
        instances: list[dict[str, Any]] = []
        success_count = 0
        fail_count = 0

        for index in sorted_indices:
            result, success, error = instance_data[index]

            if success:
                response.append(result)
                success_count += 1
                instances.append({"index": index, "status": "success"})
            else:
                response.append(None)
                fail_count += 1
                instances.append(
                    {
                        "index": index,
                        "status": "failed",
                        "error": error,
                    }
                )

        # Determine overall status
        if fail_count == 0:
            status = "success"
        elif success_count == 0:
            status = "failed"
        else:
            status = "partial"

        return {
            "response": response,
            "status": status,
            "fan_out": True,
            "instances": instances,
        }

    def get_error_summary(self, original_agent_name: str) -> dict[str, Any]:
        """Get error details for failed instances.

        Args:
            original_agent_name: Original agent name

        Returns:
            Dict with total_instances, failed_count, success_count, errors
        """
        if original_agent_name not in self._results:
            return {
                "total_instances": 0,
                "failed_count": 0,
                "success_count": 0,
                "errors": [],
            }

        instance_data = self._results[original_agent_name]
        errors: list[dict[str, Any]] = []
        success_count = 0
        fail_count = 0

        for index, (_, success, error) in instance_data.items():
            if success:
                success_count += 1
            else:
                fail_count += 1
                errors.append({"index": index, "error": error})

        return {
            "total_instances": len(instance_data),
            "failed_count": fail_count,
            "success_count": success_count,
            "errors": errors,
        }

    def has_pending_instances(
        self, original_agent_name: str, expected_count: int
    ) -> bool:
        """Check if any instances are still pending.

        Args:
            original_agent_name: Original agent name
            expected_count: Expected total number of instances

        Returns:
            True if not all instances have been recorded
        """
        recorded = self.get_recorded_count(original_agent_name)
        return recorded < expected_count

    def get_recorded_count(self, original_agent_name: str) -> int:
        """Get number of recorded instances for an agent.

        Args:
            original_agent_name: Original agent name

        Returns:
            Count of recorded instances
        """
        if original_agent_name not in self._results:
            return 0
        return len(self._results[original_agent_name])

    def clear(self, original_agent_name: str) -> None:
        """Clear all recorded results for an agent.

        Args:
            original_agent_name: Original agent name to clear
        """
        if original_agent_name in self._results:
            del self._results[original_agent_name]

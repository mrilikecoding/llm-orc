"""Benchmarking framework for comparing async vs threading approaches for agent parallelism.

This module implements the research investigation described in Issue #43.
"""

import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable
from dataclasses import dataclass
from collections.abc import Awaitable

import psutil


@dataclass
class PerformanceMetrics:
    """Performance metrics for execution comparison."""
    
    total_execution_time: float
    agent_execution_times: list[float]
    memory_usage: float  # MB
    cpu_utilization: float  # percentage
    error_count: int
    cancellation_responsiveness: float | None = None


class MockAgent:
    """Mock agent for benchmarking different execution approaches."""
    
    def __init__(self, name: str, simulated_latency: float = 1.0) -> None:
        """Initialize mock agent with simulated API latency."""
        self.name = name
        self.simulated_latency = simulated_latency
        self.execution_count = 0
    
    async def execute_async(self, input_data: str) -> str:
        """Async execution with simulated I/O bound API call."""
        start_time = time.time()
        self.execution_count += 1
        
        # Simulate LLM API call latency (I/O bound)
        await asyncio.sleep(self.simulated_latency)
        
        execution_time = time.time() - start_time
        return f"Agent {self.name} response (took {execution_time:.2f}s)"
    
    def execute_sync(self, input_data: str) -> str:
        """Synchronous execution with simulated I/O bound API call."""
        start_time = time.time()
        self.execution_count += 1
        
        # Simulate LLM API call latency (I/O bound) 
        time.sleep(self.simulated_latency)
        
        execution_time = time.time() - start_time
        return f"Agent {self.name} response (took {execution_time:.2f}s)"


class ParallelizationBenchmark:
    """Benchmark framework for comparing execution approaches."""
    
    def __init__(self) -> None:
        """Initialize benchmarking framework."""
        self.process = psutil.Process()
    
    def _measure_resources_start(self) -> dict[str, float]:
        """Capture initial resource measurements."""
        return {
            "memory": self.process.memory_info().rss / 1024 / 1024,  # MB
            "cpu_percent": self.process.cpu_percent(),
        }
    
    def _measure_resources_end(self, start_resources: dict[str, float]) -> dict[str, float]:
        """Capture final resource measurements and calculate differences."""
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        end_cpu = self.process.cpu_percent()
        
        return {
            "memory_usage": end_memory - start_resources["memory"],
            "cpu_utilization": max(end_cpu, start_resources["cpu_percent"]),
        }
    
    async def benchmark_current_async_sequential(
        self, agents: list[MockAgent], input_data: str
    ) -> PerformanceMetrics:
        """Benchmark current async approach (sequential execution within phases)."""
        start_resources = self._measure_resources_start()
        start_time = time.time()
        
        agent_times: list[float] = []
        error_count = 0
        
        # Current implementation: execute agents sequentially
        for agent in agents:
            try:
                agent_start = time.time()
                await agent.execute_async(input_data)
                agent_duration = time.time() - agent_start
                agent_times.append(agent_duration)
            except Exception:
                error_count += 1
                agent_times.append(0.0)
        
        total_time = time.time() - start_time
        end_resources = self._measure_resources_end(start_resources)
        
        return PerformanceMetrics(
            total_execution_time=total_time,
            agent_execution_times=agent_times,
            memory_usage=end_resources["memory_usage"],
            cpu_utilization=end_resources["cpu_utilization"],
            error_count=error_count,
        )
    
    async def benchmark_async_parallel(
        self, agents: list[MockAgent], input_data: str
    ) -> PerformanceMetrics:
        """Benchmark async approach with true parallelization using asyncio.gather."""
        start_resources = self._measure_resources_start()
        start_time = time.time()
        
        # Create tasks for parallel execution
        tasks = [agent.execute_async(input_data) for agent in agents]
        
        # Track individual agent execution times
        agent_start_times = [time.time() for _ in agents]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            agent_times = []
            error_count = 0
            
            for i, result in enumerate(results):
                agent_duration = time.time() - agent_start_times[i]
                if isinstance(result, Exception):
                    error_count += 1
                    agent_times.append(0.0)
                else:
                    agent_times.append(agent_duration)
            
        except Exception:
            error_count = len(agents)
            agent_times = [0.0] * len(agents)
        
        total_time = time.time() - start_time
        end_resources = self._measure_resources_end(start_resources)
        
        return PerformanceMetrics(
            total_execution_time=total_time,
            agent_execution_times=agent_times,
            memory_usage=end_resources["memory_usage"],
            cpu_utilization=end_resources["cpu_utilization"],
            error_count=error_count,
        )
    
    def benchmark_threading(
        self, agents: list[MockAgent], input_data: str
    ) -> PerformanceMetrics:
        """Benchmark multi-threading approach with ThreadPoolExecutor."""
        start_resources = self._measure_resources_start()
        start_time = time.time()
        
        agent_times: list[float] = []
        error_count = 0
        
        with ThreadPoolExecutor(max_workers=len(agents)) as executor:
            # Submit all tasks
            future_to_agent = {
                executor.submit(agent.execute_sync, input_data): agent 
                for agent in agents
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_agent):
                agent_start = time.time()
                try:
                    result = future.result()
                    agent_duration = time.time() - agent_start
                    agent_times.append(agent_duration)
                except Exception:
                    error_count += 1
                    agent_times.append(0.0)
        
        total_time = time.time() - start_time
        end_resources = self._measure_resources_end(start_resources)
        
        return PerformanceMetrics(
            total_execution_time=total_time,
            agent_execution_times=agent_times,
            memory_usage=end_resources["memory_usage"],
            cpu_utilization=end_resources["cpu_utilization"],
            error_count=error_count,
        )
    
    async def benchmark_hybrid_async_threading(
        self, agents: list[MockAgent], input_data: str
    ) -> PerformanceMetrics:
        """Benchmark hybrid approach: async coordination with thread execution."""
        start_resources = self._measure_resources_start()
        start_time = time.time()
        
        loop = asyncio.get_event_loop()
        agent_times: list[float] = []
        error_count = 0
        
        with ThreadPoolExecutor(max_workers=len(agents)) as executor:
            # Create async tasks that run sync functions in threads
            tasks = [
                loop.run_in_executor(executor, agent.execute_sync, input_data)
                for agent in agents
            ]
            
            agent_start_times = [time.time() for _ in agents]
            
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(results):
                    agent_duration = time.time() - agent_start_times[i]
                    if isinstance(result, Exception):
                        error_count += 1
                        agent_times.append(0.0)
                    else:
                        agent_times.append(agent_duration)
                        
            except Exception:
                error_count = len(agents)
                agent_times = [0.0] * len(agents)
        
        total_time = time.time() - start_time
        end_resources = self._measure_resources_end(start_resources)
        
        return PerformanceMetrics(
            total_execution_time=total_time,
            agent_execution_times=agent_times,
            memory_usage=end_resources["memory_usage"],
            cpu_utilization=end_resources["cpu_utilization"],
            error_count=error_count,
        )


@dataclass
class BenchmarkScenario:
    """Configuration for a benchmark scenario."""
    
    name: str
    agent_count: int
    simulated_latency: float  # seconds per agent
    description: str


class ParallelizationStudy:
    """Main study coordinator for Issue #43 investigation."""
    
    def __init__(self) -> None:
        """Initialize the parallelization study."""
        self.benchmark = ParallelizationBenchmark()
        self.scenarios = [
            BenchmarkScenario(
                name="small_fast",
                agent_count=3,
                simulated_latency=0.5,
                description="3 agents with 0.5s latency each",
            ),
            BenchmarkScenario(
                name="medium_typical",
                agent_count=5,
                simulated_latency=1.0,
                description="5 agents with 1.0s latency each",
            ),
            BenchmarkScenario(
                name="large_slow",
                agent_count=10,
                simulated_latency=1.5,
                description="10 agents with 1.5s latency each",
            ),
            BenchmarkScenario(
                name="stress_test",
                agent_count=15,
                simulated_latency=2.0,
                description="15 agents with 2.0s latency each",
            ),
        ]
    
    def create_agents(self, scenario: BenchmarkScenario) -> list[MockAgent]:
        """Create mock agents for a scenario."""
        return [
            MockAgent(f"agent_{i}", scenario.simulated_latency)
            for i in range(scenario.agent_count)
        ]
    
    async def run_scenario_comparison(
        self, scenario: BenchmarkScenario
    ) -> dict[str, PerformanceMetrics]:
        """Run all approaches for a single scenario."""
        print(f"\n🧪 Running scenario: {scenario.name}")
        print(f"   Description: {scenario.description}")
        
        input_data = "Test input for performance comparison"
        results: dict[str, PerformanceMetrics] = {}
        
        # Test 1: Current async sequential approach
        print("   ⏳ Testing current async sequential...")
        agents = self.create_agents(scenario)
        results["current_async_sequential"] = (
            await self.benchmark.benchmark_current_async_sequential(agents, input_data)
        )
        
        # Test 2: Async parallel approach
        print("   ⏳ Testing async parallel...")
        agents = self.create_agents(scenario)
        results["async_parallel"] = await self.benchmark.benchmark_async_parallel(
            agents, input_data
        )
        
        # Test 3: Threading approach
        print("   ⏳ Testing threading...")
        agents = self.create_agents(scenario)
        results["threading"] = self.benchmark.benchmark_threading(agents, input_data)
        
        # Test 4: Hybrid async + threading approach
        print("   ⏳ Testing hybrid async + threading...")
        agents = self.create_agents(scenario)
        results["hybrid_async_threading"] = (
            await self.benchmark.benchmark_hybrid_async_threading(agents, input_data)
        )
        
        return results
    
    def print_scenario_results(
        self, scenario: BenchmarkScenario, results: dict[str, PerformanceMetrics]
    ) -> None:
        """Print results for a scenario in a readable format."""
        print(f"\n📊 Results for {scenario.name}:")
        print(f"   Expected sequential time: {scenario.agent_count * scenario.simulated_latency:.2f}s")
        print(f"   Expected parallel time: ~{scenario.simulated_latency:.2f}s")
        print()
        
        for approach, metrics in results.items():
            efficiency = (
                (scenario.agent_count * scenario.simulated_latency) 
                / metrics.total_execution_time * 100
            )
            print(f"   {approach:25} | "
                  f"Time: {metrics.total_execution_time:6.2f}s | "
                  f"Efficiency: {efficiency:5.1f}% | "
                  f"Memory: {metrics.memory_usage:5.1f}MB | "
                  f"CPU: {metrics.cpu_utilization:5.1f}%")
        
        # Find the fastest approach
        fastest = min(results.items(), key=lambda x: x[1].total_execution_time)
        print(f"   🏆 Fastest: {fastest[0]} ({fastest[1].total_execution_time:.2f}s)")
    
    async def run_full_study(self) -> dict[str, dict[str, PerformanceMetrics]]:
        """Run the complete parallelization study."""
        print("🚀 Starting Parallelization Study (Issue #43)")
        print("=" * 60)
        
        all_results: dict[str, dict[str, PerformanceMetrics]] = {}
        
        for scenario in self.scenarios:
            scenario_results = await self.run_scenario_comparison(scenario)
            all_results[scenario.name] = scenario_results
            self.print_scenario_results(scenario, scenario_results)
        
        return all_results
    
    def generate_summary_report(
        self, all_results: dict[str, dict[str, PerformanceMetrics]]
    ) -> str:
        """Generate a summary report of all results."""
        report = []
        report.append("# Parallelization Study Results (Issue #43)")
        report.append("")
        report.append("## Executive Summary")
        report.append("")
        
        # Calculate overall performance across scenarios
        approach_totals: dict[str, list[float]] = {}
        
        for scenario_name, scenario_results in all_results.items():
            for approach, metrics in scenario_results.items():
                if approach not in approach_totals:
                    approach_totals[approach] = []
                approach_totals[approach].append(metrics.total_execution_time)
        
        # Find best approach overall
        avg_times = {
            approach: sum(times) / len(times) 
            for approach, times in approach_totals.items()
        }
        best_approach = min(avg_times.items(), key=lambda x: x[1])
        
        report.append(f"**Best overall approach: {best_approach[0]}** "
                     f"(avg: {best_approach[1]:.2f}s)")
        report.append("")
        
        # Detailed scenario breakdown
        report.append("## Detailed Results by Scenario")
        report.append("")
        
        for scenario_name, scenario_results in all_results.items():
            scenario = next(s for s in self.scenarios if s.name == scenario_name)
            report.append(f"### {scenario.name}")
            report.append(f"*{scenario.description}*")
            report.append("")
            report.append("| Approach | Time (s) | Memory (MB) | CPU (%) | Efficiency (%) |")
            report.append("|----------|----------|-------------|---------|----------------|")
            
            for approach, metrics in scenario_results.items():
                efficiency = (
                    (scenario.agent_count * scenario.simulated_latency) 
                    / metrics.total_execution_time * 100
                )
                report.append(
                    f"| {approach} | {metrics.total_execution_time:.2f} | "
                    f"{metrics.memory_usage:.1f} | {metrics.cpu_utilization:.1f} | "
                    f"{efficiency:.1f} |"
                )
            report.append("")
        
        return "\n".join(report)


# Main execution function for running the study
async def main() -> None:
    """Run the parallelization study."""
    study = ParallelizationStudy()
    results = await study.run_full_study()
    
    print("\n" + "=" * 60)
    print("📝 Generating summary report...")
    
    report = study.generate_summary_report(results)
    
    # Save report to file
    report_path = "/Users/nathangreen/Development/eddi-lab/llm-orc/benchmarks/parallelization_study_results.md"
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"📄 Report saved to: {report_path}")
    print("\n" + "=" * 60)
    print("✅ Study complete!")


if __name__ == "__main__":
    asyncio.run(main())
# rag_generation.py
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class QueryComponents:
    performance_metrics: str
    configuration_context: str
    problem_focus: str

class RAGQueryGenerator:
    def __init__(self):
        self.performance_thresholds = {
            "tps": {
                "low": 50,
                "medium": 100,
                "high": 200
            },
            "latency": {
                "low": 100,
                "medium": 500,
                "high": 1000
            },
            "timeouts": {
                "low": 1,
                "medium": 5,
                "high": 10
            }
        }

    def analyze_metrics(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Analyze metrics to determine their severity levels."""
        analysis = {}
        
        # Analyze TPS
        if metrics["tps"] < self.performance_thresholds["tps"]["low"]:
            analysis["tps"] = "critically low"
        elif metrics["tps"] < self.performance_thresholds["tps"]["medium"]:
            analysis["tps"] = "low"
        elif metrics["tps"] > self.performance_thresholds["tps"]["high"]:
            analysis["tps"] = "high"
        else:
            analysis["tps"] = "moderate"

        # Analyze Latency
        if metrics["latency"] > self.performance_thresholds["latency"]["high"]:
            analysis["latency"] = "critically high"
        elif metrics["latency"] > self.performance_thresholds["latency"]["medium"]:
            analysis["latency"] = "high"
        elif metrics["latency"] < self.performance_thresholds["latency"]["low"]:
            analysis["latency"] = "low"
        else:
            analysis["latency"] = "moderate"

        # Analyze Timeouts
        if metrics["timeouts"] > self.performance_thresholds["timeouts"]["high"]:
            analysis["timeouts"] = "critically high"
        elif metrics["timeouts"] > self.performance_thresholds["timeouts"]["medium"]:
            analysis["timeouts"] = "high"
        else:
            analysis["timeouts"] = "normal"

        return analysis

    def generate_performance_query(self, metrics: Dict[str, Any]) -> str:
        """Generate a query focused on performance metrics."""
        analysis = self.analyze_metrics(metrics)
        
        query_components = []
        
        if analysis["tps"] in ["critically low", "low"]:
            query_components.append(f"how to improve low TPS ({metrics['tps']})")
        
        if analysis["latency"] in ["critically high", "high"]:
            query_components.append(f"reduce high latency ({metrics['latency']}ms)")
            
        if analysis["timeouts"] in ["critically high", "high"]:
            query_components.append(f"handle frequent timeouts ({metrics['timeouts']} occurrences)")
            
        if metrics["block_size"] < 50:
            query_components.append("optimize small block size configuration")
            
        base_query = "Hyperledger Fabric performance optimization for "
        return base_query + " and ".join(query_components)

    def generate_config_query(self, current_config: Dict[str, Any], problem_areas: List[str]) -> str:
        """Generate a query focused on configuration improvements."""
        query_components = []
        
        for area in problem_areas:
            if area == "block_size":
                query_components.append(f"optimal block size configuration (current: {current_config.get('block_size')})")
            elif area == "consensus":
                query_components.append(f"consensus optimization for {current_config.get('consensus_type', 'unknown')}")
            elif area == "endorsement":
                query_components.append("endorsement policy performance impact")
                
        base_query = "Hyperledger Fabric configuration recommendations for "
        return base_query + " and ".join(query_components)

    def generate_resource_query(self, metrics: Dict[str, Any], current_config: Dict[str, Any]) -> str:
        """Generate a query focused on resource utilization."""
        components = []
        
        if metrics.get("cpu_utilization", 0) > 80:
            components.append("high CPU utilization")
        if metrics.get("memory_utilization", 0) > 80:
            components.append("high memory usage")
        if metrics.get("disk_latency", 0) > 100:
            components.append("disk I/O bottlenecks")
            
        base_query = "Hyperledger Fabric resource optimization for "
        return base_query + " and ".join(components) if components else ""

    def generate_queries(self, metrics: Dict[str, Any], problem_areas: List[str]) -> List[str]:
        """Generate multiple specialized queries for different aspects of the system."""
        queries = []
        
        # Add performance query
        perf_query = self.generate_performance_query(metrics)
        if perf_query:
            queries.append(perf_query)
            
        # Add configuration query
        config_query = self.generate_config_query(metrics["current_config"], problem_areas)
        if config_query:
            queries.append(config_query)
            
        # Add resource query
        resource_query = self.generate_resource_query(metrics, metrics["current_config"])
        if resource_query:
            queries.append(resource_query)
            
        return queries
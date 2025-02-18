# system_info_acquisition_module.py
from typing import Dict
import json
from system_metrics import SystemMetrics

class SystemInfoAcquisitionModule:
    def __init__(self):
        self.metrics_schema = {
            "tps": float,
            "latency": float,
            "block_size": int,
            "concurrent_requests": int,
            "timeouts": int,
            "current_config": dict
        }

    def validate_metrics(self, raw_metrics: Dict) -> bool:
        """Validate if the input metrics match the required schema."""
        try:
            for key, type_class in self.metrics_schema.items():
                if key not in raw_metrics:
                    return False
                if not isinstance(raw_metrics[key], type_class):
                    return False
            return True
        except Exception:
            return False

    def extract_metrics(self, raw_metrics: Dict) -> SystemMetrics:
        """Extract and transform relevant metrics from raw data."""
        if not self.validate_metrics(raw_metrics):
            raise ValueError("Invalid metrics format")

        return SystemMetrics(
            tps=raw_metrics["tps"],
            latency=raw_metrics["latency"],
            block_size=raw_metrics["block_size"],
            concurrent_requests=raw_metrics["concurrent_requests"],
            timeouts=raw_metrics["timeouts"],
            current_config=raw_metrics["current_config"]
        )

    def process(self, raw_metrics: Dict) -> Dict:
        """Process metrics and return in a standardized format."""
        try:
            metrics = self.extract_metrics(raw_metrics)
            return {
                "status": "success",
                "metrics": {
                    "tps": metrics.tps,
                    "latency": metrics.latency,
                    "block_size": metrics.block_size,
                    "concurrent_requests": metrics.concurrent_requests,
                    "timeouts": metrics.timeouts,
                    "current_config": metrics.current_config
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

if __name__ == "__main__":
    # Example usage
    example_metrics = {
        "tps": 100.5,
        "latency": 250.0,
        "block_size": 10,
        "concurrent_requests": 1000,
        "timeouts": 5,
        "current_config": {
            "block_size": 10,
            "consensus_type": "kafka",
            "endorsement_policy": "AND('Org1.member', 'Org2.member')"
        }
    }
    
    module = SystemInfoAcquisitionModule()
    result = module.process(example_metrics)
    print(json.dumps(result, indent=2))
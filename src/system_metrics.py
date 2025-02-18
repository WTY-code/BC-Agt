# system_metrics.py
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class SystemMetrics:
    tps: float
    latency: float
    block_size: int
    concurrent_requests: int
    timeouts: int
    current_config: Dict[str, Any]